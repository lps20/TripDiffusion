import ast
import torch
import torch.nn.functional as F
import pandas as pd
import logging
from tqdm import tqdm

def generate_synthetic_trips(num_samples):
    """
    Randomly generate synthetic trip data for training.
    """
    data = []
    for _ in range(num_samples):
        start_type = torch.randint(0, 5, (1,))
        start_zcode_num = torch.randint(0, 9, (1,))
        act_num = torch.randint(0, 9, (1,))
        mode_num = torch.randint(0, 5, (1,))
        end_zcode_num = torch.randint(0, 77, (1,))
        start_time_num_6 = torch.randint(0, 241, (1,))
        trip_time_num_6 = torch.randint(0, 241, (1,))
        x0 = torch.tensor([start_type, start_zcode_num, act_num, mode_num, end_zcode_num, 
                           start_time_num_6, trip_time_num_6], dtype=torch.long).flatten()
        relation = torch.randint(0, 3, (1,))
        sex = torch.randint(0, 2, (1,))
        age_code = torch.randint(0, 5, (1,))
        job_type = torch.randint(0, 4, (1,))
        cond = torch.tensor([relation, sex, age_code, job_type], dtype=torch.long).flatten()
        data.append((x0, cond))
    return data

def load_data(file_path, features_info, cond_info):
    """
    Load trip data from a CSV file.
    CSV file should contain the following columns:
      - Trip features: start_type, start_zcode_num, act_num, mode_num, end_zcode_num,
                    start_time_num_6, trip_time_num_6
      - Conditional features: relation, sex, age_code, job_type
    """
    df = pd.read_csv(file_path)
    data_features = [feat["name"] for feat in features_info]
    data_cond = [cond["name"] for cond in cond_info]
    data = []

    for _, row in df.iterrows():
        x0 = torch.tensor([row[feature] for feature in data_features], dtype=torch.long).flatten()
        cond = torch.tensor([row[feature] for feature in data_cond], dtype=torch.long).flatten()
        data.append((x0, cond))

    return data


def _resolve_causal_weights(causal_weight):
    defaults = {'st': 1.0, 'mode': 1.0}
    if causal_weight is None:
        return defaults
    if isinstance(causal_weight, str):
        causal_weight = ast.literal_eval(causal_weight)
    if not isinstance(causal_weight, dict):
        raise ValueError("causal_weight must be a dict like {'st': 1.0, 'mode': 1.0}")

    resolved = defaults.copy()
    for key in defaults:
        if key in causal_weight:
            resolved[key] = float(causal_weight[key])
    return resolved


def _build_batch_sampling_weights(dataset, features_info, sampling_feature, sampling_power):
    feature_to_idx = {feat["name"]: i for i, feat in enumerate(features_info)}
    if sampling_feature not in feature_to_idx:
        raise ValueError(f"sampling_feature '{sampling_feature}' not found in features_info")
    feat_idx = feature_to_idx[sampling_feature]

    feat_values = torch.tensor([int(item[0][feat_idx]) for item in dataset], dtype=torch.long)
    unique_vals, counts = torch.unique(feat_values, return_counts=True)
    count_map = {int(v.item()): int(c.item()) for v, c in zip(unique_vals, counts)}

    sample_weights = []
    for v in feat_values.tolist():
        freq = max(count_map[v], 1)
        sample_weights.append((1.0 / freq) ** sampling_power)
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    sample_weights = sample_weights / sample_weights.sum().clamp(min=1e-12)
    return sample_weights


def _build_timestep_probs(T, t_sampling):
    if t_sampling == 'uniform':
        return None
    steps = torch.arange(1, T + 1, dtype=torch.float32)
    if t_sampling == 'sqrt':
        weights = torch.sqrt(steps)
    elif t_sampling == 'late':
        weights = steps
    else:
        raise ValueError(f"Unknown t_sampling strategy: {t_sampling}")
    return weights / weights.sum()


def train_model(model, optimizer, dataset, features_info, 
                lambda_weight, lambda_joint, T, epochs, batch_size, device,
                loss_type='standard',causal_weight=None,
                model_save_path=None, patience=10, min_delta=1e-4,
                batch_sampling='sequential', sampling_feature='act_num',
                sampling_power=1.0, t_sampling='uniform'):
    """
    Model training process:
      - For each batch, do the diffusion process based on random step t.
      - Forword propagate and compute the entropy loss (CE loss and VB loss).
      - Backpropagate and update the model parameters.
    """
    logger = logging.getLogger(__name__)
    
    if hasattr(model, 'module'):
        attr_model = model.module
    else:
        attr_model = model

    model.train()

    # --- Early Stopping Variables ---
    best_loss = float('inf')
    patience_counter = 0

    valid_batch_sampling = {'sequential', 'shuffle', 'balanced'}
    if batch_sampling not in valid_batch_sampling:
        raise ValueError(f"batch_sampling must be one of {valid_batch_sampling}, got '{batch_sampling}'")

    causal_weights = _resolve_causal_weights(causal_weight)
    t_probs = _build_timestep_probs(T, t_sampling)
    sample_weights = None
    if batch_sampling == 'balanced':
        sample_weights = _build_batch_sampling_weights(
            dataset=dataset,
            features_info=features_info,
            sampling_feature=sampling_feature,
            sampling_power=sampling_power
        )

    logger.info(
        "Sampling config: batch_sampling=%s, sampling_feature=%s, sampling_power=%.3f, t_sampling=%s",
        batch_sampling, sampling_feature, sampling_power, t_sampling
    )

    num_samples = len(dataset)
    for epoch in range(epochs):
        total_loss = 0.0

        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        if batch_sampling == 'sequential':
            epoch_indices = torch.arange(num_samples, dtype=torch.long)
        elif batch_sampling == 'shuffle':
            epoch_indices = torch.randperm(num_samples)
        else:
            epoch_indices = torch.multinomial(sample_weights, num_samples, replacement=True)

        for i in pbar:
            batch_indices = epoch_indices[i:i+batch_size].tolist()
            batch = [dataset[idx] for idx in batch_indices]
            x0_batch = torch.stack([item[0] for item in batch]).to(device)
            cond_batch = torch.stack([item[1] for item in batch]).to(device)
            bsz = x0_batch.size(0)

            # sample timesteps
            if t_probs is None:
                t_batch = torch.randint(1, T+1, (bsz,), device=device)
            else:
                t_batch = torch.multinomial(t_probs, bsz, replacement=True).to(device) + 1

            # run your forward diffusion to get x_t and x_{t-1}
            x_t_minus_1_list = []
            x_t_list = []
            # For each sample, perform the forward diffusion process
            for feat_index, feat in enumerate(features_info):
                name = feat["name"]
                
                # Get current feature values at x0
                x0_feat = x0_batch[:, feat_index]  # (bsz,)

                # 1. Get cumulative matrices Q_bar(t-1): x0 -> x_t-1
                cum_matrices = getattr(attr_model, f'cum_trans_{name}')[t_batch - 1]
                # Get one-step transition matrices Q(t): x_t-1 -> x_t
                step_matrices = getattr(attr_model, f'trans_{name}')[t_batch - 1]
                state_dim = step_matrices.size(-1)

                # 2. Sample x_t-1 given x0
                probs_tm1 = cum_matrices.gather(
                    1, x0_feat.view(-1, 1, 1).expand(-1, 1, state_dim)
                ).squeeze(1)

                x_tm1_feat = torch.multinomial(probs_tm1.clamp(min=0), 1).squeeze(1) # (B,)
                x_t_minus_1_list.append(x_tm1_feat)

                # 3. Sample x_t given x_{t-1}
                probs_t = step_matrices.gather(
                    1, x_tm1_feat.view(-1, 1, 1).expand(-1, 1, state_dim)
                ).squeeze(1)
                x_t_feat = torch.multinomial(probs_t.clamp(min=0), 1).squeeze(1) # (B,)
                x_t_list.append(x_t_feat)

            # after this loop:
            x_t_minus_1 = torch.stack(x_t_minus_1_list, dim=1)
            x_t = torch.stack(x_t_list, dim=1)

            # 2) model forward
            logits, joint_logits = model(x_t, cond_batch, t_batch)

            if loss_type == 'causal':
                # === New Causal Loss Strategy ===
                # Logic: L = L_CE(Act) + w1 * L_CE(ST) + w2 * L_CE(Mode) + L_vb
                # Explicit Joint Loss is REMOVED because the architecture handles dependencies.
                
                loss_groups = {'act': 0.0, 'st': 0.0, 'mode': 0.0}
                vb_loss = 0.0
                
                # Check if model has group definitions (CausalChainTransformer)
                if not hasattr(attr_model, 'group_act_names'):
                    raise ValueError("Model does not support causal grouping. Use 'standard' loss.")

                for feat_index, feat in enumerate(features_info):
                    name = feat["name"]
                    
                    # A) CE Loss (Reconstruction)
                    logits_x0 = logits[name]
                    target_x0 = x0_batch[:, feat_index]
                    feat_ce = F.cross_entropy(logits_x0, target_x0)
                    
                    # Accumulate into groups
                    if name in attr_model.group_act_names:
                        loss_groups['act'] += feat_ce
                    elif name in attr_model.group_st_names:
                        loss_groups['st'] += feat_ce
                    elif name in attr_model.group_mode_names:
                        loss_groups['mode'] += feat_ce
                    
                    # B) VB Loss (Optional but recommended for Diffusion)
                    probs_x0 = F.softmax(logits_x0, dim=-1)
                    M_batch = getattr(attr_model, f'post_{name}')[t_batch - 1] # Use registered buffer
                    probs_xtm1 = torch.bmm(probs_x0.unsqueeze(1), M_batch).squeeze(1)
                    logits_xtm1 = torch.log(probs_xtm1 + 1e-8)
                    target_xtm1 = x_t_minus_1[:, feat_index]
                    vb_loss += F.cross_entropy(logits_xtm1, target_xtm1)
                
                # Normalize VB loss
                vb_loss = vb_loss / len(features_info)
                
                # Aggregate Causal CE Loss
                # Formula: L_act + lambda1 * L_st + lambda2 * L_mode
                causal_ce_loss = (loss_groups['act'] + 
                                  causal_weights['st'] * loss_groups['st'] + 
                                  causal_weights['mode'] * loss_groups['mode'])
                
                # Total Loss
                loss = vb_loss + causal_ce_loss
                
                # Logging info for progress bar
                pbar.set_postfix({'vb': f'{vb_loss.item():.2f}', 'c_ce': f'{causal_ce_loss.item():.2f}'})

            elif loss_type == 'standard':
                ce_loss = 0.0
                vb_loss = 0.0

                # 3) per‐feature losses
                for feat_index, feat in enumerate(features_info):
                    name = feat["name"]
                    # --- a) CE loss on x0
                    logits_x0 = logits[name]                          # (bsz, K)
                    target_x0 = x0_batch[:, feat_index]               # (bsz,)
                    ce_loss += F.cross_entropy(logits_x0, target_x0)

                    # --- b) VB-loss via predicted p(x_{t-1}|x_t)
                    # 1) compute p_theta(x0|xt)
                    probs_x0 = F.softmax(logits_x0, dim=-1)           # (bsz, K)

                    # 2) gather the right M for each sample's t
                    #    model.posterior[name] has shape (T, K, K)
                    #    t_batch is in [1..T], so we index at t-1
                    # M_all    = model.posterior[name].to(device)                  # (T, K, K)
                    # M_batch  = M_all[t_batch - 1]                    # (bsz, K, K)
                    if hasattr(attr_model, f'post_{name}'):
                        M_batch = getattr(attr_model, f'post_{name}')[t_batch - 1]
                    else:
                        posterior = attr_model.posterior[name]
                        if posterior.device != t_batch.device:
                            M_batch = posterior[(t_batch - 1).cpu()].to(device)
                        else:
                            M_batch = posterior[t_batch - 1].to(device)

                    # 3) do batched mat-mul: (bsz,1,K) @ (bsz,K,K) -> (bsz,1,K) -> (bsz,K)
                    probs_xtm1 = torch.bmm(probs_x0.unsqueeze(1), M_batch).squeeze(1)

                    # 4) turn back into “logits” and cross-entropy against sampled x_{t-1}
                    logits_xtm1 = torch.log(probs_xtm1 + 1e-8)        # (bsz, K)
                    target_xtm1 = x_t_minus_1[:, feat_index]          # (bsz,)
                    vb_loss += F.cross_entropy(logits_xtm1, target_xtm1)
                ce_loss = ce_loss / len(features_info)
                vb_loss = vb_loss / len(features_info)

                # 4) joint losses for important feature pairs
                joint_loss_val = 0.0
                if len(attr_model.joint_pairs) > 0 and lambda_joint > 0:
                    for idx, (feat_idx1, feat_idx2) in enumerate(attr_model.joint_pairs):
                        K2 = features_info[feat_idx2]["num_classes"]
                        # Create Joint Target Labels: label = val1 * K2 + val2
                        # This converts the combination of two features into a unique integer ID
                        target_1 = x0_batch[:, feat_idx1]
                        target_2 = x0_batch[:, feat_idx2]
                        target_joint = target_1 * K2 + target_2
                        
                        # Calculate Cross Entropy
                        joint_loss_val += F.cross_entropy(joint_logits[idx], target_joint)
                    
                    # Average Joint Loss
                    joint_loss_val = joint_loss_val / len(attr_model.joint_pairs)


                loss = vb_loss + lambda_weight * ce_loss + lambda_joint * joint_loss_val
                pbar.set_postfix({'loss': f'{loss.item():.2f}'})

            # Backrop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz
        avg_loss = total_loss / num_samples
        msg = f"Epoch {epoch+1}/{epochs}: Average loss = {avg_loss:.4f}"

        # --- Early Stopping Check ---
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            msg += " (New best model saved!)"
            
            # 4. 保存模型状态
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            msg += f" (Patience: {patience_counter}/{patience})"

        logger.info(msg) 
        pbar.update(1)

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs (patience {patience}).")
            break
    end_msg = "Training completed."
    logger.info(end_msg)

    if model_save_path:
        logger.info(f"Loading best model weights from {model_save_path}")

        state_dict = torch.load(model_save_path, map_location=device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
            
        if hasattr(model, 'module'):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)

def sample_trip(model, cond_tensor,device):
    """
    Based on given conditional features, sample a trip using the trained model.
    The sampling process is done by running the reverse diffusion process from T to 1.
    """
    model.eval()
    num_features = len(model.features_info)
    x_t = torch.empty((1, num_features), dtype=torch.long).to(device)
    for i, feat in enumerate(model.features_info):
        K = feat["num_classes"]
        x_t[0, i] = torch.randint(0, K, (1,)).to(device)
    with torch.no_grad():
        for t in range(model.T, 0, -1):
            # prepare timestep tensor
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # 1) model forward → logits for p(x0 | xt)
            logits = model(x_t, cond_tensor.unsqueeze(0).to(device), t_tensor)
            # convert to probabilities, squeeze batch
            probs = {
                name: F.softmax(logits[name], dim=1)[0]  # shape (K,)
                for name in logits
            }

            # 2) sample x_{t-1} for each feature
            x_prev = torch.empty_like(x_t)
            for feat_index, feat in enumerate(model.features_info):
                name = feat["name"]
                K = feat["num_classes"]

                # current noisy value at time t
                i = x_t[0, feat_index].item()

                # p_theta(x0 | x_t = i)
                p_theta = probs[name]              # (K,)

                # fetch one‐step kernels
                Q_t       = model.transitions[name][t-1].to(device)    # (K_j, K_i)
                Q_bar_tm1 = model.cum_transitions[name][t-1].to(device)# (K_z, K_j)
                Q_bar_t   = model.cum_transitions[name][t].to(device)  # (K_z, K_i)

                # build the 3-D numerator: (K_z, K_j, K_i)
                num   = Q_bar_tm1.unsqueeze(2) * Q_t.unsqueeze(0)
                # build denominator (K_z,1,K_i) and avoid zeros
                denom = Q_bar_t.unsqueeze(1).clamp(min=1e-12)

                # full posterior: (K_z, K_j, K_i)
                M_all = num / denom

                # slice out the current x_t = i column → (K_z, K_j)
                M_slice = M_all[:, :, i]

                # weight[j] = sum_z p_theta[z] * M_slice[z, j]
                # -> (1,K_z) @ (K_z, K_j) = (1, K_j) -> squeeze -> (K_j,)
                weight = p_theta.unsqueeze(0).mm(M_slice).squeeze(0)

                # fallback to uniform if underflow
                if weight.sum() <= 0:
                    prob = torch.ones(K, device=device) / K
                else:
                    prob = weight / weight.sum()

                # multinomial draw
                x_prev[0, feat_index] = torch.multinomial(prob, num_samples=1)

            # set up for next reverse step
            x_t = x_prev

    return x_t[0]

def fast_sample_trips(model, cond_batch, device):
    """
    Batch sampling: Generates multiple trips in parallel.
    cond_batch: Tensor of shape (Batch_Size, Num_Cond_Features)
    """
    if hasattr(model, 'module'):
        attr_model = model.module
    else:
        attr_model = model
    model.eval()
    bsz = cond_batch.shape[0]
    num_features = len(attr_model.features_info)
    mask_token_ids = getattr(attr_model, "mask_token_ids", None)
    
    # 1. Initialize x_T randomly
    x_t = torch.empty((bsz, num_features), dtype=torch.long).to(device)
    for i, feat in enumerate(attr_model.features_info):
        name = feat["name"]
        trans_mat = getattr(attr_model, f'trans_{name}')
        state_dim = trans_mat.size(-1)
        mask_id = None if mask_token_ids is None else mask_token_ids.get(name)
        if mask_id is not None:
            x_t[:, i] = mask_id
        else:
            x_t[:, i] = torch.randint(0, state_dim, (bsz,)).to(device)

    with torch.no_grad():
        # Reverse diffusion process T -> 1
        for t in range(attr_model.T, 0, -1):
            # Construct time tensor (Batch_Size,)
            t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)

            # 2. Model Forward (process entire batch at once)
            # Note: If your model forward returns (logits, joint_logits), unpack here
            model_output = model(x_t, cond_batch, t_batch)
            if isinstance(model_output, tuple):
                logits, _ = model_output # Ignore joint_logits
            else:
                logits = model_output

            # 3. Sample each feature
            x_prev_list = []
            for feat_index, feat in enumerate(attr_model.features_info):
                name = feat["name"]
                
                # Get current Batch's x_t values and predicted p(x0)
                curr_x_t = x_t[:, feat_index]            # (B,)
                logits_feat = logits[name]               # (B, K)
                p_theta = F.softmax(logits_feat, dim=1)  # (B, K) -> predicted p(x0)

                # Get matrices (from buffer or list)
                # Q_t: x_{t-1} -> x_t
                Q_t = getattr(attr_model, f'trans_{name}')[t-1]       # (K, K)
                # Q_bar_tm1: x_0 -> x_{t-1}
                Q_bar_tm1 = getattr(attr_model, f'cum_trans_{name}')[t-1] # (K, K)
                # Q_bar_t: x_0 -> x_t
                Q_bar_t = getattr(attr_model, f'cum_trans_{name}')[t]     # (K, K)
                # --- Vectorized Posterior Calculation ---
                # Formula: p(x_{t-1}|x_t) \propto Q_t(x_{t-1}, x_t) * sum_x0 [ (p(x0)/Q_bar_t(x0, x_t)) * Q_bar_tm1(x0, x_{t-1}) ]
                
                # A. Calculate denominator Q_bar_t(x0, x_t) -> select columns corresponding to x_t
                # Q_bar_t is (Rows=x0, Cols=xt), we need to select Cols=curr_x_t
                # Result shape: (B, K) (denominator for each sample in batch for x0 distribution)
                denom = Q_bar_t[:, curr_x_t].t() # Transpose -> (B, K)

                # B. Calculate ratio p(x0) / Q_bar_t
                ratio = p_theta / denom.clamp(min=1e-12) # (B, K)

                # C. Matrix multiplication sum: ratio @ Q_bar_tm1
                # (B, K) @ (K, K) -> (B, K)
                # This step completes the weighted sum over x0
                mix_prob = torch.matmul(ratio, Q_bar_tm1) # (B, K) represents x_{t-1} distribution part

                # D. Multiply by forward transition probabilities Q_t(x_{t-1}, x_t)
                # Q_t is (Rows=xt-1, Cols=xt), we need to select Cols=curr_x_t
                q_t_probs = Q_t[:, curr_x_t].t() # (B, K)

                # E. Final unnormalized probabilities
                out_probs = mix_prob * q_t_probs # (B, K)

                # F. Normalize and sample
                out_probs = out_probs / out_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
                state_dim = out_probs.size(1)
                
                # Handle NaNs due to numerical instability
                out_probs = torch.where(torch.isnan(out_probs), torch.ones_like(out_probs)/state_dim, out_probs)

                # Batch Multinomial Sampling
                next_val = torch.multinomial(out_probs, 1).squeeze(1) # (B,)
                x_prev_list.append(next_val)

            # Update x_t
            x_t = torch.stack(x_prev_list, dim=1) # (B, Num_Features)

    # Convert residual mask tokens back to valid class ids for evaluation/export.
    if mask_token_ids is not None:
        for feat_index, feat in enumerate(attr_model.features_info):
            name = feat["name"]
            mask_id = mask_token_ids.get(name)
            if mask_id is None:
                continue
            valid_classes = feat["num_classes"]
            is_mask = x_t[:, feat_index] == mask_id
            if is_mask.any():
                replacement = torch.randint(0, valid_classes, (int(is_mask.sum().item()),), device=device)
                x_t[is_mask, feat_index] = replacement

    return x_t

def sample_trip(model, df, num_samples, device):
    """
    Generate samples from the entire dataset without clustering.
    
    Args:
        model: Trained TripDiffusionModel
        df: The dataframe (test data) to sample conditions from
        num_samples: Total number of samples to generate
        device: Torch device
    """
    if hasattr(model, 'module'):
        attr_model = model.module
    else:
        attr_model = model

    model.eval()
    
    results = []
    truth_trip = []

    cond_features = [cond["name"] for cond in attr_model.cond_info]
    trip_features = [feat["name"] for feat in attr_model.features_info]

    print(f"Sampling {num_samples} conditions from dataset...")
    sampled_rows = df.sample(n=num_samples, replace=True)

    all_conds = []
    all_truth = []
    
    for _, row in sampled_rows.iterrows():
        c = [int(row[feat]) for feat in cond_features]
        t = [int(row[feat]) for feat in trip_features]
        all_conds.append(c)
        all_truth.append(t)
    
    all_conds_tensor = torch.tensor(all_conds, dtype=torch.long).to(device)

    MAX_BATCH_SIZE = 512 
    generated_trips_list = []
    
    for i in tqdm(range(0, num_samples, MAX_BATCH_SIZE), desc="Generating trips"):
        batch_cond = all_conds_tensor[i : i + MAX_BATCH_SIZE]
        
        batch_generated = fast_sample_trips(model, batch_cond, device)

        generated_trips_list.append(batch_generated.cpu())

    all_generated = torch.cat(generated_trips_list, dim=0).tolist()
    
    for idx in range(num_samples):
        results.append({
            "condition": all_conds[idx],
            "trip": all_generated[idx]
        })
        truth_trip.append({
            "condition": all_conds[idx],
            "trip": all_truth[idx]
        })
    
    return results, truth_trip

def save_generated_samples(generated_samples, output_file):
    """
    Modified to save a flat list of samples (no clusters).
    """
    import csv

    headers = ["relation", "sex", "age_code", "job_type", 
               "start_type", "start_zcode_num", "act_num", 
               "mode_num", "end_type", "end_zcode_num", 
               "start_time_num_6", "trip_time_num_6"]

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for sample in generated_samples:
            cond = sample["condition"]
            trip = sample["trip"]
            row = cond + trip 
            writer.writerow(row)


