import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import csv
import logging
from tqdm import tqdm

from model.Net import TripDiffusionModel

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


def train_model(model, optimizer, dataset, features_info, lambda_weight, T, epochs, batch_size, device):
    """
    Model training process:
      - For each batch, do the diffusion process based on random step t.
      - Forword propagate and compute the entropy loss (CE loss and VB loss).
      - Backpropagate and update the model parameters.
    """
    logger = logging.getLogger(__name__)

    model.train()

    num_samples = len(dataset)
    for epoch in range(epochs):
        total_loss = 0.0
        for i in tqdm(range(0, num_samples, batch_size),
                      desc=f"Epoch {epoch+1}/{epochs}", leave=False):

            batch = dataset[i:i+batch_size]
            x0_batch = torch.stack([item[0] for item in batch]).to(device)
            cond_batch = torch.stack([item[1] for item in batch]).to(device)
            bsz = x0_batch.size(0)

            # sample timesteps
            t_batch = torch.randint(1, T+1, (bsz,), device=device)

            # run your forward diffusion to get x_t and x_{t-1}
            x_prev = x0_batch.clone()
            x_t_minus_1 = torch.zeros_like(x0_batch)
            # For each sample, perform the forward diffusion process
            for idx in range(bsz):
                t = t_batch[idx].item()
                for step in range(1, t+1):
                    if step == t:
                        x_t_minus_1[idx] = x_prev[idx].clone()  # store the state at t-1
                    # For each feature: x_prev -> x_next
                    x_next_feat = []
                    for feat_index, feat in enumerate(features_info):
                        feat_type = feat["type"]
                        current_val = x_prev[idx, feat_index].item()
                        if feat_type == "categorical":
                            beta = model.beta_schedule[step-1].item()
                            if torch.rand(1).item() < (1 - beta):
                                new_val = current_val  # Keep the same value
                            else:
                                # Uniformly sample a new value from the transition matrix
                                K = feat["num_classes"]
                                if K > 1:
                                    new_val = torch.randint(0, K-1, (1,)).item()
                                    if new_val >= current_val:
                                        new_val += 1
                                else:
                                    new_val = current_val
                            x_next_feat.append(new_val)
                        elif feat_type == "ordinal":
                            sigma = model.sigma_schedule[step-1].item()
                            noise = int(round(torch.randn(1).item() * sigma))
                            new_val = int(current_val + noise)
                            new_val = max(0, min(feat["num_classes"]-1, new_val))
                            x_next_feat.append(new_val)
                    x_next_feat = torch.tensor(x_next_feat, dtype=torch.long)
                    x_prev[idx] = x_next_feat  # Update state
            # after this loop:
            x_t = x_prev

            # 2) model forward
            logits = model(x_t, cond_batch, t_batch)
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
                M_all    = model.posterior[name].to(device)                  # (T, K, K)
                M_batch  = M_all[t_batch - 1]                    # (bsz, K, K)

                # 3) do batched mat-mul: (bsz,1,K) @ (bsz,K,K) -> (bsz,1,K) -> (bsz,K)
                probs_xtm1 = torch.bmm(probs_x0.unsqueeze(1), M_batch).squeeze(1)

                # 4) turn back into “logits” and cross-entropy against sampled x_{t-1}
                logits_xtm1 = torch.log(probs_xtm1 + 1e-8)        # (bsz, K)
                target_xtm1 = x_t_minus_1[:, feat_index]          # (bsz,)
                vb_loss += F.cross_entropy(logits_xtm1, target_xtm1)
            ce_loss = ce_loss / len(features_info)
            vb_loss = vb_loss / len(features_info)
            loss = vb_loss + lambda_weight * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz
        avg_loss = total_loss / num_samples
        msg = f"Epoch {epoch+1}/{epochs}: Average loss = {avg_loss:.4f}"
        logger.info(msg) 
    end_msg = "Training completed."
    logger.info(end_msg)

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

def sample_condition_from_cluster(clustered_df, cluster_id,features_info, cond_info):
    cond_features = [cond["name"] for cond in cond_info]
    trip_features = [feat["name"] for feat in features_info]
    cluster_data = clustered_df[clustered_df["Cluster"] == cluster_id]
    if cluster_data.empty:
        raise ValueError(f"No data in class {cluster_id} ")
    sample_row = cluster_data.sample(n=1).iloc[0]
    condition = [int(sample_row[feat]) for feat in cond_features]
    true_trip = [int(sample_row[feat]) for feat in trip_features]
    return torch.tensor(condition, dtype=torch.long), true_trip

def sample_trip_by_clusters(model, clustered_df, num_samples_each, device):
    results = {}
    clusters = sorted(clustered_df["Cluster"].unique())
    truth_trip = {}
    for cluster_id in tqdm(clusters, desc="Processing clusters"):
        results[cluster_id] = []
        truth_trip[cluster_id] = []
        for _ in tqdm(range(num_samples_each), desc=f"Cluster {cluster_id} samples", leave=False):
            cond_tensor, true_trip = sample_condition_from_cluster(clustered_df, cluster_id, model.features_info, model.cond_info)
            truth_trip[cluster_id].append({
                "condition": cond_tensor.tolist(),
                "trip": true_trip
            })
            trip = sample_trip(model, cond_tensor,device)
            results[cluster_id].append({
                "condition": cond_tensor.tolist(),
                "trip": trip.tolist()
            })
    return results, truth_trip

def save_generated_samples(generated_samples, output_file):
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["relation", "sex", "age_code", "job_type", 
                         "start_type",  "start_zcode_num", "act_num", "mode_num",  "end_type", "end_zcode_num","start_time_num_6",
                         "trip_time_num_6", "Cluster"])
        for cluster_id, samples in generated_samples.items():
            for sample in samples:
                cond = sample["condition"]
                trip = sample["trip"]
                row = cond + trip + [cluster_id]
                writer.writerow(row)


