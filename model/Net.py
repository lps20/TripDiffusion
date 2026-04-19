import torch
import torch.nn as nn
import torch.optim as optim


class TripDiffusionModel(nn.Module):
    def __init__(self, features_info, cond_info, T, joint_pairs=None):
        super().__init__()
        self.features_info = features_info
        self.cond_info = cond_info
        self.T = T
        self.joint_pairs = joint_pairs if joint_pairs is not None else []  # Add important feature pairs

        # Embeddings for each trip feature (categorical or ordinal)
        self.feature_embeddings = nn.ModuleDict()
        for feat in features_info:
            name = feat["name"]
            num_classes = feat["num_classes"]
            embed_dim = 16  # embedding size for features (could tune per feature)
            if feat["type"] == "ordinal" and num_classes > 100:
                embed_dim = 32  # use larger embedding for wide ordinal range
            self.feature_embeddings[name] = nn.Embedding(num_classes, embed_dim)
        
        # Embeddings for conditional demographic features
        self.cond_embeddings = nn.ModuleDict()
        for cond in cond_info:
            name = cond["name"]
            num_classes = cond["num_classes"]
            cond_embed_dim = 8
            self.cond_embeddings[name] = nn.Embedding(num_classes, cond_embed_dim)
        
        # Time-step embedding
        self.time_embedding = nn.Embedding(T+1, 64)  # +1 so we can index time steps 0..T

        # Define the neural network layers
        # Input dimension = sum of all embedding dims (features + cond + time)
        total_feature_embed_dim = sum(
            emb.embedding_dim for emb in self.feature_embeddings.values()
        )
        total_cond_embed_dim = sum(
            emb.embedding_dim for emb in self.cond_embeddings.values()
        )
        input_dim = total_feature_embed_dim + total_cond_embed_dim + self.time_embedding.embedding_dim
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output heads for each feature
        self.output_heads = nn.ModuleDict()
        for feat in features_info:
            name = feat["name"]
            num_classes = feat["num_classes"]
            self.output_heads[name] = nn.Linear(hidden_dim, num_classes)

        # Joint heads
        self.joint_heads = nn.ModuleList()
        for (idx1, idx2) in self.joint_pairs:
            feat1 = features_info[idx1]
            feat2 = features_info[idx2]
            joint_dim = feat1["num_classes"] * feat2["num_classes"]
            # Create a linear layer for joint prediction
            self.joint_heads.append(nn.Linear(hidden_dim, joint_dim))
        
        ## Diffusion forward process parameters (for q)
        # Noise schedules (beta for categorical, sigma for ordinal)
        # Using simple linear schedules for demonstration
        beta_schedule = torch.linspace(0.1, 0.5, steps=T)         # e.g., from 0.1 to 0.5
        sigma_schedule = torch.linspace(5.0, 50.0, steps=T)       # e.g., from 5 to 50 (for Gaussian noise)

        # assume beta_schedule, sigma_schedule are torch.Tensor of shape (T,)
        self.register_buffer('beta_schedule',  beta_schedule)
        self.register_buffer('sigma_schedule', sigma_schedule)

        # Precompute transition matrices Q_t for each feature type and cumulative products Q_bar
        self.transitions = {}         # Q_t matrices per feature per step
        self.cum_transitions = {}     # Q_bar matrices per feature for t steps
        for feat in features_info:
            name = feat["name"]
            K = feat["num_classes"]
            feat_type = feat["type"]
            # Initialize cumulative transition for 0 steps as Identity
            Q_bar_prev = torch.eye(K)
            self.transitions[name] = []
            self.cum_transitions[name] = [Q_bar_prev]
            for t in range(T):
                if feat_type == "categorical":
                    # Uniform transition matrix for step t+1
                    beta_t = self.beta_schedule[t].item()
                    if K == 1:
                        Q_t = torch.eye(1)
                    else:
                        Q_t = torch.full((K, K), beta_t/(K-1))
                        Q_t.fill_diagonal_(1 - beta_t)  # each state stays with prob 1-beta, transitions uniformly otherwise
                elif feat_type == "ordinal":
                    # Discretized Gaussian transition for step t+1
                    sigma_t = self.sigma_schedule[t].item()
                    # Compute Gaussian probabilities for all pairs (i->j)
                    idx = torch.arange(K).unsqueeze(1)  # shape (K,1)
                    jdx = torch.arange(K).unsqueeze(0)  # shape (1,K)
                    dist_sq = (idx - jdx).float().pow(2)
                    Q_t = torch.exp(- dist_sq / (2 * sigma_t**2))
                    Q_t = Q_t / Q_t.sum(dim=1, keepdim=True)  # normalize each row
                # Append and update cumulative
                self.transitions[name].append(Q_t)
                Q_bar = Q_bar_prev @ Q_t  # matrix multiplication
                self.cum_transitions[name].append(Q_bar)
                Q_bar_prev = Q_bar
        #   build posterior matrices for every t, for every feat
        #   posterior[name][t] has shape (K, K, K), 
        #   where posterior[name][t][k, j, i] = q(x_{t-1}=j | x_t=i, x_0=k)
        self.posterior = {}
        for feat in features_info:
            name = feat["name"]
            K = feat["num_classes"]
            post_list = []
            for t in range(1, T+1):
                Q_t      = self.transitions[name][t-1]      # (K,K): from x_{t-1}->x_t
                Q_bar_tm1= self.cum_transitions[name][t-1]  # (K,K): from x0->x_{t-1}
                Q_bar_t  = self.cum_transitions[name][t]    # (K,K): from x0->x_t

                # shape manip:
                #   Q_bar_tm1.unsqueeze(2):    (K, K, 1)
                #   Q_t.        unsqueeze(0):  (1, K, K)
                # → broadcast mul → (K, K, K) with axes (x0, x_{t-1}, x_t)
                num = Q_bar_tm1.unsqueeze(2) * Q_t.unsqueeze(0)

                # denom[k,i] = Q_bar_t[k,i], reshape to (K,1,K) so we divide num[k, j, i]
                denom = Q_bar_t.unsqueeze(1).clamp(min=1e-12)

                M_2d = (num/denom).sum(dim=0)                     # (K,K) after summing out x_t
                post_list.append(M_2d)

            # stack into a single tensor of shape (T, K, K, K)
            self.posterior[name] = torch.stack(post_list, dim=0)

        # Precompute transition matrices Q_t and Q_bar
        self.transitions = nn.ParameterDict()     # use nn.ParameterDict to register as buffers
        self.cum_transitions = nn.ParameterDict()

        for feat in features_info:
            name = feat["name"]
            K = feat["num_classes"]
            feat_type = feat["type"]
            
            # 临时列表用于构建 tensor
            trans_list = []
            cum_trans_list = []
            
            Q_bar_prev = torch.eye(K)
            
            # Action：cum_transitions need to include t=0 (Identity) to t=T
            # transitions need to include t=1 to t=T (corresponding to indices 0 to T-1)
            cum_trans_list.append(Q_bar_prev) 

            for t in range(T):
                if feat_type == "categorical":
                    beta_t = self.beta_schedule[t].item()
                    if K == 1:
                        Q_t = torch.eye(1)
                    else:
                        Q_t = torch.full((K, K), beta_t/(K-1))
                        Q_t.fill_diagonal_(1 - beta_t)
                elif feat_type == "ordinal":
                    sigma_t = self.sigma_schedule[t].item()
                    idx = torch.arange(K).unsqueeze(1)
                    jdx = torch.arange(K).unsqueeze(0)
                    dist_sq = (idx - jdx).float().pow(2)
                    Q_t = torch.exp(- dist_sq / (2 * sigma_t**2))
                    Q_t = Q_t / Q_t.sum(dim=1, keepdim=True)
                
                trans_list.append(Q_t)
                Q_bar = Q_bar_prev @ Q_t
                cum_trans_list.append(Q_bar)
                Q_bar_prev = Q_bar
            
            # Stack and register as buffers (this will automatically move with the model to GPU)
            # transitions shape: (T, K, K)
            self.register_buffer(f'trans_{name}', torch.stack(trans_list))
            # cum_transitions shape: (T+1, K, K)
            self.register_buffer(f'cum_trans_{name}', torch.stack(cum_trans_list))

            # posterior shape: (T, K, K), compatible with train_utils fast path
            post_list = []
            for t in range(1, T + 1):
                Q_t = trans_list[t - 1]
                Q_bar_tm1 = cum_trans_list[t - 1]
                Q_bar_t = cum_trans_list[t]
                num = Q_bar_tm1.unsqueeze(2) * Q_t.unsqueeze(0)
                denom = Q_bar_t.unsqueeze(1).clamp(min=1e-12)
                post_list.append((num / denom).sum(dim=0))
            post_tensor = torch.stack(post_list, dim=0)
            self.register_buffer(f'post_{name}', post_tensor, persistent=False)
            self.posterior[name] = post_tensor

    def forward(self, x_t, cond, t):
        """
        Perform a forward pass of the network.
        x_t: tensor of shape (batch, num_features) containing the noised trip data at time t.
        cond: tensor of shape (batch, num_cond) with the conditional features.
        t: tensor of shape (batch,) with the diffusion step indices for each sample.
        Returns a dict of logits for each feature (shape (batch, num_classes) per feature).
        """
        batch_size = x_t.size(0)
        # Embed each feature of x_t
        feature_embeds = []
        for i, feat in enumerate(self.features_info):
            name = feat["name"]
            # Note: x_t[:, i] is a batch of indices for this feature
            embed = self.feature_embeddings[name](x_t[:, i])
            feature_embeds.append(embed)
        feature_embeds = torch.cat(feature_embeds, dim=1)  # concat all feature embeddings
        
        # Embed conditional features
        cond_embeds = []
        for j, cond_feat in enumerate(self.cond_info):
            name = cond_feat["name"]
            cond_embeds.append(self.cond_embeddings[name](cond[:, j]))
        cond_embeds = torch.cat(cond_embeds, dim=1) if cond_embeds else None

        # Embed time step
        t_emb = self.time_embedding(t)  # shape (batch, time_embed_dim)
        
        # Concatenate feature embeds, conditional embeds, and time embed
        if cond_embeds is not None:
            h = torch.cat([feature_embeds, cond_embeds, t_emb], dim=1)
        else:
            h = torch.cat([feature_embeds, t_emb], dim=1)
        # Pass through the MLP
        h = self.net(h)
        # Compute output logits for each feature
        logits = {}
        for feat in self.features_info:
            name = feat["name"]
            logits[name] = self.output_heads[name](h)

        joint_logits = []
        for i, head in enumerate(self.joint_heads):
            joint_logits.append(head(h))
            
        return logits, joint_logits
