import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalChainBlock(nn.Module):
    """
    A single layer of the Causal Chain Transformer.
    Manages 3 streams: Act (Purpose), ST (Spatio-Temporal), Mode.
    Enforces the dependency: Cond -> Act -> ST -> Mode
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # --- 1. Act Stream (Root) ---
        # Self Attention
        self.attn_act_self = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_act_1 = nn.LayerNorm(d_model)
        # Cross Attention (Queries: Act, Keys/Values: Condition)
        self.attn_act_cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_act_2 = nn.LayerNorm(d_model)
        # FFN
        self.ffn_act = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_act_3 = nn.LayerNorm(d_model)

        # --- 2. ST Stream (Intermediate) ---
        # Self Attention (ST constraints correlate with each other)
        self.attn_st_self = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_st_1 = nn.LayerNorm(d_model)
        # Cross Attention (Queries: ST, Keys/Values: Concat(Condition, Act))
        self.attn_st_cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_st_2 = nn.LayerNorm(d_model)
        # FFN
        self.ffn_st = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_st_3 = nn.LayerNorm(d_model)

        # --- 3. Mode Stream (Leaf) ---
        # Self Attention
        self.attn_mode_self = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_mode_1 = nn.LayerNorm(d_model)
        # Cross Attention (Queries: Mode, Keys/Values: Concat(Condition, Act, ST))
        self.attn_mode_cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_mode_2 = nn.LayerNorm(d_model)
        # FFN
        self.ffn_mode = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_mode_3 = nn.LayerNorm(d_model)

    def forward(self, h_act, h_st, h_mode, h_cond):
        """
        h_act: [Batch, 1, D]
        h_st:  [Batch, Seq_ST, D]
        h_mode:[Batch, 1, D]
        h_cond:[Batch, 1, D] (Global Context including Time)
        """
        
        # === Stream 1: Act ===
        # 1.1 Self
        res = h_act
        h_act, _ = self.attn_act_self(h_act, h_act, h_act)
        h_act = self.norm_act_1(res + h_act)
        
        # 1.2 Cross (Look at Cond)
        res = h_act
        h_act, _ = self.attn_act_cross(query=h_act, key=h_cond, value=h_cond)
        h_act = self.norm_act_2(res + h_act)
        
        # 1.3 FFN
        h_act = self.norm_act_3(h_act + self.ffn_act(h_act))

        # === Stream 2: ST ===
        # 2.1 Self
        res = h_st
        h_st, _ = self.attn_st_self(h_st, h_st, h_st)
        h_st = self.norm_st_1(res + h_st)
        
        # 2.2 Cross (Look at Cond + Act)
        # Context for ST is Condition AND the updated Act
        ctx_st = torch.cat([h_cond, h_act], dim=1) 
        res = h_st
        h_st, _ = self.attn_st_cross(query=h_st, key=ctx_st, value=ctx_st)
        h_st = self.norm_st_2(res + h_st)
        
        # 2.3 FFN
        h_st = self.norm_st_3(h_st + self.ffn_st(h_st))

        # === Stream 3: Mode ===
        # 3.1 Self
        res = h_mode
        h_mode, _ = self.attn_mode_self(h_mode, h_mode, h_mode)
        h_mode = self.norm_mode_1(res + h_mode)
        
        # 3.2 Cross (Look at Cond + Act + ST)
        # Context for Mode is Everything above
        ctx_mode = torch.cat([h_cond, h_act, h_st], dim=1)
        res = h_mode
        h_mode, _ = self.attn_mode_cross(query=h_mode, key=ctx_mode, value=ctx_mode)
        h_mode = self.norm_mode_2(res + h_mode)
        
        # 3.3 FFN
        h_mode = self.norm_mode_3(h_mode + self.ffn_mode(h_mode))

        return h_act, h_st, h_mode


class TripDiffusionModel(nn.Module):
    def __init__(self, features_info, cond_info, T, joint_pairs=None):
        super().__init__()
        self.features_info = features_info
        self.cond_info = cond_info
        self.T = T
        self.joint_pairs = joint_pairs if joint_pairs is not None else []
        
        # Model Hyperparameters
        self.d_model = 256
        self.num_layers = 6
        self.nhead = 8
        
        # --- 1. Identify Groups for Causal Chain ---
        # We define lists of feature names belonging to each group
        self.group_act_names = ["act_num"]
        self.group_mode_names = ["mode_num"]
        # All other features fall into Spatio-Temporal
        self.group_st_names = [f["name"] for f in features_info 
                               if f["name"] not in self.group_act_names 
                               and f["name"] not in self.group_mode_names]
        
        # Create a mapping from feature name to index in x_t
        self.feat_idx_map = {f["name"]: i for i, f in enumerate(features_info)}
        
        # --- 2. Embeddings ---
        # Feature Embeddings (Project to d_model)
        self.feature_embeddings = nn.ModuleDict()
        for feat in features_info:
            name = feat["name"]
            num_classes = feat["num_classes"]
            # We project all features to d_model for the Transformer
            self.feature_embeddings[name] = nn.Embedding(num_classes, self.d_model)
            
        # Cond Embeddings
        self.cond_embeddings = nn.ModuleDict()
        for cond in cond_info:
            name = cond["name"]
            num_classes = cond["num_classes"]
            # Keep small intermediate dim, then project to d_model later
            self.cond_embeddings[name] = nn.Embedding(num_classes, 16)
            
        # Time Embedding
        self.time_embedding = nn.Embedding(T+1, self.d_model)

        # Projection for Condition Context (Cond Embeds -> d_model)
        total_cond_dim = sum(16 for _ in cond_info)
        self.cond_projector = nn.Linear(total_cond_dim, self.d_model)

        # --- 3. Transformer Backbone ---
        self.layers = nn.ModuleList([
            CausalChainBlock(self.d_model, self.nhead) for _ in range(self.num_layers)
        ])

        # --- 4. Output Heads ---
        self.output_heads = nn.ModuleDict()
        for feat in features_info:
            name = feat["name"]
            num_classes = feat["num_classes"]
            self.output_heads[name] = nn.Linear(self.d_model, num_classes)

        # Joint heads
        self.joint_heads = nn.ModuleList()
        self.joint_head_pairs = [] # Store pairs to know which features to concat
        for (idx1, idx2) in self.joint_pairs:
            feat1 = features_info[idx1]
            feat2 = features_info[idx2]
            joint_dim = feat1["num_classes"] * feat2["num_classes"]
            # Input dim is d_model * 2 because we concat two features
            self.joint_heads.append(nn.Linear(self.d_model * 2, joint_dim))
            self.joint_head_pairs.append((feat1["name"], feat2["name"]))

        # --- 5. Diffusion Logic (Preserved exactly from original) ---
        beta_schedule = torch.linspace(0.1, 0.5, steps=T)
        sigma_schedule = torch.linspace(5.0, 50.0, steps=T)
        self.register_buffer('beta_schedule',  beta_schedule)
        self.register_buffer('sigma_schedule', sigma_schedule)

        # Temporary lists to hold data for self.posterior calculation
        # We need to reconstruct the Python list logic first to calculate posterior, 
        # then register buffers.
        temp_transitions = {} 
        temp_cum_transitions = {}

        for feat in features_info:
            name = feat["name"]
            K = feat["num_classes"]
            feat_type = feat["type"]
            
            trans_list_for_buffer = []
            cum_trans_list_for_buffer = []
            
            # For posterior calc (Python lists)
            temp_transitions[name] = []
            
            Q_bar_prev = torch.eye(K)
            temp_cum_transitions[name] = [Q_bar_prev]
            cum_trans_list_for_buffer.append(Q_bar_prev)

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
                
                temp_transitions[name].append(Q_t)
                trans_list_for_buffer.append(Q_t)
                
                Q_bar = Q_bar_prev @ Q_t
                temp_cum_transitions[name].append(Q_bar)
                cum_trans_list_for_buffer.append(Q_bar)
                
                Q_bar_prev = Q_bar
            
            # Register Buffers
            self.register_buffer(f'trans_{name}', torch.stack(trans_list_for_buffer))
            self.register_buffer(f'cum_trans_{name}', torch.stack(cum_trans_list_for_buffer))

        # Build Posterior (Using temp lists)
        for feat in features_info:
            name = feat["name"]
            K = feat["num_classes"]
            post_list = []
            for t in range(1, T+1):
                Q_t      = temp_transitions[name][t-1]
                Q_bar_tm1= temp_cum_transitions[name][t-1]
                Q_bar_t  = temp_cum_transitions[name][t]

                num = Q_bar_tm1.unsqueeze(2) * Q_t.unsqueeze(0)
                denom = Q_bar_t.unsqueeze(1).clamp(min=1e-12)
                M_2d = (num/denom).sum(dim=0)
                post_list.append(M_2d)
            
            # Register posterior as buffer (it might be large, but useful for sampling)
            # Original code stored it in self.posterior dict, we can keep it there or register
            # Storing in self.posterior as tensors on correct device
            self.register_buffer(f'post_{name}', torch.stack(post_list, dim=0))

    def get_posterior(self, name):
        # Helper to retrieve registered posterior buffer by name
        return getattr(self, f'post_{name}')

    def forward(self, x_t, cond, t):
        """
        x_t: [Batch, Num_Features]
        cond: [Batch, Num_Conds]
        t: [Batch]
        """
        batch_size = x_t.size(0)

        # --- 1. Embed Inputs ---
        
        # A. Embed Features & Route to Groups
        emb_map = {} # Store embeddings by feature name
        for name, idx in self.feat_idx_map.items():
            # x_t[:, idx] is the column for this feature
            emb_map[name] = self.feature_embeddings[name](x_t[:, idx])
        
        # Stack Group: Act [Batch, 1, D]
        h_act = torch.stack([emb_map[name] for name in self.group_act_names], dim=1)
        
        # Stack Group: ST [Batch, Len_ST, D]
        # Order matters here, we follow the order in group_st_names
        h_st = torch.stack([emb_map[name] for name in self.group_st_names], dim=1)
        
        # Stack Group: Mode [Batch, 1, D]
        h_mode = torch.stack([emb_map[name] for name in self.group_mode_names], dim=1)

        # B. Embed Conditions & Time (Global Context)
        cond_embeds = []
        for j, cond_feat in enumerate(self.cond_info):
            name = cond_feat["name"]
            cond_embeds.append(self.cond_embeddings[name](cond[:, j]))
        
        # Concatenate and Project Conditions
        # [Batch, Total_Cond_Raw_Dim]
        raw_cond = torch.cat(cond_embeds, dim=1) 
        # [Batch, D]
        proj_cond = self.cond_projector(raw_cond)
        
        # Add Time Embedding
        t_emb = self.time_embedding(t) # [Batch, D]
        
        # Final Global Context: [Batch, 1, D]
        # We add time to the condition context so it guides everything
        h_cond = (proj_cond + t_emb).unsqueeze(1) 

        # --- 2. Pass through Causal Chain Transformer ---
        for layer in self.layers:
            h_act, h_st, h_mode = layer(h_act, h_st, h_mode, h_cond)

        # --- 3. Output Heads ---
        # We need to map the hidden states back to feature names
        
        logits = {}
        
        # Map Act Outputs
        for i, name in enumerate(self.group_act_names):
            # h_act[:, i, :] -> [Batch, D]
            logits[name] = self.output_heads[name](h_act[:, i, :])
            
        # Map Mode Outputs
        for i, name in enumerate(self.group_mode_names):
            logits[name] = self.output_heads[name](h_mode[:, i, :])
            
        # Map ST Outputs
        for i, name in enumerate(self.group_st_names):
            logits[name] = self.output_heads[name](h_st[:, i, :])

        # --- 4. Joint Heads ---
        joint_logits = []
        # To compute joint logits, we need to find the specific hidden states for the pair
        # We construct a temporary lookup for hidden states
        hidden_state_map = {}
        for i, name in enumerate(self.group_act_names):
            hidden_state_map[name] = h_act[:, i, :]
        for i, name in enumerate(self.group_st_names):
            hidden_state_map[name] = h_st[:, i, :]
        for i, name in enumerate(self.group_mode_names):
            hidden_state_map[name] = h_mode[:, i, :]
            
        for i, (name1, name2) in enumerate(self.joint_head_pairs):
            h1 = hidden_state_map[name1]
            h2 = hidden_state_map[name2]
            # Concat [Batch, 2*D]
            h_pair = torch.cat([h1, h2], dim=1)
            joint_logits.append(self.joint_heads[i](h_pair))

        return logits, joint_logits