import torch
import torch.nn as nn


class TripDiffusionModel(nn.Module):
    """
    Pure Transformer backbone for trip diffusion:
    - No causal-chain split.
    - All trip features are processed as one token sequence.
    """

    def __init__(self, features_info, cond_info, T, joint_pairs=None):
        super().__init__()
        self.features_info = features_info
        self.cond_info = cond_info
        self.T = T
        self.joint_pairs = joint_pairs if joint_pairs is not None else []

        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dropout = 0.1

        self.feat_idx_map = {f["name"]: i for i, f in enumerate(features_info)}
        self.feat_names = [f["name"] for f in features_info]
        self.num_feature_tokens = len(self.feat_names)

        # Feature token embeddings
        self.feature_embeddings = nn.ModuleDict()
        for feat in features_info:
            self.feature_embeddings[feat["name"]] = nn.Embedding(feat["num_classes"], self.d_model)

        # Condition + time to global context token
        self.cond_embeddings = nn.ModuleDict()
        for cond in cond_info:
            self.cond_embeddings[cond["name"]] = nn.Embedding(cond["num_classes"], 16)
        self.cond_projector = nn.Linear(16 * len(cond_info), self.d_model)
        self.time_embedding = nn.Embedding(T + 1, self.d_model)

        self.feature_pos_embed = nn.Parameter(torch.randn(1, self.num_feature_tokens, self.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.output_heads = nn.ModuleDict()
        for feat in features_info:
            self.output_heads[feat["name"]] = nn.Linear(self.d_model, feat["num_classes"])

        self.joint_heads = nn.ModuleList()
        self.joint_head_pairs = []
        for (idx1, idx2) in self.joint_pairs:
            feat1 = features_info[idx1]
            feat2 = features_info[idx2]
            joint_dim = feat1["num_classes"] * feat2["num_classes"]
            self.joint_heads.append(nn.Linear(self.d_model * 2, joint_dim))
            self.joint_head_pairs.append((feat1["name"], feat2["name"]))

        # Diffusion schedule
        beta_schedule = torch.linspace(0.1, 0.5, steps=T)
        sigma_schedule = torch.linspace(5.0, 50.0, steps=T)
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("sigma_schedule", sigma_schedule)

        # Build transition / cumulative transition / posterior buffers
        temp_transitions = {}
        temp_cum_transitions = {}
        for feat in features_info:
            name = feat["name"]
            K = feat["num_classes"]
            feat_type = feat["type"]

            trans_list = []
            cum_list = []
            temp_transitions[name] = []

            q_bar_prev = torch.eye(K)
            temp_cum_transitions[name] = [q_bar_prev]
            cum_list.append(q_bar_prev)

            for t in range(T):
                if feat_type == "categorical":
                    beta_t = float(self.beta_schedule[t].item())
                    if K == 1:
                        q_t = torch.eye(1)
                    else:
                        q_t = torch.full((K, K), beta_t / (K - 1))
                        q_t.fill_diagonal_(1 - beta_t)
                elif feat_type == "ordinal":
                    sigma_t = float(self.sigma_schedule[t].item())
                    idx = torch.arange(K).unsqueeze(1)
                    jdx = torch.arange(K).unsqueeze(0)
                    dist_sq = (idx - jdx).float().pow(2)
                    q_t = torch.exp(-dist_sq / (2 * sigma_t ** 2))
                    q_t = q_t / q_t.sum(dim=1, keepdim=True)
                else:
                    raise ValueError(f"Unsupported feature type: {feat_type}")

                temp_transitions[name].append(q_t)
                trans_list.append(q_t)
                q_bar = q_bar_prev @ q_t
                temp_cum_transitions[name].append(q_bar)
                cum_list.append(q_bar)
                q_bar_prev = q_bar

            self.register_buffer(f"trans_{name}", torch.stack(trans_list, dim=0))
            self.register_buffer(f"cum_trans_{name}", torch.stack(cum_list, dim=0))

        for feat in features_info:
            name = feat["name"]
            post_list = []
            for t in range(1, T + 1):
                q_t = temp_transitions[name][t - 1]
                q_bar_tm1 = temp_cum_transitions[name][t - 1]
                q_bar_t = temp_cum_transitions[name][t]
                num = q_bar_tm1.unsqueeze(2) * q_t.unsqueeze(0)
                denom = q_bar_t.unsqueeze(1).clamp(min=1e-12)
                post_list.append((num / denom).sum(dim=0))
            self.register_buffer(f"post_{name}", torch.stack(post_list, dim=0))

    def forward(self, x_t, cond, t):
        batch_size = x_t.size(0)

        feat_tokens = []
        for name in self.feat_names:
            idx = self.feat_idx_map[name]
            feat_tokens.append(self.feature_embeddings[name](x_t[:, idx]))
        feat_tokens = torch.stack(feat_tokens, dim=1)
        feat_tokens = feat_tokens + self.feature_pos_embed

        cond_embeds = []
        for j, cond_feat in enumerate(self.cond_info):
            name = cond_feat["name"]
            cond_embeds.append(self.cond_embeddings[name](cond[:, j]))
        cond_raw = torch.cat(cond_embeds, dim=1)
        cond_ctx = self.cond_projector(cond_raw)
        t_emb = self.time_embedding(t)
        global_token = (cond_ctx + t_emb).unsqueeze(1)

        tokens = torch.cat([global_token, feat_tokens], dim=1)
        hidden = self.backbone(tokens)
        feat_hidden = hidden[:, 1:, :]

        logits = {}
        hidden_state_map = {}
        for i, name in enumerate(self.feat_names):
            h = feat_hidden[:, i, :]
            hidden_state_map[name] = h
            logits[name] = self.output_heads[name](h)

        joint_logits = []
        for i, (name1, name2) in enumerate(self.joint_head_pairs):
            h_pair = torch.cat([hidden_state_map[name1], hidden_state_map[name2]], dim=1)
            joint_logits.append(self.joint_heads[i](h_pair))

        return logits, joint_logits
