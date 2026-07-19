import torch
import torch.nn as nn

# Ordered ST sub-chains within the ST stream (loc then time).
ST_LOC_CHAIN_NAMES = ["start_type", "start_zcode_num", "end_type", "end_zcode_num"]
ST_TIME_CHAIN_NAMES = ["start_time_num_6", "trip_time_num_6"]

# Preset two-phase cascades over ST tokens.
# phase1 tokens are updated first; phase2 then conditions on finished phase1.
ST_CASCADE_PRESETS = {
    # Current default / paper st_cascade.
    "loc_then_time": (ST_LOC_CHAIN_NAMES, ST_TIME_CHAIN_NAMES),
    # Reverse phase order: schedule first, then location.
    "time_then_loc": (ST_TIME_CHAIN_NAMES, ST_LOC_CHAIN_NAMES),
    # Destination-first OD, then times.
    "end_first_loc": (
        ["end_type", "end_zcode_num", "start_type", "start_zcode_num"],
        ST_TIME_CHAIN_NAMES,
    ),
    # Zone codes before land-use types.
    "zcode_first": (
        ["start_zcode_num", "end_zcode_num", "start_type", "end_type"],
        ST_TIME_CHAIN_NAMES,
    ),
    # Types for both ends, then zones, then times.
    "types_then_z": (
        ["start_type", "end_type", "start_zcode_num", "end_zcode_num"],
        ST_TIME_CHAIN_NAMES,
    ),
    # Single trip-order chain (no second phase).
    "start_then_end": (
        [
            "start_type",
            "start_zcode_num",
            "start_time_num_6",
            "end_type",
            "end_zcode_num",
            "trip_time_num_6",
        ],
        [],
    ),
}


def resolve_st_cascade_phases(chain_name):
    """Return (phase1_names, phase2_names) for a cascade preset."""
    if chain_name not in ST_CASCADE_PRESETS:
        raise ValueError(
            f"Unknown st_cascade_chain={chain_name!r}. "
            f"Choose from: {sorted(ST_CASCADE_PRESETS)}"
        )
    return ST_CASCADE_PRESETS[chain_name]


class _CascadeTokenStep(nn.Module):
    """Update one token from causal predecessors + global context."""

    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, ctx):
        res = x
        x1, _ = self.self_attn(x, x, x)
        x = self.norm1(res + x1)

        res = x
        x2, _ = self.cross_attn(query=x, key=ctx, value=ctx)
        x = self.norm2(res + x2)

        return self.norm3(x + self.ffn(x))


class SoftCausalAdapterBlock(nn.Module):
    """
    Lightweight causal adapter with soft constraints:
    - Act can see global context.
    - ST can see global context + Act.
    - Mode can see global context + Act + ST.
    Global shared feature tokens are always available in each stream.
    """

    def __init__(self, d_model, nhead, dropout=0.2, gate_init=None, st_cascade=False, st_loc_chain_idx=None, st_time_chain_idx=None, hard_stream_cascade=False):
        super().__init__()
        gate_init = gate_init or {}
        self.st_cascade = st_cascade
        self.hard_stream_cascade = bool(hard_stream_cascade)
        self.st_loc_chain_idx = st_loc_chain_idx or []
        self.st_time_chain_idx = st_time_chain_idx or []
        self.act_self = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.st_self = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.mode_self = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.act_cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.st_cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.mode_cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.act_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.st_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.mode_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.act_norm1 = nn.LayerNorm(d_model)
        self.act_norm2 = nn.LayerNorm(d_model)
        self.act_norm3 = nn.LayerNorm(d_model)
        self.st_norm1 = nn.LayerNorm(d_model)
        self.st_norm2 = nn.LayerNorm(d_model)
        self.st_norm3 = nn.LayerNorm(d_model)
        self.mode_norm1 = nn.LayerNorm(d_model)
        self.mode_norm2 = nn.LayerNorm(d_model)
        self.mode_norm3 = nn.LayerNorm(d_model)

        # Start close to identity, then let training learn causal strength.
        # With freeze_gates=True these stay fixed (soft-causal on/off ablation).
        self.gate_act = nn.Parameter(torch.tensor(float(gate_init.get("act", -1.0))))
        self.gate_st = nn.Parameter(torch.tensor(float(gate_init.get("st", -1.0))))
        self.gate_mode = nn.Parameter(torch.tensor(float(gate_init.get("mode", -1.0))))
        if bool(gate_init.get("freeze", False)):
            self.gate_act.requires_grad_(False)
            self.gate_st.requires_grad_(False)
            self.gate_mode.requires_grad_(False)

        if self.st_cascade:
            # Keep module names for backward-compatible checkpoints.
            self.loc_cascade_steps = nn.ModuleList(
                [_CascadeTokenStep(d_model, nhead, dropout=dropout) for _ in self.st_loc_chain_idx]
            )
            self.time_cascade_steps = nn.ModuleList(
                [_CascadeTokenStep(d_model, nhead, dropout=dropout) for _ in self.st_time_chain_idx]
            )

    def _update_st_cascade(self, h_st, st_ctx):
        """Two-phase ST token cascade: phase1 then phase2 (phase2 sees phase1)."""
        h_st_new = h_st.clone()

        phase1_ctx = st_ctx
        for step_idx, st_idx in enumerate(self.st_loc_chain_idx):
            token = h_st[:, st_idx : st_idx + 1, :]
            token_new = self.loc_cascade_steps[step_idx](token, phase1_ctx)
            h_st_new[:, st_idx : st_idx + 1, :] = token_new
            phase1_ctx = torch.cat([phase1_ctx, token_new], dim=1)

        if self.st_time_chain_idx:
            if self.st_loc_chain_idx:
                phase2_ctx = torch.cat([st_ctx, h_st_new[:, self.st_loc_chain_idx, :]], dim=1)
            else:
                phase2_ctx = st_ctx
            for step_idx, st_idx in enumerate(self.st_time_chain_idx):
                token = h_st[:, st_idx : st_idx + 1, :]
                token_new = self.time_cascade_steps[step_idx](token, phase2_ctx)
                h_st_new[:, st_idx : st_idx + 1, :] = token_new
                phase2_ctx = torch.cat([phase2_ctx, token_new], dim=1)

        return h_st_new

    def _update_stream(self, x, self_attn, cross_attn, ffn, norm1, norm2, norm3, ctx):
        res = x
        x1, _ = self_attn(x, x, x)
        x = norm1(res + x1)

        res = x
        x2, _ = cross_attn(query=x, key=ctx, value=ctx)
        x = norm2(res + x2)

        x = norm3(x + ffn(x))
        return x

    def _gated_blend(self, x_old, x_new, gate):
        alpha = torch.sigmoid(gate)
        return x_old + alpha * (x_new - x_old)

    def forward(self, h_act, h_st, h_mode, h_cond, h_shared):
        if self.hard_stream_cascade:
            # True stream-level hard cascade: act -> st -> mode (each sees updated upstream).
            act_ctx = torch.cat([h_cond, h_shared], dim=1)
            act_new = self._update_stream(
                h_act, self.act_self, self.act_cross, self.act_ffn, self.act_norm1, self.act_norm2, self.act_norm3, act_ctx
            )
            h_act = act_new

            st_ctx = torch.cat([h_cond, h_shared, h_act], dim=1)
            if self.st_cascade:
                st_new = self._update_st_cascade(h_st, st_ctx)
            else:
                st_new = self._update_stream(
                    h_st, self.st_self, self.st_cross, self.st_ffn, self.st_norm1, self.st_norm2, self.st_norm3, st_ctx
                )
            h_st = st_new

            mode_ctx = torch.cat([h_cond, h_shared, h_act, h_st], dim=1)
            mode_new = self._update_stream(
                h_mode, self.mode_self, self.mode_cross, self.mode_ffn, self.mode_norm1, self.mode_norm2, self.mode_norm3, mode_ctx
            )
            h_mode = mode_new
            return h_act, h_st, h_mode

        # Soft / parallel causal adapters (default): contexts use pre-update streams, then gated residual.
        act_ctx = torch.cat([h_cond, h_shared], dim=1)
        st_ctx = torch.cat([h_cond, h_shared, h_act], dim=1)
        mode_ctx = torch.cat([h_cond, h_shared, h_act, h_st], dim=1)

        act_new = self._update_stream(
            h_act, self.act_self, self.act_cross, self.act_ffn, self.act_norm1, self.act_norm2, self.act_norm3, act_ctx
        )
        if self.st_cascade:
            st_new = self._update_st_cascade(h_st, st_ctx)
        else:
            st_new = self._update_stream(
                h_st, self.st_self, self.st_cross, self.st_ffn, self.st_norm1, self.st_norm2, self.st_norm3, st_ctx
            )
        mode_new = self._update_stream(
            h_mode, self.mode_self, self.mode_cross, self.mode_ffn, self.mode_norm1, self.mode_norm2, self.mode_norm3, mode_ctx
        )

        h_act = self._gated_blend(h_act, act_new, self.gate_act)
        h_st = self._gated_blend(h_st, st_new, self.gate_st)
        h_mode = self._gated_blend(h_mode, mode_new, self.gate_mode)
        return h_act, h_st, h_mode


class TripDiffusionModel(nn.Module):
    def __init__(
        self,
        features_info,
        cond_info,
        T,
        joint_pairs=None,
        gate_init=None,
        freeze_gates=False,
        st_cascade=False,
        st_cascade_chain="loc_then_time",
        hard_stream_cascade=False,
        use_joint_heads=True,
        d_model=128,
        shared_layers=2,
        causal_layers=2,
    ):
        super().__init__()
        gate_init = dict(gate_init or {})
        if freeze_gates:
            gate_init["freeze"] = True
        self.features_info = features_info
        self.cond_info = cond_info
        self.T = T
        self.joint_pairs = joint_pairs if joint_pairs is not None else []
        self.st_cascade = st_cascade
        self.st_cascade_chain = st_cascade_chain if st_cascade else None
        self.hard_stream_cascade = bool(hard_stream_cascade)
        self.use_joint_heads = use_joint_heads
        self.freeze_gates = bool(freeze_gates)

        # Keep group definitions for compatibility with causal loss path.
        self.group_act_names = ["act_num"]
        self.group_mode_names = ["mode_num"]
        self.group_st_names = [
            f["name"]
            for f in features_info
            if f["name"] not in self.group_act_names and f["name"] not in self.group_mode_names
        ]

        self.d_model = d_model
        self.nhead = max(1, d_model // 32)
        self.shared_layers = shared_layers
        self.causal_layers = causal_layers
        self.dropout = 0.2

        self.feat_names = [f["name"] for f in features_info]
        self.feat_idx_map = {f["name"]: i for i, f in enumerate(features_info)}
        self.group_act_idx = [self.feat_idx_map[n] for n in self.group_act_names]
        self.group_st_idx = [self.feat_idx_map[n] for n in self.group_st_names]
        self.group_mode_idx = [self.feat_idx_map[n] for n in self.group_mode_names]

        st_name_to_pos = {name: i for i, name in enumerate(self.group_st_names)}
        if st_cascade:
            phase1_names, phase2_names = resolve_st_cascade_phases(st_cascade_chain)
        else:
            phase1_names, phase2_names = ST_LOC_CHAIN_NAMES, ST_TIME_CHAIN_NAMES
        missing = [n for n in list(phase1_names) + list(phase2_names) if n not in st_name_to_pos]
        if missing:
            raise ValueError(f"Cascade chain refers to unknown ST features: {missing}")
        # st_loc_chain_idx / st_time_chain_idx = phase1 / phase2 (names kept for checkpoints).
        self.st_loc_chain_idx = [st_name_to_pos[n] for n in phase1_names]
        self.st_time_chain_idx = [st_name_to_pos[n] for n in phase2_names]
        self.st_cascade_phase1_names = list(phase1_names)
        self.st_cascade_phase2_names = list(phase2_names)

        self.feature_embeddings = nn.ModuleDict()
        for feat in features_info:
            self.feature_embeddings[feat["name"]] = nn.Embedding(feat["num_classes"], self.d_model)
        self.feature_pos_embed = nn.Parameter(torch.randn(1, len(self.feat_names), self.d_model) * 0.02)

        self.cond_embeddings = nn.ModuleDict()
        for cond in cond_info:
            self.cond_embeddings[cond["name"]] = nn.Embedding(cond["num_classes"], 16)
        self.cond_projector = nn.Linear(16 * len(cond_info), self.d_model)
        self.time_embedding = nn.Embedding(T + 1, self.d_model)

        shared_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.shared_encoder = nn.TransformerEncoder(shared_layer, num_layers=self.shared_layers)
        self.causal_adapters = nn.ModuleList(
            [
                SoftCausalAdapterBlock(
                    self.d_model,
                    self.nhead,
                    dropout=self.dropout,
                    gate_init=gate_init,
                    st_cascade=st_cascade,
                    st_loc_chain_idx=self.st_loc_chain_idx,
                    st_time_chain_idx=self.st_time_chain_idx,
                    hard_stream_cascade=self.hard_stream_cascade,
                )
                for _ in range(self.causal_layers)
            ]
        )

        self.output_heads = nn.ModuleDict()
        for feat in features_info:
            self.output_heads[feat["name"]] = nn.Linear(self.d_model, feat["num_classes"])

        self.joint_heads = nn.ModuleList()
        self.joint_head_pairs = []
        if self.use_joint_heads:
            for idx1, idx2 in self.joint_pairs:
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

    def get_posterior(self, name):
        return getattr(self, f"post_{name}")

    def get_gate_values(self):
        """Return learned gate parameters for each causal adapter layer."""
        gates = []
        for layer_idx, block in enumerate(self.causal_adapters):
            gate_act = block.gate_act.detach().cpu()
            gate_st = block.gate_st.detach().cpu()
            gate_mode = block.gate_mode.detach().cpu()
            gates.append(
                {
                    "layer": layer_idx,
                    "gate_act_raw": float(gate_act),
                    "gate_st_raw": float(gate_st),
                    "gate_mode_raw": float(gate_mode),
                    "alpha_act": float(torch.sigmoid(gate_act)),
                    "alpha_st": float(torch.sigmoid(gate_st)),
                    "alpha_mode": float(torch.sigmoid(gate_mode)),
                    "frozen": not bool(block.gate_act.requires_grad),
                }
            )
        return gates

    def forward(self, x_t, cond, t):
        feat_tokens = []
        for name in self.feat_names:
            idx = self.feat_idx_map[name]
            feat_tokens.append(self.feature_embeddings[name](x_t[:, idx]))
        feat_tokens = torch.stack(feat_tokens, dim=1) + self.feature_pos_embed

        cond_embeds = []
        for j, cond_feat in enumerate(self.cond_info):
            name = cond_feat["name"]
            cond_embeds.append(self.cond_embeddings[name](cond[:, j]))
        cond_raw = torch.cat(cond_embeds, dim=1)
        cond_ctx = self.cond_projector(cond_raw)
        t_emb = self.time_embedding(t)
        global_token = (cond_ctx + t_emb).unsqueeze(1)

        shared_tokens = torch.cat([global_token, feat_tokens], dim=1)
        shared_hidden = self.shared_encoder(shared_tokens)
        h_cond = shared_hidden[:, :1, :]
        h_shared = shared_hidden[:, 1:, :]

        h_act = h_shared[:, self.group_act_idx, :]
        h_st = h_shared[:, self.group_st_idx, :]
        h_mode = h_shared[:, self.group_mode_idx, :]

        for block in self.causal_adapters:
            h_act, h_st, h_mode = block(h_act, h_st, h_mode, h_cond, h_shared)

        hidden_state_map = {}
        for i, name in enumerate(self.group_act_names):
            hidden_state_map[name] = h_act[:, i, :]
        for i, name in enumerate(self.group_st_names):
            hidden_state_map[name] = h_st[:, i, :]
        for i, name in enumerate(self.group_mode_names):
            hidden_state_map[name] = h_mode[:, i, :]

        logits = {}
        for name in self.feat_names:
            logits[name] = self.output_heads[name](hidden_state_map[name])

        joint_logits = []
        if self.use_joint_heads:
            for i, (name1, name2) in enumerate(self.joint_head_pairs):
                h_pair = torch.cat([hidden_state_map[name1], hidden_state_map[name2]], dim=1)
                joint_logits.append(self.joint_heads[i](h_pair))

        return logits, joint_logits
