"""Continuous Embedding-DDPM for discrete trip tables.

Pipeline:
  discrete trip ids  --embed-->  x0 in R^{F x D}
  Gaussian DDPM forward / reverse in embedding space
  decode each token by similarity to its embedding table (argmax by default)

This is intentionally different from the discrete D3PM / Q_t path used by d3pm_*.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(min=1e-5, max=0.999).float()


def _shift_betas(betas: torch.Tensor, shift: float) -> torch.Tensor:
    """Shift a schedule in log-SNR space: logSNR'(t) = logSNR(t) - 2 log(shift)."""
    alpha_bar = torch.cumprod(1.0 - betas.double(), dim=0).clamp(1e-12, 1 - 1e-12)
    log_snr = torch.log(alpha_bar) - torch.log1p(-alpha_bar)
    alpha_bar_new = torch.sigmoid(log_snr - 2.0 * math.log(shift))
    prev = torch.cat([torch.ones(1, dtype=alpha_bar_new.dtype), alpha_bar_new[:-1]])
    return (1 - alpha_bar_new / prev).clamp(min=1e-5, max=0.999).float()


def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EmbeddingDDPM(nn.Module):
    """Gaussian DDPM over learned feature embeddings (Transformer or MLP denoiser)."""

    def __init__(
        self,
        features_info: Sequence[Dict[str, Any]],
        cond_info: Sequence[Dict[str, Any]],
        T: int = 100,
        d_model: int = 128,
        backbone: str = "transformer",
        nhead: int = 8,
        num_layers: int = 4,
        mlp_hidden: Optional[List[int]] = None,
        dropout: float = 0.1,
        beta_schedule: str = "cosine",
        feature_embedding_mode: str = "learned",
        fixed_codebook_type: str = "random",
        decode_metric: str = "dot",
        x0_ce_weight: float = 0.0,
        sample_method: str = "ddpm",
        embed_dim: Optional[int] = None,
        snr_shift: float = 1.0,
        parameterization: str = "eps",
        cond_tokens: bool = False,
        cond_dropout: float = 0.0,
        guidance_scale: float = 0.0,
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unsupported backbone={backbone!r}")
        if feature_embedding_mode not in {"learned", "fixed"}:
            raise ValueError(f"Unsupported feature_embedding_mode={feature_embedding_mode!r}")
        if fixed_codebook_type not in {"random", "structured"}:
            raise ValueError(f"Unsupported fixed_codebook_type={fixed_codebook_type!r}")
        if decode_metric not in {"dot", "cosine"}:
            raise ValueError(f"Unsupported decode_metric={decode_metric!r}")
        if sample_method not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sample_method={sample_method!r}")
        if parameterization not in {"eps", "x0"}:
            raise ValueError(f"Unsupported parameterization={parameterization!r}")
        if cond_dropout > 0 and not cond_tokens:
            raise ValueError("cond_dropout requires cond_tokens=True (needs a null condition token).")
        self.features_info = list(features_info)
        self.cond_info = list(cond_info)
        self.T = int(T)
        self.d_model = int(d_model)
        self.backbone_type = backbone
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.mlp_hidden = list(mlp_hidden) if mlp_hidden is not None else None
        self.beta_schedule = beta_schedule
        self.feature_embedding_mode = feature_embedding_mode
        self.fixed_codebook_type = fixed_codebook_type
        self.decode_metric = decode_metric
        self.x0_ce_weight = float(x0_ce_weight)
        self.sample_method = sample_method
        # Diffusion-space width per feature. None keeps the legacy behaviour (= d_model).
        self.embed_dim = int(embed_dim) if embed_dim else self.d_model
        # log-SNR shift: logSNR'(t) = logSNR(t) - 2 log(snr_shift). >1 adds noise earlier.
        self.snr_shift = float(snr_shift)
        self.parameterization = parameterization
        self.cond_tokens = bool(cond_tokens)
        self.cond_dropout = float(cond_dropout)
        self.guidance_scale = float(guidance_scale)
        self.feat_names = [f["name"] for f in self.features_info]
        self.num_features = len(self.feat_names)

        self.feature_embeddings = nn.ModuleDict(
            {f["name"]: nn.Embedding(f["num_classes"], self.embed_dim) for f in self.features_info}
        )
        if self.feature_embedding_mode == "fixed":
            # DDPM assumes x0 has O(1) variance per coordinate. Fixed spherical
            # codebooks prevent the epsilon objective from shrinking embeddings
            # toward zero and keep unseen/rare categories on the same norm.
            for feature in self.features_info:
                embedding = self.feature_embeddings[feature["name"]]
                with torch.no_grad():
                    if self.fixed_codebook_type == "structured" and feature.get("type") == "ordinal":
                        positions = torch.arange(
                            feature["num_classes"], dtype=embedding.weight.dtype
                        ).unsqueeze(1)
                        frequencies = torch.arange(
                            1, self.embed_dim // 2 + 1, dtype=embedding.weight.dtype
                        ).unsqueeze(0)
                        angles = 2.0 * math.pi * positions * frequencies / feature["num_classes"]
                        codebook = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
                        codebook = codebook[:, : self.embed_dim]
                    elif self.fixed_codebook_type == "structured":
                        random_matrix = torch.randn(
                            self.embed_dim,
                            feature["num_classes"],
                            dtype=embedding.weight.dtype,
                        )
                        codebook = torch.linalg.qr(random_matrix, mode="reduced").Q.t()
                    else:
                        codebook = embedding.weight
                    codebook = codebook - codebook.mean(dim=0, keepdim=True)
                    embedding.weight.copy_(
                        F.normalize(codebook, dim=-1) * math.sqrt(self.embed_dim)
                    )
                embedding.weight.requires_grad_(False)
        self.cond_embeddings = nn.ModuleDict(
            {c["name"]: nn.Embedding(c["num_classes"], 16) for c in self.cond_info}
        )
        self.cond_projector = nn.Linear(16 * max(len(self.cond_info), 1), self.d_model)
        if self.cond_tokens:
            # One token per conditioning field; index num_classes is the null (dropped) condition.
            self.cond_token_embeddings = nn.ModuleDict(
                {c["name"]: nn.Embedding(c["num_classes"] + 1, self.d_model) for c in self.cond_info}
            )
            self.cond_type_embed = nn.Parameter(torch.randn(1, len(self.cond_info), self.d_model) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.SiLU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        self.feature_pos_embed = nn.Parameter(
            torch.randn(1, self.num_features, self.d_model) * 0.02
        )

        if backbone == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=self.d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.denoiser = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.in_proj = (
                nn.Linear(self.embed_dim, self.d_model) if self.embed_dim != self.d_model else nn.Identity()
            )
            self.eps_head = nn.Linear(self.d_model, self.embed_dim)
        else:
            hidden = mlp_hidden or [512, 512, 512]
            if self.cond_tokens or self.parameterization != "eps":
                raise ValueError("cond_tokens / x0 parameterization are implemented for the transformer backbone only.")
            in_dim = self.num_features * self.embed_dim + self.d_model  # flat x_t + cond/time
            dims = [in_dim] + list(hidden) + [self.num_features * self.embed_dim]
            layers: List[nn.Module] = []
            for i in range(len(dims) - 2):
                layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.denoiser = nn.Sequential(*layers)
            self.in_proj = nn.Identity()
            self.eps_head = nn.Identity()
        if self.parameterization == "x0":
            self.x0_heads = nn.ModuleDict(
                {f["name"]: nn.Linear(self.d_model, f["num_classes"]) for f in self.features_info}
            )

        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.T)
        elif beta_schedule == "linear":
            betas = _linear_beta_schedule(self.T)
        else:
            raise ValueError(f"Unknown beta_schedule={beta_schedule}")
        if self.snr_shift != 1.0:
            betas = _shift_betas(betas, self.snr_shift)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        # posterior variance for sampling
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance", torch.log(posterior_var.clamp(min=1e-20)))

    def encode_features(self, x_ids: torch.Tensor) -> torch.Tensor:
        """(B, F) long -> (B, F, D) embeddings."""
        tokens = []
        for i, name in enumerate(self.feat_names):
            tokens.append(self.feature_embeddings[name](x_ids[:, i]))
        return torch.stack(tokens, dim=1)

    def encode_cond(self, cond_ids: torch.Tensor) -> torch.Tensor:
        if not self.cond_info:
            return torch.zeros(cond_ids.size(0), self.d_model, device=cond_ids.device)
        embeds = [
            self.cond_embeddings[c["name"]](cond_ids[:, j]) for j, c in enumerate(self.cond_info)
        ]
        return self.cond_projector(torch.cat(embeds, dim=1))

    def decode_features(self, x_emb: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        """(B, F, D) -> (B, F) discrete ids via embedding similarity."""
        outs = []
        for i, name in enumerate(self.feat_names):
            weight = self.feature_embeddings[name].weight  # (K, D)
            query = x_emb[:, i, :]
            if self.decode_metric == "cosine":
                query = F.normalize(query, dim=-1)
                weight = F.normalize(weight, dim=-1)
            logits = torch.matmul(query, weight.t())  # (B, K)
            if temperature and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                outs.append(torch.multinomial(probs, num_samples=1).squeeze(1))
            else:
                outs.append(torch.argmax(logits, dim=-1))
        return torch.stack(outs, dim=1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        # t in [0, T-1]
        sa = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sa * x0 + so * noise, noise

    def _null_cond(self, cond_ids: torch.Tensor) -> torch.Tensor:
        null = torch.tensor([c["num_classes"] for c in self.cond_info], device=cond_ids.device)
        return null.unsqueeze(0).expand_as(cond_ids)

    def _hidden(self, x_t: torch.Tensor, cond_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Shared denoiser trunk; returns (B, F, d_model) hidden states for the feature tokens."""
        t_emb = self.time_mlp(t)
        if self.backbone_type == "transformer":
            tokens = self.in_proj(x_t) + self.feature_pos_embed
            if self.cond_tokens:
                cond_tok = torch.stack(
                    [self.cond_token_embeddings[c["name"]](cond_ids[:, j]) for j, c in enumerate(self.cond_info)],
                    dim=1,
                ) + self.cond_type_embed
                prefix = torch.cat([t_emb.unsqueeze(1), cond_tok], dim=1)
            else:
                prefix = (self.encode_cond(cond_ids) + t_emb).unsqueeze(1)
            hidden = self.denoiser(torch.cat([prefix, tokens], dim=1))
            return hidden[:, prefix.size(1):, :]
        flat = torch.cat([x_t.reshape(x_t.size(0), -1), self.encode_cond(cond_ids) + t_emb], dim=1)
        return self.denoiser(flat).view_as(x_t)

    def _x0_logits(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        return [self.x0_heads[name](hidden[:, i, :]) for i, name in enumerate(self.feat_names)]

    def _expected_embedding(self, logits: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [F.softmax(lg, dim=-1) @ self.feature_embeddings[name].weight for lg, name in zip(logits, self.feat_names)],
            dim=1,
        )

    def predict_eps(self, x_t: torch.Tensor, cond_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Epsilon prediction (eps parameterization only; kept for legacy callers)."""
        return self.eps_head(self._hidden(x_t, cond_ids, t))

    def predict_x0(self, x_t: torch.Tensor, cond_ids: torch.Tensor, t: torch.Tensor):
        """Return (x0_hat embedding, per-feature logits or None) under either parameterization."""
        hidden = self._hidden(x_t, cond_ids, t)
        if self.parameterization == "x0":
            logits = self._x0_logits(hidden)
            return self._expected_embedding(logits), logits
        eps = self.eps_head(hidden)
        sa = self.sqrt_alpha_bar[t].view(-1, 1, 1).clamp(min=1e-8)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return (x_t - so * eps) / sa, None

    def _guided_x0(self, x_t: torch.Tensor, cond_ids: torch.Tensor, t: torch.Tensor):
        x0, logits = self.predict_x0(x_t, cond_ids, t)
        w = self.guidance_scale
        if w <= 0 or not self.cond_tokens:
            return x0, logits
        x0_u, logits_u = self.predict_x0(x_t, self._null_cond(cond_ids), t)
        if logits is not None:
            logits = [(1 + w) * lc - w * lu for lc, lu in zip(logits, logits_u)]
            return self._expected_embedding(logits), logits
        return (1 + w) * x0 - w * x0_u, None

    def forward_loss(self, x_ids: torch.Tensor, cond_ids: torch.Tensor) -> torch.Tensor:
        bsz = x_ids.size(0)
        device = x_ids.device
        t = torch.randint(0, self.T, (bsz,), device=device, dtype=torch.long)
        if self.cond_dropout > 0 and self.training:
            drop = torch.rand(bsz, device=device) < self.cond_dropout
            cond_ids = torch.where(drop.unsqueeze(1), self._null_cond(cond_ids), cond_ids)
        x0 = self.encode_features(x_ids)
        x_t, noise = self.q_sample(x0, t)
        hidden = self._hidden(x_t, cond_ids, t)
        if self.parameterization == "x0":
            logits = self._x0_logits(hidden)
            return torch.stack([F.cross_entropy(lg, x_ids[:, i]) for i, lg in enumerate(logits)]).mean()
        pred = self.eps_head(hidden)
        loss = F.mse_loss(pred, noise)
        if self.x0_ce_weight > 0:
            sa = self.sqrt_alpha_bar[t].view(-1, 1, 1).clamp(min=1e-4)
            so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            x0_pred = (x_t - so * pred) / sa
            ce_losses = []
            for i, name in enumerate(self.feat_names):
                query = F.normalize(x0_pred[:, i, :], dim=-1)
                codebook = F.normalize(self.feature_embeddings[name].weight, dim=-1)
                logits = 10.0 * torch.matmul(query, codebook.t())
                ce_losses.append(F.cross_entropy(logits, x_ids[:, i], reduction="none"))
            ce_per_row = torch.stack(ce_losses, dim=1).mean(dim=1)
            # Classification is meaningful when x_t retains signal. Weighting
            # by alpha_bar avoids forcing majority-class guesses at pure noise.
            signal_weight = self.alpha_bar[t].detach()
            ce_loss = (ce_per_row * signal_weight).sum() / signal_weight.sum().clamp(min=1e-6)
            loss = loss + self.x0_ce_weight * ce_loss
        return loss

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, cond_ids: torch.Tensor, t: int):
        """Ancestral step via the posterior q(x_{t-1} | x_t, x0_hat). Returns (x_{t-1}, logits).

        For the eps parameterization this equals the standard DDPM mean
        1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps).
        """
        t_batch = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        if self.parameterization == "eps" and not (self.guidance_scale > 0 and self.cond_tokens):
            # Legacy epsilon form, kept bit-for-bit so earlier checkpoints reproduce exactly.
            eps = self.predict_eps(x_t, cond_ids, t_batch)
            mean = (1.0 / torch.sqrt(self.alphas[t])) * (
                x_t - (self.betas[t] / torch.sqrt(1.0 - self.alpha_bar[t]).clamp(min=1e-8)) * eps
            )
            if t == 0:
                return mean, None
            return mean + torch.sqrt(self.posterior_variance[t]) * torch.randn_like(x_t), None
        x0, logits = self._guided_x0(x_t, cond_ids, t_batch)
        if t == 0:
            return x0, logits
        ab_t = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t - 1]
        coef_x0 = torch.sqrt(ab_prev) * self.betas[t] / (1.0 - ab_t)
        coef_xt = torch.sqrt(self.alphas[t]) * (1.0 - ab_prev) / (1.0 - ab_t)
        mean = coef_x0 * x0 + coef_xt * x_t
        return mean + torch.sqrt(self.posterior_variance[t]) * torch.randn_like(x_t), logits

    @torch.no_grad()
    def ddim_sample(self, x_t: torch.Tensor, cond_ids: torch.Tensor, t: int):
        """Deterministic DDIM step. Returns (x_{t-1}, logits)."""
        t_batch = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        x0, logits = self._guided_x0(x_t, cond_ids, t_batch)
        if t == 0:
            return x0, logits
        ab_t = self.alpha_bar[t]
        eps = (x_t - torch.sqrt(ab_t) * x0) / torch.sqrt(1.0 - ab_t).clamp(min=1e-8)
        ab_prev = self.alpha_bar[t - 1]
        return torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps, logits

    @torch.no_grad()
    def sample(
        self,
        cond_ids: torch.Tensor,
        temperature: float = 0.0,
        progress: bool = False,
    ) -> torch.Tensor:
        device = cond_ids.device
        bsz = cond_ids.size(0)
        x = torch.randn(bsz, self.num_features, self.embed_dim, device=device)
        steps = range(self.T - 1, -1, -1)
        if progress:
            steps = tqdm(list(steps), desc="Embedding-DDPM sample", leave=False)
        logits = None
        step = self.ddim_sample if self.sample_method == "ddim" else self.p_sample
        for t in steps:
            x, logits = step(x, cond_ids, t)
        if logits is not None:
            # x0 parameterization: read the final categorical prediction directly.
            if temperature and temperature > 0:
                return torch.stack(
                    [torch.multinomial(F.softmax(lg / temperature, dim=-1), 1).squeeze(1) for lg in logits], dim=1
                )
            return torch.stack([lg.argmax(dim=-1) for lg in logits], dim=1)
        return self.decode_features(x, temperature=temperature)


def _build_id_tensors(
    df,
    feat_cols: Sequence[str],
    cond_cols: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd

    x = torch.tensor(df[list(feat_cols)].astype(int).values, dtype=torch.long)
    c = torch.tensor(df[list(cond_cols)].astype(int).values, dtype=torch.long)
    return x, c


def train_embedding_ddpm(
    model: EmbeddingDDPM,
    train_df,
    feat_cols: Sequence[str],
    cond_cols: Sequence[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    model_save_path: Optional[str] = None,
) -> EmbeddingDDPM:
    x_all, c_all = _build_id_tensors(train_df, feat_cols, cond_cols)
    loader = DataLoader(
        TensorDataset(x_all, c_all),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x_ids, cond_ids in loader:
            x_ids = x_ids.to(device)
            cond_ids = cond_ids.to(device)
            loss = model.forward_loss(x_ids, cond_ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * x_ids.size(0)
            n += x_ids.size(0)
        avg = total / max(n, 1)
        improved = avg < best_loss
        if improved:
            best_loss = avg
            if model_save_path:
                torch.save(
                    {
                        "model_type": "embedding_ddpm",
                        "backbone": model.backbone_type,
                        "state_dict": model.state_dict(),
                        "features_info": model.features_info,
                        "cond_info": model.cond_info,
                        "T": model.T,
                        "d_model": model.d_model,
                        "nhead": model.nhead,
                        "num_layers": model.num_layers,
                        "dropout": model.dropout,
                        "mlp_hidden": model.mlp_hidden,
                        "beta_schedule": model.beta_schedule,
                        "feature_embedding_mode": model.feature_embedding_mode,
                        "fixed_codebook_type": model.fixed_codebook_type,
                        "decode_metric": model.decode_metric,
                        "x0_ce_weight": model.x0_ce_weight,
                        "sample_method": model.sample_method,
                        "embed_dim": model.embed_dim,
                        "snr_shift": model.snr_shift,
                        "parameterization": model.parameterization,
                        "cond_tokens": model.cond_tokens,
                        "cond_dropout": model.cond_dropout,
                        "guidance_scale": model.guidance_scale,
                    },
                    model_save_path,
                )
        logging.info(
            "Embedding-DDPM epoch %d/%d: loss=%.6f%s",
            epoch,
            epochs,
            avg,
            " (New best model saved!)" if improved and model_save_path else "",
        )

    if model_save_path:
        ckpt = torch.load(model_save_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        logging.info("Loaded best Embedding-DDPM weights from %s", model_save_path)
    return model


@torch.no_grad()
def sample_embedding_ddpm(
    model: EmbeddingDDPM,
    test_df,
    feat_cols: Sequence[str],
    cond_cols: Sequence[str],
    n_samples: int,
    device: torch.device,
    batch_size: int = 2048,
    temperature: float = 0.0,
    match_test_one_to_one: bool = False,
):
    import pandas as pd

    model = model.to(device)
    model.eval()
    if match_test_one_to_one or n_samples is None or n_samples <= 0:
        cond_src = test_df
        n_samples = len(test_df)
    else:
        # unconditional-style: sample conditions with replacement from test
        cond_src = test_df.sample(n=n_samples, replace=True, random_state=0).reset_index(drop=True)

    rows = []
    for start in tqdm(range(0, n_samples, batch_size), desc="Embedding-DDPM generate"):
        end = min(start + batch_size, n_samples)
        batch_df = cond_src.iloc[start:end]
        cond_ids = torch.tensor(batch_df[list(cond_cols)].astype(int).values, dtype=torch.long, device=device)
        trip_ids = model.sample(cond_ids, temperature=temperature, progress=False).cpu()
        for i in range(trip_ids.size(0)):
            row = {c: int(batch_df.iloc[i][c]) for c in cond_cols}
            for j, name in enumerate(feat_cols):
                row[name] = int(trip_ids[i, j].item())
            rows.append(row)
    return pd.DataFrame(rows)
