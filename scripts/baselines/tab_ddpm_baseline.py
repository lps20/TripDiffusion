"""
Lightweight TabDDPM adapter for trip tabular baselines.

Uses the official TabDDPM core (Gaussian + multinomial diffusion) from:
https://github.com/yandex-research/tab-ddpm

Supports optional demographic conditioning: diffuse trip columns only,
condition the denoiser on one-hot demographics, and sample 1:1 from test conds.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from third_party.tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion, TransformerDiffusion


def _split_tabddpm_columns(
    all_columns: List[str],
    time_columns: List[str],
    schema_by_name: Dict[str, Dict[str, Any]],
    condition_columns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], np.ndarray]:
    cond_set = set(condition_columns or [])
    target_cols = [c for c in all_columns if c not in cond_set]
    num_cols = [c for c in target_cols if c in time_columns]
    cat_cols = [c for c in target_cols if c not in time_columns]
    cat_sizes = np.array([int(schema_by_name[c]["num_classes"]) for c in cat_cols], dtype=np.int64)
    return num_cols, cat_cols, cat_sizes


def _encode_tabddpm_frame(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    schema_by_name: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for col in num_cols:
        upper = float(schema_by_name[col]["num_classes"] - 1)
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, upper).values.astype(np.float32)
        if upper > 0:
            vals = vals / upper
        parts.append(vals.reshape(-1, 1))
    for col in cat_cols:
        upper = int(schema_by_name[col]["num_classes"] - 1)
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int).clip(0, upper).values
        parts.append(vals.reshape(-1, 1).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _encode_condition_onehot(
    df: pd.DataFrame,
    condition_columns: List[str],
    schema_by_name: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for col in condition_columns:
        k = int(schema_by_name[col]["num_classes"])
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int).clip(0, k - 1).values
        parts.append(np.eye(k, dtype=np.float32)[vals])
    if not parts:
        return np.zeros((len(df), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _decode_tabddpm_array(
    arr: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    schema_by_name: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    n_num = len(num_cols)
    out: Dict[str, List[int]] = {}
    for i, col in enumerate(num_cols):
        upper = int(schema_by_name[col]["num_classes"] - 1)
        vals = np.rint(arr[:, i] * upper).astype(int)
        out[col] = np.clip(vals, 0, upper).tolist()
    for j, col in enumerate(cat_cols):
        upper = int(schema_by_name[col]["num_classes"] - 1)
        vals = np.rint(arr[:, n_num + j]).astype(int)
        out[col] = np.clip(vals, 0, upper).tolist()
    return pd.DataFrame(out)


def _update_ema(target_params, source_params, rate: float = 0.999) -> None:
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def _build_tabddpm_denoiser(
    d_in: int,
    backbone: str,
    hidden_layers: Optional[List[int]] = None,
    tf_d_model: int = 128,
    tf_nhead: int = 8,
    tf_layers: int = 4,
    tf_n_tokens: int = 16,
    is_y_cond: bool = False,
    cond_dim: int = 0,
):
    backbone = (backbone or "mlp").lower()
    if backbone == "mlp":
        layers = hidden_layers or [256, 512, 512, 256]
        return MLPDiffusion(
            d_in=d_in,
            num_classes=0,
            is_y_cond=is_y_cond,
            cond_dim=cond_dim,
            rtdl_params={"d_layers": layers, "dropout": 0.0},
        ), {"backbone": "mlp", "hidden_layers": layers, "is_y_cond": is_y_cond, "cond_dim": cond_dim}
    if backbone in {"tf", "transformer"}:
        model = TransformerDiffusion(
            d_in=d_in,
            num_classes=0,
            is_y_cond=is_y_cond,
            cond_dim=cond_dim,
            d_model=tf_d_model,
            nhead=tf_nhead,
            num_layers=tf_layers,
            n_tokens=tf_n_tokens,
        )
        return model, {
            "backbone": "transformer",
            "tf_d_model": tf_d_model,
            "tf_nhead": tf_nhead,
            "tf_layers": tf_layers,
            "tf_n_tokens": tf_n_tokens,
            "is_y_cond": is_y_cond,
            "cond_dim": cond_dim,
        }
    raise ValueError(f"Unsupported TabDDPM backbone: {backbone!r}")


def run_tabddpm(
    train_df: pd.DataFrame,
    all_columns: List[str],
    time_columns: List[str],
    schema: List[Dict[str, Any]],
    n_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_timesteps: int,
    seed: int,
    model_save_path: Optional[str] = None,
    hidden_layers: Optional[List[int]] = None,
    backbone: str = "mlp",
    tf_d_model: int = 128,
    tf_nhead: int = 8,
    tf_layers: int = 4,
    tf_n_tokens: int = 16,
    condition_columns: Optional[List[str]] = None,
    sample_cond_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    from utils.multi_seed import set_global_seed

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema_by_name = {field["name"]: field for field in schema}
    condition_columns = list(condition_columns or [])
    is_conditional = len(condition_columns) > 0

    num_cols, cat_cols, cat_sizes = _split_tabddpm_columns(
        all_columns, time_columns, schema_by_name, condition_columns=condition_columns
    )
    num_numerical = len(num_cols)

    x_np = _encode_tabddpm_frame(train_df, num_cols, cat_cols, schema_by_name)
    if is_conditional:
        y_np = _encode_condition_onehot(train_df, condition_columns, schema_by_name)
        cond_dim = int(y_np.shape[1])
        dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    else:
        cond_dim = 0
        dataset = TensorDataset(torch.from_numpy(x_np))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    data_iter = iter(loader)
    steps = max(epochs * max(len(loader), 1), 1)

    d_in = int(cat_sizes.sum() + num_numerical)
    layers = hidden_layers or [256, 512, 512, 256]
    model, backbone_meta = _build_tabddpm_denoiser(
        d_in=d_in,
        backbone=backbone,
        hidden_layers=layers,
        tf_d_model=tf_d_model,
        tf_nhead=tf_nhead,
        tf_layers=tf_layers,
        tf_n_tokens=tf_n_tokens,
        is_y_cond=is_conditional,
        cond_dim=cond_dim,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_sizes,
        num_numerical_features=num_numerical,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type="mse",
        scheduler="cosine",
        device=device,
    ).to(device)
    diffusion.train()

    ema_model = deepcopy(diffusion._denoise_fn)
    for param in ema_model.parameters():
        param.detach_()

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-4)
    y_dummy = torch.tensor([1.0], device=device)

    logging.info(
        "TabDDPM: backbone=%s, conditional=%s, cond_cols=%s, num_cols=%s, cat_cols=%d, "
        "T=%d, steps=%d, d_in=%d, cond_dim=%d, params=%d",
        backbone_meta.get("backbone"),
        is_conditional,
        condition_columns,
        num_cols,
        len(cat_cols),
        num_timesteps,
        steps,
        d_in,
        cond_dim,
        n_params,
    )

    step = 0
    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if is_conditional:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out_dict = {"y": batch_y}
        else:
            (batch_x,) = batch
            batch_x = batch_x.to(device)
            out_dict = {"y": torch.zeros(batch_x.size(0), dtype=torch.long, device=device)}

        optimizer.zero_grad()
        loss_multi, loss_gauss = diffusion.mixed_loss(batch_x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        optimizer.step()
        _update_ema(ema_model.parameters(), diffusion._denoise_fn.parameters())

        frac_done = (step + 1) / steps
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * (1 - frac_done)

        if (step + 1) % max(steps // 10, 1) == 0 or step + 1 == steps:
            logging.info(
                "TabDDPM step %d/%d | mloss=%.4f | gloss=%.4f",
                step + 1,
                steps,
                float(loss_multi.item()),
                float(loss_gauss.item()),
            )
        step += 1

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_type": "tabddpm",
                "state_dict": diffusion._denoise_fn.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "cat_sizes": cat_sizes.tolist(),
                "num_timesteps": num_timesteps,
                "hidden_layers": layers,
                "all_columns": all_columns,
                "condition_columns": condition_columns,
                **backbone_meta,
            },
            model_save_path,
        )
        logging.info("Saved TabDDPM checkpoint: %s", model_save_path)

    diffusion.eval()
    ema_model.eval()
    diffusion._denoise_fn = ema_model

    if is_conditional:
        cond_src = sample_cond_df if sample_cond_df is not None else train_df
        if sample_cond_df is not None or n_samples <= 0:
            cond_src = cond_src.reset_index(drop=True)
            n_samples = len(cond_src)
        else:
            cond_src = cond_src.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)
        y_all = torch.from_numpy(_encode_condition_onehot(cond_src, condition_columns, schema_by_name)).to(device)
        x_gen, _ = diffusion.sample_all(n_samples, batch_size, y_dummy, ddim=False, y=y_all)
        decoded = _decode_tabddpm_array(x_gen.numpy(), num_cols, cat_cols, schema_by_name)
        for col in condition_columns:
            decoded[col] = cond_src[col].astype(int).values
        return decoded[all_columns]

    generated = []
    remaining = n_samples
    while remaining > 0:
        bsz = min(batch_size, remaining)
        x_gen, _ = diffusion.sample_all(bsz, bsz, y_dummy, ddim=False)
        generated.append(x_gen.numpy())
        remaining -= bsz

    x_all = np.concatenate(generated, axis=0)[:n_samples]
    decoded = _decode_tabddpm_array(x_all, num_cols, cat_cols, schema_by_name)
    return decoded[all_columns]


def _build_tabddpm_from_checkpoint(
    checkpoint: Dict[str, Any],
    schema: List[Dict[str, Any]],
    device: torch.device,
):
    schema_by_name = {field["name"]: field for field in schema}
    num_cols = list(checkpoint["num_cols"])
    cat_cols = list(checkpoint["cat_cols"])
    cat_sizes = np.array(checkpoint["cat_sizes"], dtype=np.int64)
    num_timesteps = int(checkpoint["num_timesteps"])
    layers = list(checkpoint.get("hidden_layers") or [256, 512, 512, 256])
    num_numerical = len(num_cols)
    is_y_cond = bool(checkpoint.get("is_y_cond", False))
    cond_dim = int(checkpoint.get("cond_dim") or 0)

    d_in = int(cat_sizes.sum() + num_numerical)
    model, _ = _build_tabddpm_denoiser(
        d_in=d_in,
        backbone=str(checkpoint.get("backbone") or "mlp"),
        hidden_layers=layers,
        tf_d_model=int(checkpoint.get("tf_d_model") or 128),
        tf_nhead=int(checkpoint.get("tf_nhead") or 8),
        tf_layers=int(checkpoint.get("tf_layers") or 4),
        tf_n_tokens=int(checkpoint.get("tf_n_tokens") or 16),
        is_y_cond=is_y_cond,
        cond_dim=cond_dim,
    )
    model = model.to(device)
    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_sizes,
        num_numerical_features=num_numerical,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type="mse",
        scheduler="cosine",
        device=device,
    ).to(device)

    state_key = "ema_state_dict" if "ema_state_dict" in checkpoint else "state_dict"
    diffusion._denoise_fn.load_state_dict(checkpoint[state_key])
    diffusion.eval()
    diffusion._denoise_fn.eval()
    return diffusion, num_cols, cat_cols, schema_by_name


def sample_tabddpm_from_checkpoint(
    checkpoint_path: str,
    schema: List[Dict[str, Any]],
    n_samples: int,
    batch_size: int = 500,
    device: Optional[torch.device] = None,
    sample_cond_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_type") != "tabddpm":
        raise ValueError(f"Expected tabddpm checkpoint, got {checkpoint.get('model_type')!r}")

    diffusion, num_cols, cat_cols, schema_by_name = _build_tabddpm_from_checkpoint(
        checkpoint, schema, device
    )
    all_columns = list(checkpoint["all_columns"])
    condition_columns = list(checkpoint.get("condition_columns") or [])
    y_dummy = torch.tensor([1.0], device=device)

    if condition_columns:
        if sample_cond_df is None:
            raise ValueError("Conditional TabDDPM checkpoint requires sample_cond_df.")
        cond_src = sample_cond_df.reset_index(drop=True)
        n_samples = len(cond_src)
        y_all = torch.from_numpy(
            _encode_condition_onehot(cond_src, condition_columns, schema_by_name)
        ).to(device)
        x_gen, _ = diffusion.sample_all(n_samples, batch_size, y_dummy, ddim=False, y=y_all)
        decoded = _decode_tabddpm_array(x_gen.detach().cpu().numpy(), num_cols, cat_cols, schema_by_name)
        for col in condition_columns:
            decoded[col] = cond_src[col].astype(int).values
        return decoded[all_columns]

    generated = []
    remaining = n_samples
    while remaining > 0:
        bsz = min(batch_size, remaining)
        x_gen, _ = diffusion.sample_all(bsz, bsz, y_dummy, ddim=False)
        generated.append(x_gen.detach().cpu().numpy())
        remaining -= bsz

    x_all = np.concatenate(generated, axis=0)[:n_samples]
    decoded = _decode_tabddpm_array(x_all, num_cols, cat_cols, schema_by_name)
    return decoded[all_columns]
