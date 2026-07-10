"""
Lightweight TabDDPM adapter for trip tabular baselines.

Uses the official TabDDPM core (Gaussian + multinomial diffusion) from:
https://github.com/yandex-research/tab-ddpm
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

from third_party.tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion


def _split_tabddpm_columns(
    all_columns: List[str],
    time_columns: List[str],
    schema_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str], np.ndarray]:
    num_cols = [c for c in all_columns if c in time_columns]
    cat_cols = [c for c in all_columns if c not in time_columns]
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
) -> pd.DataFrame:
    from utils.multi_seed import set_global_seed

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema_by_name = {field["name"]: field for field in schema}
    num_cols, cat_cols, cat_sizes = _split_tabddpm_columns(all_columns, time_columns, schema_by_name)
    num_numerical = len(num_cols)

    x_np = _encode_tabddpm_frame(train_df, num_cols, cat_cols, schema_by_name)
    x_tensor = torch.from_numpy(x_np)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True, drop_last=False)
    data_iter = iter(loader)
    steps = max(epochs * max(len(loader), 1), 1)

    d_in = int(cat_sizes.sum() + num_numerical)
    layers = hidden_layers or [256, 512, 512, 256]
    model = MLPDiffusion(
        d_in=d_in,
        num_classes=0,
        is_y_cond=False,
        rtdl_params={"d_layers": layers, "dropout": 0.0},
    ).to(device)

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
        "TabDDPM: num_cols=%s, cat_cols=%d, T=%d, steps=%d, d_in=%d",
        num_cols,
        len(cat_cols),
        num_timesteps,
        steps,
        d_in,
    )

    step = 0
    while step < steps:
        try:
            (batch_x,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_x,) = next(data_iter)

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
            },
            model_save_path,
        )
        logging.info("Saved TabDDPM checkpoint: %s", model_save_path)

    diffusion.eval()
    ema_model.eval()
    diffusion._denoise_fn = ema_model

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
