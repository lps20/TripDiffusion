import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _BASELINES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from project_paths import setup

setup()

import argparse
import inspect
import importlib
import importlib.metadata
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils.test_utils
import utils.train_utils
from utils.multi_seed import (
    add_multiseed_arguments,
    aggregate_by_model,
    parse_seeds,
    save_multiseed_artifacts,
)


TRIP_SCHEMA = [
    {"name": "start_type", "num_classes": 5},
    {"name": "start_time_num_6", "num_classes": 241},
    {"name": "start_zcode_num", "num_classes": 77},
    {"name": "act_num", "num_classes": 9},
    {"name": "mode_num", "num_classes": 9},
    {"name": "trip_time_num_6", "num_classes": 241},
    {"name": "end_type", "num_classes": 5},
    {"name": "end_zcode_num", "num_classes": 77},
]

COND_COLUMNS = ["relation", "sex", "age_code", "job_type"]
LOC_COLUMNS = ["start_type", "start_zcode_num", "end_type", "end_zcode_num"]
TIME_COLUMNS = ["start_time_num_6", "trip_time_num_6"]
ACT_COLUMN = "act_num"
MODE_COLUMN = "mode_num"
TRIP_COLUMNS = [f["name"] for f in TRIP_SCHEMA]
ALL_COLUMNS = COND_COLUMNS + TRIP_COLUMNS
FULL_SCHEMA = [
    {"name": "relation", "num_classes": 5},
    {"name": "sex", "num_classes": 2},
    {"name": "age_code", "num_classes": 13},
    {"name": "job_type", "num_classes": 9},
    *TRIP_SCHEMA,
]


def _set_seed(seed: int) -> None:
    from utils.multi_seed import set_global_seed

    set_global_seed(seed)


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}

    params = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _import_ctgan_class():
    try:
        module = importlib.import_module("ctgan")
        return getattr(module, "CTGAN")
    except Exception as exc:
        raise ImportError(
            "CTGAN is not installed. Install with: pip install ctgan"
        ) from exc


def _import_tvae_class():
    try:
        module = importlib.import_module("sdv.single_table")
        return getattr(module, "TVAESynthesizer")
    except Exception as exc:
        raise ImportError(
            "SDV (TVAE) is not installed. Install with: pip install sdv"
        ) from exc


def _build_sdv_metadata(train_df: pd.DataFrame, columns: List[str]):
    try:
        from sdv.metadata import SingleTableMetadata
    except Exception as exc:
        raise ImportError(
            "SDV metadata is not available. Install with: pip install sdv"
        ) from exc

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df[columns])
    for col in columns:
        metadata.update_column(col, sdtype="categorical")
    return metadata


def _patch_sklearn_onehot_for_datgan() -> None:
    """
    DATGAN expects sklearn.preprocessing.OneHotEncoder(..., sparse=...).
    Newer sklearn versions renamed this argument to `sparse_output`.
    """
    try:
        import sklearn.preprocessing as sk_pre
    except Exception:
        return

    onehot_cls = getattr(sk_pre, "OneHotEncoder", None)
    if onehot_cls is None:
        return

    try:
        sig = inspect.signature(onehot_cls.__init__)
    except (TypeError, ValueError):
        return

    has_sparse = "sparse" in sig.parameters
    has_sparse_output = "sparse_output" in sig.parameters
    if has_sparse or (not has_sparse_output):
        return

    base_cls = onehot_cls

    class CompatOneHotEncoder(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, sparse=None, **kwargs):
            if sparse is not None and "sparse_output" not in kwargs:
                kwargs["sparse_output"] = sparse
            super().__init__(*args, **kwargs)

    sk_pre.OneHotEncoder = CompatOneHotEncoder


def _patch_tensorflow_checkpoint_save_for_datgan() -> None:
    """
    DATGAN's training loop always saves a checkpoint at the last epoch.
    On some Windows setups this rename operation fails; skip checkpoint failure
    so baseline generation can still finish.
    """
    try:
        import tensorflow as tf
    except Exception:
        return

    ckpt_cls = getattr(tf.train, "CheckpointManager", None)
    if ckpt_cls is None:
        return
    if getattr(ckpt_cls, "_datgan_safe_save_patched", False):
        return

    original_save = ckpt_cls.save

    def _safe_save(self, *args, **kwargs):
        try:
            return original_save(self, *args, **kwargs)
        except Exception as exc:
            logging.warning("DATGAN checkpoint save failed and was skipped: %s", exc)
            return None

    ckpt_cls.save = _safe_save
    ckpt_cls._datgan_safe_save_patched = True


def _import_datgan_class():
    _patch_sklearn_onehot_for_datgan()
    candidates = [
        ("datgan", "DATGAN"),
        ("datgan.datgan", "DATGAN"),
        ("DATGAN", "DATGAN"),
    ]
    import_errors: List[str] = []

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                _patch_tensorflow_checkpoint_save_for_datgan()
                return getattr(module, class_name)
            import_errors.append(f"{module_name}: class `{class_name}` not found")
        except ModuleNotFoundError:
            continue
        except Exception as exc:
            import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    hints = []
    try:
        tf_ver = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        tf_ver = None
    try:
        np_ver = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        np_ver = None
    try:
        pb_ver = importlib.metadata.version("protobuf")
    except importlib.metadata.PackageNotFoundError:
        pb_ver = None

    if tf_ver is not None or np_ver is not None or pb_ver is not None:
        hints.append(f"Detected versions -> tensorflow={tf_ver}, numpy={np_ver}, protobuf={pb_ver}")
    hints.append("For DATGAN(TensorFlow 2.x), use compatible deps, e.g.:")
    hints.append(
        "pip install \"numpy<2\" \"protobuf<=3.20.3\" \"tensorflow==2.8.*\" --upgrade --force-reinstall"
    )

    detail = "; ".join(import_errors) if import_errors else "No DATGAN module candidate could be imported."
    raise ImportError(
        "DATGAN is installed but not importable due to dependency/API mismatch. "
        f"Details: {detail}. {' | '.join(hints)}"
    )


def _sanitize_df_by_schema(df: pd.DataFrame, schema: List[Dict[str, Any]]) -> pd.DataFrame:
    from utils.data_encoding import normalize_category_values

    output = df.copy()
    for feat in schema:
        col = feat["name"]
        if col not in output.columns:
            raise ValueError(f"Generated data missing required column: {col}")
        output[col] = normalize_category_values(output[col], feat["num_classes"], col)
    return output


def _full_field_info() -> List[Dict[str, Any]]:
    return FULL_SCHEMA


def _df_to_index_tensor(df: pd.DataFrame, fields: List[Dict[str, Any]]) -> torch.Tensor:
    arr = np.zeros((len(df), len(fields)), dtype=np.int64)
    for i, field in enumerate(fields):
        name = field["name"]
        k = field["num_classes"]
        vals = pd.to_numeric(df[name], errors="coerce").fillna(0).round().astype(int).clip(0, k - 1).values
        arr[:, i] = vals
    return torch.from_numpy(arr)


def _index_tensor_to_df(index_tensor: torch.Tensor, fields: List[Dict[str, Any]]) -> pd.DataFrame:
    data = index_tensor.detach().cpu().numpy()
    out = {}
    for i, field in enumerate(fields):
        out[field["name"]] = data[:, i].astype(int)
    return pd.DataFrame(out)


def _onehot_batch(x_idx: torch.Tensor, fields: List[Dict[str, Any]]) -> torch.Tensor:
    parts = []
    for i, field in enumerate(fields):
        k = field["num_classes"]
        parts.append(F.one_hot(x_idx[:, i], num_classes=k).float())
    return torch.cat(parts, dim=1)


class _TabularVAE(nn.Module):
    def __init__(self, fields: List[Dict[str, Any]], hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.fields = fields
        self.total_dim = int(sum(f["num_classes"] for f in fields))
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_heads = nn.ModuleList([nn.Linear(hidden_dim, f["num_classes"]) for f in fields])

    def encode(self, x_onehot: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        h = self.encoder(x_onehot)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> List[torch.Tensor]:
        h = self.decoder_backbone(z)
        return [head(h) for head in self.decoder_heads]

    def forward(self, x_onehot: torch.Tensor) -> (List[torch.Tensor], torch.Tensor, torch.Tensor):
        mu, logvar = self.encode(x_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar


def _run_vae(
    train_df: pd.DataFrame,
    n_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    hidden_dim: int,
    beta_kl: float,
    seed: int,
    model_save_path: Optional[str] = None,
) -> pd.DataFrame:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fields = _full_field_info()
    x_idx = _df_to_index_tensor(train_df, fields)
    dataset = TensorDataset(x_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = _TabularVAE(fields=fields, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_idx,) in loader:
            batch_idx = batch_idx.to(device)
            x_onehot = _onehot_batch(batch_idx, fields)

            logits_list, mu, logvar = model(x_onehot)
            recon_loss = 0.0
            for i, logits in enumerate(logits_list):
                recon_loss = recon_loss + F.cross_entropy(logits, batch_idx[:, i])
            recon_loss = recon_loss / len(fields)

            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + beta_kl * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_idx.size(0)

        avg_loss = total_loss / max(len(dataset), 1)
        logging.info("VAE epoch %d/%d avg_loss=%.6f", epoch + 1, epochs, avg_loss)

    if model_save_path:
        _save_vae_checkpoint(
            model_save_path,
            model=model,
            fields=fields,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            beta_kl=beta_kl,
        )

    model.eval()
    out_rows = []
    remaining = n_samples
    with torch.no_grad():
        while remaining > 0:
            bsz = min(batch_size, remaining)
            z = torch.randn((bsz, latent_dim), device=device)
            logits_list = model.decode_logits(z)
            sampled_cols = []
            for logits in logits_list:
                probs = F.softmax(logits, dim=1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                sampled_cols.append(sampled)
            x_gen = torch.stack(sampled_cols, dim=1)
            out_rows.append(x_gen.cpu())
            remaining -= bsz

    x_all = torch.cat(out_rows, dim=0)
    return _index_tensor_to_df(x_all, fields)


def _fit_model_with_discrete_columns(model: Any, train_df: pd.DataFrame, discrete_columns: List[str]) -> None:
    if hasattr(model, "fit"):
        fit_fn = model.fit
    elif hasattr(model, "train"):
        fit_fn = model.train
    else:
        raise AttributeError("Model has neither `fit` nor `train` method.")

    # Some libraries wrap `fit` and hide real signatures behind (*args, **kwargs),
    # so we try one keyword at a time instead of passing all candidates together.
    keyword_candidates = [
        ("discrete_columns", discrete_columns),
        ("categorical_columns", discrete_columns),
        ("cat_cols", discrete_columns),
        ("cat_columns", discrete_columns),
    ]

    last_unexpected_kw_error: Optional[Exception] = None
    for kw_name, kw_value in keyword_candidates:
        try:
            fit_fn(train_df, **{kw_name: kw_value})
            return
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg:
                last_unexpected_kw_error = exc
                continue
            raise

    try:
        fit_fn(train_df)
        return
    except TypeError as exc:
        if last_unexpected_kw_error is not None:
            raise last_unexpected_kw_error
        raise exc


def _sample_model(model: Any, n_samples: int, columns: List[str]) -> pd.DataFrame:
    if hasattr(model, "sample"):
        sample_fn = model.sample
    elif hasattr(model, "generate"):
        sample_fn = model.generate
    else:
        raise AttributeError("Model has neither `sample` nor `generate` method.")

    keyword_candidates = [
        ("n", n_samples),
        ("num_rows", n_samples),
        ("num_samples", n_samples),
        ("samples", n_samples),
    ]
    generated = None
    last_unexpected_kw_error: Optional[Exception] = None
    for kw_name, kw_value in keyword_candidates:
        try:
            generated = sample_fn(**{kw_name: kw_value})
            break
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg:
                last_unexpected_kw_error = exc
                continue
            raise

    if generated is None:
        try:
            generated = sample_fn(n_samples)
        except TypeError as exc:
            if last_unexpected_kw_error is not None:
                raise last_unexpected_kw_error
            raise exc

    if isinstance(generated, pd.DataFrame):
        return generated

    generated = np.asarray(generated)
    if generated.ndim != 2:
        raise ValueError("Generated output must be 2D tabular data.")
    if generated.shape[1] != len(columns):
        raise ValueError(
            f"Generated output has {generated.shape[1]} columns, expected {len(columns)}."
        )
    return pd.DataFrame(generated, columns=columns)


def _build_datgan_metadata(continuous_time: bool = True) -> Dict[str, Dict[str, Any]]:
    """Column metadata for DATGAN; time fields as discrete-bounded continuous when requested."""
    schema_by_name = {field["name"]: field for field in FULL_SCHEMA}
    metadata: Dict[str, Dict[str, Any]] = {}
    continuous_cols = set(TIME_COLUMNS) if continuous_time else set()
    for col in ALL_COLUMNS:
        if col in continuous_cols:
            upper = int(schema_by_name[col]["num_classes"]) - 1
            metadata[col] = {
                "type": "continuous",
                "discrete": True,
                "bounds": [0, upper],
                "enforce_bounds": True,
            }
        else:
            metadata[col] = {"type": "categorical"}
    return metadata


def _build_datgan_cascade_dag():
    """
    DAG aligned with HCD v2 cascade:
      demo -> act -> (loc, time) -> mode
    with act->time/loc and loc->mode emphasis.
    """
    import networkx as nx

    dag = nx.DiGraph()

    for parent, child in zip(COND_COLUMNS, COND_COLUMNS[1:]):
        dag.add_edge(parent, child)

    for demo in COND_COLUMNS:
        dag.add_edge(demo, ACT_COLUMN)

    for demo in COND_COLUMNS:
        for loc_col in LOC_COLUMNS:
            dag.add_edge(demo, loc_col)
        for time_col in TIME_COLUMNS:
            dag.add_edge(demo, time_col)

    for loc_col in LOC_COLUMNS:
        dag.add_edge(ACT_COLUMN, loc_col)
    for time_col in TIME_COLUMNS:
        dag.add_edge(ACT_COLUMN, time_col)

    dag.add_edge("start_type", "start_zcode_num")
    dag.add_edge("start_zcode_num", "end_type")
    dag.add_edge("end_type", "end_zcode_num")
    dag.add_edge("start_time_num_6", "trip_time_num_6")

    dag.add_edge(ACT_COLUMN, MODE_COLUMN)
    for loc_col in LOC_COLUMNS:
        dag.add_edge(loc_col, MODE_COLUMN)
    for time_col in TIME_COLUMNS:
        dag.add_edge(time_col, MODE_COLUMN)
    for demo in COND_COLUMNS:
        dag.add_edge(demo, MODE_COLUMN)

    missing = set(ALL_COLUMNS) - set(dag.nodes)
    if missing:
        raise ValueError(f"DATGAN cascade DAG missing nodes: {sorted(missing)}")

    order = list(nx.topological_sort(dag))
    logging.info("DATGAN cascade DAG topological order: %s", order)
    return dag


def _resolve_datgan_dag(dag_mode: str):
    if dag_mode == "cascade":
        return _build_datgan_cascade_dag()
    if dag_mode == "linear":
        return None
    raise ValueError(f"Unsupported datgan_dag mode: {dag_mode}")


def _baseline_model_paths(output_dir: str, model_name: str) -> Dict[str, str]:
    tag = model_name.upper()
    return {
        "ctgan": os.path.join(output_dir, f"{tag}_model.pkl"),
        "tvae": os.path.join(output_dir, f"{tag}_model.pkl"),
        "vae": os.path.join(output_dir, f"{tag}_model.pt"),
        "ddpm_tf": os.path.join(output_dir, f"{tag}_model.pth"),
        "ddpm_mlp": os.path.join(output_dir, f"{tag}_model.pth"),
        "tabddpm": os.path.join(output_dir, f"{tag}_model.pth"),
        "tabddpm_tf": os.path.join(output_dir, f"{tag}_model.pth"),
        "datgan_dir": os.path.join(output_dir, f"{tag}_model"),
    }


def _save_vae_checkpoint(
    path: str,
    model: "_TabularVAE",
    fields: List[Dict[str, Any]],
    hidden_dim: int,
    latent_dim: int,
    beta_kl: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_type": "tabular_vae",
            "state_dict": model.state_dict(),
            "fields": fields,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "beta_kl": beta_kl,
        },
        path,
    )
    logging.info("Saved VAE checkpoint: %s", path)


def _save_ddpm_checkpoint(
    path: str,
    model: nn.Module,
    features_info: List[Dict[str, Any]],
    cond_info: List[Dict[str, Any]],
    T: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_type": "ddpm_transformer",
            "state_dict": model.state_dict(),
            "features_info": features_info,
            "cond_info": cond_info,
            "T": T,
        },
        path,
    )
    logging.info("Saved DDPM-Transformer checkpoint: %s", path)


def _run_ctgan(
    train_df: pd.DataFrame,
    all_columns: List[str],
    n_samples: int,
    epochs: int,
    batch_size: int,
    model_save_path: Optional[str] = None,
) -> pd.DataFrame:
    ctgan_class = _import_ctgan_class()
    init_kwargs = _filter_kwargs(
        ctgan_class,
        {"epochs": epochs, "batch_size": batch_size, "verbose": True},
    )
    model = ctgan_class(**init_kwargs)
    _fit_model_with_discrete_columns(model, train_df[all_columns], all_columns)
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        model.save(model_save_path)
        logging.info("Saved CTGAN checkpoint: %s", model_save_path)
    return _sample_model(model, n_samples, all_columns)


def _run_tvae(
    train_df: pd.DataFrame,
    all_columns: List[str],
    n_samples: int,
    epochs: int,
    batch_size: int,
    model_save_path: Optional[str] = None,
) -> pd.DataFrame:
    tvae_class = _import_tvae_class()
    metadata = _build_sdv_metadata(train_df, all_columns)
    init_kwargs = _filter_kwargs(
        tvae_class,
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": True,
            "embedding_dim": 128,
            "compress_dims": (128, 128),
            "decompress_dims": (128, 128),
        },
    )
    model = tvae_class(metadata, **init_kwargs)
    model.fit(train_df[all_columns])
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        model.save(model_save_path)
        logging.info("Saved TVAE checkpoint: %s", model_save_path)
    return _sample_model(model, n_samples, all_columns)


def _run_datgan(
    train_df: pd.DataFrame,
    all_columns: List[str],
    n_samples: int,
    output_dir: str,
    epochs: int,
    batch_size: int,
    dag_mode: str = "cascade",
    continuous_time: bool = True,
    save_model: bool = True,
    condition_columns: Optional[List[str]] = None,
    sample_cond_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    datgan_class = _import_datgan_class()
    os.makedirs(output_dir, exist_ok=True)
    datgan_output = os.path.join(output_dir, "")
    condition_columns = list(condition_columns or [])

    init_kwargs = _filter_kwargs(
        datgan_class,
        {
            "name": "datgan_baseline",
            "output": datgan_output,
            "num_epochs": epochs,
            "batch_size": batch_size,
            "save_checkpoints": save_model,
            "restore_session": False,
            "verbose": 1,
            "conditional_inputs": condition_columns if condition_columns else None,
        },
    )
    model = datgan_class(**init_kwargs)

    metadata = _build_datgan_metadata(continuous_time=continuous_time)
    dag = _resolve_datgan_dag(dag_mode)
    fit_fn = getattr(model, "fit", None)
    if fit_fn is None:
        raise AttributeError("DATGAN model has no `fit` method.")

    fit_kwargs = _filter_kwargs(
        fit_fn,
        {
            "metadata": metadata,
            "dag": dag,
            "preprocessed_data_path": os.path.join(output_dir, "encoded_data"),
        },
    )
    logging.info(
        "DATGAN fit config: dag_mode=%s, continuous_time=%s, conditional=%s, "
        "cond_cols=%s, epochs=%d, batch_size=%d",
        dag_mode,
        continuous_time,
        bool(condition_columns),
        condition_columns,
        epochs,
        batch_size,
    )
    fit_fn(train_df[all_columns], **fit_kwargs)
    if save_model:
        logging.info("Saved DATGAN artifacts under: %s", output_dir)

    if condition_columns:
        if sample_cond_df is None:
            raise ValueError("Conditional DATGAN requires sample_cond_df for 1:1 sampling.")
        cond_src = sample_cond_df.reset_index(drop=True)
        inputs = cond_src[condition_columns].copy()
        n_out = len(inputs)
        # DATGAN.sample concatenates every batch into one growing DataFrame; that OOMs
        # on full-test (~500k). Sample in chunks and concat once at the end.
        chunk_size = max(int(batch_size) * 20, int(batch_size))
        logging.info(
            "DATGAN conditional sample: n=%d, chunk_size=%d, randomize=False, cond_cols=%s",
            n_out,
            chunk_size,
            condition_columns,
        )
        chunks: List[pd.DataFrame] = []
        for start in range(0, n_out, chunk_size):
            end = min(start + chunk_size, n_out)
            part_inputs = inputs.iloc[start:end].reset_index(drop=True)
            part = model.sample(
                num_samples=len(part_inputs),
                inputs=part_inputs,
                cond_dict={},
                randomize=False,
                timeout=False,
            )
            if not isinstance(part, pd.DataFrame):
                part = pd.DataFrame(part, columns=all_columns)
            for col in condition_columns:
                part[col] = part_inputs[col].values
            chunks.append(part[all_columns])
            if (end % (chunk_size * 5) == 0) or end == n_out:
                logging.info("DATGAN conditional sample progress: %d/%d", end, n_out)
        generated = pd.concat(chunks, ignore_index=True)
        return generated[all_columns]

    return _sample_model(model, n_samples, all_columns)


def _run_ddpm_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    T: int,
    lambda_weight: float,
    lambda_joint: float,
    seed: int,
    model_save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Continuous Embedding-DDPM with Transformer denoiser (not discrete D3PM)."""
    del lambda_weight, lambda_joint  # unused; continuous MSE objective
    _set_seed(seed)
    from model.EmbeddingDDPM_Net import EmbeddingDDPM, sample_embedding_ddpm, train_embedding_ddpm

    logging.info(
        "DDPM-Transformer = continuous Embedding-DDPM (Gaussian in embedding space, T=%d).",
        T,
    )

    ddpm_features_info = [
        {"name": "start_type", "type": "categorical", "num_classes": 5},
        {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "act_num", "type": "categorical", "num_classes": 9},
        {"name": "mode_num", "type": "categorical", "num_classes": 9},
        {"name": "end_type", "type": "categorical", "num_classes": 5},
        {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
        {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
    ]
    ddpm_cond_info = [
        {"name": "relation", "num_classes": 5},
        {"name": "sex", "num_classes": 2},
        {"name": "age_code", "num_classes": 13},
        {"name": "job_type", "num_classes": 9},
    ]
    feat_cols = [f["name"] for f in ddpm_features_info]
    cond_cols = [c["name"] for c in ddpm_cond_info]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingDDPM(
        features_info=ddpm_features_info,
        cond_info=ddpm_cond_info,
        T=T,
        d_model=128,
        backbone="transformer",
        nhead=8,
        num_layers=4,
        beta_schedule="cosine",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Embedding-DDPM (transformer) parameters: %d (%.2f M)", n_params, n_params / 1e6)

    model = train_embedding_ddpm(
        model=model,
        train_df=train_df,
        feat_cols=feat_cols,
        cond_cols=cond_cols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        model_save_path=model_save_path,
    )
    return sample_embedding_ddpm(
        model=model,
        test_df=test_df,
        feat_cols=feat_cols,
        cond_cols=cond_cols,
        n_samples=n_samples,
        device=device,
        batch_size=max(batch_size, 512),
        match_test_one_to_one=True,
    )


def _run_ddpm_mlp(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    T: int,
    lambda_weight: float,
    lambda_joint: float,
    seed: int,
    model_save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Continuous Embedding-DDPM with MLP denoiser."""
    del lambda_weight, lambda_joint
    _set_seed(seed)
    from model.EmbeddingDDPM_Net import EmbeddingDDPM, sample_embedding_ddpm, train_embedding_ddpm

    logging.info(
        "DDPM-MLP = continuous Embedding-DDPM (Gaussian in embedding space, T=%d).",
        T,
    )

    ddpm_features_info = [
        {"name": "start_type", "type": "categorical", "num_classes": 5},
        {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "act_num", "type": "categorical", "num_classes": 9},
        {"name": "mode_num", "type": "categorical", "num_classes": 9},
        {"name": "end_type", "type": "categorical", "num_classes": 5},
        {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
        {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
    ]
    ddpm_cond_info = [
        {"name": "relation", "num_classes": 5},
        {"name": "sex", "num_classes": 2},
        {"name": "age_code", "num_classes": 13},
        {"name": "job_type", "num_classes": 9},
    ]
    feat_cols = [f["name"] for f in ddpm_features_info]
    cond_cols = [c["name"] for c in ddpm_cond_info]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingDDPM(
        features_info=ddpm_features_info,
        cond_info=ddpm_cond_info,
        T=T,
        d_model=128,
        backbone="mlp",
        mlp_hidden=[512, 512, 512],
        beta_schedule="cosine",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Embedding-DDPM (mlp) parameters: %d (%.2f M)", n_params, n_params / 1e6)

    model = train_embedding_ddpm(
        model=model,
        train_df=train_df,
        feat_cols=feat_cols,
        cond_cols=cond_cols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        model_save_path=model_save_path,
    )
    return sample_embedding_ddpm(
        model=model,
        test_df=test_df,
        feat_cols=feat_cols,
        cond_cols=cond_cols,
        n_samples=n_samples,
        device=device,
        batch_size=max(batch_size, 512),
        match_test_one_to_one=True,
    )


def _evaluate_and_save(
    model_name: str,
    generated_df: pd.DataFrame,
    sampled_truth_df: pd.DataFrame,
    train_real_df: pd.DataFrame,
    test_real_df: pd.DataFrame,
    output_dir: str,
    seed: int,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    generated_df = _sanitize_df_by_schema(generated_df, FULL_SCHEMA)
    sampled_truth_df = _sanitize_df_by_schema(sampled_truth_df, FULL_SCHEMA)
    train_real_df = _sanitize_df_by_schema(train_real_df, FULL_SCHEMA)
    test_real_df = _sanitize_df_by_schema(test_real_df, FULL_SCHEMA)

    generated_trips = generated_df[TRIP_COLUMNS].values.tolist()
    truth_trips = sampled_truth_df[TRIP_COLUMNS].values.tolist()

    cond_info_for_eval = [{"name": c} for c in COND_COLUMNS]
    jsd_lvr_tstr_metrics = utils.test_utils.evaluate_generated_trips(
        truth_trips,
        generated_trips,
        TRIP_SCHEMA,
        cond_info=cond_info_for_eval,
        generated_df=generated_df[ALL_COLUMNS],
        train_real_df=train_real_df,
        test_real_df=test_real_df,
        random_state=seed
    )

    metrics = jsd_lvr_tstr_metrics

    model_tag = model_name.upper()
    if save_outputs:
        output_csv = os.path.join(output_dir, f"{model_tag}_gene.csv")
        generated_df[ALL_COLUMNS].to_csv(output_csv, index=False)

        metric_json = os.path.join(output_dir, f"{model_tag}_metrics.json")
        with open(metric_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    flat_row = utils.test_utils.flatten_evaluation_metrics(
        model_name=model_tag,
        metrics=metrics,
        extra_fields={"seed": seed},
        include_formatted=True,
    )
    return flat_row


def _append_summary(output_dir: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path = os.path.join(output_dir, "baseline_metrics.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        keep_old = old_df[~old_df["model"].isin(new_df["model"])]
        merged = pd.concat([keep_old, new_df], ignore_index=True)
        merged.to_csv(summary_path, index=False)
    else:
        new_df.to_csv(summary_path, index=False)


def _run_model_for_seed(
    model_name: str,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sampled_truth_df: pd.DataFrame,
    seed: int,
    output_dir: str,
    save_outputs: bool,
) -> pd.DataFrame:
    model_paths = _baseline_model_paths(output_dir, model_name)
    save_models = not args.no_save_models

    if model_name == "ctgan":
        return _run_ctgan(
            train_df=train_df,
            all_columns=ALL_COLUMNS,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=model_paths["ctgan"] if save_models else None,
        )
    if model_name == "tvae":
        return _run_tvae(
            train_df=train_df,
            all_columns=ALL_COLUMNS,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=model_paths["tvae"] if save_models else None,
        )
    if model_name == "tabddpm":
        from tab_ddpm_baseline import run_tabddpm

        return run_tabddpm(
            train_df=train_df,
            all_columns=ALL_COLUMNS,
            time_columns=TIME_COLUMNS,
            schema=FULL_SCHEMA,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_timesteps=args.tabddpm_t,
            seed=seed,
            model_save_path=model_paths["tabddpm"] if save_models else None,
            hidden_layers=args.tabddpm_hidden_layers,
            backbone="mlp",
            condition_columns=COND_COLUMNS if args.tabddpm_conditional else None,
            sample_cond_df=test_df if args.tabddpm_conditional else None,
        )
    if model_name == "tabddpm_tf":
        from tab_ddpm_baseline import run_tabddpm

        return run_tabddpm(
            train_df=train_df,
            all_columns=ALL_COLUMNS,
            time_columns=TIME_COLUMNS,
            schema=FULL_SCHEMA,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_timesteps=args.tabddpm_t,
            seed=seed,
            model_save_path=model_paths["tabddpm_tf"] if save_models else None,
            backbone="transformer",
            tf_d_model=args.tabddpm_tf_d_model,
            tf_nhead=args.tabddpm_tf_nhead,
            tf_layers=args.tabddpm_tf_layers,
            tf_n_tokens=args.tabddpm_tf_n_tokens,
            condition_columns=COND_COLUMNS,
            sample_cond_df=test_df,
        )
    if model_name == "vae":
        return _run_vae(
            train_df=train_df,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.vae_latent_dim,
            hidden_dim=args.vae_hidden_dim,
            beta_kl=args.vae_beta_kl,
            seed=seed,
            model_save_path=model_paths["vae"] if save_models else None,
        )
    if model_name == "ddpm_tf":
        return _run_ddpm_transformer(
            train_df=train_df,
            test_df=test_df,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            T=args.ddpm_t,
            lambda_weight=args.ddpm_lambda_weight,
            lambda_joint=args.ddpm_lambda_joint,
            seed=seed,
            model_save_path=model_paths["ddpm_tf"] if save_models else None,
        )
    if model_name == "ddpm_mlp":
        return _run_ddpm_mlp(
            train_df=train_df,
            test_df=test_df,
            n_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            T=args.ddpm_t,
            lambda_weight=args.ddpm_lambda_weight,
            lambda_joint=args.ddpm_lambda_joint,
            seed=seed,
            model_save_path=model_paths["ddpm_mlp"] if save_models else None,
        )
    if model_name == "datgan":
        datgan_dir = model_paths["datgan_dir"] if save_models else args.datgan_output_dir
        cond_cols = COND_COLUMNS if args.datgan_conditional else None
        return _run_datgan(
            train_df=train_df,
            all_columns=ALL_COLUMNS,
            n_samples=args.num_samples,
            output_dir=datgan_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dag_mode=args.datgan_dag,
            continuous_time=args.datgan_continuous_time,
            save_model=save_models,
            condition_columns=cond_cols,
            sample_cond_df=test_df if cond_cols else None,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Arguments: %s", vars(args))

    seeds = parse_seeds(args.seed, args.seeds, args.num_seeds, args.seed_start)
    logging.info("Running baselines across seeds: %s", seeds)

    train_df = pd.read_csv(args.traindata)
    test_df = pd.read_csv(args.testdata)

    required_columns = set(ALL_COLUMNS)
    for dataset_name, df in [("train", train_df), ("test", test_df)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{dataset_name} dataset missing columns: {sorted(missing)}")

    per_seed_rows: List[Dict[str, Any]] = []
    conditional_models = {"ddpm_tf", "ddpm_mlp", "tabddpm_tf"}
    if args.tabddpm_conditional:
        conditional_models = conditional_models | {"tabddpm"}
    if args.datgan_conditional:
        conditional_models = conditional_models | {"datgan"}

    for seed in seeds:
        _set_seed(seed)
        # Conditional diffusion baselines: evaluate 1:1 against the full test set.
        use_full_test = bool(set(args.models) & conditional_models) or args.num_samples <= 0
        if use_full_test:
            sampled_truth_df = test_df.reset_index(drop=True)
            # Keep args.num_samples consistent for unconditional models in mixed runs.
            if args.num_samples <= 0:
                args.num_samples = len(test_df)
        else:
            sampled_truth_df = test_df.sample(
                n=args.num_samples,
                replace=True,
                random_state=seed,
            ).reset_index(drop=True)
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        save_outputs = len(seeds) == 1

        for model_name in args.models:
            logging.info("Running baseline %s (seed=%d)", model_name.upper(), seed)
            truth_df = sampled_truth_df
            if model_name in conditional_models:
                truth_df = test_df.reset_index(drop=True)
            generated_df = _run_model_for_seed(
                model_name=model_name,
                args=args,
                train_df=train_df,
                test_df=test_df,
                sampled_truth_df=truth_df,
                seed=seed,
                output_dir=seed_dir if not save_outputs else args.output_dir,
                save_outputs=save_outputs,
            )
            row = _evaluate_and_save(
                model_name=model_name,
                generated_df=generated_df,
                sampled_truth_df=truth_df,
                train_real_df=train_df,
                test_real_df=test_df,
                output_dir=seed_dir if not save_outputs else args.output_dir,
                seed=seed,
                save_outputs=save_outputs,
            )
            per_seed_rows.append(row)
            logging.info(
                "%s seed=%d | joint_js=%.6f | marginal_jsd=%.6f | LVR=%.4f | MNL sim=%.4f",
                row["model"],
                seed,
                row.get("joint_js", float("nan")),
                row.get("mean_marginal_jsd", float("nan")),
                row.get("logical_validity_rate", float("nan")),
                row.get("mnl_behavioral_similarity", float("nan")),
            )

    per_seed_path = os.path.join(args.output_dir, "baseline_metrics_per_seed.csv")
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)

    if len(seeds) > 1:
        summary_rows = aggregate_by_model(per_seed_rows, model_key="model")
        summary_path = os.path.join(args.output_dir, "baseline_metrics_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        save_multiseed_artifacts(
            args.output_dir,
            [
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "joint_js": row.get("joint_js"),
                    "mean_marginal_jsd": row.get("mean_marginal_jsd"),
                    "logical_validity_rate": row.get("logical_validity_rate"),
                    "mnl_behavioral_similarity": row.get("mnl_behavioral_similarity"),
                    "tstr_trtr_f1_ratio": row.get("mnl_behavioral_similarity"),
                }
                for row in per_seed_rows
            ],
            summary_rows=summary_rows,
            basename="baseline_multiseed",
        )
        logging.info("Multi-seed baseline summary:")
        for summary_row in summary_rows:
            logging.info(
                "%s | joint_js=%s | marginal_jsd=%s | LVR=%s | MNL sim=%s",
                summary_row["model"],
                summary_row.get("joint_js"),
                summary_row.get("mean_marginal_jsd"),
                summary_row.get("logical_validity_rate"),
                summary_row.get("mnl_behavioral_similarity"),
            )
        _append_summary(args.output_dir, summary_rows)
    else:
        _append_summary(args.output_dir, per_seed_rows)

    logging.info("All selected baselines completed. Outputs in: %s", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate tabular baselines: CTGAN / TVAE / TabDDPM / DATGAN / VAE / DDPM-Transformer."
        )
    )
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--output_dir", type=str, default="exp/baseline")
    parser.add_argument("--datgan_output_dir", type=str, default="exp/baseline/datgan_model")
    parser.add_argument(
        "--no_save_models",
        action="store_true",
        help="Skip saving trained baseline model checkpoints.",
    )
    parser.add_argument(
        "--datgan_dag",
        type=str,
        default="cascade",
        choices=["cascade", "linear"],
        help="DATGAN DAG: cascade matches HCD (demo->act->loc/time->mode); linear uses column order.",
    )
    parser.add_argument(
        "--datgan_continuous_time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat start/trip time columns as continuous in DATGAN metadata.",
    )
    parser.add_argument(
        "--datgan_conditional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train/sample DATGAN with demographics as conditional_inputs (true conditional, 1:1 test).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["datgan"],
        choices=["ctgan", "tvae", "tabddpm", "tabddpm_tf", "datgan", "vae", "ddpm_tf", "ddpm_mlp"],
        help="Which baselines to run.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Sample count for unconditional baselines. Use 0 for |test|. "
        "Conditional models (ddpm_*/tabddpm_tf) always generate full-test 1:1.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vae_latent_dim", type=int, default=64)
    parser.add_argument("--vae_hidden_dim", type=int, default=256)
    parser.add_argument("--vae_beta_kl", type=float, default=0.01)
    parser.add_argument(
        "--ddpm_t",
        type=int,
        default=100,
        help="Diffusion steps for Embedding-DDPM baselines (ddpm_tf / ddpm_mlp).",
    )
    parser.add_argument("--tabddpm_t", type=int, default=100, help="Diffusion steps for TabDDPM baseline.")
    parser.add_argument(
        "--tabddpm_hidden_layers",
        nargs="+",
        type=int,
        default=[256, 512, 512, 256],
        help="MLP hidden layer sizes for TabDDPM denoiser.",
    )
    parser.add_argument("--tabddpm_tf_d_model", type=int, default=128)
    parser.add_argument("--tabddpm_tf_nhead", type=int, default=8)
    parser.add_argument("--tabddpm_tf_layers", type=int, default=4)
    parser.add_argument("--tabddpm_tf_n_tokens", type=int, default=16)
    parser.add_argument(
        "--tabddpm_conditional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make MLP TabDDPM conditional on demographics (tabddpm_tf is always conditional).",
    )
    parser.add_argument("--ddpm_lambda_weight", type=float, default=1.0)
    parser.add_argument("--ddpm_lambda_joint", type=float, default=0.0)
    add_multiseed_arguments(parser, default_num_seeds=5)
    args = parser.parse_args()
    main(args)
