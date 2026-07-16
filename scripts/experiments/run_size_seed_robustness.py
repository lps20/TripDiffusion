"""Multi size × multi seed robustness: CTGAN/TVAE/TabDDPM/DATGAN/ddpm_tf/d3pm_tf/HCD.

Layout:
  exp/robustness_size/{N}k/seed_{s}/train_subset_{N}.csv
  exp/robustness_size/{N}k/seed_{s}/{model}/...
  exp/robustness_size/summary_all.csv
  exp/robustness_size/multi_seed_summary.csv   # mean±std by (size, model)

Reuses completed cells under exp/robustness_20k/seed_42 when size=20000, seed=42.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import numpy as np
import pandas as pd

FEATURES_INFO = [
    {"name": "start_type", "type": "categorical", "num_classes": 5},
    {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
    {"name": "act_num", "type": "categorical", "num_classes": 9},
    {"name": "mode_num", "type": "categorical", "num_classes": 9},
    {"name": "end_type", "type": "categorical", "num_classes": 5},
    {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
    {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
    {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
]
TRIP_COLS = [f["name"] for f in FEATURES_INFO]
COND_COLS = ["relation", "sex", "age_code", "job_type"]

DEFAULT_MODELS = [
    "ctgan",
    "tvae",
    "tabddpm",
    "datgan",
    "ddpm_tf",
    "d3pm_tf",
    "hcd_st_cascade",
]
TABULAR_MODELS = {"ctgan", "tvae", "tabddpm", "datgan", "ddpm_tf"}
LEGACY_20K_ROOT = Path("exp/robustness_20k")
HEADLINE_KEYS = [
    "joint_js",
    "mean_marginal_jsd",
    "mean_ordinal_emd",
    "start_zcode_jsd",
    "end_zcode_jsd",
    "logical_validity_rate",
    "mnl_behavioral_similarity",
]


def _py() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _size_tag(n: int) -> str:
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def _headline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    ev = metrics.get("evaluation", metrics)
    jsd = ev.get("single_feature_jsd") or {}
    return {
        "joint_js": ev.get("joint_js"),
        "mean_marginal_jsd": ev.get("mean_marginal_jsd"),
        "mean_ordinal_emd": ev.get("mean_ordinal_emd"),
        "start_zcode_jsd": jsd.get("start_zcode_num"),
        "end_zcode_jsd": jsd.get("end_zcode_num"),
        "logical_validity_rate": ev.get("logical_validity_rate"),
        "mnl_behavioral_similarity": ev.get("mnl_behavioral_similarity"),
    }


def _ensure_subset(full_train: str, subset_path: Path, n: int, seed: int) -> Path:
    if subset_path.exists():
        logging.info("Reusing train subset: %s", subset_path)
        return subset_path
    # Prefer exact legacy 20k/seed_42 subset for comparability with prior run.
    legacy = LEGACY_20K_ROOT / f"seed_{seed}" / f"train_subset_{n}.csv"
    if n == 20000 and seed == 42 and legacy.exists():
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, subset_path)
        logging.info("Copied legacy subset %s -> %s", legacy, subset_path)
        return subset_path
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(full_train)
    out = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    out.to_csv(subset_path, index=False)
    logging.info("Wrote %d-row subset to %s", len(out), subset_path)
    return subset_path


def _legacy_model_dir(model: str, seed: int) -> Optional[Path]:
    if seed != 42:
        return None
    # Prior 20k script used hcd_st_cascade / d3pm_tf / tabular names.
    cand = LEGACY_20K_ROOT / f"seed_{seed}" / model
    return cand if cand.exists() else None


def _metrics_path_for(model: str, out_dir: Path) -> Path:
    if model in TABULAR_MODELS:
        return out_dir / f"{model.upper()}_metrics.json"
    return out_dir / "generated_samples_metrics.json"


def _gene_path_for(model: str, out_dir: Path) -> Path:
    if model in TABULAR_MODELS:
        return out_dir / f"{model.upper()}_gene.csv"
    return out_dir / "generated_samples.csv"


def _maybe_reuse_legacy(model: str, out_dir: Path, seed: int, size: int) -> bool:
    """Copy completed legacy 20k/seed_42 artifacts into the new layout if present."""
    if size != 20000 or seed != 42:
        return False
    if _metrics_path_for(model, out_dir).exists():
        return False
    legacy = _legacy_model_dir(model, seed)
    if legacy is None:
        return False
    legacy_metrics = _metrics_path_for(model, legacy)
    if not legacy_metrics.exists():
        return False
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(legacy, out_dir)
    logging.info("Reused legacy run %s -> %s", legacy, out_dir)
    return True


def _run_hcd(train: str, test: str, out_dir: Path, seed: int, epochs: int) -> Dict[str, Any]:
    metrics = out_dir / "generated_samples_metrics.json"
    if metrics.exists():
        logging.info("Skip HCD: existing %s", metrics)
        return json.loads(metrics.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _py()
        + [
            "scripts/train/run_hcd_v2.py",
            "--traindata",
            train,
            "--testdata",
            test,
            "--epochs",
            str(epochs),
            "--batch_size",
            "500",
            "--lr",
            "0.001",
            "--lambda_weight",
            "1.0",
            "--lambda_joint",
            "0.0",
            "--T",
            "10",
            "--joint_pairs",
            "[]",
            "--batch_sampling",
            "shuffle",
            "--patience",
            "100",
            "--min_delta",
            "0.0",
            "--num_samples",
            "0",
            "--seed",
            str(seed),
            "--num_seeds",
            "1",
            "--st_cascade",
            "--exp_dir",
            str(out_dir),
        ]
    )
    return json.loads(metrics.read_text(encoding="utf-8"))


def _run_d3pm_tf(train: str, test: str, out_dir: Path, seed: int, epochs: int) -> Dict[str, Any]:
    metrics = out_dir / "generated_samples_metrics.json"
    if metrics.exists():
        logging.info("Skip D3PM_TF: existing %s", metrics)
        return json.loads(metrics.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _py()
        + [
            "scripts/train/run_transformer.py",
            "--traindata",
            train,
            "--testdata",
            test,
            "--epochs",
            str(epochs),
            "--batch_size",
            "500",
            "--lr",
            "0.001",
            "--lambda_weight",
            "1.0",
            "--lambda_joint",
            "0.0",
            "--T",
            "10",
            "--joint_pairs",
            "[]",
            "--batch_sampling",
            "shuffle",
            "--patience",
            "100",
            "--min_delta",
            "0.0",
            "--num_samples",
            "0",
            "--seed",
            str(seed),
            "--num_seeds",
            "1",
            "--exp_dir",
            str(out_dir),
        ]
    )
    return json.loads(metrics.read_text(encoding="utf-8"))


def _reeval_gene_csv(
    model_name: str,
    gene_csv: Path,
    train: str,
    test: str,
    seed: int,
    eval_sampling: str,
) -> Dict[str, Any]:
    import utils.test_utils
    from scripts.baselines.run_tabular_baselines import ALL_COLUMNS, FULL_SCHEMA, _sanitize_df_by_schema

    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    generated_df = _sanitize_df_by_schema(pd.read_csv(gene_csv), FULL_SCHEMA)
    truth_trips = test_df[TRIP_COLS].astype(int).values.tolist()
    generated_trips = generated_df[TRIP_COLS].astype(int).values.tolist()
    cond_info = [{"name": c} for c in COND_COLS]
    metrics = utils.test_utils.evaluate_generated_trips(
        truth_trips,
        generated_trips,
        FEATURES_INFO,
        cond_info=cond_info,
        generated_df=generated_df[ALL_COLUMNS],
        train_real_df=train_df,
        test_real_df=test_df,
        random_state=seed,
    )
    return {
        "seed": seed,
        "model": model_name,
        "num_samples": len(generated_df),
        "eval_sampling": eval_sampling,
        "train_subset_rows": int(train_df.shape[0]),
        "evaluation": metrics,
    }


def _run_tabular_baseline(
    model_name: str,
    train: str,
    test: str,
    out_dir: Path,
    seed: int,
    epochs: int,
    n_gene: int,
) -> Dict[str, Any]:
    tag = model_name.upper()
    metrics_path = out_dir / f"{tag}_metrics.json"
    gene_csv = out_dir / f"{tag}_gene.csv"
    if metrics_path.exists() and gene_csv.exists():
        logging.info("Skip %s: existing metrics", model_name)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _py()
        + [
            "scripts/baselines/run_tabular_baselines.py",
            "--traindata",
            train,
            "--testdata",
            test,
            "--output_dir",
            str(out_dir),
            "--models",
            model_name,
            "--epochs",
            str(epochs),
            "--batch_size",
            "500",
            "--num_samples",
            str(n_gene),
            "--seed",
            str(seed),
            "--num_seeds",
            "1",
            "--ddpm_lambda_weight",
            "1.0",
            "--ddpm_lambda_joint",
            "0.0",
            "--ddpm_t",
            "10",
        ]
    )
    payload = _reeval_gene_csv(
        model_name=model_name,
        gene_csv=gene_csv,
        train=train,
        test=test,
        seed=seed,
        eval_sampling="unconditional_full_test_reference",
    )
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_one_model(
    model: str,
    train: str,
    test: str,
    out_dir: Path,
    seed: int,
    epochs: int,
    n_gene: int,
) -> Dict[str, Any]:
    if model == "hcd_st_cascade":
        return _run_hcd(train, test, out_dir, seed, epochs)
    if model == "d3pm_tf":
        return _run_d3pm_tf(train, test, out_dir, seed, epochs)
    if model in TABULAR_MODELS:
        return _run_tabular_baseline(model, train, test, out_dir, seed, epochs, n_gene)
    raise ValueError(f"Unknown model: {model}")


def _upsert_summary(summary_path: Path, row: Dict[str, Any]) -> None:
    new_df = pd.DataFrame([row])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        mask = (
            (old["model"] == row["model"])
            & (old["seed"] == row["seed"])
            & (old["train_subset_rows"] == row["train_subset_rows"])
        )
        old = old.loc[~mask].copy()
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(["train_subset_rows", "model", "seed"]).reset_index(drop=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(summary_path, index=False)


def _write_multiseed_summary(summary_path: Path, out_path: Path) -> None:
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    rows: List[Dict[str, Any]] = []
    for (size, model), g in df.groupby(["train_subset_rows", "model"], sort=True):
        row: Dict[str, Any] = {
            "train_subset_rows": int(size),
            "model": model,
            "n_seeds": int(len(g)),
        }
        for key in HEADLINE_KEYS:
            if key not in g.columns:
                continue
            vals = pd.to_numeric(g[key], errors="coerce").dropna().tolist()
            if not vals:
                row[f"{key}_mean"] = np.nan
                row[f"{key}_std"] = np.nan
                row[key] = None
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            row[key] = f"{mean:.6f} ± {std:.6f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info("Wrote multiseed summary: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5/10/20k × multi-seed robustness for HCD and baselines."
    )
    parser.add_argument("--output_root", type=str, default="exp/robustness_size")
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 20000],
    )
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the planned cells without training.",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.info("Args: %s", vars(args))
    logging.info("Log file: %s", log_path)

    test_n = len(pd.read_csv(args.testdata))
    summary_path = root / "summary_all.csv"
    multi_path = root / "multi_seed_summary.csv"

    planned = [
        (size, seed, model)
        for size in args.sizes
        for seed in seeds
        for model in args.models
    ]
    logging.info("Planned cells: %d", len(planned))

    for i, (size, seed, model) in enumerate(planned, start=1):
        size_root = root / _size_tag(size) / f"seed_{seed}"
        subset = size_root / f"train_subset_{size}.csv"
        out_dir = size_root / model
        logging.info(
            "===== [%d/%d] size=%d seed=%d model=%s =====",
            i,
            len(planned),
            size,
            seed,
            model,
        )
        if args.dry_run:
            continue

        _ensure_subset(args.full_train, subset, size, seed)
        _maybe_reuse_legacy(model, out_dir, seed, size)

        try:
            payload = _run_one_model(
                model=model,
                train=str(subset),
                test=args.testdata,
                out_dir=out_dir,
                seed=seed,
                epochs=args.epochs,
                n_gene=test_n,
            )
        except Exception:
            logging.exception("FAILED size=%d seed=%d model=%s", size, seed, model)
            continue

        row = {
            "model": model,
            "seed": seed,
            "train_subset_rows": size,
            **_headline(payload),
        }
        _upsert_summary(summary_path, row)
        _write_multiseed_summary(summary_path, multi_path)
        logging.info(
            "DONE %s | size=%d seed=%d | joint_js=%s | MNL=%s | LVR=%s",
            model,
            size,
            seed,
            row.get("joint_js"),
            row.get("mnl_behavioral_similarity"),
            row.get("logical_validity_rate"),
        )

    if summary_path.exists():
        print(pd.read_csv(summary_path).to_string(index=False))
        logging.info("Saved %s", summary_path)
    if multi_path.exists():
        print(pd.read_csv(multi_path).to_string(index=False))
        logging.info("Saved %s", multi_path)


if __name__ == "__main__":
    main()
