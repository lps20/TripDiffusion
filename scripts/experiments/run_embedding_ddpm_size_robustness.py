"""5/10/20k × 5-seed sample-size robustness for Embedding-DDPM (fixed codebook).

Matches the existing robustness_size protocol: shared train subsets, 1:1
conditional sampling against the full test set, summary_all / multi_seed_summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import setup

setup()

import utils.test_utils
from scripts.baselines.run_tabular_baselines import ALL_COLUMNS, FULL_SCHEMA, _sanitize_df_by_schema
from scripts.experiments.run_sequential_econometric_20k import (
    COND_COLUMNS,
    FEATURES_INFO,
    TRIP_COLUMNS,
    ensure_subset,
)
from scripts.experiments.run_size_seed_robustness import (
    _headline,
    _size_tag,
    _upsert_summary,
    _write_multiseed_summary,
)

REVISION_ROBUSTNESS = ROOT.parent / "D3PM_revision" / "revision_exp" / "robustness_size"
MODEL_NAME = "ddpm_tf"


def _py() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def resolve_subset(size: int, seed: int, full_train: Path, output_root: Path) -> Path:
    filename = f"train_subset_{size}.csv"
    candidates = [
        output_root / _size_tag(size) / f"seed_{seed}" / filename,
        REVISION_ROBUSTNESS / _size_tag(size) / f"seed_{seed}" / filename,
        Path("exp/robustness_20k") / f"seed_{seed}" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    dest = output_root / _size_tag(size) / f"seed_{seed}" / filename
    return ensure_subset(full_train, dest, size, seed)


def reeval(
    gene_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    size: int,
) -> Dict[str, Any]:
    generated = _sanitize_df_by_schema(pd.read_csv(gene_path), FULL_SCHEMA)
    metrics = utils.test_utils.evaluate_generated_trips(
        test[TRIP_COLUMNS].astype(int).values.tolist(),
        generated[TRIP_COLUMNS].astype(int).values.tolist(),
        FEATURES_INFO,
        cond_info=[{"name": column} for column in COND_COLUMNS],
        generated_df=generated[ALL_COLUMNS],
        train_real_df=train,
        test_real_df=test,
        random_state=seed,
    )
    return {
        "seed": seed,
        "model": MODEL_NAME,
        "config": "fixed_codebook_cosine_T10",
        "train_subset_rows": size,
        "num_samples": len(generated),
        "eval_sampling": "match_test_one_to_one",
        "evaluation": metrics,
    }


def run_cell(
    subset_path: Path,
    test_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    seed: int,
    size: int,
    epochs: int,
    batch_size: int,
    ddpm_t: int,
    force: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_path = out_dir / "DDPM_TF_gene.csv"
    metrics_path = out_dir / "DDPM_TF_metrics.json"
    model_path = out_dir / "DDPM_TF_model.pth"

    if gene_path.exists() and metrics_path.exists() and model_path.exists() and not force:
        logging.info("Skip existing cell: %s", out_dir)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "evaluation" not in payload:
            payload = reeval(gene_path, train, test, seed, size)
            metrics_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        return payload

    logging.info(
        "Training Embedding-DDPM size=%d seed=%d T=%d fixed+cosine",
        size,
        seed,
        ddpm_t,
    )
    _run(
        _py()
        + [
            "scripts/baselines/run_tabular_baselines.py",
            "--traindata",
            str(subset_path),
            "--testdata",
            str(test_path),
            "--output_dir",
            str(out_dir),
            "--models",
            "ddpm_tf",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            "0.001",
            "--num_samples",
            "0",
            "--seed",
            str(seed),
            "--num_seeds",
            "1",
            "--ddpm_t",
            str(ddpm_t),
        ]
    )
    payload = reeval(gene_path, train, test, seed, size)
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REVISION_ROBUSTNESS if REVISION_ROBUSTNESS.exists() else Path("revision_exp/robustness_size"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 10000, 20000])
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--ddpm_t", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_root / "embedding_ddpm_run.log", encoding="utf-8"),
        ],
    )

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    test = _sanitize_df_by_schema(pd.read_csv(args.test), FULL_SCHEMA)
    summary_path = args.output_root / "summary_all.csv"
    multi_path = args.output_root / "multi_seed_summary.csv"
    planned = [(size, seed) for size in args.sizes for seed in seeds]
    logging.info(
        "Planned Embedding-DDPM cells: %d -> %s (T=%d, fixed codebook)",
        len(planned),
        args.output_root,
        args.ddpm_t,
    )

    for index, (size, seed) in enumerate(planned, start=1):
        logging.info(
            "===== [%d/%d] size=%d seed=%d Embedding-DDPM =====",
            index,
            len(planned),
            size,
            seed,
        )
        subset = resolve_subset(size, seed, args.full_train, args.output_root)
        logging.info("Train subset: %s", subset)
        train = _sanitize_df_by_schema(pd.read_csv(subset), FULL_SCHEMA)
        out_dir = args.output_root / _size_tag(size) / f"seed_{seed}" / MODEL_NAME
        payload = run_cell(
            subset_path=subset,
            test_path=args.test,
            train=train,
            test=test,
            out_dir=out_dir,
            seed=seed,
            size=size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            ddpm_t=args.ddpm_t,
            force=args.force,
        )
        evaluation = payload.get("evaluation", payload)
        row = {
            "model": MODEL_NAME,
            "seed": seed,
            "train_subset_rows": size,
            **_headline({"evaluation": evaluation}),
        }
        _upsert_summary(summary_path, row)
        _write_multiseed_summary(summary_path, multi_path)
        logging.info(
            "DONE size=%d seed=%d | joint_js=%s | marginal=%s | EMD=%s | LVR=%s | MNL=%s",
            size,
            seed,
            row.get("joint_js"),
            row.get("mean_marginal_jsd"),
            evaluation.get("mean_ordinal_emd"),
            row.get("logical_validity_rate"),
            row.get("mnl_behavioral_similarity"),
        )

    if summary_path.exists():
        frame = pd.read_csv(summary_path)
        frame = frame[frame["model"] == MODEL_NAME]
        print(frame.to_string(index=False))
    if multi_path.exists():
        multi = pd.read_csv(multi_path)
        multi = multi[multi["model"] == MODEL_NAME]
        print(multi.to_string(index=False))


if __name__ == "__main__":
    main()
