"""Retrain Embedding-DDPM on full data with fixed codebook + cosine decode.

Uses the collapse-fix defaults already wired into run_tabular_baselines:
  feature_embedding_mode=fixed, decode_metric=cosine, T=10 (via --ddpm_t).

After each seed, re-evaluates with explicit ordinal FEATURES_INFO so EMD is reported.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import pandas as pd

import utils.test_utils
from scripts.baselines.run_tabular_baselines import ALL_COLUMNS, FULL_SCHEMA, _sanitize_df_by_schema

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
FEAT_COLS = [feature["name"] for feature in FEATURES_INFO]
COND_COLS = ["relation", "sex", "age_code", "job_type"]


def _py() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _headline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = metrics.get("evaluation", metrics)
    return {
        "joint_js": evaluation.get("joint_js"),
        "mean_marginal_jsd": evaluation.get("mean_marginal_jsd"),
        "mean_ordinal_emd": evaluation.get("mean_ordinal_emd"),
        "logical_validity_rate": evaluation.get("logical_validity_rate"),
        "mnl_behavioral_similarity": evaluation.get("mnl_behavioral_similarity"),
    }


def _reeval(
    gene_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> Dict[str, Any]:
    generated = _sanitize_df_by_schema(pd.read_csv(gene_path), FULL_SCHEMA)
    metrics = utils.test_utils.evaluate_generated_trips(
        test[FEAT_COLS].astype(int).values.tolist(),
        generated[FEAT_COLS].astype(int).values.tolist(),
        FEATURES_INFO,
        cond_info=[{"name": column} for column in COND_COLS],
        generated_df=generated[ALL_COLUMNS],
        train_real_df=train,
        test_real_df=test,
        random_state=seed,
    )
    return {
        "seed": seed,
        "model": "ddpm_tf",
        "config": "fixed_codebook_cosine_T10",
        "evaluation": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/embedding_ddpm_full/T10_cond_fixed",
    )
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ddpm_t", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--force", action="store_true", help="Retrain even if seed metrics exist.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
    )

    train = _sanitize_df_by_schema(pd.read_csv(args.traindata), FULL_SCHEMA)
    test = _sanitize_df_by_schema(pd.read_csv(args.testdata), FULL_SCHEMA)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        metrics_path = seed_dir / "DDPM_TF_metrics.json"
        gene_path = seed_dir / "DDPM_TF_gene.csv"
        model_path = seed_dir / "DDPM_TF_model.pth"
        if (
            metrics_path.exists()
            and gene_path.exists()
            and model_path.exists()
            and not args.force
        ):
            logging.info("Skip train seed=%d; re-evaluate existing samples", seed)
            payload = _reeval(gene_path, train, test, seed)
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.append({"seed": seed, "model": "ddpm_tf", **_headline(payload)})
            continue

        logging.info(
            "===== Embedding-DDPM full seed=%d T=%d fixed+cosine =====",
            seed,
            args.ddpm_t,
        )
        seed_dir.mkdir(parents=True, exist_ok=True)
        _run(
            _py()
            + [
                "scripts/baselines/run_tabular_baselines.py",
                "--traindata",
                args.traindata,
                "--testdata",
                args.testdata,
                "--output_dir",
                str(seed_dir),
                "--models",
                "ddpm_tf",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--num_samples",
                "0",
                "--seed",
                str(seed),
                "--num_seeds",
                "1",
                "--ddpm_t",
                str(args.ddpm_t),
            ]
        )
        payload = _reeval(gene_path, train, test, seed)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = {"seed": seed, "model": "ddpm_tf", **_headline(payload)}
        rows.append(row)
        logging.info(
            "DONE seed=%d | joint_js=%s | marginal=%s | EMD=%s | LVR=%s | MNL=%s",
            seed,
            row.get("joint_js"),
            row.get("mean_marginal_jsd"),
            row.get("mean_ordinal_emd"),
            row.get("logical_validity_rate"),
            row.get("mnl_behavioral_similarity"),
        )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "baseline_metrics_per_seed.csv", index=False)

    summary: Dict[str, Any] = {
        "model": "DDPM_TF",
        "config": "fixed_codebook_cosine_T10",
        "n_seeds": len(per_seed),
        "seeds": seeds,
    }
    for metric in [
        "joint_js",
        "mean_marginal_jsd",
        "mean_ordinal_emd",
        "logical_validity_rate",
        "mnl_behavioral_similarity",
    ]:
        values = pd.to_numeric(per_seed[metric], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[metric] = f"{mean:.4f} ± {std:.4f}" if mean is not None else None

    (output_dir / "multi_seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(output_dir / "baseline_metrics_summary.csv", index=False)
    print(per_seed.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
