"""Seed-42 robustness: train on 20k subset with optimal HCD / baseline recipes."""

from __future__ import annotations

import argparse
import json
import logging
import os
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


def _py() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _ensure_subset(full_train: str, subset_path: Path, n: int, seed: int) -> Path:
    if subset_path.exists():
        logging.info("Reusing train subset: %s", subset_path)
        return subset_path
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(full_train)
    out = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    out.to_csv(subset_path, index=False)
    logging.info("Wrote %d-row subset to %s", len(out), subset_path)
    return subset_path


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


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


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
        "train_subset_rows": None,
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
    # Train + generate via tabular baseline runner (uses n_gene for sampling size).
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
    # Re-evaluate with HCD-aligned feature column order vs full test.
    payload = _reeval_gene_csv(
        model_name=model_name,
        gene_csv=gene_csv,
        train=train,
        test=test,
        seed=seed,
        eval_sampling="unconditional_full_test_reference",
    )
    payload["train_subset_rows"] = int(pd.read_csv(train).shape[0])
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="20k-train seed-42 HCD vs baselines.")
    parser.add_argument("--output_root", type=str, default="exp/robustness_20k")
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--subset_rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hcd_st_cascade", "d3pm_tf", "ctgan", "tvae", "tabddpm", "datgan"],
        choices=["hcd_st_cascade", "d3pm_tf", "ctgan", "tvae", "tabddpm", "datgan", "ddpm_tf_baseline"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = Path(args.output_root)
    seed_root = root / f"seed_{args.seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    subset = _ensure_subset(
        args.full_train,
        seed_root / f"train_subset_{args.subset_rows}.csv",
        args.subset_rows,
        args.seed,
    )
    test_n = len(pd.read_csv(args.testdata))
    rows: List[Dict[str, Any]] = []

    for name in args.models:
        logging.info("===== %s =====", name)
        if name == "hcd_st_cascade":
            out = seed_root / "hcd_st_cascade"
            payload = _run_hcd(str(subset), args.testdata, out, args.seed, args.epochs)
            row = {"model": "hcd_st_cascade", "seed": args.seed, **_headline(payload)}
        elif name in {"d3pm_tf", "ddpm_tf_baseline"}:
            out = seed_root / "d3pm_tf"
            payload = _run_d3pm_tf(str(subset), args.testdata, out, args.seed, args.epochs)
            row = {"model": "d3pm_tf", "seed": args.seed, **_headline(payload)}
        else:
            out = seed_root / name
            payload = _run_tabular_baseline(
                model_name=name,
                train=str(subset),
                test=args.testdata,
                out_dir=out,
                seed=args.seed,
                epochs=args.epochs,
                n_gene=test_n,
            )
            row = {"model": name, "seed": args.seed, **_headline(payload)}
        row["train_subset_rows"] = args.subset_rows
        rows.append(row)
        logging.info(
            "%s | joint_js=%.6f | marg=%.6f | EMD=%s | MNL=%s | LVR=%s",
            row["model"],
            row.get("joint_js") or float("nan"),
            row.get("mean_marginal_jsd") or float("nan"),
            row.get("mean_ordinal_emd"),
            row.get("mnl_behavioral_similarity"),
            row.get("logical_validity_rate"),
        )

    summary = pd.DataFrame(rows).sort_values("joint_js")
    summary_path = seed_root / "summary_20k.csv"
    summary.to_csv(summary_path, index=False)
    (seed_root / "summary_20k.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Saved %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
