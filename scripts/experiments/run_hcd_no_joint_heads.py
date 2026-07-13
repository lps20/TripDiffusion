"""Ablation: HCD with joint heads vs marginal sample-statistics joint loss (no joint heads)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

import pandas as pd

from scripts.experiments.run_hcd_v2_improvement import _ensure_train_subset, _extract_row


EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "baseline_joint_heads",
        "description": "Standard HCD v2 with explicit joint heads",
    },
    {
        "name": "no_joint_heads_batch_stats",
        "description": "No joint heads; batch co-occurrence KL on marginal products",
        "no_joint_heads": True,
        "joint_loss_mode": "batch_stats",
    },
    {
        "name": "no_joint_heads_product",
        "description": "No joint heads; per-sample product-of-marginals CE",
        "no_joint_heads": True,
        "joint_loss_mode": "product",
    },
]


def _training_command_prefix() -> list[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _run_experiment(args: argparse.Namespace, exp: Dict[str, Any], run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    metrics_file = os.path.join(run_dir, "generated_samples_metrics.json")
    cmd = _training_command_prefix() + [
        "scripts/train/run_hcd_v2.py",
        "--traindata",
        args.traindata,
        "--testdata",
        args.testdata,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--lambda_weight",
        str(args.lambda_weight),
        "--lambda_joint",
        str(args.lambda_joint),
        "--T",
        str(args.T),
        "--joint_pairs",
        args.joint_pairs,
        "--batch_sampling",
        args.batch_sampling,
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--num_seeds",
        "1",
        "--exp_dir",
        run_dir,
        "--metrics_file",
        metrics_file,
    ]
    if exp.get("no_joint_heads"):
        cmd.append("--no_joint_heads")
        cmd.extend(["--joint_loss_mode", str(exp.get("joint_loss_mode", "batch_stats"))])
        if args.d_model is not None:
            cmd.extend(["--d_model", str(args.d_model)])
        if args.shared_layers is not None:
            cmd.extend(["--shared_layers", str(args.shared_layers)])
        if args.causal_layers is not None:
            cmd.extend(["--causal_layers", str(args.causal_layers)])
    if exp.get("feature_loss_weights"):
        cmd.extend(["--feature_loss_weights", exp["feature_loss_weights"]])
    if exp.get("st_cascade"):
        cmd.append("--st_cascade")
    if args.joint_sampling_at_inference:
        cmd.append("--joint_sampling_at_inference")

    logging.info("Running experiment %s: %s", exp["name"], exp["description"])
    logging.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare joint-head HCD vs marginal-statistics joint loss.")
    parser.add_argument("--output_root", type=str, default="exp/hcd_v2_improvement/no_joint_heads_50k")
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--train_subset", type=int, default=50000)
    parser.add_argument("--eval_subset", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_weight", type=float, default=2.0)
    parser.add_argument("--lambda_joint", type=float, default=0.5)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument(
        "--joint_pairs",
        type=str,
        default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]",
    )
    parser.add_argument("--batch_sampling", type=str, default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--shared_layers", type=int, default=None)
    parser.add_argument("--causal_layers", type=int, default=None)
    parser.add_argument("--joint_sampling_at_inference", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[e["name"] for e in EXPERIMENTS],
        choices=[e["name"] for e in EXPERIMENTS],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train_subset_path = output_root / f"train_{args.train_subset}.csv"
    test_subset_path = output_root / f"test_{args.eval_subset}.csv"
    _ensure_train_subset(args.traindata, str(train_subset_path), args.train_subset, args.seed)
    _ensure_train_subset(args.testdata, str(test_subset_path), args.eval_subset, args.seed)

    args.traindata = str(train_subset_path)
    args.testdata = str(test_subset_path)
    args.num_samples = args.eval_subset

    rows = []
    for exp in EXPERIMENTS:
        if exp["name"] not in args.experiments:
            continue
        run_dir = str(output_root / exp["name"])
        _run_experiment(args, exp, run_dir)
        metrics_path = os.path.join(run_dir, "generated_samples_metrics.json")
        row = _extract_row(exp["name"], metrics_path)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "no_joint_heads_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved summary: %s", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
