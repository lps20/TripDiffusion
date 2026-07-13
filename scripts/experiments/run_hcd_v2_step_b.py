"""Step B: ST loc/time cascade ablation (50k train subset, 10k eval)."""

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
import subprocess
import sys
from typing import Any, Dict, List

import pandas as pd

from scripts.experiments.run_hcd_v2_improvement import _ensure_train_subset, _extract_row


STEP_B_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "baseline_v2",
        "description": "Standard HCD v2 (parallel ST stream)",
    },
    {
        "name": "st_cascade",
        "description": "ST loc chain + time chain inside causal adapters",
        "st_cascade": True,
    },
    {
        "name": "st_cascade_zcode2x",
        "description": "ST cascade + 2x zcode CE/VB weights",
        "st_cascade": True,
        "feature_loss_weights": '{"start_zcode_num": 2.0, "end_zcode_num": 2.0}',
    },
]


def _python_executable() -> str:
    """Build command prefix for tripdiffusion conda env when available."""
    import shutil

    conda = shutil.which("conda")
    if conda:
        return conda
    return sys.executable


def _training_command_prefix() -> list[str]:
    python_bin = _python_executable()
    if os.path.basename(python_bin).lower() == "conda":
        return [python_bin, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
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
        str(exp.get("lambda_weight", args.lambda_weight)),
        "--lambda_joint",
        str(exp.get("lambda_joint", args.lambda_joint)),
        "--T",
        str(args.T),
        "--num_samples",
        str(args.num_samples),
        "--random_condition_sampling",
        "--exp_dir",
        run_dir,
        "--joint_pairs",
        args.joint_pairs,
        "--batch_sampling",
        str(exp.get("batch_sampling", args.batch_sampling)),
        "--sampling_feature",
        str(exp.get("sampling_feature", args.sampling_feature)),
        "--gate_init_act",
        str(exp.get("gate_init_act", args.gate_init_act)),
        "--gate_init_st",
        str(exp.get("gate_init_st", args.gate_init_st)),
        "--gate_init_mode",
        str(exp.get("gate_init_mode", args.gate_init_mode)),
        "--metrics_file",
        metrics_file,
        "--seed",
        str(args.seed),
        "--num_seeds",
        "1",
        "--parallel",
        "False",
    ]
    if exp.get("st_cascade"):
        cmd.append("--st_cascade")
    if exp.get("feature_loss_weights"):
        cmd.extend(["--feature_loss_weights", exp["feature_loss_weights"]])
    logging.info("=== Step B: %s ===", exp["name"])
    logging.info("Desc: %s", exp.get("description", ""))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HCD v2 Step B ST cascade ablations.")
    parser.add_argument("--output_dir", type=str, default="exp/hcd_v2_improvement/step_b_50k")
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--traindata", type=str, default=None)
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--subset_rows", type=int, default=50000)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_weight", type=float, default=2.0)
    parser.add_argument("--lambda_joint", type=float, default=0.5)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--joint_pairs", type=str, default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]")
    parser.add_argument("--batch_sampling", type=str, default="sequential")
    parser.add_argument("--sampling_feature", type=str, default="act_num")
    parser.add_argument("--gate_init_act", type=float, default=-1.0)
    parser.add_argument("--gate_init_st", type=float, default=-1.0)
    parser.add_argument("--gate_init_mode", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.traindata is None:
        subset_path = os.path.join(args.output_dir, f"train_subset_{args.subset_rows}.csv")
        args.traindata = _ensure_train_subset(args.full_train, subset_path, args.subset_rows, args.seed)

    selected = STEP_B_EXPERIMENTS
    if args.experiments:
        names = set(args.experiments)
        selected = [e for e in STEP_B_EXPERIMENTS if e["name"] in names]
        if not selected:
            raise ValueError(f"No matching experiments in {names}")

    rows: List[Dict[str, Any]] = []
    for exp in selected:
        run_dir = os.path.join(args.output_dir, exp["name"])
        _run_experiment(args, exp, run_dir)
        metrics_file = os.path.join(run_dir, "generated_samples_metrics.json")
        row = _extract_row(exp["name"], metrics_file)
        row["st_cascade"] = bool(exp.get("st_cascade"))
        rows.append(row)
        logging.info(
            "Result %s | zcode=%.4f/%.4f | joint_js=%.4f | marginal=%.4f | MNL=%s",
            exp["name"],
            row.get("start_zcode_jsd") or float("nan"),
            row.get("end_zcode_jsd") or float("nan"),
            row.get("joint_js") or float("nan"),
            row.get("mean_marginal_jsd") or float("nan"),
            row.get("mnl_behavioral_similarity"),
        )

    summary_path = os.path.join(args.output_dir, "step_b_summary.csv")
    summary_df = pd.DataFrame(rows).sort_values("joint_js")
    summary_df.to_csv(summary_path, index=False)
    summary_json = os.path.join(args.output_dir, "step_b_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logging.info("Saved summary: %s", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
