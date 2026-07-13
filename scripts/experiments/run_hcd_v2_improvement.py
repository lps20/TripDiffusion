"""HCD v2 improvement ablations: joint loss, balanced zcode sampling, gate init, zcode loss weights."""

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


PHASE1_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "baseline",
        "description": "Revision default: sequential act sampling, lambda_joint=0.5",
    },
    {
        "name": "joint_1p5",
        "description": "Stronger loc/time joint supervision",
        "lambda_joint": 1.5,
    },
    {
        "name": "balanced_zcode",
        "description": "Inverse-frequency batch sampling on start_zcode_num",
        "batch_sampling": "balanced",
        "sampling_feature": "start_zcode_num",
    },
    {
        "name": "joint_balanced",
        "description": "joint_1p5 + balanced_zcode",
        "lambda_joint": 1.5,
        "batch_sampling": "balanced",
        "sampling_feature": "start_zcode_num",
    },
    {
        "name": "gate_st_open",
        "description": "Initialize ST-stream gate alpha~0.5 instead of ~0.27",
        "gate_init_st": 0.0,
    },
    {
        "name": "zcode_loss2x",
        "description": "Double CE/VB weight on location zcodes",
        "feature_loss_weights": '{"start_zcode_num": 2.0, "end_zcode_num": 2.0}',
    },
]


def _ensure_train_subset(full_train: str, subset_path: str, n_rows: int, seed: int) -> str:
    if os.path.exists(subset_path):
        return subset_path
    os.makedirs(os.path.dirname(subset_path) or ".", exist_ok=True)
    df = pd.read_csv(full_train)
    n = min(n_rows, len(df))
    df.sample(n=n, random_state=seed).reset_index(drop=True).to_csv(subset_path, index=False)
    logging.info("Created train subset (%d rows): %s", n, subset_path)
    return subset_path


def _run_experiment(args: argparse.Namespace, exp: Dict[str, Any], run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    metrics_file = os.path.join(run_dir, "generated_samples_metrics.json")
    cmd = [
        sys.executable,
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
        "--sampling_power",
        str(exp.get("sampling_power", args.sampling_power)),
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
    if exp.get("feature_loss_weights"):
        cmd.extend(["--feature_loss_weights", exp["feature_loss_weights"]])
    logging.info("=== Running %s ===", exp["name"])
    logging.info("Desc: %s", exp.get("description", ""))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _extract_row(exp_name: str, metrics_file: str) -> Dict[str, Any]:
    with open(metrics_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ev = payload.get("evaluation", {})
    jsd = ev.get("single_feature_jsd") or {}
    gates = payload.get("gate_values") or []
    alpha_st = None
    if gates:
        alpha_st = float(gates[0].get("alpha_st", float("nan")))
    return {
        "experiment": exp_name,
        "joint_js": ev.get("joint_js"),
        "mean_marginal_jsd": ev.get("mean_marginal_jsd"),
        "start_zcode_jsd": jsd.get("start_zcode_num"),
        "end_zcode_jsd": jsd.get("end_zcode_num"),
        "act_jsd": jsd.get("act_num"),
        "mode_jsd": jsd.get("mode_num"),
        "mean_ordinal_emd": ev.get("mean_ordinal_emd"),
        "logical_validity_rate": ev.get("logical_validity_rate"),
        "mnl_behavioral_similarity": ev.get("mnl_behavioral_similarity"),
        "alpha_st_layer0": alpha_st,
        "num_samples": payload.get("num_samples"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HCD v2 improvement ablations.")
    parser.add_argument("--output_dir", type=str, default="exp/hcd_v2_improvement/phase1_50k")
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--traindata", type=str, default=None, help="Override train CSV (default: 50k subset)")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--subset_rows", type=int, default=50000)
    parser.add_argument("--experiments", nargs="+", default=None, help="Subset of experiment names to run")
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
    parser.add_argument("--sampling_power", type=float, default=1.0)
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

    selected = PHASE1_EXPERIMENTS
    if args.experiments:
        names = set(args.experiments)
        selected = [e for e in PHASE1_EXPERIMENTS if e["name"] in names]
        if not selected:
            raise ValueError(f"No matching experiments in {names}")

    rows: List[Dict[str, Any]] = []
    for exp in selected:
        run_dir = os.path.join(args.output_dir, exp["name"])
        _run_experiment(args, exp, run_dir)
        metrics_file = os.path.join(run_dir, "generated_samples_metrics.json")
        rows.append(_extract_row(exp["name"], metrics_file))
        logging.info(
            "Result %s | zcode_jsd=%.4f/%.4f | joint_js=%.4f | MNL=%s",
            exp["name"],
            rows[-1].get("start_zcode_jsd") or float("nan"),
            rows[-1].get("end_zcode_jsd") or float("nan"),
            rows[-1].get("joint_js") or float("nan"),
            rows[-1].get("mnl_behavioral_similarity"),
        )

    summary_path = os.path.join(args.output_dir, "phase1_summary.csv")
    summary_df = pd.DataFrame(rows).sort_values("start_zcode_jsd")
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved summary: %s", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
