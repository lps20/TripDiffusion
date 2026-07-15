"""HCD v2 ablations on locked match_ddpm training recipe (50k screen)."""

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

from scripts.experiments.run_hcd_v2_improvement import _ensure_train_subset, _extract_row

DEFAULT_JOINT_PAIRS = "[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]"

EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "baseline_match",
        "description": "Locked match_ddpm: shuffle, λ_w=1, λ_j=0, empty joint pairs",
    },
    {
        "name": "gate_st_open",
        "description": "Initialize ST gate α≈0.5",
        "gate_init_st": 0.0,
    },
    {
        "name": "gate_all_open",
        "description": "Initialize act/st/mode gates α≈0.5",
        "gate_init_act": 0.0,
        "gate_init_st": 0.0,
        "gate_init_mode": 0.0,
    },
    {
        "name": "time_loss2x",
        "description": "2x CE/VB weight on ordinal time features",
        "feature_loss_weights": '{"start_time_num_6": 2.0, "trip_time_num_6": 2.0}',
    },
    {
        "name": "st_cascade",
        "description": "ST loc/time cascade in causal adapters",
        "st_cascade": True,
    },
    {
        "name": "mild_joint_batch_stats",
        "description": "No joint heads; mild batch_stats joint loss",
        "no_joint_heads": True,
        "lambda_joint": 0.5,
        "joint_pairs": DEFAULT_JOINT_PAIRS,
        "joint_loss_mode": "batch_stats",
        # Keep backbone at match_ddpm size (do not inflate to 192/3/3)
        "d_model": 128,
        "shared_layers": 2,
        "causal_layers": 2,
    },
    {
        "name": "t_sampling_late",
        "description": "Bias t sampling toward late denoising steps",
        "t_sampling": "late",
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
        str(exp.get("lambda_weight", args.lambda_weight)),
        "--lambda_joint",
        str(exp.get("lambda_joint", args.lambda_joint)),
        "--T",
        str(args.T),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--num_samples",
        str(args.num_samples),
        "--random_condition_sampling",
        "--exp_dir",
        run_dir,
        "--joint_pairs",
        str(exp.get("joint_pairs", args.joint_pairs)),
        "--batch_sampling",
        str(exp.get("batch_sampling", args.batch_sampling)),
        "--sampling_feature",
        str(exp.get("sampling_feature", args.sampling_feature)),
        "--t_sampling",
        str(exp.get("t_sampling", args.t_sampling)),
        "--gate_init_act",
        str(exp.get("gate_init_act", args.gate_init_act)),
        "--gate_init_st",
        str(exp.get("gate_init_st", args.gate_init_st)),
        "--gate_init_mode",
        str(exp.get("gate_init_mode", args.gate_init_mode)),
        "--joint_loss_mode",
        str(exp.get("joint_loss_mode", args.joint_loss_mode)),
        "--metrics_file",
        metrics_file,
        "--seed",
        str(args.seed),
        "--num_seeds",
        "1",
        "--parallel",
        "False",
    ]
    if exp.get("no_joint_heads"):
        cmd.append("--no_joint_heads")
        if exp.get("d_model") is not None:
            cmd.extend(["--d_model", str(exp["d_model"])])
        if exp.get("shared_layers") is not None:
            cmd.extend(["--shared_layers", str(exp["shared_layers"])])
        if exp.get("causal_layers") is not None:
            cmd.extend(["--causal_layers", str(exp["causal_layers"])])
    if exp.get("st_cascade"):
        cmd.append("--st_cascade")
    if exp.get("feature_loss_weights"):
        cmd.extend(["--feature_loss_weights", exp["feature_loss_weights"]])

    logging.info("=== Ablation: %s ===", exp["name"])
    logging.info("Desc: %s", exp.get("description", ""))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HCD v2 match_ddpm-recipe ablations on 50k train / 10k eval."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/hcd_v2_improvement/match_ddpm_ablations_50k",
    )
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--traindata", type=str, default=None)
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--subset_rows", type=int, default=50000)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--lambda_joint", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--joint_pairs", type=str, default="[]")
    parser.add_argument("--batch_sampling", type=str, default="shuffle")
    parser.add_argument("--sampling_feature", type=str, default="act_num")
    parser.add_argument("--t_sampling", type=str, default="uniform")
    parser.add_argument("--gate_init_act", type=float, default=-1.0)
    parser.add_argument("--gate_init_st", type=float, default=-1.0)
    parser.add_argument("--gate_init_mode", type=float, default=-1.0)
    parser.add_argument("--joint_loss_mode", type=str, default="batch_stats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-run even if generated_samples_metrics.json already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.traindata is None:
        # Prefer existing 50k subset used by other improvement runs.
        preferred = "exp/hcd_v2_improvement/phase1_50k/train_subset_50000.csv"
        if os.path.exists(preferred):
            args.traindata = preferred
            logging.info("Reusing train subset: %s", preferred)
        else:
            subset_path = os.path.join(args.output_dir, f"train_subset_{args.subset_rows}.csv")
            args.traindata = _ensure_train_subset(
                args.full_train, subset_path, args.subset_rows, args.seed
            )

    selected = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments)
        selected = [e for e in EXPERIMENTS if e["name"] in names]
        if not selected:
            raise ValueError(f"No matching experiments in {names}")

    rows: List[Dict[str, Any]] = []
    for exp in selected:
        run_dir = os.path.join(args.output_dir, exp["name"])
        metrics_file = os.path.join(run_dir, "generated_samples_metrics.json")
        if os.path.exists(metrics_file) and not args.force_rerun:
            logging.info("Skip existing metrics for %s (%s)", exp["name"], metrics_file)
            row = _extract_row(exp["name"], metrics_file)
        else:
            _run_experiment(args, exp, run_dir)
            row = _extract_row(exp["name"], metrics_file)
        rows.append(row)
        logging.info(
            "Result %s | joint_js=%.6f | EMD=%.4f | MNL=%s | LVR=%s | alpha_st0=%s",
            exp["name"],
            row.get("joint_js") or float("nan"),
            row.get("mean_ordinal_emd") or float("nan"),
            row.get("mnl_behavioral_similarity"),
            row.get("logical_validity_rate"),
            row.get("alpha_st_layer0"),
        )

    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df = pd.DataFrame(rows).sort_values("joint_js")
    summary_df.to_csv(summary_path, index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logging.info("Saved summary: %s", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
