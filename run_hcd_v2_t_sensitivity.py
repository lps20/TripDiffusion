import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_t_values(t_values: str) -> List[int]:
    vals = [int(v.strip()) for v in t_values.split(",") if v.strip()]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid T values provided.")
    return vals


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _run_single_t(args: argparse.Namespace, t_value: int, run_dir: str, metrics_file: str) -> None:
    cmd = [
        sys.executable,
        "run_hcd_v2.py",
        "--traindata",
        args.traindata,
        "--testdata",
        args.testdata,
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
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
        str(t_value),
        "--num_samples",
        str(args.num_samples),
        "--exp_dir",
        run_dir,
        "--joint_pairs",
        args.joint_pairs,
        "--loss_type",
        args.loss_type,
        "--batch_sampling",
        args.batch_sampling,
        "--sampling_feature",
        args.sampling_feature,
        "--sampling_power",
        str(args.sampling_power),
        "--t_sampling",
        args.t_sampling,
        "--metrics_file",
        metrics_file,
        "--seed",
        str(args.seed),
    ]
    if args.causal_weight is not None:
        cmd.extend(["--causal_weight", args.causal_weight])
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.eval_only:
        cmd.append("--eval_only")

    logging.info("Running T=%d experiment: %s", t_value, " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_metrics(metrics_file: str, fallback_num_samples: int) -> Dict[str, float]:
    with open(metrics_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    eval_metrics = payload.get("evaluation", {})
    joint_js = eval_metrics.get("joint_js")
    sampling_seconds = payload.get("sampling_seconds")
    num_samples = int(payload.get("num_samples", fallback_num_samples))
    per_10k = payload.get("sampling_seconds_per_10k")
    if per_10k is None and sampling_seconds is not None:
        per_10k = float(sampling_seconds) * (10000.0 / max(float(num_samples), 1.0))

    return {
        "T": int(payload.get("T")),
        "joint_js": np.nan if joint_js is None else float(joint_js),
        "sampling_seconds": np.nan if sampling_seconds is None else float(sampling_seconds),
        "sampling_seconds_per_10k": np.nan if per_10k is None else float(per_10k),
        "num_samples": num_samples,
    }


def _plot_sensitivity(df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    x = df["T"].to_numpy(dtype=float)
    y_jsd = df["joint_js"].to_numpy(dtype=float)
    y_time = df["sampling_seconds_per_10k"].to_numpy(dtype=float)

    axes[0].plot(x, y_jsd, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_xlabel("Diffusion Steps T")
    axes[0].set_ylabel("Joint JSD")
    axes[0].set_title("JSD Sensitivity to T")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].set_xticks(df["T"].tolist())

    axes[1].plot(x, y_time, marker="o", linewidth=2, color="#d62728")
    axes[1].set_xlabel("Diffusion Steps T")
    axes[1].set_ylabel("Inference Time (sec / 10k samples)")
    axes[1].set_title("Sampling Efficiency vs T")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_xticks(df["T"].tolist())

    for ax, ys in zip(axes, [y_jsd, y_time]):
        for xv, yv in zip(x, ys):
            if np.isfinite(yv):
                ax.annotate(f"{yv:.4f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    os.makedirs(args.base_exp_dir, exist_ok=True)
    t_values = _parse_t_values(args.t_values)

    rows: List[Dict[str, float]] = []
    for t in t_values:
        run_dir = os.path.join(args.base_exp_dir, f"T_{t}")
        metrics_file = os.path.join(run_dir, "generated_samples_metrics.json")

        if args.skip_runs:
            if not os.path.exists(metrics_file):
                raise FileNotFoundError(f"Missing metrics file for T={t}: {metrics_file}")
            logging.info("Skip run enabled. Reusing metrics for T=%d.", t)
        elif args.reuse_existing and os.path.exists(metrics_file):
            logging.info("Reusing existing run for T=%d: %s", t, metrics_file)
        else:
            os.makedirs(run_dir, exist_ok=True)
            _run_single_t(args=args, t_value=t, run_dir=run_dir, metrics_file=metrics_file)

        row = _load_metrics(metrics_file=metrics_file, fallback_num_samples=args.num_samples)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    summary_csv = os.path.join(args.base_exp_dir, "t_sensitivity_summary.csv")
    df.to_csv(summary_csv, index=False)
    logging.info("Saved summary CSV: %s", summary_csv)

    fig_png = os.path.join(args.base_exp_dir, "hcd_v2_t_sensitivity.png")
    fig_pdf = os.path.join(args.base_exp_dir, "hcd_v2_t_sensitivity.pdf")
    _plot_sensitivity(df=df, out_png=fig_png, out_pdf=fig_pdf)
    logging.info("Saved sensitivity plots: %s, %s", fig_png, fig_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCD v2 sensitivity analysis for diffusion steps T.")
    parser.add_argument("--t_values", type=str, default="10,50,100,200", help="Comma-separated T values, e.g. '10,50,100,200'")
    parser.add_argument("--base_exp_dir", type=str, default="exp/hcd_v2_t_sensitivity", help="Base directory for sweep runs and outputs")
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse existing run outputs if metrics file exists")
    parser.add_argument("--skip_runs", action="store_true", help="Do not launch runs; only read existing metrics and replot")

    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--lambda_joint", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=10000, help="Samples generated per T run")
    parser.add_argument("--joint_pairs", type=str, default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]")
    parser.add_argument("--loss_type", type=str, default="standard")
    parser.add_argument("--causal_weight", type=str, default=None)
    parser.add_argument("--batch_sampling", type=str, default="sequential", choices=["sequential", "shuffle", "balanced"])
    parser.add_argument("--sampling_feature", type=str, default="act_num")
    parser.add_argument("--sampling_power", type=float, default=1.0)
    parser.add_argument("--t_sampling", type=str, default="uniform", choices=["uniform", "sqrt", "late"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
