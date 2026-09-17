"""Run the minimum stream/ST ordering study on paired 20k subsets and five seeds."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

STREAM_CONFIGS = {
    "soft_parallel": [],
    "act_st_mode": ["--hard_stream_cascade", "--stream_order", "act_st_mode"],
    "mode_st_act": ["--hard_stream_cascade", "--stream_order", "mode_st_act"],
    "st_act_mode": ["--hard_stream_cascade", "--stream_order", "st_act_mode"],
}
ST_CONFIGS = {
    "parallel_st": [],
    "loc_then_time": ["--st_cascade", "--st_cascade_chain", "loc_then_time"],
    "time_then_loc": ["--st_cascade", "--st_cascade_chain", "time_then_loc"],
    "types_then_z": ["--st_cascade", "--st_cascade_chain", "types_then_z"],
}


def python_cmd() -> list[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def headline(payload: dict[str, Any]) -> dict[str, Any]:
    ev = payload.get("evaluation", payload)
    return {
        "joint_js": ev.get("joint_js"),
        "mean_marginal_jsd": ev.get("mean_marginal_jsd"),
        "mean_ordinal_emd": ev.get("mean_ordinal_emd"),
        "logical_validity_rate": ev.get("logical_validity_rate"),
        "mnl_behavioral_similarity": ev.get("mnl_behavioral_similarity"),
    }


def ensure_subset(full_train: Path, subset: Path, n: int, seed: int) -> Path:
    if subset.exists():
        return subset
    subset.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(full_train)
    frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True).to_csv(
        subset, index=False
    )
    return subset


def run_one(
    study: str,
    config: str,
    extra: list[str],
    subset: Path,
    test: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    metrics = out_dir / "generated_samples_metrics.json"
    if not metrics.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        resume_flags: list[str] = []
        checkpoint = out_dir / "model.pth"
        training_log = out_dir / "training.log"
        if checkpoint.exists() and training_log.exists():
            completed = [
                int(value)
                for value in re.findall(
                    r"Epoch (\d+)/\d+: Average loss", training_log.read_text(encoding="utf-8")
                )
            ]
            if completed:
                resume_epoch = max(completed)
                resume_flags = [
                    "--checkpoint", str(checkpoint),
                    "--resume_epoch", str(resume_epoch),
                ]
                logging.info("Resuming %s/%s seed=%d after epoch %d", study, config, seed, resume_epoch)
        cmd = python_cmd() + [
            "scripts/train/run_hcd_v2.py",
            "--traindata", str(subset),
            "--testdata", str(test),
            "--epochs", str(epochs),
            "--batch_size", "500",
            "--lr", "0.001",
            "--lambda_weight", "1.0",
            "--lambda_joint", "0.0",
            "--T", "10",
            "--joint_pairs", "[]",
            "--batch_sampling", "shuffle",
            "--patience", "100",
            "--min_delta", "0.0",
            "--num_samples", "0",
            "--seed", str(seed),
            "--num_seeds", "1",
            "--exp_dir", str(out_dir),
            *resume_flags,
            *extra,
        ]
        logging.info("Running %s/%s seed=%d", study, config, seed)
        subprocess.run(cmd, cwd=ROOT, check=True)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    return {"study": study, "config": config, "seed": seed, **headline(payload)}


def aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    metrics = [
        "joint_js", "mean_marginal_jsd", "mean_ordinal_emd",
        "logical_validity_rate", "mnl_behavioral_similarity",
    ]
    records = []
    for (study, config), group in pd.DataFrame(rows).groupby(["study", "config"]):
        rec: dict[str, Any] = {"study": study, "config": config, "n_seeds": len(group)}
        for metric in metrics:
            rec[f"{metric}_mean"] = group[metric].mean()
            rec[f"{metric}_std"] = group[metric].std(ddof=1)
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--testdata", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--output_root", type=Path, default=Path("revision_exp/ordering"))
    parser.add_argument("--subset_rows", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--studies", nargs="+", choices=["stream", "st"], default=["stream", "st"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    studies = []
    if "stream" in args.studies:
        # Hold the within-ST order fixed to the expert loc->time order.
        studies.append(("stream", STREAM_CONFIGS, ["--st_cascade", "--st_cascade_chain", "loc_then_time"]))
    if "st" in args.studies:
        studies.append(("st", ST_CONFIGS, []))

    for seed in seeds:
        subset = ensure_subset(
            args.full_train,
            args.output_root / "subsets" / f"train_subset_{args.subset_rows}_seed_{seed}.csv",
            args.subset_rows,
            seed,
        )
        for study, configs, common in studies:
            for config, extra in configs.items():
                rows.append(
                    run_one(
                        study, config, [*common, *extra], subset, args.testdata,
                        args.output_root / study / config / f"seed_{seed}",
                        seed, args.epochs,
                    )
                )
                pd.DataFrame(rows).to_csv(args.output_root / "per_seed.csv", index=False)
                aggregate(rows).to_csv(args.output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
