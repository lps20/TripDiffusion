"""Promote selected HCD match_ddpm ablations to full-data seed 42."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

VARIANTS = {
    "st_cascade": {
        "extra": ["--st_cascade"],
        "exp_dir": "exp/revision_hcd_opt/st_cascade/seed_42",
    },
    "time_loss2x": {
        "extra": [
            "--feature_loss_weights",
            '{"start_time_num_6": 2.0, "trip_time_num_6": 2.0}',
        ],
        "exp_dir": "exp/revision_hcd_opt/time_loss2x/seed_42",
    },
}


def _prefix() -> list[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["st_cascade", "time_loss2x"],
        choices=sorted(VARIANTS.keys()),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for name in args.variants:
        cfg = VARIANTS[name]
        metrics = Path(cfg["exp_dir"]) / "generated_samples_metrics.json"
        if metrics.exists():
            logging.info("Skip %s: metrics already exist", name)
            continue
        cmd = _prefix() + [
            "scripts/train/run_hcd_v2.py",
            "--traindata",
            "data/train_data.csv",
            "--testdata",
            "data/test_data.csv",
            "--epochs",
            "100",
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
            "42",
            "--num_seeds",
            "1",
            "--exp_dir",
            cfg["exp_dir"],
        ] + cfg["extra"]
        logging.info("=== Full promote: %s ===", name)
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
        logging.info("Finished %s", name)


if __name__ == "__main__":
    main()
