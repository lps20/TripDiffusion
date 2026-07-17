"""Ablate ST cascade chain presets on the shared 20k / seed_42 subset."""

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

from model.HCD_Net_v2 import ST_CASCADE_PRESETS

DEFAULT_CHAINS = [
    "loc_then_time",  # current default; reused from robustness_20k when possible
    "time_then_loc",
    "end_first_loc",
    "zcode_first",
    "types_then_z",
    "start_then_end",
]
LEGACY_LOC_THEN_TIME = Path("exp/robustness_20k/seed_42/hcd_st_cascade")
LEGACY_SUBSET = Path("exp/robustness_20k/seed_42/train_subset_20000.csv")


def _py() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


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


def _ensure_subset(full_train: str, subset_path: Path, n: int, seed: int) -> Path:
    if subset_path.exists():
        logging.info("Reusing subset %s", subset_path)
        return subset_path
    if LEGACY_SUBSET.exists() and n == 20000 and seed == 42:
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_SUBSET, subset_path)
        logging.info("Copied legacy subset -> %s", subset_path)
        return subset_path
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(full_train)
    out = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    out.to_csv(subset_path, index=False)
    logging.info("Wrote %d-row subset to %s", len(out), subset_path)
    return subset_path


def _maybe_reuse_legacy(chain: str, out_dir: Path) -> bool:
    if chain != "loc_then_time":
        return False
    metrics = out_dir / "generated_samples_metrics.json"
    if metrics.exists():
        return False
    legacy_metrics = LEGACY_LOC_THEN_TIME / "generated_samples_metrics.json"
    if not legacy_metrics.exists():
        return False
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(LEGACY_LOC_THEN_TIME, out_dir)
    logging.info("Reused legacy loc_then_time run -> %s", out_dir)
    return True


def _run_chain(
    chain: str,
    train: str,
    test: str,
    out_dir: Path,
    seed: int,
    epochs: int,
) -> Dict[str, Any]:
    metrics_path = out_dir / "generated_samples_metrics.json"
    if metrics_path.exists():
        logging.info("Skip %s: existing metrics", chain)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    phase1, phase2 = ST_CASCADE_PRESETS[chain]
    logging.info("Chain %s | phase1=%s | phase2=%s", chain, phase1, phase2)
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
            "--st_cascade_chain",
            chain,
            "--exp_dir",
            str(out_dir),
        ]
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="20k ST cascade chain ablations.")
    parser.add_argument("--output_root", type=str, default="exp/cascade_chain_20k")
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--subset_rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--chains",
        nargs="+",
        default=DEFAULT_CHAINS,
        choices=sorted(ST_CASCADE_PRESETS.keys()),
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.info("Args: %s", vars(args))

    seed_root = root / f"seed_{args.seed}"
    subset = _ensure_subset(
        args.full_train,
        seed_root / f"train_subset_{args.subset_rows}.csv",
        args.subset_rows,
        args.seed,
    )

    rows: List[Dict[str, Any]] = []
    for i, chain in enumerate(args.chains, start=1):
        logging.info("===== [%d/%d] chain=%s =====", i, len(args.chains), chain)
        out_dir = seed_root / chain
        _maybe_reuse_legacy(chain, out_dir)
        try:
            payload = _run_chain(
                chain=chain,
                train=str(subset),
                test=args.testdata,
                out_dir=out_dir,
                seed=args.seed,
                epochs=args.epochs,
            )
        except Exception:
            logging.exception("FAILED chain=%s", chain)
            continue

        phase1, phase2 = ST_CASCADE_PRESETS[chain]
        row = {
            "chain": chain,
            "phase1": " -> ".join(phase1),
            "phase2": " -> ".join(phase2) if phase2 else "",
            "seed": args.seed,
            "train_subset_rows": args.subset_rows,
            **_headline(payload),
        }
        rows.append(row)
        logging.info(
            "DONE %s | joint_js=%s | MNL=%s | LVR=%s | EMD=%s",
            chain,
            row.get("joint_js"),
            row.get("mnl_behavioral_similarity"),
            row.get("logical_validity_rate"),
            row.get("mean_ordinal_emd"),
        )
        summary = pd.DataFrame(rows).sort_values("joint_js")
        summary.to_csv(seed_root / "summary_chains.csv", index=False)
        (seed_root / "summary_chains.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if rows:
        print(pd.DataFrame(rows).sort_values("joint_js").to_string(index=False))


if __name__ == "__main__":
    main()
