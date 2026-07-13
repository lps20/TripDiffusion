"""Re-generate and evaluate revision baselines with HCD-aligned eval settings."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

from scripts.eval.generate_and_eval_baseline_checkpoint import (
    CHECKPOINT_NAMES,
    generate_and_evaluate,
)
from utils.multi_seed import aggregate_headline_metrics, extract_headline_metrics


def _discover_seed_dirs(root: Path) -> list[Path]:
    seed_dirs = sorted(root.glob("seed_*"))
    if seed_dirs:
        return seed_dirs
    raise FileNotFoundError(f"No seed_* directories found under {root}")


def _parse_seed(seed_dir: Path) -> int:
    return int(seed_dir.name.split("_", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-generate/evaluate CTGAN, DDPM-TF, TabDDPM with HCD-aligned full-test eval."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ctgan", "ddpm_tf", "tabddpm"],
        choices=["ctgan", "ddpm_tf", "tabddpm"],
        help="Baseline models to re-evaluate.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="exp/revision_baseline",
        help="Revision baseline root containing ctgan/ddpm_tf/tabddpm subdirectories.",
    )
    parser.add_argument("--train_data", type=str, default="data/train_data.csv")
    parser.add_argument("--test_data", type=str, default="data/test_data.csv")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Generated rows per seed. 0 = full test size (HCD revision protocol).",
    )
    parser.add_argument("--sample_batch_size", type=int, default=5000)
    parser.add_argument("--eval_only", action="store_true", help="Skip generation if *_gene.csv exists.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = Path(args.root)
    summaries = {}

    for model_name in args.models:
        model_root = root / model_name
        if not model_root.exists():
            logging.warning("Skip %s: directory not found at %s", model_name, model_root)
            continue

        per_seed_rows = []
        for seed_dir in _discover_seed_dirs(model_root):
            seed = _parse_seed(seed_dir)
            checkpoint = seed_dir / CHECKPOINT_NAMES[model_name]
            if not checkpoint.exists():
                logging.warning("Skip %s: missing checkpoint %s", seed_dir, checkpoint)
                continue

            logging.info("Re-evaluating %s seed=%d from %s", model_name, seed, checkpoint)
            payload = generate_and_evaluate(
                model_name=model_name,
                checkpoint=str(checkpoint),
                output_dir=seed_dir,
                train_data=args.train_data,
                test_data=args.test_data,
                num_samples=args.num_samples,
                sample_batch_size=args.sample_batch_size,
                seed=seed,
                eval_only=args.eval_only,
            )
            row = {
                "seed": seed,
                "eval_sampling": payload.get("eval_sampling"),
                **extract_headline_metrics(payload.get("evaluation", {})),
            }
            per_seed_rows.append(row)
            logging.info(
                "%s seed=%d | joint_js=%.6f | marginal_jsd=%.6f | LVR=%.4f | MNL sim=%s",
                model_name,
                seed,
                row.get("joint_js") or float("nan"),
                row.get("mean_marginal_jsd") or float("nan"),
                row.get("logical_validity_rate") or float("nan"),
                row.get("mnl_behavioral_similarity"),
            )

        summary = aggregate_headline_metrics(per_seed_rows)
        summary["model"] = model_name
        summary["num_seeds"] = len(per_seed_rows)
        summary["eval_num_samples"] = args.num_samples
        summaries[model_name] = summary

        summary_path = model_root / "multi_seed_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info("Saved %s multi-seed summary: %s", model_name, summary_path)

    combined_path = root / "multi_seed_summary_all_baselines.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logging.info("Saved combined summary: %s", combined_path)


if __name__ == "__main__":
    main()
