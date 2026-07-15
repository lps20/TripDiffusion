"""Re-generate and evaluate TVAE / DATGAN with HCD-aligned full-test eval."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

from utils.multi_seed import aggregate_headline_metrics, extract_headline_metrics


def _discover_seed_dirs(root: Path) -> list[Path]:
    seed_dirs = sorted(root.glob("seed_*"))
    if seed_dirs:
        return seed_dirs
    raise FileNotFoundError(f"No seed_* directories found under {root}")


def _parse_seed(seed_dir: Path) -> int:
    return int(seed_dir.name.split("_", 1)[1])


def _python_prefix() -> list[str]:
    import shutil

    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate TVAE and DATGAN at full test scale.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tvae", "datgan"],
        choices=["tvae", "datgan"],
    )
    parser.add_argument("--root", type=str, default="exp/revision_baseline")
    parser.add_argument("--train_data", type=str, default="data/train_data.csv")
    parser.add_argument("--test_data", type=str, default="data/test_data.csv")
    parser.add_argument("--num_samples", type=int, default=534445)
    parser.add_argument("--sample_batch_size", type=int, default=5000)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = Path(args.root)
    py = _python_prefix()
    summaries = {}

    for model_name in args.models:
        model_root = root / model_name
        per_seed_rows = []

        for seed_dir in _discover_seed_dirs(model_root):
            seed = _parse_seed(seed_dir)
            if model_name == "tvae":
                checkpoint = seed_dir / "TVAE_model.pkl"
                script = "scripts/eval/generate_and_eval_tvae_checkpoint.py"
                cmd = py + [
                    script,
                    "--checkpoint",
                    str(checkpoint),
                    "--output_dir",
                    str(seed_dir),
                    "--num_samples",
                    str(args.num_samples),
                    "--sample_batch_size",
                    str(args.sample_batch_size),
                    "--train_data",
                    args.train_data,
                    "--test_data",
                    args.test_data,
                    "--seed",
                    str(seed),
                ]
            else:
                checkpoint = seed_dir / "DATGAN_model"
                script = "scripts/eval/generate_and_eval_datgan_checkpoint.py"
                cmd = py + [
                    script,
                    "--checkpoint_dir",
                    str(checkpoint),
                    "--output_dir",
                    str(seed_dir),
                    "--num_samples",
                    str(args.num_samples),
                    "--sample_batch_size",
                    str(args.sample_batch_size),
                    "--train_data",
                    args.train_data,
                    "--test_data",
                    args.test_data,
                    "--seed",
                    str(seed),
                ]

            if not checkpoint.exists():
                logging.warning("Skip %s: missing %s", seed_dir, checkpoint)
                continue
            if args.eval_only:
                cmd.append("--eval_only")

            logging.info("Running %s seed=%d", model_name, seed)
            subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

            metrics_path = seed_dir / f"{model_name.upper()}_metrics.json"
            with open(metrics_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
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
        logging.info("Saved %s summary: %s", model_name, summary_path)

    combined_path = root / "multi_seed_summary_tvae_datgan.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logging.info("Saved combined summary: %s", combined_path)


if __name__ == "__main__":
    main()
