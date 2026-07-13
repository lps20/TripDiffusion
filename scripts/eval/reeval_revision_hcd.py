"""Re-run 1:1 full-test generation and metrics for trained HCD v2 checkpoints."""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

from scripts.train.run_hcd_v2 import run_once
from utils.multi_seed import aggregate_headline_metrics, extract_headline_metrics


def _discover_seed_dirs(root: Path) -> list[Path]:
    seed_dirs = sorted(root.glob("seed_*"))
    if seed_dirs:
        return seed_dirs
    if (root / "model.pth").exists():
        return [root]
    raise FileNotFoundError(f"No seed_* directories or model.pth found under {root}")


def _parse_seed(seed_dir: Path) -> int:
    name = seed_dir.name
    if name.startswith("seed_"):
        return int(name.split("_", 1)[1])
    return 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved HCD v2 models with 1:1 full-test generation.")
    parser.add_argument(
        "--root",
        type=str,
        default="exp/revision_hcd",
        help="Experiment root containing seed_* subdirectories.",
    )
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument(
        "--joint_pairs",
        type=str,
        default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = Path(args.root)
    seed_dirs = _discover_seed_dirs(root)
    per_seed_rows = []

    for seed_dir in seed_dirs:
        seed = _parse_seed(seed_dir)
        model_path = seed_dir / "model.pth"
        if not model_path.exists():
            logging.warning("Skip %s: missing model.pth", seed_dir)
            continue

        run_args = argparse.Namespace(
            traindata=args.traindata,
            testdata=args.testdata,
            patience=10,
            min_delta=1e-4,
            epochs=100,
            batch_size=256,
            lr=1e-3,
            lambda_weight=2.0,
            lambda_joint=0.5,
            T=args.T,
            parallel=False,
            num_samples=0,
            random_condition_sampling=False,
            exp_dir=str(seed_dir),
            joint_pairs=args.joint_pairs,
            loss_type="standard",
            causal_weight=None,
            batch_sampling="sequential",
            sampling_feature="act_num",
            sampling_power=1.0,
            t_sampling="uniform",
            checkpoint=str(model_path),
            eval_only=True,
            metrics_file=str(seed_dir / "generated_samples_metrics.json"),
        )

        logging.info("Re-evaluating %s (seed=%d) with 1:1 full-test generation", seed_dir, seed)
        payload = run_once(run_args, seed, str(seed_dir))
        eval_metrics = payload.get("evaluation", {})
        row = {"seed": seed, **extract_headline_metrics(eval_metrics)}
        per_seed_rows.append(row)
        logging.info(
            "seed=%d | joint_js=%.6f | marginal_jsd=%.6f | LVR=%.4f | MNL sim=%s",
            seed,
            row.get("joint_js") or float("nan"),
            row.get("mean_marginal_jsd") or float("nan"),
            row.get("logical_validity_rate") or float("nan"),
            row.get("mnl_behavioral_similarity"),
        )

    summary = aggregate_headline_metrics(per_seed_rows)
    summary_path = root / "multi_seed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Saved multi-seed summary: %s", summary_path)
    logging.info("Completed re-evaluation for %d seeds under %s", len(per_seed_rows), root)


if __name__ == "__main__":
    main()
