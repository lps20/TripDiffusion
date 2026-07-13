"""Re-evaluate saved HCD v2 checkpoints with joint-pair Gibbs sampling at inference."""

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
    parser = argparse.ArgumentParser(
        description="Re-evaluate HCD v2 with joint-pair Gibbs sampling (Step A, no gate change)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="exp/revision_hcd",
        help="Checkpoint root containing seed_* subdirectories.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="exp/hcd_v2_improvement/joint_sampling_1to1",
        help="Directory for joint-sampling eval outputs.",
    )
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument(
        "--joint_pairs",
        type=str,
        default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]",
    )
    parser.add_argument(
        "--joint_sample_steps",
        type=str,
        default="[1]",
        help="Reverse-diffusion timesteps for joint Gibbs, e.g. '[1]' or '[1,2]'.",
    )
    parser.add_argument("--joint_gibbs_iters", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = Path(args.root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_dirs = _discover_seed_dirs(root)
    per_seed_rows = []

    for seed_dir in seed_dirs:
        seed = _parse_seed(seed_dir)
        model_path = seed_dir / "model.pth"
        if not model_path.exists():
            logging.warning("Skip %s: missing model.pth", seed_dir)
            continue

        out_dir = output_root / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

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
            exp_dir=str(out_dir),
            joint_pairs=args.joint_pairs,
            loss_type="standard",
            causal_weight=None,
            batch_sampling="sequential",
            sampling_feature="act_num",
            sampling_power=1.0,
            t_sampling="uniform",
            gate_init_act=-1.0,
            gate_init_st=-1.0,
            gate_init_mode=-1.0,
            feature_loss_weights=None,
            checkpoint=str(model_path),
            eval_only=True,
            joint_sampling_at_inference=True,
            joint_sample_steps=args.joint_sample_steps,
            joint_gibbs_iters=args.joint_gibbs_iters,
            metrics_file=str(out_dir / "generated_samples_metrics.json"),
        )

        logging.info(
            "Joint-sampling re-eval %s -> %s (seed=%d, gibbs_iters=%d, steps=%s)",
            seed_dir,
            out_dir,
            seed,
            args.joint_gibbs_iters,
            args.joint_sample_steps,
        )
        payload = run_once(run_args, seed, str(out_dir))
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
    summary["joint_sampling_at_inference"] = True
    summary["joint_sample_steps"] = args.joint_sample_steps
    summary["joint_gibbs_iters"] = args.joint_gibbs_iters
    summary["checkpoint_root"] = str(root)
    summary_path = output_root / "multi_seed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Saved summary: %s", summary_path)
    logging.info("Completed joint-sampling re-eval for %d seeds", len(per_seed_rows))


if __name__ == "__main__":
    main()
