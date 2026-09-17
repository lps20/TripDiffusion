"""Paired 10k stream ablation for shared-only, soft-stream, and hard-stream.

The three variants use the exact same training subset for each seed. Existing
soft-stream artifacts can be imported from a prior robustness run so only the
missing shared-only and hard-stream cells need to be trained.

Outputs:
  <output_root>/seed_<seed>/train_subset_<N>.csv
  <output_root>/seed_<seed>/<variant>/generated_samples_metrics.json
  <output_root>/summary_all.csv
  <output_root>/multi_seed_summary.csv
  <output_root>/paired_differences_seed.csv
  <output_root>/paired_summary.csv
  <output_root>/subset_manifest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import numpy as np
import pandas as pd


VARIANTS = ("shared_only", "soft_stream", "hard_stream")

METRICS = (
    "joint_js",
    "joint_js_normalized",
    "mean_marginal_jsd",
    "mean_single_feature_jsd_normalized",
    "mean_ordinal_emd",
    "logical_validity_rate",
    "mnl_behavioral_similarity",
)

LOWER_IS_BETTER = {
    "joint_js",
    "joint_js_normalized",
    "mean_marginal_jsd",
    "mean_single_feature_jsd_normalized",
    "mean_ordinal_emd",
}

HIGHER_IS_BETTER = {
    "logical_validity_rate",
    "mnl_behavioral_similarity",
}

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _python_command() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [
            conda,
            "run",
            "-n",
            "tripdiffusion",
            "--no-capture-output",
            "python",
        ]
    return [sys.executable]


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_subset(
    full_train: Path,
    destination: Path,
    subset_rows: int,
    seed: int,
    subset_source_root: Optional[Path],
) -> Dict[str, Any]:
    source = None
    if subset_source_root is not None:
        candidate = subset_source_root / f"seed_{seed}" / f"train_subset_{subset_rows}.csv"
        if candidate.exists():
            source = candidate

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source is not None and _sha256(destination) != _sha256(source):
            raise RuntimeError(
                f"Existing subset differs from archived source: {destination} vs {source}"
            )
        logging.info("Reusing subset: %s", destination)
    elif source is not None:
        shutil.copy2(source, destination)
        logging.info("Copied archived subset: %s -> %s", source, destination)
    else:
        frame = pd.read_csv(full_train)
        sampled = frame.sample(
            n=min(subset_rows, len(frame)), random_state=seed
        ).reset_index(drop=True)
        sampled.to_csv(destination, index=False)
        logging.info("Created %d-row subset: %s", len(sampled), destination)

    row_count = int(pd.read_csv(destination).shape[0])
    if row_count != subset_rows:
        raise RuntimeError(
            f"Subset row count mismatch for {destination}: {row_count} != {subset_rows}"
        )
    return {
        "seed": seed,
        "subset_rows": row_count,
        "subset_path": str(destination.resolve()),
        "subset_sha256": _sha256(destination),
        "source_subset_path": str(source.resolve()) if source is not None else None,
    }


def _copy_soft_artifact(
    soft_source_root: Path,
    seed: int,
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    candidates = (
        soft_source_root / f"seed_{seed}" / "hcd" / "generated_samples_metrics.json",
        soft_source_root
        / f"seed_{seed}"
        / "hcd_st_cascade"
        / "generated_samples_metrics.json",
        soft_source_root / f"seed_{seed}" / "soft_stream" / "generated_samples_metrics.json",
    )
    source_metrics = next((path for path in candidates if path.exists()), None)
    if source_metrics is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    target_metrics = out_dir / "generated_samples_metrics.json"
    if not target_metrics.exists():
        shutil.copy2(source_metrics, target_metrics)
    source_log = source_metrics.parent / "training.log"
    if source_log.exists() and not (out_dir / "training.log").exists():
        shutil.copy2(source_log, out_dir / "training.log")

    provenance = {
        "variant": "soft_stream",
        "source_metrics_path": str(source_metrics.resolve()),
        "source_metrics_sha256": _sha256(source_metrics),
    }
    (out_dir / "artifact_source.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    logging.info("Imported archived soft-stream metrics: %s", source_metrics)
    return json.loads(target_metrics.read_text(encoding="utf-8"))


def _variant_flags(variant: str) -> List[str]:
    if variant == "soft_stream":
        return ["--st_cascade"]
    if variant == "shared_only":
        return [
            "--st_cascade",
            "--freeze_gates",
            "--gate_init_act",
            "-20.0",
            "--gate_init_st",
            "-20.0",
            "--gate_init_mode",
            "-20.0",
        ]
    if variant == "hard_stream":
        return ["--hard_stream_cascade"]
    raise ValueError(f"Unknown variant: {variant}")


def _run_variant(
    variant: str,
    train_csv: Path,
    test_csv: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    num_samples: int,
    random_condition_sampling: bool,
) -> Dict[str, Any]:
    metrics_path = out_dir / "generated_samples_metrics.json"
    if metrics_path.exists():
        logging.info("Skip existing cell: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _python_command() + [
        "scripts/train/run_hcd_v2.py",
        "--traindata",
        str(train_csv),
        "--testdata",
        str(test_csv),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
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
        str(max(epochs, 100)),
        "--min_delta",
        "0.0",
        "--num_samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--num_seeds",
        "1",
        "--exp_dir",
        str(out_dir),
    ]
    if random_condition_sampling:
        cmd.append("--random_condition_sampling")
    cmd.extend(_variant_flags(variant))
    _run(cmd)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _metric_row(
    payload: Dict[str, Any],
    variant: str,
    seed: int,
    subset_rows: int,
    subset_sha256: str,
) -> Dict[str, Any]:
    evaluation = payload.get("evaluation", payload)
    row = {
        "variant": variant,
        "seed": seed,
        "train_subset_rows": subset_rows,
        "subset_sha256": subset_sha256,
        "mnl_status": evaluation.get("mnl_status"),
        "num_parameters": payload.get("num_parameters"),
    }
    for metric in METRICS:
        row[metric] = evaluation.get(metric)
    return row


def _write_summaries(frame: pd.DataFrame, output_root: Path) -> None:
    frame = frame.sort_values(["variant", "seed"]).reset_index(drop=True)
    frame.to_csv(output_root / "summary_all.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    for variant, group in frame.groupby("variant", sort=True):
        row: Dict[str, Any] = {
            "variant": variant,
            "n_seed_rows": int(len(group)),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        output_root / "multi_seed_summary.csv", index=False
    )

    paired_rows: List[Dict[str, Any]] = []
    for comparator in ("shared_only", "hard_stream"):
        for metric in METRICS:
            values = frame.pivot(index="seed", columns="variant", values=metric)
            if "soft_stream" not in values or comparator not in values:
                continue
            complete = values[["soft_stream", comparator]].dropna()
            for seed, pair in complete.iterrows():
                if metric in LOWER_IS_BETTER:
                    improvement = pair[comparator] - pair["soft_stream"]
                elif metric in HIGHER_IS_BETTER:
                    improvement = pair["soft_stream"] - pair[comparator]
                else:
                    raise RuntimeError(f"Metric direction missing: {metric}")
                paired_rows.append(
                    {
                        "comparison": f"soft_stream_vs_{comparator}",
                        "metric": metric,
                        "seed": int(seed),
                        "soft_value": float(pair["soft_stream"]),
                        "comparator_value": float(pair[comparator]),
                        "paired_improvement_positive_favors_soft": float(improvement),
                    }
                )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_root / "paired_differences_seed.csv", index=False)

    paired_summary: List[Dict[str, Any]] = []
    if not paired.empty:
        for (comparison, metric), group in paired.groupby(
            ["comparison", "metric"], sort=True
        ):
            values = group["paired_improvement_positive_favors_soft"].to_numpy(
                dtype=float
            )
            n_pairs = len(values)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n_pairs > 1 else np.nan
            if n_pairs > 1:
                df = n_pairs - 1
                t_critical = T_CRITICAL_975.get(df, 1.96)
                half_width = float(t_critical * std / math.sqrt(n_pairs))
                ci_low = mean - half_width
                ci_high = mean + half_width
            else:
                ci_low = np.nan
                ci_high = np.nan
            paired_summary.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "mean_paired_improvement_positive_favors_soft": mean,
                    "std_paired_improvement": std,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    pd.DataFrame(paired_summary).to_csv(
        output_root / "paired_summary.csv", index=False
    )


def _parse_seeds(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("exp/revision_stream_ablation_10k"),
    )
    parser.add_argument("--full_train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--testdata", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--subset_rows", type=int, default=10000)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--random_condition_sampling", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--subset_source_root",
        type=Path,
        default=None,
        help="Optional root containing seed_<s>/train_subset_<N>.csv.",
    )
    parser.add_argument(
        "--soft_source_root",
        type=Path,
        default=None,
        help="Optional root containing archived soft-stream seed directories.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_root / "run.log", encoding="utf-8"),
        ],
    )
    logging.info("Args: %s", vars(args))

    seeds = _parse_seeds(args.seeds)
    planned = [(seed, variant) for seed in seeds for variant in args.variants]
    logging.info("Planned cells: %s", planned)
    if args.dry_run:
        return

    rows: List[Dict[str, Any]] = []
    subset_manifest: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_root = args.output_root / f"seed_{seed}"
        subset = seed_root / f"train_subset_{args.subset_rows}.csv"
        subset_info = _ensure_subset(
            full_train=args.full_train,
            destination=subset,
            subset_rows=args.subset_rows,
            seed=seed,
            subset_source_root=args.subset_source_root,
        )
        subset_manifest.append(subset_info)

        for variant in args.variants:
            out_dir = seed_root / variant
            payload = None
            if variant == "soft_stream" and args.soft_source_root is not None:
                payload = _copy_soft_artifact(args.soft_source_root, seed, out_dir)
            if payload is None:
                payload = _run_variant(
                    variant=variant,
                    train_csv=subset,
                    test_csv=args.testdata,
                    out_dir=out_dir,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    random_condition_sampling=args.random_condition_sampling,
                )
            row = _metric_row(
                payload=payload,
                variant=variant,
                seed=seed,
                subset_rows=args.subset_rows,
                subset_sha256=subset_info["subset_sha256"],
            )
            rows.append(row)
            logging.info(
                "DONE seed=%d variant=%s joint_js=%s marginal_jsd=%s LVR=%s MNL=%s",
                seed,
                variant,
                row.get("joint_js"),
                row.get("mean_marginal_jsd"),
                row.get("logical_validity_rate"),
                row.get("mnl_behavioral_similarity"),
            )
            _write_summaries(pd.DataFrame(rows), args.output_root)

    pd.DataFrame(subset_manifest).drop_duplicates("seed").sort_values("seed").to_csv(
        args.output_root / "subset_manifest.csv", index=False
    )
    _write_summaries(pd.DataFrame(rows), args.output_root)
    logging.info("Experiment complete: %s", args.output_root)


if __name__ == "__main__":
    main()
