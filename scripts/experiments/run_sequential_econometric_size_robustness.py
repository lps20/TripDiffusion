"""5/10/20k × 5-seed sample-size robustness for Sequential Econometric.

Fits a new generator on each train subset, then samples 1:1 against the full test set.
Reuses existing robustness_size subsets when present so comparisons stay aligned.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.test_utils
from scripts.baselines.run_tabular_baselines import ALL_COLUMNS, FULL_SCHEMA, _sanitize_df_by_schema
from scripts.baselines.sequential_econometric_baseline import STAGES, SequentialEconometricGenerator
from scripts.experiments.run_sequential_econometric_20k import (
    COND_COLUMNS,
    FEATURES_INFO,
    TRIP_COLUMNS,
    ensure_subset,
)
from scripts.experiments.run_size_seed_robustness import (
    HEADLINE_KEYS,
    _headline,
    _size_tag,
    _upsert_summary,
    _write_multiseed_summary,
)

REVISION_ROBUSTNESS = (
    ROOT.parent / "D3PM_revision" / "revision_exp" / "robustness_size"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def evaluate(generated: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> Dict[str, Any]:
    return utils.test_utils.evaluate_generated_trips(
        test[TRIP_COLUMNS].astype(int).values.tolist(),
        generated[TRIP_COLUMNS].astype(int).values.tolist(),
        FEATURES_INFO,
        cond_info=[{"name": column} for column in COND_COLUMNS],
        generated_df=generated[ALL_COLUMNS],
        train_real_df=train,
        test_real_df=test,
        random_state=seed,
    )


def resolve_subset(size: int, seed: int, full_train: Path, output_root: Path) -> Path:
    filename = f"train_subset_{size}.csv"
    candidates = [
        output_root / _size_tag(size) / f"seed_{seed}" / filename,
        REVISION_ROBUSTNESS / _size_tag(size) / f"seed_{seed}" / filename,
        Path("exp/robustness_20k") / f"seed_{seed}" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    dest = output_root / _size_tag(size) / f"seed_{seed}" / filename
    return ensure_subset(full_train, dest, size, seed)


def run_cell(
    subset_path: Path,
    test: pd.DataFrame,
    out_dir: Path,
    seed: int,
    sample_batch_size: int,
    force: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_path = out_dir / "SEQUENTIAL_ECON_gene.csv"
    metrics_path = out_dir / "SEQUENTIAL_ECON_metrics.json"
    run_path = out_dir / "run_metrics.json"
    model_path = out_dir / "SEQUENTIAL_ECON_model.joblib"

    train = _sanitize_df_by_schema(pd.read_csv(subset_path), FULL_SCHEMA)

    if generated_path.exists() and metrics_path.exists() and not force:
        logging.info("Skip existing cell: %s", out_dir)
        payload = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        evaluation = metrics.get("evaluation", metrics)
        payload.setdefault("evaluation", evaluation)
        payload.setdefault("seed", seed)
        payload.setdefault("model", "sequential_econ")
        payload.setdefault("train_subset_rows", len(train))
        return payload

    if generated_path.exists() and not force:
        generated = _sanitize_df_by_schema(pd.read_csv(generated_path), FULL_SCHEMA)
        existing = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
        fit_seconds = existing.get("fit_seconds")
        sample_seconds = existing.get("sample_seconds")
    else:
        logging.info("Fitting sequential econometric on %d rows (seed=%d)", len(train), seed)
        started = time.perf_counter()
        model = SequentialEconometricGenerator(random_state=seed).fit(train)
        fit_seconds = time.perf_counter() - started
        model.save(model_path)
        logging.info("Fitted in %.1f seconds", fit_seconds)

        started = time.perf_counter()
        generated = model.sample(test, batch_size=sample_batch_size)
        sample_seconds = time.perf_counter() - started
        generated.to_csv(generated_path, index=False)
        logging.info("Generated %d rows in %.1f seconds", len(generated), sample_seconds)

    logging.info("Evaluating seed=%d n_train=%d", seed, len(train))
    metrics = evaluate(generated, train, test, seed)
    generated[ALL_COLUMNS].to_csv(generated_path, index=False)
    payload = {
        "model": "sequential_econ",
        "seed": seed,
        "train_subset_rows": len(train),
        "num_samples": len(generated),
        "eval_sampling": "match_test_one_to_one",
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "stage_order": [stage.target for stage in STAGES],
        "evaluation": metrics,
    }
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    run_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REVISION_ROBUSTNESS if REVISION_ROBUSTNESS.exists() else Path("revision_exp/robustness_size"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 10000, 20000])
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--sample_batch_size", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_root / "sequential_econ_run.log", encoding="utf-8"),
        ],
    )

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    test = _sanitize_df_by_schema(pd.read_csv(args.test), FULL_SCHEMA)
    summary_path = args.output_root / "summary_all.csv"
    multi_path = args.output_root / "multi_seed_summary.csv"
    planned = [(size, seed) for size in args.sizes for seed in seeds]
    logging.info("Planned sequential_econ cells: %d -> %s", len(planned), args.output_root)

    for index, (size, seed) in enumerate(planned, start=1):
        logging.info("===== [%d/%d] size=%d seed=%d sequential_econ =====", index, len(planned), size, seed)
        subset = resolve_subset(size, seed, args.full_train, args.output_root)
        logging.info("Train subset: %s", subset)
        out_dir = args.output_root / _size_tag(size) / f"seed_{seed}" / "sequential_econ"
        payload = run_cell(
            subset_path=subset,
            test=test,
            out_dir=out_dir,
            seed=seed,
            sample_batch_size=args.sample_batch_size,
            force=args.force,
        )
        evaluation = payload.get("evaluation", payload)
        row = {
            "model": "sequential_econ",
            "seed": seed,
            "train_subset_rows": size,
            **_headline({"evaluation": evaluation}),
        }
        _upsert_summary(summary_path, row)
        _write_multiseed_summary(summary_path, multi_path)
        logging.info(
            "DONE size=%d seed=%d | joint_js=%s | marginal=%s | EMD=%s | LVR=%s | MNL=%s",
            size,
            seed,
            row.get("joint_js"),
            row.get("mean_marginal_jsd"),
            evaluation.get("mean_ordinal_emd"),
            row.get("logical_validity_rate"),
            row.get("mnl_behavioral_similarity"),
        )

    if summary_path.exists():
        seq = pd.read_csv(summary_path)
        seq = seq[seq["model"] == "sequential_econ"]
        print(seq.to_string(index=False))
    if multi_path.exists():
        multi = pd.read_csv(multi_path)
        multi = multi[multi["model"] == "sequential_econ"]
        print(multi.to_string(index=False))


if __name__ == "__main__":
    main()
