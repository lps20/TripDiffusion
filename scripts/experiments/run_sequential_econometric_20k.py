"""Train and evaluate the sequential MNL/ordered-logit generator on seed42/20k."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.test_utils
from scripts.baselines.run_tabular_baselines import (
    ALL_COLUMNS,
    _sanitize_df_by_schema,
    FULL_SCHEMA,
)
from scripts.baselines.sequential_econometric_baseline import STAGES, SequentialEconometricGenerator

FEATURES_INFO = [
    {"name": "start_type", "type": "categorical", "num_classes": 5},
    {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
    {"name": "act_num", "type": "categorical", "num_classes": 9},
    {"name": "mode_num", "type": "categorical", "num_classes": 9},
    {"name": "end_type", "type": "categorical", "num_classes": 5},
    {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
    {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
    {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
]
TRIP_COLUMNS = [feature["name"] for feature in FEATURES_INFO]
COND_COLUMNS = ["relation", "sex", "age_code", "job_type"]


def ensure_subset(full_train: Path, subset: Path, rows: int, seed: int) -> Path:
    if subset.exists():
        return subset
    subset.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(full_train)
    frame.sample(n=min(rows, len(frame)), random_state=seed).reset_index(drop=True).to_csv(
        subset, index=False
    )
    return subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument(
        "--subset",
        type=Path,
        default=Path("exp/robustness_20k/seed_42/train_subset_20000.csv"),
    )
    parser.add_argument("--subset_rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exp/sequential_econometric_20k/seed_42"),
    )
    parser.add_argument("--sample_batch_size", type=int, default=10000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
        ],
    )
    subset_path = ensure_subset(args.full_train, args.subset, args.subset_rows, args.seed)
    train = _sanitize_df_by_schema(pd.read_csv(subset_path), FULL_SCHEMA)
    test = _sanitize_df_by_schema(pd.read_csv(args.test), FULL_SCHEMA)

    model_path = args.output_dir / "SEQUENTIAL_ECON_model.joblib"
    generated_path = args.output_dir / "SEQUENTIAL_ECON_gene.csv"
    run_metrics_path = args.output_dir / "run_metrics.json"
    if generated_path.exists():
        logging.info("Reusing generated samples: %s", generated_path)
        generated = pd.read_csv(generated_path)
        existing_run = (
            json.loads(run_metrics_path.read_text(encoding="utf-8"))
            if run_metrics_path.exists()
            else {}
        )
        fit_seconds = existing_run.get("fit_seconds")
        sample_seconds = existing_run.get("sample_seconds")
    else:
        started = time.perf_counter()
        model = SequentialEconometricGenerator(random_state=args.seed).fit(train)
        fit_seconds = time.perf_counter() - started
        model.save(model_path)
        logging.info("Model fitted and saved in %.1f seconds", fit_seconds)

        started = time.perf_counter()
        generated = model.sample(test, batch_size=args.sample_batch_size)
        sample_seconds = time.perf_counter() - started
        generated.to_csv(generated_path, index=False)
        logging.info("Generated %d rows in %.1f seconds", len(generated), sample_seconds)

    metrics = utils.test_utils.evaluate_generated_trips(
        test[TRIP_COLUMNS].astype(int).values.tolist(),
        generated[TRIP_COLUMNS].astype(int).values.tolist(),
        FEATURES_INFO,
        cond_info=[{"name": column} for column in COND_COLUMNS],
        generated_df=generated[ALL_COLUMNS],
        train_real_df=train,
        test_real_df=test,
        random_state=args.seed,
    )
    generated[ALL_COLUMNS].to_csv(generated_path, index=False)
    flat = utils.test_utils.flatten_evaluation_metrics(
        model_name="SEQUENTIAL_ECON",
        metrics=metrics,
        extra_fields={"seed": args.seed},
        include_formatted=True,
    )
    metrics_path = args.output_dir / "SEQUENTIAL_ECON_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_payload = {
        "model": "sequential_econ",
        "seed": args.seed,
        "train_subset_rows": len(train),
        "num_samples": len(generated),
        "eval_sampling": "match_test_one_to_one",
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "stage_order": [stage.target for stage in STAGES],
        "evaluation": metrics,
    }
    run_metrics_path.write_text(
        json.dumps(run_payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame([flat]).to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    print(pd.DataFrame([flat]).to_string(index=False))


if __name__ == "__main__":
    main()
