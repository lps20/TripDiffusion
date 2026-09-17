"""Fit once on full data, then evaluate five stochastic generation seeds."""

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
from scripts.baselines.sequential_econometric_baseline import SequentialEconometricGenerator
from scripts.experiments.run_sequential_econometric_20k import (
    COND_COLUMNS,
    FEATURES_INFO,
    TRIP_COLUMNS,
)
from utils.multi_seed import extract_headline_metrics


def evaluate(
    generated: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> Dict[str, Any]:
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


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "joint_js",
        "mean_marginal_jsd",
        "mean_ordinal_emd",
        "logical_validity_rate",
        "mnl_behavioral_similarity",
        "mnl_test_logloss_ratio",
        "fit_seconds",
        "sample_seconds",
    ]
    row: Dict[str, Any] = {"model": "sequential_econ", "n_seeds": len(per_seed)}
    for metric in metrics:
        values = pd.to_numeric(per_seed[metric], errors="coerce").dropna()
        row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("revision_exp/baselines/sequential_econ"),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--sample_batch_size", type=int, default=10000)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_root / "run.log", encoding="utf-8"),
        ],
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    train = _sanitize_df_by_schema(pd.read_csv(args.train), FULL_SCHEMA)
    test = _sanitize_df_by_schema(pd.read_csv(args.test), FULL_SCHEMA)

    shared_dir = args.output_root / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    model_path = shared_dir / "SEQUENTIAL_ECON_model.joblib"
    timing_path = shared_dir / "fit_timing.json"
    if model_path.exists():
        logging.info("Loading shared fitted model: %s", model_path)
        model = SequentialEconometricGenerator.load(model_path)
        timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
        fit_seconds = timing.get("fit_seconds")
    else:
        logging.info("Fitting sequential econometric generator on %d full-data rows", len(train))
        started = time.perf_counter()
        model = SequentialEconometricGenerator(random_state=seeds[0]).fit(train)
        fit_seconds = time.perf_counter() - started
        model.save(model_path)
        timing_path.write_text(
            json.dumps({"fit_seconds": fit_seconds, "n_train": len(train)}, indent=2),
            encoding="utf-8",
        )
        logging.info("Full-data model fitted in %.1f seconds", fit_seconds)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        generated_path = seed_dir / "SEQUENTIAL_ECON_gene.csv"
        metrics_path = seed_dir / "SEQUENTIAL_ECON_metrics.json"
        run_path = seed_dir / "run_metrics.json"

        if generated_path.exists():
            generated = pd.read_csv(generated_path)
            existing = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
            sample_seconds = existing.get("sample_seconds")
        else:
            logging.info("Generating seed=%d", seed)
            model.random_state = seed
            started = time.perf_counter()
            generated = model.sample(test, batch_size=args.sample_batch_size)
            sample_seconds = time.perf_counter() - started
            generated.to_csv(generated_path, index=False)
            logging.info("Generated seed=%d in %.1f seconds", seed, sample_seconds)

        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            logging.info("Evaluating seed=%d", seed)
            metrics = evaluate(generated, train, test, seed)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        payload = {
            "model": "sequential_econ",
            "seed": seed,
            "train_rows": len(train),
            "num_samples": len(generated),
            "eval_sampling": "match_test_one_to_one",
            "fit_seconds": fit_seconds,
            "sample_seconds": sample_seconds,
            "evaluation": metrics,
        }
        run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        headline = extract_headline_metrics(metrics)
        rows.append(
            {
                "model": "sequential_econ",
                "seed": seed,
                **headline,
                "mean_ordinal_emd": metrics.get("mean_ordinal_emd"),
                "mnl_test_logloss_ratio": metrics.get("mnl_test_logloss_ratio"),
                "fit_seconds": fit_seconds,
                "sample_seconds": sample_seconds,
            }
        )
        per_seed = pd.DataFrame(rows)
        per_seed.to_csv(args.output_root / "per_seed.csv", index=False)
        summary = summarize(per_seed)
        summary.to_csv(args.output_root / "summary.csv", index=False)
        (args.output_root / "summary.json").write_text(
            json.dumps(summary.replace({np.nan: None}).to_dict("records"), indent=2),
            encoding="utf-8",
        )
        logging.info(
            "Seed %d done | joint_js=%.6f marginal=%.6f EMD=%.4f LVR=%.4f MNL=%.4f",
            seed,
            headline["joint_js"],
            headline["mean_marginal_jsd"],
            metrics.get("mean_ordinal_emd"),
            headline["logical_validity_rate"],
            headline["mnl_behavioral_similarity"],
        )


if __name__ == "__main__":
    main()
