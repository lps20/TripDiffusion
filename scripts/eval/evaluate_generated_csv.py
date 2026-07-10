import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

import utils.test_utils


def _default_features_info() -> List[Dict[str, Any]]:
    return [
        {"name": "start_type", "type": "categorical", "num_classes": 5},
        {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "act_num", "type": "categorical", "num_classes": 9},
        {"name": "mode_num", "type": "categorical", "num_classes": 9},
        {"name": "end_type", "type": "categorical", "num_classes": 5},
        {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
        {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
    ]


def _default_cond_info() -> List[Dict[str, Any]]:
    return [
        {"name": "relation", "num_classes": 5},
        {"name": "sex", "num_classes": 2},
        {"name": "age_code", "num_classes": 13},
        {"name": "job_type", "num_classes": 9},
    ]


def _flatten_metrics(model_name: str, metrics: Dict[str, Any], generated_csv: str) -> Dict[str, Any]:
    row = utils.test_utils.flatten_evaluation_metrics(
        model_name=model_name,
        metrics=metrics,
        extra_fields={"generated_csv": generated_csv},
        include_formatted=True,
    )
    return row


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generated_df = pd.read_csv(args.generated_csv)
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    features_info = _default_features_info()
    cond_info = _default_cond_info()
    trip_cols = [f["name"] for f in features_info]
    cond_cols = [c["name"] for c in cond_info]

    missing_trip = [c for c in trip_cols if c not in generated_df.columns]
    if missing_trip:
        raise ValueError(f"Generated CSV missing required trip columns: {missing_trip}")

    if args.truth_source == "sample":
        n = args.num_truth_samples if args.num_truth_samples is not None else len(generated_df)
        truth_df = test_df.sample(n=n, replace=True, random_state=args.seed).reset_index(drop=True)
    else:
        truth_df = test_df.copy()

    truth_trips = truth_df[trip_cols].values.tolist()
    generated_trips = generated_df[trip_cols].values.tolist()

    has_all_cond = all(c in generated_df.columns for c in cond_cols)
    generated_eval_df = generated_df[cond_cols + trip_cols] if has_all_cond else None
    if not has_all_cond:
        logging.warning(
            "Condition columns %s not fully present in generated CSV. "
            "LVR/TSTR may be None because demographics or predictors are missing.",
            cond_cols,
        )

    metrics = utils.test_utils.evaluate_generated_trips(
        truth_trips,
        generated_trips,
        features_info,
        cond_info=cond_info,
        generated_df=generated_eval_df,
        train_real_df=train_df,
        test_real_df=test_df,
        random_state=args.seed,
    )

    output_json = args.output_json
    if output_json is None:
        stem = os.path.splitext(os.path.basename(args.generated_csv))[0]
        output_json = os.path.join(os.path.dirname(args.generated_csv), f"{stem}_metrics.json")
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics JSON: %s", output_json)

    if args.summary_csv is not None:
        os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
        model_name = args.model_name or os.path.splitext(os.path.basename(args.generated_csv))[0]
        row = _flatten_metrics(model_name=model_name, metrics=metrics, generated_csv=args.generated_csv)
        new_df = pd.DataFrame([row])
        if os.path.exists(args.summary_csv):
            old_df = pd.read_csv(args.summary_csv)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged.to_csv(args.summary_csv, index=False)
        else:
            new_df.to_csv(args.summary_csv, index=False)
        logging.info("Updated summary CSV: %s", args.summary_csv)

    logging.info(
        "Done. joint_js=%s, mean_jsd_norm=%s, mean_ordinal_emd=%s, LVR=%s, MNL sim=%s, logloss ratio=%s",
        metrics.get("joint_js"),
        metrics.get("mean_single_feature_jsd_normalized"),
        metrics.get("mean_ordinal_emd"),
        metrics.get("logical_validity_rate"),
        metrics.get("mnl_behavioral_similarity"),
        metrics.get("mnl_test_logloss_ratio"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an existing generated CSV without retraining.")
    parser.add_argument("--generated_csv", type=str, required=True, help="Path to *_gene.csv or generated_samples.csv")
    parser.add_argument("--train_data", type=str, default="data/train_data.csv", help="Path to real train CSV")
    parser.add_argument("--test_data", type=str, default="data/test_data.csv", help="Path to real test CSV")
    parser.add_argument(
        "--truth_source",
        type=str,
        default="sample",
        choices=["sample", "full_test"],
        help="How to construct truth trips for JSD: sample test rows or use full test set",
    )
    parser.add_argument(
        "--num_truth_samples",
        type=int,
        default=None,
        help="Only used when truth_source=sample. Default: same as generated rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling/TSTR")
    parser.add_argument("--output_json", type=str, default=None, help="Where to write metrics JSON")
    parser.add_argument("--summary_csv", type=str, default=None, help="Optional CSV to append one summary row")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model name used in summary CSV")
    args = parser.parse_args()
    main(args)
