"""Assess real-data augmentation for mode-choice prediction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_privacy_assessment import MODELS, load_synthetic
from utils.mnl_mode_choice import (
    MODE_CHOICE_FEATURES,
    MODE_CHOICE_TARGET,
    _prepare_mode_choice_frame,
)


def take(frame: pd.DataFrame, n: int, seed: int, replace: bool | None = None) -> pd.DataFrame:
    if replace is None:
        replace = n > len(frame)
    return frame.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True)


def aligned_log_loss(model: Any, x: np.ndarray, y: np.ndarray) -> float:
    probabilities = model.predict_proba(x)
    return float(log_loss(y, probabilities, labels=model.classes_))


def evaluate_classifier(
    classifier: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
) -> dict[str, float | int | str]:
    x_train = train_frame[MODE_CHOICE_FEATURES].to_numpy(dtype=np.float32)
    y_train = train_frame[MODE_CHOICE_TARGET].to_numpy(dtype=np.int16)
    x_test = test_frame[MODE_CHOICE_FEATURES].to_numpy(dtype=np.float32)
    y_test = test_frame[MODE_CHOICE_TARGET].to_numpy(dtype=np.int16)

    if classifier == "mnl":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed),
        )
    elif classifier == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(classifier)

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return {
        "classifier": classifier,
        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "log_loss": aligned_log_loss(model, x_test, y_test),
        "n_train_mode_rows": int(len(train_frame)),
        "n_test_mode_rows": int(len(test_frame)),
    }


def condition(
    source: str,
    model: str,
    ratio: float,
    real_subset: pd.DataFrame,
    synthetic: pd.DataFrame | None,
    seed: int,
) -> pd.DataFrame:
    if source == "real_only":
        return real_subset
    if source == "duplicated_real":
        extra = take(real_subset, int(round(len(real_subset) * ratio)), seed + 11, replace=True)
        return pd.concat([real_subset, extra], ignore_index=True)
    if synthetic is None:
        raise ValueError("synthetic frame required")
    n_syn = int(round(len(real_subset) * ratio))
    syn_part = take(synthetic, max(n_syn, 1), seed + 22)
    if source == "synthetic_only":
        return syn_part
    if source == "real_plus_synthetic":
        return pd.concat([real_subset, syn_part], ignore_index=True)
    raise ValueError(source)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "model", "real_fraction", "augmentation_ratio", "classifier"]
    metrics = ["macro_f1", "balanced_accuracy", "log_loss", "n_train_mode_rows"]
    records = []
    for values, group in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--exp_root", type=Path, default=Path("exp"))
    parser.add_argument("--output_dir", type=Path, default=Path("revision_exp/augmentation"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--real_fractions", default="0.05,0.10,0.25")
    parser.add_argument("--augmentation_ratios", default="1,2")
    parser.add_argument("--classifiers", nargs="+", choices=["mnl", "hgb"], default=["mnl", "hgb"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(args.train)
    test = _prepare_mode_choice_frame(pd.read_csv(args.test))
    train = _prepare_mode_choice_frame(train_raw)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    fractions = [float(value) for value in args.real_fractions.split(",") if value.strip()]
    ratios = [float(value) for value in args.augmentation_ratios.split(",") if value.strip()]

    per_seed_path = args.output_dir / "per_seed.csv"
    rows: list[dict[str, Any]] = []
    if per_seed_path.exists():
        rows = pd.read_csv(per_seed_path).to_dict("records")
    completed = {
        (
            str(row["source"]), str(row["model"]), int(row["seed"]),
            float(row["real_fraction"]), float(row["augmentation_ratio"]),
            str(row["classifier"]),
        )
        for row in rows
    }

    for seed in seeds:
        real_subsets = {
            fraction: take(train, max(100, int(round(len(train) * fraction))), seed)
            for fraction in fractions
        }

        # Controls are shared by all generators.
        for fraction, real_subset in real_subsets.items():
            controls = [("real_only", "real", 0.0)]
            controls += [("duplicated_real", "real", ratio) for ratio in ratios]
            for source, model_name, ratio in controls:
                train_condition = condition(source, model_name, ratio, real_subset, None, seed)
                for classifier in args.classifiers:
                    key = (source, model_name, seed, fraction, ratio, classifier)
                    if key in completed:
                        continue
                    result = evaluate_classifier(classifier, train_condition, test, seed)
                    rows.append(
                        {
                            "source": source, "model": model_name, "seed": seed,
                            "real_fraction": fraction, "augmentation_ratio": ratio,
                            **result,
                        }
                    )

        max_needed = max(int(round(len(real_subsets[f]) * max(ratios))) for f in fractions)
        for model_name in args.models:
            logging.info("Loading augmentation synthetic model=%s seed=%d", model_name, seed)
            synthetic_raw = load_synthetic(
                model_name,
                seed,
                train_raw,
                max_needed,
                args.output_dir / "_synthetic_cache",
                args.exp_root,
            )
            synthetic = _prepare_mode_choice_frame(synthetic_raw)
            for fraction, real_subset in real_subsets.items():
                scenarios = [("real_plus_synthetic", ratio) for ratio in ratios]
                scenarios.append(("synthetic_only", 1.0))
                for source, ratio in scenarios:
                    train_condition = condition(
                        source, model_name, ratio, real_subset, synthetic, seed
                    )
                    for classifier in args.classifiers:
                        key = (source, model_name, seed, fraction, ratio, classifier)
                        if key in completed:
                            continue
                        result = evaluate_classifier(classifier, train_condition, test, seed)
                        rows.append(
                            {
                                "source": source, "model": model_name, "seed": seed,
                                "real_fraction": fraction, "augmentation_ratio": ratio,
                                **result,
                            }
                        )
                        frame = pd.DataFrame(rows)
                        frame.to_csv(per_seed_path, index=False)
                        summary = aggregate(frame)
                        summary.to_csv(args.output_dir / "summary.csv", index=False)
                        (args.output_dir / "summary.json").write_text(
                            json.dumps(summary.to_dict("records"), indent=2), encoding="utf-8"
                        )


if __name__ == "__main__":
    main()
