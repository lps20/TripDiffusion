"""Re-evaluate legacy generated CSVs with the new MNL TSTR behavioral metrics."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse
import json
from pathlib import Path

import pandas as pd

import utils.test_utils as tu

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
COND_INFO = [
    {"name": "relation", "num_classes": 5},
    {"name": "sex", "num_classes": 2},
    {"name": "age_code", "num_classes": 13},
    {"name": "job_type", "num_classes": 9},
]
TRIP_COLS = [f["name"] for f in FEATURES_INFO]
ALL_COLS = [c["name"] for c in COND_INFO] + TRIP_COLS

DEFAULT_MODELS = [
    ("HCD_V2", "exp/hcd_v2/generated_samples.csv"),
    ("DDPM_TF", "exp/baseline/DDPM_TF_gene.csv"),
    ("VAE", "exp/baseline/VAE_gene.csv"),
    ("CTGAN", "exp/baseline/CTGAN_gene.csv"),
    ("DATGAN", "exp/baseline/DATGAN_gene.csv"),
]


def _load_generated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[ALL_COLS]


def evaluate_one(
    model_name: str,
    generated_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict:
    generated_df = _load_generated(generated_path)
    n = len(generated_df)
    truth_df = test_df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    truth_trips = truth_df[TRIP_COLS].values.tolist()
    generated_trips = generated_df[TRIP_COLS].values.tolist()

    metrics = tu.evaluate_generated_trips(
        truth_trips,
        generated_trips,
        FEATURES_INFO,
        cond_info=COND_INFO,
        generated_df=generated_df,
        train_real_df=train_df,
        test_real_df=test_df,
        random_state=seed,
    )

    row = {
        "model": model_name,
        "generated_csv": str(generated_path).replace("\\", "/"),
        "n_generated": n,
        "mnl_status": metrics.get("mnl_status"),
        "mnl_behavioral_similarity": metrics.get("mnl_behavioral_similarity"),
        "mnl_coef_cosine_similarity": metrics.get("mnl_coef_cosine_similarity"),
        "mnl_coef_rmse": metrics.get("mnl_coef_rmse"),
        "mnl_ame_cosine_similarity": metrics.get("mnl_ame_cosine_similarity"),
        "mnl_ame_rmse": metrics.get("mnl_ame_rmse"),
        "mnl_elasticity_cosine_similarity": metrics.get("mnl_elasticity_cosine_similarity"),
        "mnl_elasticity_rmse": metrics.get("mnl_elasticity_rmse"),
        "mnl_tstr_test_logloss": metrics.get("mnl_tstr_test_logloss"),
        "mnl_trtr_test_logloss": metrics.get("mnl_trtr_test_logloss"),
        "mnl_test_logloss_ratio": metrics.get("mnl_test_logloss_ratio"),
        "joint_js": metrics.get("joint_js"),
        "mean_marginal_jsd": metrics.get("mean_marginal_jsd"),
        "logical_validity_rate": metrics.get("logical_validity_rate"),
        "legacy_tstr_trtr_f1_ratio": metrics.get("tstr_trtr_f1_ratio"),
    }
    return row, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy generated data with new MNL TSTR metrics.")
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--output_csv", type=str, default="exp/mnl_tstr_legacy_comparison.csv")
    parser.add_argument("--output_json_dir", type=str, default="exp/mnl_tstr_legacy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(".").resolve()
    train_df = pd.read_csv(root / args.traindata)
    test_df = pd.read_csv(root / args.testdata)

    rows = []
    json_dir = root / args.output_json_dir
    json_dir.mkdir(parents=True, exist_ok=True)

    for model_name, rel_path in DEFAULT_MODELS:
        path = root / rel_path
        if not path.exists():
            print(f"SKIP {model_name}: missing {path}")
            continue
        print(f"Evaluating {model_name} ...")
        row, metrics = evaluate_one(model_name, path, train_df, test_df, args.seed)
        rows.append(row)
        with open(json_dir / f"{model_name}_mnl_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    out_df = pd.DataFrame(rows).sort_values("mnl_behavioral_similarity", ascending=False)
    out_path = root / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\n=== MNL TSTR comparison (legacy generated data) ===")
    display_cols = [
        "model",
        "mnl_behavioral_similarity",
        "mnl_ame_cosine_similarity",
        "mnl_elasticity_cosine_similarity",
        "mnl_test_logloss_ratio",
        "joint_js",
        "logical_validity_rate",
    ]
    print(out_df[display_cols].to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
