"""Regenerate and re-evaluate full-test Embedding-DDPM samples with ordinal EMD."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.test_utils
from model.EmbeddingDDPM_Net import EmbeddingDDPM, sample_embedding_ddpm
from scripts.baselines.run_tabular_baselines import ALL_COLUMNS, FULL_SCHEMA, _sanitize_df_by_schema


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EmbeddingDDPM, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EmbeddingDDPM(
        features_info=checkpoint["features_info"],
        cond_info=checkpoint["cond_info"],
        T=int(checkpoint["T"]),
        d_model=int(checkpoint["d_model"]),
        backbone=str(checkpoint["backbone"]),
        nhead=int(checkpoint.get("nhead", 8)),
        num_layers=int(checkpoint.get("num_layers", 4)),
        mlp_hidden=checkpoint.get("mlp_hidden"),
        dropout=float(checkpoint.get("dropout", 0.1)),
        beta_schedule=str(checkpoint.get("beta_schedule", "cosine")),
        feature_embedding_mode=str(checkpoint.get("feature_embedding_mode", "learned")),
        fixed_codebook_type=str(checkpoint.get("fixed_codebook_type", "random")),
        decode_metric=str(checkpoint.get("decode_metric", "dot")),
        x0_ce_weight=float(checkpoint.get("x0_ce_weight", 0.0)),
        sample_method=str(checkpoint.get("sample_method", "ddpm")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(
    generated: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features_info: List[Dict[str, Any]],
    cond_info: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    feature_columns = [feature["name"] for feature in features_info]
    return utils.test_utils.evaluate_generated_trips(
        test[feature_columns].astype(int).values.tolist(),
        generated[feature_columns].astype(int).values.tolist(),
        features_info,
        cond_info=[{"name": condition["name"]} for condition in cond_info],
        generated_df=generated[ALL_COLUMNS],
        train_real_df=train,
        test_real_df=test,
        random_state=seed,
    )


def summary_frame(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "joint_js",
        "mean_marginal_jsd",
        "mean_ordinal_emd",
        "logical_validity_rate",
        "mnl_behavioral_similarity",
        "mnl_test_logloss_ratio",
        "generation_seconds",
    ]
    result: Dict[str, Any] = {"model": "DDPM_TF", "n_seeds": len(rows)}
    for metric in metrics:
        values = pd.to_numeric(rows[metric], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
        result[metric] = (
            f"{mean:.4f} ± {std:.4f}" if mean is not None else None
        )
    return pd.DataFrame([result])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=Path("exp/embedding_ddpm_full/T10_cond"),
    )
    parser.add_argument("--train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.experiment_dir / "regenerate_eval.log", encoding="utf-8"),
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = _sanitize_df_by_schema(pd.read_csv(args.train), FULL_SCHEMA)
    test = _sanitize_df_by_schema(pd.read_csv(args.test), FULL_SCHEMA)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        seed_dir = args.experiment_dir / f"seed_{seed}"
        checkpoint_path = seed_dir / "DDPM_TF_model.pth"
        generated_path = seed_dir / "DDPM_TF_gene.csv"
        metrics_path = seed_dir / "DDPM_TF_metrics.json"
        logging.info("Loading seed=%d checkpoint: %s", seed, checkpoint_path)
        model, checkpoint = load_model(checkpoint_path, device)
        feature_columns = [feature["name"] for feature in checkpoint["features_info"]]
        condition_columns = [condition["name"] for condition in checkpoint["cond_info"]]

        started = time.perf_counter()
        generated = sample_embedding_ddpm(
            model=model,
            test_df=test,
            feat_cols=feature_columns,
            cond_cols=condition_columns,
            n_samples=len(test),
            device=device,
            batch_size=args.batch_size,
            match_test_one_to_one=True,
        )
        generation_seconds = time.perf_counter() - started
        generated = _sanitize_df_by_schema(generated, FULL_SCHEMA)
        generated[ALL_COLUMNS].to_csv(generated_path, index=False)
        logging.info(
            "Generated seed=%d rows=%d in %.1f seconds",
            seed,
            len(generated),
            generation_seconds,
        )

        metrics = evaluate(
            generated,
            train,
            test,
            checkpoint["features_info"],
            checkpoint["cond_info"],
            seed,
        )
        payload = {
            "model": "DDPM_TF",
            "seed": seed,
            "T": int(checkpoint["T"]),
            "backbone": str(checkpoint["backbone"]),
            "num_samples": len(generated),
            "eval_sampling": "match_test_one_to_one",
            "generation_seconds": generation_seconds,
            "evaluation": metrics,
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = {
            "model": "DDPM_TF",
            "seed": seed,
            "joint_js": metrics.get("joint_js"),
            "mean_marginal_jsd": metrics.get("mean_marginal_jsd"),
            "mean_ordinal_emd": metrics.get("mean_ordinal_emd"),
            "logical_validity_rate": metrics.get("logical_validity_rate"),
            "mnl_behavioral_similarity": metrics.get("mnl_behavioral_similarity"),
            "mnl_test_logloss_ratio": metrics.get("mnl_test_logloss_ratio"),
            "generation_seconds": generation_seconds,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.experiment_dir / "baseline_metrics_per_seed.csv", index=False)
        logging.info(
            "Seed %d done | joint=%.6f marginal=%.6f EMD=%.4f LVR=%.4f MNL=%s",
            seed,
            row["joint_js"],
            row["mean_marginal_jsd"],
            row["mean_ordinal_emd"],
            row["logical_validity_rate"],
            row["mnl_behavioral_similarity"],
        )

    per_seed = pd.DataFrame(rows)
    summary = summary_frame(per_seed)
    per_seed.to_csv(args.experiment_dir / "baseline_metrics_per_seed.csv", index=False)
    per_seed.to_csv(args.experiment_dir / "baseline_multiseed_per_seed.csv", index=False)
    summary.to_csv(args.experiment_dir / "baseline_metrics.csv", index=False)
    summary.to_csv(args.experiment_dir / "baseline_metrics_summary.csv", index=False)
    summary.to_csv(args.experiment_dir / "baseline_multiseed_summary.csv", index=False)
    summary_record = {
        key: (None if pd.isna(value) else value)
        for key, value in summary.iloc[0].to_dict().items()
    }
    (args.experiment_dir / "multi_seed_summary.json").write_text(
        json.dumps(
            {
                "model": "DDPM_TF",
                "n_seeds": len(per_seed),
                "seeds": seeds,
                "metrics": summary_record,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
