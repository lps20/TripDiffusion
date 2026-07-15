"""Generate and evaluate CTGAN / DDPM-TF / TabDDPM from saved checkpoints."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import pandas as pd
import torch

import utils.test_utils
from scripts.baselines.run_tabular_baselines import (
    ALL_COLUMNS,
    FULL_SCHEMA,
    TRIP_COLUMNS,
    _sample_model,
    _sanitize_df_by_schema,
)
from scripts.baselines.tab_ddpm_baseline import sample_tabddpm_from_checkpoint
from utils.multi_seed import set_global_seed

MODEL_TAGS = {
    "ctgan": "CTGAN",
    "ddpm_tf": "DDPM_TF",
    "tabddpm": "TABDDPM",
}

CHECKPOINT_NAMES = {
    "ctgan": "CTGAN_model.pkl",
    "ddpm_tf": "DDPM_TF_model.pth",
    "tabddpm": "TABDDPM_model.pth",
}

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

DDPM_TRIP_COLS = [f["name"] for f in FEATURES_INFO]
DDPM_COND_COLS = [c["name"] for c in COND_INFO]


def _load_ctgan(checkpoint: str):
    module = importlib.import_module("ctgan")
    ctgan_class = getattr(module, "CTGAN")
    return ctgan_class.load(checkpoint)


def _generate_ctgan(checkpoint: str, num_samples: int, sample_batch_size: int) -> pd.DataFrame:
    model = _load_ctgan(checkpoint)
    chunks = []
    remaining = num_samples
    while remaining > 0:
        bsz = min(sample_batch_size, remaining)
        chunks.append(_sample_model(model, bsz, ALL_COLUMNS))
        remaining -= bsz
    generated = pd.concat(chunks, ignore_index=True)
    return _sanitize_df_by_schema(generated, FULL_SCHEMA)


def _generate_tabddpm(checkpoint: str, num_samples: int, sample_batch_size: int) -> pd.DataFrame:
    generated = sample_tabddpm_from_checkpoint(
        checkpoint_path=checkpoint,
        schema=FULL_SCHEMA,
        n_samples=num_samples,
        batch_size=sample_batch_size,
    )
    return _sanitize_df_by_schema(generated, FULL_SCHEMA)


def _load_ddpm_tf(checkpoint: str, device: torch.device):
    from model.Transformer_Net import TripDiffusionModel as DDPMTransformerModel

    ckpt = torch.load(checkpoint, map_location=device)
    if ckpt.get("model_type") != "ddpm_transformer":
        raise ValueError(f"Expected ddpm_transformer checkpoint, got {ckpt.get('model_type')!r}")

    model = DDPMTransformerModel(
        ckpt["features_info"],
        ckpt["cond_info"],
        int(ckpt["T"]),
        joint_pairs=[],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _generate_ddpm_tf(
    checkpoint: str,
    test_df: pd.DataFrame,
    num_samples: int,
    device: torch.device,
) -> pd.DataFrame:
    import utils.train_utils

    model = _load_ddpm_tf(checkpoint, device)
    match_test = num_samples <= 0
    generated_samples, _ = utils.train_utils.sample_trip(
        model=model,
        df=test_df,
        num_samples=num_samples,
        device=device,
        match_test_one_to_one=match_test,
    )

    rows = []
    for sample in generated_samples:
        row = {}
        for i, col in enumerate(DDPM_COND_COLS):
            row[col] = int(sample["condition"][i])
        for i, col in enumerate(DDPM_TRIP_COLS):
            row[col] = int(sample["trip"][i])
        rows.append(row)
    return _sanitize_df_by_schema(pd.DataFrame(rows), FULL_SCHEMA)


def _evaluate(
    model_name: str,
    generated_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    seed: int,
    eval_sampling: str,
    truth_trips: List[List[int]],
    generated_trips: List[List[int]],
    generated_samples=None,
) -> Dict[str, Any]:
    cond_info = [{"name": c["name"]} for c in COND_INFO]
    metrics = utils.test_utils.evaluate_generated_trips(
        truth_trips,
        generated_trips,
        FEATURES_INFO,
        generated_samples=generated_samples,
        cond_info=cond_info,
        generated_df=generated_df[ALL_COLUMNS],
        train_real_df=train_df,
        test_real_df=test_df,
        random_state=seed,
    )
    return {
        "seed": seed,
        "model": model_name,
        "num_samples": len(generated_df),
        "eval_sampling": eval_sampling,
        "evaluation": metrics,
    }


def generate_and_evaluate(
    model_name: str,
    checkpoint: str,
    output_dir: Path,
    train_data: str,
    test_data: str,
    num_samples: int,
    sample_batch_size: int,
    seed: int,
    eval_only: bool = False,
) -> Dict[str, Any]:
    model_tag = MODEL_TAGS[model_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = output_dir / f"{model_tag}_gene.csv"
    metrics_json = output_dir / f"{model_tag}_metrics.json"

    set_global_seed(seed)
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    full_test_n = len(test_df)

    if model_name in {"ctgan", "tabddpm"}:
        target_n = full_test_n if num_samples <= 0 else num_samples
        eval_sampling = "unconditional_full_test_reference"
    else:
        target_n = full_test_n if num_samples <= 0 else num_samples
        eval_sampling = "match_test_one_to_one"

    if not eval_only:
        logging.info(
            "Generating %s samples for %s (eval_sampling=%s)...",
            target_n,
            model_name,
            eval_sampling,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "ctgan":
            generated_df = _generate_ctgan(checkpoint, target_n, sample_batch_size)
        elif model_name == "tabddpm":
            generated_df = _generate_tabddpm(checkpoint, target_n, sample_batch_size)
        elif model_name == "ddpm_tf":
            generated_df = _generate_ddpm_tf(
                checkpoint,
                test_df=test_df,
                num_samples=0 if num_samples <= 0 else num_samples,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        generated_df[ALL_COLUMNS].to_csv(gene_csv, index=False)
        logging.info("Saved generated CSV: %s", gene_csv)
    else:
        logging.info("Eval only: loading existing %s", gene_csv)
        generated_df = pd.read_csv(gene_csv)
        generated_df = _sanitize_df_by_schema(generated_df, FULL_SCHEMA)

    # Feature-index order MUST match FEATURES_INFO (HCD revision order).
    # TRIP_COLUMNS from tabular baselines uses a different column order and would
    # mis-assign per-feature JSD / mean_marginal_jsd if used here.
    eval_trip_cols = [f["name"] for f in FEATURES_INFO]
    truth_df = test_df.reset_index(drop=True)
    truth_trips = truth_df[eval_trip_cols].astype(int).values.tolist()
    generated_trips = generated_df[eval_trip_cols].astype(int).values.tolist()

    generated_samples = None
    if model_name == "ddpm_tf":
        generated_samples = [
            {
                "condition": [int(row[c]) for c in DDPM_COND_COLS],
                "trip": [int(row[c]) for c in DDPM_TRIP_COLS],
            }
            for _, row in generated_df.iterrows()
        ]

    payload = _evaluate(
        model_name=model_name,
        generated_df=generated_df,
        test_df=test_df,
        train_df=train_df,
        seed=seed,
        eval_sampling=eval_sampling,
        truth_trips=truth_trips,
        generated_trips=generated_trips,
        generated_samples=generated_samples,
    )
    payload["checkpoint"] = str(checkpoint)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics JSON: %s", metrics_json)

    metrics = payload["evaluation"]
    logging.info(
        "%s | joint_js=%.6f | marginal_jsd=%.6f | LVR=%.4f | MNL sim=%s",
        model_tag,
        metrics.get("joint_js", float("nan")),
        metrics.get("mean_marginal_jsd", float("nan")),
        metrics.get("logical_validity_rate", float("nan")),
        metrics.get("mnl_behavioral_similarity"),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate a saved tabular baseline checkpoint.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODEL_TAGS.keys()),
        help="Baseline model type.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for *_gene.csv and *_metrics.json.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of generated rows. 0 uses full test size (534445), matching HCD revision eval.",
    )
    parser.add_argument("--sample_batch_size", type=int, default=5000)
    parser.add_argument("--train_data", type=str, default="data/train_data.csv")
    parser.add_argument("--test_data", type=str, default="data/test_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true", help="Skip generation; evaluate existing *_gene.csv.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generate_and_evaluate(
        model_name=args.model,
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir),
        train_data=args.train_data,
        test_data=args.test_data,
        num_samples=args.num_samples,
        sample_batch_size=args.sample_batch_size,
        seed=args.seed,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
