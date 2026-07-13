"""Generate and evaluate from a saved TVAE (SDV) checkpoint."""

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

import pandas as pd

import utils.test_utils
from scripts.baselines.run_tabular_baselines import (
    ALL_COLUMNS,
    FULL_SCHEMA,
    TRIP_COLUMNS,
    _sanitize_df_by_schema,
)


def _load_tvae_checkpoint(checkpoint: str):
    """Load SDV TVAE pickle, patching faker Provider attrs removed in newer faker."""
    import uuid

    import faker.providers.misc as misc

    for name in ("uuid1", "uuid3", "uuid4", "uuid5", "uuid6", "uuid7", "uuid8"):
        if not hasattr(misc.Provider, name):
            setattr(misc.Provider, name, lambda self, n=name: getattr(uuid, n)())

    from sdv.single_table import TVAESynthesizer

    return TVAESynthesizer.load(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from saved TVAE checkpoint and evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to TVAE_model.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for gene.csv and metrics.json")
    parser.add_argument("--num_samples", type=int, default=534445)
    parser.add_argument("--sample_batch_size", type=int, default=5000)
    parser.add_argument("--train_data", type=str, default="data/train_data.csv")
    parser.add_argument("--test_data", type=str, default="data/test_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip generation; evaluate existing TVAE_gene.csv in output_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = output_dir / "TVAE_gene.csv"
    metrics_json = output_dir / "TVAE_metrics.json"

    if not args.eval_only:
        logging.info("Loading TVAE checkpoint: %s", args.checkpoint)
        synthesizer = _load_tvae_checkpoint(args.checkpoint)
        logging.info("Generating %d samples (batch_size=%d)...", args.num_samples, args.sample_batch_size)
        generated_df = synthesizer.sample(num_rows=args.num_samples, batch_size=args.sample_batch_size)
        generated_df = _sanitize_df_by_schema(generated_df, FULL_SCHEMA)
        generated_df[ALL_COLUMNS].to_csv(gene_csv, index=False)
        logging.info("Saved generated CSV: %s", gene_csv)
    else:
        logging.info("Eval only: loading existing %s", gene_csv)
        generated_df = pd.read_csv(gene_csv)
        generated_df = _sanitize_df_by_schema(generated_df, FULL_SCHEMA)

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    truth_df = test_df.reset_index(drop=True)

    cond_info = [{"name": c} for c in ["relation", "sex", "age_code", "job_type"]]
    features_info = [
        {"name": "start_type", "type": "categorical", "num_classes": 5},
        {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "act_num", "type": "categorical", "num_classes": 9},
        {"name": "mode_num", "type": "categorical", "num_classes": 9},
        {"name": "end_type", "type": "categorical", "num_classes": 5},
        {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
        {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
    ]

    truth_trips = truth_df[TRIP_COLUMNS].values.tolist()
    generated_trips = generated_df[TRIP_COLUMNS].values.tolist()

    metrics = utils.test_utils.evaluate_generated_trips(
        truth_trips,
        generated_trips,
        features_info,
        cond_info=cond_info,
        generated_df=generated_df[ALL_COLUMNS],
        train_real_df=train_df,
        test_real_df=test_df,
        random_state=args.seed,
    )

    payload = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "eval_sampling": "unconditional_full_test_reference",
        "checkpoint": str(args.checkpoint),
        "evaluation": metrics,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics JSON: %s", metrics_json)
    logging.info(
        "TVAE | joint_js=%.6f | marginal_jsd=%.6f | LVR=%.4f | MNL sim=%s",
        metrics.get("joint_js", float("nan")),
        metrics.get("mean_marginal_jsd", float("nan")),
        metrics.get("logical_validity_rate", float("nan")),
        metrics.get("mnl_behavioral_similarity"),
    )


if __name__ == "__main__":
    main()
