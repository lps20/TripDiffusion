"""Generate and evaluate from a saved DATGAN model directory."""

import argparse
import json
import logging
import os
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
    _build_datgan_metadata,
    _filter_kwargs,
    _import_datgan_class,
    _resolve_datgan_dag,
    _sample_model,
    _sanitize_df_by_schema,
)


def _load_datgan_from_dir(model_dir: str, train_data: str, dag_mode: str = "cascade", continuous_time: bool = True):
    datgan_class = _import_datgan_class()
    from datgan.synthesizer.synthesizer import Synthesizer
    from datgan.utils.dag import get_order_variables, transform_dag, verify_dag
    output = os.path.join(model_dir, "")
    init_kwargs = _filter_kwargs(
        datgan_class,
        {
            "name": "datgan_eval",
            "output": output,
            "restore_session": True,
            "save_checkpoints": False,
            "verbose": 0,
        },
    )
    model = datgan_class(**init_kwargs)
    train_df = pd.read_csv(train_data, nrows=1000)[ALL_COLUMNS]
    metadata = _build_datgan_metadata(continuous_time=continuous_time)
    dag = _resolve_datgan_dag(dag_mode)
    encoded_path = os.path.join(model_dir, "encoded_data")

    model.preprocess(train_df, metadata, encoded_path)
    model.metadata = model.encoded_data.metadata
    model.dag = transform_dag(dag, model.conditional_inputs)
    verify_dag(train_df, model.dag)
    model.var_order, model.n_sources = get_order_variables(model.dag)
    model._DATGAN__default_parameter_values(train_df)

    model.synthesizer = Synthesizer(
        model.output,
        model.metadata,
        model.dag,
        model.batch_size,
        model.z_dim,
        model.noise,
        model.learning_rate,
        model.g_period,
        model.l2_reg,
        model.num_gen_rnn,
        model.num_gen_hidden,
        model.num_dis_layers,
        model.num_dis_hidden,
        model.label_smoothing,
        model.loss_function,
        model.var_order,
        model.n_sources,
        model.conditional_inputs,
        model.save_checkpoints,
        model.restore_session,
        model.verbose,
    )
    model.synthesizer.initialize()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from saved DATGAN checkpoint dir and evaluate.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to DATGAN_model directory (contains checkpoints/).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for gene.csv and metrics.json")
    parser.add_argument("--num_samples", type=int, default=534445)
    parser.add_argument("--sample_batch_size", type=int, default=5000)
    parser.add_argument("--train_data", type=str, default="data/train_data.csv")
    parser.add_argument("--test_data", type=str, default="data/test_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip generation; evaluate existing DATGAN_gene.csv in output_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = output_dir / "DATGAN_gene.csv"
    metrics_json = output_dir / "DATGAN_metrics.json"

    if not args.eval_only:
        logging.info("Loading DATGAN from: %s", args.checkpoint_dir)
        model = _load_datgan_from_dir(args.checkpoint_dir, args.train_data)
        chunks = []
        remaining = args.num_samples
        while remaining > 0:
            bsz = min(args.sample_batch_size, remaining)
            logging.info("Generating batch of %d (%d remaining)...", bsz, remaining)
            chunk = _sample_model(model, bsz, ALL_COLUMNS)
            chunks.append(chunk)
            remaining -= bsz
        generated_df = pd.concat(chunks, ignore_index=True)
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

    # Keep trip list order identical to features_info (HCD revision order).
    eval_trip_cols = [f["name"] for f in features_info]
    truth_trips = truth_df[eval_trip_cols].astype(int).values.tolist()
    generated_trips = generated_df[eval_trip_cols].astype(int).values.tolist()

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
        "checkpoint_dir": str(args.checkpoint_dir),
        "evaluation": metrics,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics JSON: %s", metrics_json)
    logging.info(
        "DATGAN | joint_js=%.6f | marginal_jsd=%.6f | LVR=%.4f | MNL sim=%s",
        metrics.get("joint_js", float("nan")),
        metrics.get("mean_marginal_jsd", float("nan")),
        metrics.get("logical_validity_rate", float("nan")),
        metrics.get("mnl_behavioral_similarity"),
    )


if __name__ == "__main__":
    main()
