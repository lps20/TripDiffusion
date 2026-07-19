"""Load trained conditional DATGAN (20k/seed42) and run chunked 1:1 eval on full test."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES = _REPO_ROOT / "scripts" / "baselines"
for p in (_REPO_ROOT, _BASELINES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project_paths import setup

setup()

import pandas as pd

from scripts.baselines.run_tabular_baselines import (
    ALL_COLUMNS,
    COND_COLUMNS,
    FULL_SCHEMA,
    _build_datgan_cascade_dag,
    _build_datgan_metadata,
    _evaluate_and_save,
    _import_datgan_class,
    _sanitize_df_by_schema,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = Path("exp/datgan_cond_20k/seed_42")
    model_dir = root / "DATGAN_model"
    train_csv = root / "train_subset_20000.csv"
    test_csv = Path("data/test_data.csv")
    encoded = model_dir / "encoded_data"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    datgan_class = _import_datgan_class()
    model = datgan_class(
        output=str(model_dir) + "/",
        num_epochs=100,
        batch_size=500,
        save_checkpoints=False,
        restore_session=True,
        verbose=1,
        conditional_inputs=list(COND_COLUMNS),
    )
    dag = _build_datgan_cascade_dag()
    logging.info("Loading DATGAN checkpoint from %s", model_dir)
    model.load(
        train_df[ALL_COLUMNS],
        dag=dag,
        preprocessed_data_path=str(encoded),
    )

    condition_columns = list(COND_COLUMNS)
    inputs = test_df[condition_columns].copy()
    n_out = len(inputs)
    chunk_size = 10000
    logging.info("Chunked conditional sample n=%d chunk_size=%d", n_out, chunk_size)
    chunks = []
    for start in range(0, n_out, chunk_size):
        end = min(start + chunk_size, n_out)
        part_inputs = inputs.iloc[start:end].reset_index(drop=True)
        part = model.sample(
            num_samples=len(part_inputs),
            inputs=part_inputs,
            cond_dict={},
            randomize=False,
            timeout=False,
        )
        if not isinstance(part, pd.DataFrame):
            part = pd.DataFrame(part, columns=ALL_COLUMNS)
        for col in condition_columns:
            part[col] = part_inputs[col].values
        chunks.append(part[ALL_COLUMNS])
        logging.info("Sample progress %d/%d", end, n_out)

    generated = pd.concat(chunks, ignore_index=True)
    generated = _sanitize_df_by_schema(generated, FULL_SCHEMA)
    out_dir = root / "datgan_cond"
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = out_dir / "DATGAN_gene.csv"
    generated.to_csv(gene_csv, index=False)

    row = _evaluate_and_save(
        model_name="datgan",
        generated_df=generated,
        sampled_truth_df=test_df,
        train_real_df=train_df,
        test_real_df=test_df,
        output_dir=str(out_dir),
        seed=42,
        save_outputs=True,
    )
    summary = {
        "model": "datgan_conditional",
        "seed": 42,
        "train_subset_rows": int(len(train_df)),
        "joint_js": row.get("joint_js"),
        "mean_marginal_jsd": row.get("mean_marginal_jsd"),
        "logical_validity_rate": row.get("logical_validity_rate"),
        "mnl_behavioral_similarity": row.get("mnl_behavioral_similarity"),
    }
    (root / "summary_datgan_cond.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(root / "summary_datgan_cond.csv", index=False)
    logging.info(
        "DONE conditional DATGAN | joint=%.6f | marginal=%.6f | LVR=%.4f | MNL=%.4f",
        float(summary["joint_js"]),
        float(summary["mean_marginal_jsd"]),
        float(summary["logical_validity_rate"]),
        float(summary["mnl_behavioral_similarity"] or float("nan")),
    )
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
