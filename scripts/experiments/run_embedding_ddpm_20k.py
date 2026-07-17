"""Train continuous Embedding-DDPM (tf + mlp) on shared 20k / seed_42 subset."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import pandas as pd

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
TRIP_COLS = [f["name"] for f in FEATURES_INFO]
COND_COLS = ["relation", "sex", "age_code", "job_type"]
LEGACY_SUBSET = Path("exp/robustness_20k/seed_42/train_subset_20000.csv")


def _py() -> List[str]:
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "tripdiffusion", "--no-capture-output", "python"]
    return [sys.executable]


def _run(cmd: List[str]) -> None:
    logging.info("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _headline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    ev = metrics.get("evaluation", metrics)
    jsd = ev.get("single_feature_jsd") or {}
    return {
        "joint_js": ev.get("joint_js"),
        "mean_marginal_jsd": ev.get("mean_marginal_jsd"),
        "mean_ordinal_emd": ev.get("mean_ordinal_emd"),
        "start_zcode_jsd": jsd.get("start_zcode_num"),
        "end_zcode_jsd": jsd.get("end_zcode_num"),
        "logical_validity_rate": ev.get("logical_validity_rate"),
        "mnl_behavioral_similarity": ev.get("mnl_behavioral_similarity"),
    }


def _ensure_subset(path: Path, full_train: str, n: int, seed: int) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if LEGACY_SUBSET.exists() and n == 20000 and seed == 42:
        shutil.copy2(LEGACY_SUBSET, path)
        return path
    df = pd.read_csv(full_train)
    df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True).to_csv(path, index=False)
    return path


def _reeval(model_name: str, gene_csv: Path, train: str, test: str, seed: int) -> Dict[str, Any]:
    import utils.test_utils
    from scripts.baselines.run_tabular_baselines import ALL_COLUMNS, FULL_SCHEMA, _sanitize_df_by_schema

    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    generated_df = _sanitize_df_by_schema(pd.read_csv(gene_csv), FULL_SCHEMA)
    metrics = utils.test_utils.evaluate_generated_trips(
        test_df[TRIP_COLS].astype(int).values.tolist(),
        generated_df[TRIP_COLS].astype(int).values.tolist(),
        FEATURES_INFO,
        cond_info=[{"name": c} for c in COND_COLS],
        generated_df=generated_df[ALL_COLUMNS],
        train_real_df=train_df,
        test_real_df=test_df,
        random_state=seed,
    )
    return {"seed": seed, "model": model_name, "evaluation": metrics}


def _run_one(model: str, train: str, test: str, out_dir: Path, seed: int, epochs: int, T: int, n_gene: int) -> Dict[str, Any]:
    tag = model.upper()
    metrics_path = out_dir / f"{tag}_metrics.json"
    gene_csv = out_dir / f"{tag}_gene.csv"
    if metrics_path.exists() and gene_csv.exists():
        logging.info("Skip %s", model)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _py()
        + [
            "scripts/baselines/run_tabular_baselines.py",
            "--traindata",
            train,
            "--testdata",
            test,
            "--output_dir",
            str(out_dir),
            "--models",
            model,
            "--epochs",
            str(epochs),
            "--batch_size",
            "500",
            "--num_samples",
            str(n_gene),
            "--seed",
            str(seed),
            "--num_seeds",
            "1",
            "--ddpm_t",
            str(T),
            "--ddpm_lambda_weight",
            "1.0",
            "--ddpm_lambda_joint",
            "0.0",
        ]
    )
    payload = _reeval(model, gene_csv, train, test, seed)
    payload["train_subset_rows"] = int(pd.read_csv(train).shape[0])
    payload["ddpm_t"] = T
    payload["diffusion"] = "embedding_gaussian"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="exp/embedding_ddpm_20k")
    parser.add_argument("--full_train", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--subset_rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ddpm_t", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=["ddpm_tf", "ddpm_mlp"], choices=["ddpm_tf", "ddpm_mlp"])
    args = parser.parse_args()

    root = Path(args.output_root) / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_root) / "run.log", encoding="utf-8"),
        ],
    )
    subset = _ensure_subset(
        root / f"train_subset_{args.subset_rows}.csv",
        args.full_train,
        args.subset_rows,
        args.seed,
    )
    test_n = len(pd.read_csv(args.testdata))
    rows = []
    for model in args.models:
        logging.info("===== Embedding-DDPM %s =====", model)
        payload = _run_one(
            model, str(subset), args.testdata, root / model, args.seed, args.epochs, args.ddpm_t, test_n
        )
        row = {"model": model, "seed": args.seed, "train_subset_rows": args.subset_rows, "ddpm_t": args.ddpm_t, **_headline(payload)}
        rows.append(row)
        logging.info("DONE %s | joint_js=%s | MNL=%s | LVR=%s", model, row.get("joint_js"), row.get("mnl_behavioral_similarity"), row.get("logical_validity_rate"))
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary_embedding_ddpm.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
