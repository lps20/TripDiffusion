import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import pandas as pd


MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "hcd_v2": {
        "script": "run_hcd_v2.py",
        "label": "HCD_V2",
    },
    "ddpm_tf": {
        "script": "run_transformer.py",
        "label": "DDPM_TF",
    },
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sample_subset(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    return df.sample(frac=frac, replace=False, random_state=seed).reset_index(drop=True)


def _run_command(cmd: List[str], cwd: str) -> None:
    logging.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _read_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_metrics(model_name: str, subset_pct: int, subset_size: int, exp_dir: str, metrics: Dict) -> Dict:
    row = {
        "model": model_name,
        "subset_pct": subset_pct,
        "subset_size": subset_size,
        "exp_dir": exp_dir,
        "joint_js": metrics.get("joint_js"),
        "logical_validity_rate": metrics.get("logical_validity_rate"),
        "tstr_macro_f1": metrics.get("tstr_macro_f1"),
        "trtr_macro_f1": metrics.get("trtr_macro_f1"),
        "tstr_accuracy": metrics.get("tstr_accuracy"),
        "trtr_accuracy": metrics.get("trtr_accuracy"),
        "tstr_trtr_f1_ratio": metrics.get("tstr_trtr_f1_ratio"),
    }
    for feat_name, value in metrics.get("single_feature_jsd", {}).items():
        row[f"jsd_{feat_name}"] = value
    for rule_name, value in metrics.get("invalid_rule_breakdown", {}).items():
        row[f"lvr_{rule_name}"] = value
    row["n_total"] = metrics.get("n_total")
    row["n_valid"] = metrics.get("n_valid")
    row["n_invalid"] = metrics.get("n_invalid")
    return row


def _upsert_summary(summary_path: str, row: Dict) -> None:
    new_df = pd.DataFrame([row])
    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        mask = (old_df["model"] == row["model"]) & (old_df["subset_pct"] == row["subset_pct"])
        old_df = old_df.loc[~mask].copy()
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(["model", "subset_pct"]).reset_index(drop=True)
    merged.to_csv(summary_path, index=False)


def main(args: argparse.Namespace) -> None:
    root_dir = os.path.abspath(args.root_dir)
    exp_root = os.path.join(root_dir, args.exp_root)
    subset_dir = os.path.join(exp_root, "subsets")
    summary_path = os.path.join(exp_root, "robustness_summary.csv")
    _ensure_dir(exp_root)
    _ensure_dir(subset_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Arguments: %s", vars(args))

    train_path = os.path.join(root_dir, args.traindata)
    test_path = os.path.join(root_dir, args.testdata)
    train_df = pd.read_csv(train_path)

    fractions: List[Tuple[int, float]] = [(1, 0.01), (5, 0.05)]

    for subset_pct, frac in fractions:
        subset_seed = args.seed + subset_pct
        subset_df = _sample_subset(train_df, frac=frac, seed=subset_seed)
        subset_name = f"train_{subset_pct}pct.csv"
        subset_path = os.path.join(subset_dir, subset_name)
        subset_df.to_csv(subset_path, index=False)
        logging.info("Prepared subset %s with %d rows", subset_name, len(subset_df))

        for model_key in args.models:
            model_cfg = MODEL_CONFIGS[model_key]
            model_label = model_cfg["label"]
            exp_dir = os.path.join(exp_root, f"{model_label.lower()}_{subset_pct}pct")
            _ensure_dir(exp_dir)
            metrics_json = os.path.join(exp_dir, "generated_samples_metrics.json")

            if args.skip_existing and os.path.exists(metrics_json):
                metrics = _read_metrics(metrics_json)
                row = _flatten_metrics(
                    model_name=model_label,
                    subset_pct=subset_pct,
                    subset_size=len(subset_df),
                    exp_dir=exp_dir,
                    metrics=metrics,
                )
                _upsert_summary(summary_path, row)
                logging.info("Skipping existing completed run %s at %d%%", model_label, subset_pct)
                continue

            train_cmd = [
                sys.executable,
                model_cfg["script"],
                "--traindata",
                subset_path,
                "--testdata",
                test_path,
                "--exp_dir",
                exp_dir,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_samples",
                str(args.num_samples),
                "--lr",
                str(args.lr),
            ]
            if model_key == "hcd_v2" and args.hcd_loss_type is not None:
                train_cmd.extend(["--loss_type", args.hcd_loss_type])
            _run_command(train_cmd, cwd=root_dir)

            generated_csv = os.path.join(exp_dir, "generated_samples.csv")
            eval_cmd = [
                sys.executable,
                "evaluate_generated_csv.py",
                "--generated_csv",
                generated_csv,
                "--train_data",
                subset_path,
                "--test_data",
                test_path,
                "--output_json",
                metrics_json,
                "--model_name",
                f"{model_label}_{subset_pct}pct",
            ]
            _run_command(eval_cmd, cwd=root_dir)

            metrics = _read_metrics(metrics_json)
            row = _flatten_metrics(
                model_name=model_label,
                subset_pct=subset_pct,
                subset_size=len(subset_df),
                exp_dir=exp_dir,
                metrics=metrics,
            )
            _upsert_summary(summary_path, row)
            logging.info(
                "Finished %s at %d%%: joint_js=%s, LVR=%s, TSTR_F1=%s",
                model_label,
                subset_pct,
                metrics.get("joint_js"),
                metrics.get("logical_validity_rate"),
                metrics.get("tstr_macro_f1"),
            )

    logging.info("Saved robustness summary: %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robustness to training sample size for HCD v2 and DDPM+TF."
    )
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--exp_root", type=str, default="exp/robustness_sample_size")
    parser.add_argument("--models", nargs="+", default=["hcd_v2", "ddpm_tf"], choices=["hcd_v2", "ddpm_tf"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hcd_loss_type", type=str, default="standard", choices=["standard", "causal"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    main(args)
