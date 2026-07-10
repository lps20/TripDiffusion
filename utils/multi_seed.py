"""Utilities for multi-seed training/evaluation with mean ± std reporting."""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional for import-time only
    torch = None

HEADLINE_METRICS = (
    "joint_js",
    "mean_marginal_jsd",
    "logical_validity_rate",
    "mnl_behavioral_similarity",
)


def parse_seeds(seed: int = 42, seeds: Optional[str] = None, num_seeds: int = 1, seed_start: int = 42) -> List[int]:
    """Resolve the list of experiment seeds from CLI-style arguments."""
    if seeds:
        parsed = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        if not parsed:
            raise ValueError("No valid seeds found in --seeds.")
        return parsed
    if num_seeds <= 1:
        return [int(seed)]
    return [int(seed_start) + i for i in range(int(num_seeds))]


def add_multiseed_arguments(parser, default_num_seeds: int = 5) -> None:
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when num_seeds=1)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated explicit seeds, e.g. '42,43,44,45,46' (overrides num_seeds)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=default_num_seeds,
        help="Number of consecutive seeds starting at seed_start",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=42,
        help="First seed when using --num_seeds > 1",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def extract_headline_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull paper headline metrics from a full evaluation dict."""
    single_jsd = metrics.get("single_feature_jsd") or {}
    marginal = metrics.get("mean_marginal_jsd")
    if marginal is None and single_jsd:
        marginal = float(np.mean(list(single_jsd.values())))

    def _as_float(value):
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(out) else out

    return {
        "joint_js": _as_float(metrics.get("joint_js")),
        "mean_marginal_jsd": _as_float(marginal),
        "logical_validity_rate": _as_float(metrics.get("logical_validity_rate")),
        "mnl_behavioral_similarity": _as_float(metrics.get("mnl_behavioral_similarity")),
        "tstr_trtr_f1_ratio": _as_float(metrics.get("mnl_behavioral_similarity")),
    }


def aggregate_metric_values(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    arr = np.array([float(v) for v in values if v is not None and not np.isnan(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0, "values": []}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(arr.size), "values": arr.tolist()}


def format_mean_std(mean: Optional[float], std: Optional[float], precision: int = 4) -> str:
    if mean is None:
        return "N/A"
    if std is None or std == 0.0:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def aggregate_headline_metrics(per_seed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_seeds": len(per_seed_rows), "metrics": {}}
    for key in HEADLINE_METRICS:
        agg = aggregate_metric_values([row.get(key) for row in per_seed_rows])
        agg["formatted"] = (
            format_mean_std(agg["mean"], agg["std"]) if agg["mean"] is not None else None
        )
        summary["metrics"][key] = agg
    return summary


def aggregate_by_model(per_seed_rows: Sequence[Dict[str, Any]], model_key: str = "model") -> List[Dict[str, Any]]:
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_seed_rows:
        model = str(row.get(model_key, "unknown"))
        rows_by_model.setdefault(model, []).append(row)

    summary_rows = []
    for model, rows in sorted(rows_by_model.items()):
        agg = aggregate_headline_metrics(rows)
        flat: Dict[str, Any] = {"model": model, "n_seeds": agg["n_seeds"]}
        for metric_name, stats in agg["metrics"].items():
            flat[f"{metric_name}_mean"] = stats["mean"]
            flat[f"{metric_name}_std"] = stats["std"]
            flat[f"{metric_name}"] = stats["formatted"]
        summary_rows.append(flat)
    return summary_rows


def log_headline_summary(summary: Dict[str, Any], title: str = "Multi-seed headline metrics") -> None:
    logging.info("%s (n=%d seeds):", title, summary.get("n_seeds", 0))
    for metric_name, stats in summary.get("metrics", {}).items():
        logging.info("  %s: %s", metric_name, stats.get("formatted", "N/A"))


def save_multiseed_artifacts(
    output_dir: str,
    per_seed_rows: Sequence[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
    summary_rows: Optional[Sequence[Dict[str, Any]]] = None,
    basename: str = "multi_seed",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    per_seed_path = os.path.join(output_dir, f"{basename}_per_seed.csv")
    if per_seed_rows:
        fieldnames: List[str] = []
        for row in per_seed_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_seed_rows)
    else:
        open(per_seed_path, "w", encoding="utf-8").close()
    paths["per_seed_csv"] = per_seed_path

    if summary is not None:
        summary_json = os.path.join(output_dir, f"{basename}_summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        paths["summary_json"] = summary_json

    if summary_rows is not None:
        summary_csv = os.path.join(output_dir, f"{basename}_summary.csv")
        rows = list(summary_rows)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            open(summary_csv, "w", encoding="utf-8").close()
        paths["summary_csv"] = summary_csv

    return paths


def run_multiseed(
    args: Any,
    seeds: Sequence[int],
    run_once: Callable[[Any, int, str], Dict[str, Any]],
    output_root: str,
    experiment_name: str = "experiment",
) -> Dict[str, Any]:
    """
    Run an experiment function for each seed and aggregate headline metrics.

    run_once(args, seed, exp_dir) must return a dict that includes headline metrics
  or a nested 'evaluation' dict.
    """
    per_seed_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = os.path.join(output_root, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        logging.info("=== %s | seed=%d | dir=%s ===", experiment_name, seed, seed_dir)

        seed_args = copy.copy(args)
        seed_args.seed = seed
        seed_args.exp_dir = seed_dir

        result = run_once(seed_args, seed, seed_dir)
        evaluation = result.get("evaluation", result)
        headline = extract_headline_metrics(evaluation)
        row = {"seed": seed, "exp_dir": seed_dir, **headline}
        if "model" in result:
            row["model"] = result["model"]
        per_seed_rows.append(row)
        logging.info(
            "Seed %d done: joint_js=%.6f, marginal_jsd=%.6f, LVR=%.4f, MNL sim=%.4f",
            seed,
            headline["joint_js"] or float("nan"),
            headline["mean_marginal_jsd"] or float("nan"),
            headline["logical_validity_rate"] or float("nan"),
            headline["mnl_behavioral_similarity"] or float("nan"),
        )

    summary = aggregate_headline_metrics(per_seed_rows)
    log_headline_summary(summary, title=f"{experiment_name} headline metrics")
    paths = save_multiseed_artifacts(output_root, per_seed_rows, summary=summary, basename="multi_seed")
    logging.info("Saved multi-seed artifacts: %s", paths)
    return {"per_seed_rows": per_seed_rows, "summary": summary, "paths": paths}
