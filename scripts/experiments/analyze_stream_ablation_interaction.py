"""Combine full-data and 10k stream ablations and estimate interaction CIs.

Positive paired improvement always favors soft-stream. The interaction is
defined as paired_improvement_10k - paired_improvement_full_data, so a positive
value means the soft-stream advantage is larger in the 10k regime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.experiments.run_stream_ablation_10k import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRICS,
    T_CRITICAL_975,
)


def _read_metrics(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("evaluation", payload)


def _full_rows(
    seeds: List[int],
    soft_root: Path,
    shared_root: Path,
    hard_root: Path,
) -> List[Dict[str, Any]]:
    roots = {
        "soft_stream": soft_root,
        "shared_only": shared_root,
        "hard_stream": hard_root,
    }
    rows: List[Dict[str, Any]] = []
    for variant, root in roots.items():
        for seed in seeds:
            metrics_path = root / f"seed_{seed}" / "generated_samples_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)
            evaluation = _read_metrics(metrics_path)
            row: Dict[str, Any] = {
                "regime": "full_data",
                "variant": variant,
                "seed": seed,
                "mnl_status": evaluation.get("mnl_status"),
                "source_metrics_path": str(metrics_path.resolve()),
            }
            for metric in METRICS:
                row[metric] = evaluation.get(metric)
            rows.append(row)
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (regime, variant), group in frame.groupby(["regime", "variant"], sort=True):
        row: Dict[str, Any] = {
            "regime": regime,
            "variant": variant,
            "n_seed_rows": int(len(group)),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_improvements(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for regime, regime_frame in frame.groupby("regime", sort=True):
        for comparator in ("shared_only", "hard_stream"):
            for metric in METRICS:
                pivot = regime_frame.pivot(
                    index="seed", columns="variant", values=metric
                )
                if "soft_stream" not in pivot or comparator not in pivot:
                    continue
                complete = pivot[["soft_stream", comparator]].dropna()
                for seed, pair in complete.iterrows():
                    if metric in LOWER_IS_BETTER:
                        improvement = pair[comparator] - pair["soft_stream"]
                    elif metric in HIGHER_IS_BETTER:
                        improvement = pair["soft_stream"] - pair[comparator]
                    else:
                        raise RuntimeError(f"Metric direction missing: {metric}")
                    rows.append(
                        {
                            "regime": regime,
                            "comparison": f"soft_stream_vs_{comparator}",
                            "metric": metric,
                            "seed": int(seed),
                            "paired_improvement_positive_favors_soft": float(improvement),
                        }
                    )
    return pd.DataFrame(rows)


def _ci_summary(
    frame: pd.DataFrame,
    group_columns: List[str],
    value_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_column].to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else np.nan
        if n > 1:
            t_critical = T_CRITICAL_975.get(n - 1, 1.96)
            half_width = float(t_critical * std / math.sqrt(n))
            ci_low = mean - half_width
            ci_high = mean + half_width
        else:
            ci_low = np.nan
            ci_high = np.nan
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_pairs": n,
                "mean": mean,
                "std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _interaction_rows(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (comparison, metric), group in paired.groupby(
        ["comparison", "metric"], sort=True
    ):
        pivot = group.pivot(
            index="seed",
            columns="regime",
            values="paired_improvement_positive_favors_soft",
        )
        if "10k" not in pivot or "full_data" not in pivot:
            continue
        complete = pivot[["10k", "full_data"]].dropna()
        for seed, pair in complete.iterrows():
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "seed": int(seed),
                    "improvement_10k": float(pair["10k"]),
                    "improvement_full_data": float(pair["full_data"]),
                    "interaction_10k_minus_full_positive_means_larger_soft_advantage": float(
                        pair["10k"] - pair["full_data"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small_summary",
        type=Path,
        default=Path("exp/revision_stream_ablation_10k/summary_all.csv"),
    )
    parser.add_argument("--full_soft_root", type=Path, required=True)
    parser.add_argument("--full_shared_root", type=Path, required=True)
    parser.add_argument("--full_hard_root", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("exp/revision_stream_ablation_10k"),
    )
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    small = pd.read_csv(args.small_summary)
    small.insert(0, "regime", "10k")
    full = pd.DataFrame(
        _full_rows(
            seeds=seeds,
            soft_root=args.full_soft_root,
            shared_root=args.full_shared_root,
            hard_root=args.full_hard_root,
        )
    )
    combined = pd.concat([small, full], ignore_index=True, sort=False)
    combined = combined.sort_values(["regime", "variant", "seed"]).reset_index(
        drop=True
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_root / "cross_regime_summary_all.csv", index=False)
    _summary(combined).to_csv(
        args.output_root / "cross_regime_multi_seed_summary.csv", index=False
    )

    paired = _paired_improvements(combined)
    paired.to_csv(
        args.output_root / "cross_regime_paired_differences_seed.csv", index=False
    )
    _ci_summary(
        paired,
        ["regime", "comparison", "metric"],
        "paired_improvement_positive_favors_soft",
    ).to_csv(args.output_root / "cross_regime_paired_summary.csv", index=False)

    interactions = _interaction_rows(paired)
    interactions.to_csv(
        args.output_root / "interaction_differences_seed.csv", index=False
    )
    _ci_summary(
        interactions,
        ["comparison", "metric"],
        "interaction_10k_minus_full_positive_means_larger_soft_advantage",
    ).to_csv(args.output_root / "interaction_summary.csv", index=False)

    print(_summary(combined).to_string(index=False))
    print()
    print(
        _ci_summary(
            interactions,
            ["comparison", "metric"],
            "interaction_10k_minus_full_positive_means_larger_soft_advantage",
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
