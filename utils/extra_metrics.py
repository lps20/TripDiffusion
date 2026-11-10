"""Additional evaluation metrics used by the tuning protocol and the paired analyses.

- Pairwise (second-order) JSD over variable pairs, reported as the mean over all pairs and
  over cross-stream pairs (pairs whose variables belong to different behavioural streams:
  demographics / activity / space-time / mode). Cross-stream pairs are exactly the
  dependencies the cascade is meant to encode.
- Diversity / memorisation: share of distinct 8-field trip tuples and the exact-copy rate
  of (condition + trip) records against a training table.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

COND = ["relation", "sex", "age_code", "job_type"]
TRIP = ["start_type", "start_zcode_num", "act_num", "mode_num", "end_type", "end_zcode_num",
        "start_time_num_6", "trip_time_num_6"]
SIZES = {"relation": 5, "sex": 2, "age_code": 13, "job_type": 9, "start_type": 5, "start_zcode_num": 77,
         "act_num": 9, "mode_num": 9, "end_type": 5, "end_zcode_num": 77, "start_time_num_6": 241,
         "trip_time_num_6": 241}
STREAM = {**{c: "demo" for c in COND}, "act_num": "act", "mode_num": "mode",
          **{c: "st" for c in ["start_type", "start_zcode_num", "end_type", "end_zcode_num",
                               "start_time_num_6", "trip_time_num_6"]}}


def _jsd_from_counts(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log(a[nz] / b[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _pair_counts(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    k = df[a].to_numpy(dtype=np.int64) * SIZES[b] + df[b].to_numpy(dtype=np.int64)
    return np.bincount(k, minlength=SIZES[a] * SIZES[b]).astype(float)


def pairwise_jsd(real: pd.DataFrame, gen: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Mean JSD of 2-D joint distributions over all pairs / cross-stream pairs.

    Pairs made only of demographic columns are skipped (they are given, not generated)."""
    cols = list(cols or (COND + TRIP))
    all_v, cross_v = [], []
    for a, b in combinations(cols, 2):
        if STREAM[a] == "demo" and STREAM[b] == "demo":
            continue
        v = _jsd_from_counts(_pair_counts(real, a, b), _pair_counts(gen, a, b))
        all_v.append(v)
        if STREAM[a] != STREAM[b]:
            cross_v.append(v)
    return {"pairwise_jsd_mean": float(np.mean(all_v)), "pairwise_jsd_cross_stream": float(np.mean(cross_v))}


def tuple_key(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    k = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        k = k * SIZES[c] + df[c].to_numpy(dtype=np.int64)
    return pd.Series(k)


def diversity(gen: pd.DataFrame, train: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    out = {"unique_trip_share": float(tuple_key(gen, TRIP).nunique() / max(len(gen), 1))}
    if train is not None:
        train_keys = set(tuple_key(train, COND + TRIP))
        out["copy_rate"] = float(tuple_key(gen, COND + TRIP).isin(train_keys).mean())
    return out
