"""Does a generator actually use the demographic condition?

For each generated CSV (row-aligned with the truth CSV) report, next to the real data:
  - car-driver share (mode 5) for age_code<=1 (minors) vs age_code>=4
  - mutual information I(demographic; act) and I(demographic; mode), demographic = 4-tuple
A generator that ignores the condition shows MI ~ 0 and equal car-driver shares.
Usage: python scripts/eval/condition_check.py --truth data/val_subset_50k.csv NAME=path.csv ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COND = ["relation", "sex", "age_code", "job_type"]


def mi(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    p = ct / ct.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def stats(df: pd.DataFrame) -> dict:
    demo = df[COND].astype(str).agg("-".join, axis=1)
    minor, adult = df.age_code <= 1, df.age_code >= 4
    return {
        "car_driver_minor": float((df.loc[minor, "mode_num"] == 5).mean()),
        "car_driver_adult": float((df.loc[adult, "mode_num"] == 5).mean()),
        "MI(demo;act)": mi(demo, df.act_num),
        "MI(demo;mode)": mi(demo, df.mode_num),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("gen", nargs="+")
    a = ap.parse_args()
    rows = [{"source": "REAL", **stats(pd.read_csv(a.truth))}]
    for g in a.gen:
        n, p = g.split("=", 1)
        rows.append({"source": n, **stats(pd.read_csv(p))})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
