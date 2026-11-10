"""Summarise D3PM-SC3T trick screen (+ Embedding-DDPM C5 and real-data floor) on the fixed 100k test subset."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import utils.test_utils as tu  # noqa: E402

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(REPO / "exp/d3pm_tricks_20k"))
ap.add_argument("--truth", default=None, help="Row-aligned truth CSV (default: <out>/test_subset_100k.csv)")
ap.add_argument("--train", default=str(REPO / "../D3PM_revision/revision_exp/robustness_size/20k/seed_42/train_subset_20000.csv"))
ap.add_argument("--extra", action="append", default=[], help="name=path/to/generated.csv (metrics.json next to it)")
ap.add_argument("--no_c5", action="store_true")
args = ap.parse_args()
OUT = Path(args.out)
FEAT = ["start_type", "start_zcode_num", "act_num", "mode_num", "end_type", "end_zcode_num",
        "start_time_num_6", "trip_time_num_6"]
COND = ["relation", "sex", "age_code", "job_type"]
RADIX = {"relation": 5, "sex": 2, "age_code": 13, "job_type": 9, "start_type": 5, "start_zcode_num": 77,
         "act_num": 9, "mode_num": 9, "end_type": 5, "end_zcode_num": 77, "start_time_num_6": 241,
         "trip_time_num_6": 241}


def key(d, cols):
    k = np.zeros(len(d), dtype=np.int64)
    for c in cols:
        k = k * RADIX[c] + d[c].to_numpy(dtype=np.int64)
    return pd.Series(k)


def entropy(s):
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum())


truth = pd.read_csv(args.truth or OUT / "test_subset_100k.csv")
train20k = pd.read_csv(args.train)
train_keys = set(key(train20k, COND + FEAT))
truth_trip = truth[FEAT].astype(int).values.tolist()

srcs = {}
for d in sorted(glob.glob(str(OUT / "*/generated_samples.csv"))):
    srcs[Path(d).parent.name] = pd.read_csv(d)
c5 = REPO / "exp/embedding_ddpm_screen_20k/C5_x0_d16_T50_shift2_ctok/generated.csv"
if c5.exists() and not args.no_c5:
    srcs["EMB_C5"] = pd.read_csv(c5)
extra_metrics = {}
for e in args.extra:
    n, pth = e.split("=", 1)
    srcs[n] = pd.read_csv(pth)
    extra_metrics[n] = Path(pth).parent / "metrics.json"

rows = []
for name, d in srcs.items():
    assert (d[COND].to_numpy() == truth[COND].to_numpy()).all(), f"{name}: conditions not row-aligned"
    gen_trip = d[FEAT].astype(int).values.tolist()
    mj = c5.parent / "metrics.json" if name == "EMB_C5" else extra_metrics.get(name, OUT / name / "generated_samples_metrics.json")
    ev = json.load(open(mj))["evaluation"] if mj.exists() else {}
    k8 = key(d, FEAT)
    rows.append({
        "run": name,
        "joint_js": tu.evaluate_joint_js_divergence(truth_trip, gen_trip),
        "marg_jsd_norm_e3": 1e3 * ev.get("mean_single_feature_jsd_normalized", np.nan),
        "emd": ev.get("mean_ordinal_emd", np.nan),
        "LVR": ev.get("logical_validity_rate", np.nan),
        "coef_cos": ev.get("mnl_coef_cosine_similarity", np.nan),
        "ame_cos": ev.get("mnl_ame_cosine_similarity", np.nan),
        "ll_ratio": ev.get("mnl_test_logloss_ratio", np.nan),
        "uniq8": k8.nunique() / len(d),
        "H8": entropy(k8),
        "copy": float(key(d, COND + FEAT).isin(train_keys).mean()),
    })
df = pd.DataFrame(rows)
df.to_csv(OUT / "summary.csv", index=False)
print(df.round(4).to_string(index=False))
