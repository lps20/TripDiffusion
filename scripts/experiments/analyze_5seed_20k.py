"""Uniform re-evaluation + paired 5-seed comparison on the fixed 100k test subset (20k training).

Every generated CSV is evaluated with the same evaluator and the same MNL reference
(full real training data), so model differences are not confounded by evaluation set-up.

Usage:
  python scripts/experiments/analyze_5seed_20k.py --model ABS=exp/d3pm_abs_5seed_20k/ABS/seed_{seed}/generated_samples.csv \
      --model D3PM=exp/d3pm_abs_5seed_20k/D3PM_orig/seed_{seed}/generated_samples.csv \
      --model C5=exp/d3pm_abs_5seed_20k/C5/seed_{seed}/C5_x0_d16_T50_shift2_ctok/generated.csv --ref ABS
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from project_paths import setup  # noqa: E402

setup()
import utils.test_utils as tu  # noqa: E402
from scripts.train.run_d3pm_sc3t import _default_cond_info, _default_features_info  # noqa: E402

FEAT = [f["name"] for f in _default_features_info()]
COND = [c["name"] for c in _default_cond_info()]
RADIX = {"relation": 5, "sex": 2, "age_code": 13, "job_type": 9, "start_type": 5, "start_zcode_num": 77,
         "act_num": 9, "mode_num": 9, "end_type": 5, "end_zcode_num": 77, "start_time_num_6": 241,
         "trip_time_num_6": 241}
METRICS = {  # name -> (key in evaluation dict or diversity dict, higher_is_better)
    "joint_js": ("joint_js", False),
    "marg_jsd_norm": ("mean_single_feature_jsd_normalized", False),
    "emd": ("mean_ordinal_emd", False),
    "LVR": ("logical_validity_rate", True),
    "coef_cos": ("mnl_coef_cosine_similarity", True),
    "ame_cos": ("mnl_ame_cosine_similarity", True),
    "elas_cos": ("mnl_elasticity_cosine_similarity", True),
    "ll_ratio": ("mnl_test_logloss_ratio", False),
    "uniq8": ("uniq8", None),
    "copy": ("copy", None),
}


def key(d, cols):
    k = np.zeros(len(d), dtype=np.int64)
    for c in cols:
        k = k * RADIX[c] + d[c].to_numpy(dtype=np.int64)
    return pd.Series(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=path with {seed} placeholder")
    ap.add_argument("--ref", required=True, help="Model name that paired differences are taken against")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--truth", default=str(REPO / "exp/d3pm_tricks_20k/test_subset_100k.csv"))
    ap.add_argument("--full_train", default=str(REPO / "data/train_data.csv"))
    ap.add_argument("--subset_tpl", default=str(REPO.parent / "D3PM_revision/revision_exp/robustness_size/20k/seed_{seed}/train_subset_20000.csv"))
    ap.add_argument("--out", default=str(REPO / "exp/d3pm_abs_5seed_20k"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(args.truth)
    truth_trip = truth[FEAT].astype(int).values.tolist()
    full_train = pd.read_csv(args.full_train)
    models = dict(m.split("=", 1) for m in args.model)

    cache_path = out / "per_seed_metrics.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "seed", "path"])
    rows = []
    for seed in args.seeds:
        sub_keys = set(key(pd.read_csv(args.subset_tpl.format(seed=seed)), COND + FEAT))
        for name, tpl in models.items():
            path = str(Path(tpl.format(seed=seed)))
            hit = cache[(cache.model == name) & (cache.seed == seed) & (cache.path == path)]
            if len(hit):
                rows.append(hit.iloc[0].to_dict())
                continue
            gen = pd.read_csv(path)
            assert (gen[COND].to_numpy() == truth[COND].to_numpy()).all(), f"{name} seed {seed} not row-aligned"
            ev = tu.evaluate_generated_trips(
                truth_trip, gen[FEAT].astype(int).values.tolist(), _default_features_info(),
                cond_info=_default_cond_info(), generated_df=gen[COND + FEAT],
                train_real_df=full_train, test_real_df=truth, random_state=seed,
            )
            ev["uniq8"] = key(gen, FEAT).nunique() / len(gen)
            ev["copy"] = float(key(gen, COND + FEAT).isin(sub_keys).mean())
            row = {"model": name, "seed": seed, "path": path, **{m: ev.get(k) for m, (k, _) in METRICS.items()}}
            rows.append(row)
            print(row, flush=True)
            pd.concat([cache, pd.DataFrame(rows)]).drop_duplicates(["model", "seed", "path"], keep="last").to_csv(cache_path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)

    mean = df.groupby("model")[list(METRICS)].agg(["mean", "std"])
    print("\n=== mean (sd) over seeds ===")
    print(mean.round(4).to_string())

    ref = df[df.model == args.ref].set_index("seed")
    paired = []
    for name in models:
        if name == args.ref:
            continue
        other = df[df.model == name].set_index("seed")
        seeds = sorted(set(ref.index) & set(other.index))
        for m, (_, hib) in METRICS.items():
            d = (ref.loc[seeds, m] - other.loc[seeds, m]).astype(float).to_numpy()
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            h = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
            better = None if hib is None else (d.mean() - h > 0 if hib else d.mean() + h < 0)
            paired.append({"ref": args.ref, "vs": name, "metric": m, "n": n, "mean_diff": d.mean(),
                           "ci_low": d.mean() - h, "ci_high": d.mean() + h, "ref_sig_better": better})
    pdf = pd.DataFrame(paired)
    pdf.to_csv(out / f"paired_vs_{args.ref}.csv", index=False)
    print(f"\n=== paired differences ({args.ref} - other), 95% t-CI ===")
    print(pdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
