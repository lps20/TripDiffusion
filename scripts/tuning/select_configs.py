"""Evaluate every tuning output on the validation subset and select one configuration per model.

Pre-registered rule (docs/tuning_protocol.md):
  metrics (all on data/val_subset_50k.csv, MNL reference = tuning training subset):
    joint JSD (-), mean entropy-normalised marginal JSD (-), mean ordinal EMD (-),
    mean cross-stream pairwise JSD (-), LVR (+), MNL coefficient cosine (+), MNL AME cosine (+)
  score  = mean of the within-model ranks over these 7 metrics (lower is better)
  failed = output missing, any of the 7 metrics NaN, or MNL status != ok
  guard  = |unique-trip share - real validation share| <= 0.10 (no mode collapse / over-dispersion);
           if no configuration of a model passes the guard, all non-failed ones are ranked
  ties   -> lower joint JSD
Outputs: exp/tuning/eval_cache.csv, exp/tuning/ranking.csv, exp/tuning/selected.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from project_paths import setup  # noqa: E402

setup()
import utils.test_utils as tu  # noqa: E402
from utils.extra_metrics import COND, TRIP, diversity, pairwise_jsd  # noqa: E402
from scripts.train.run_d3pm_sc3t import _default_cond_info, _default_features_info  # noqa: E402

RANK_METRICS = {  # name -> higher_is_better
    "joint_js": False, "marg_jsd_norm": False, "emd": False, "pair_jsd_cross": False,
    "LVR": True, "coef_cos": True, "ame_cos": True,
}
FEAT_ORDER = [f["name"] for f in _default_features_info()]


def evaluate(gen: pd.DataFrame, val: pd.DataFrame, train: pd.DataFrame) -> dict:
    assert (gen[COND].to_numpy() == val[COND].to_numpy()).all(), "generated rows not aligned with validation conditions"
    ev = tu.evaluate_generated_trips(
        val[FEAT_ORDER].astype(int).values.tolist(), gen[FEAT_ORDER].astype(int).values.tolist(),
        _default_features_info(), cond_info=_default_cond_info(), generated_df=gen[COND + FEAT_ORDER],
        train_real_df=train, test_real_df=val, random_state=42)
    pw = pairwise_jsd(val, gen)
    dv = diversity(gen, train)
    return {
        "joint_js": ev.get("joint_js"), "marg_jsd_norm": ev.get("mean_single_feature_jsd_normalized"),
        "emd": ev.get("mean_ordinal_emd"), "pair_jsd_cross": pw["pairwise_jsd_cross_stream"],
        "pair_jsd_all": pw["pairwise_jsd_mean"], "LVR": ev.get("logical_validity_rate"),
        "coef_cos": ev.get("mnl_coef_cosine_similarity"), "ame_cos": ev.get("mnl_ame_cosine_similarity"),
        "elas_cos": ev.get("mnl_elasticity_cosine_similarity"), "ll_ratio": ev.get("mnl_test_logloss_ratio"),
        "mnl_status": ev.get("mnl_status"), **dv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="exp/tuning/jobs.tsv")
    ap.add_argument("--val", default="data/val_subset_50k.csv")
    ap.add_argument("--train", default="data/tune_train_200k.csv")
    ap.add_argument("--guard", type=float, default=0.10)
    args = ap.parse_args()
    out = Path(args.jobs).parent
    val = pd.read_csv(args.val)
    train = pd.read_csv(args.train)
    real_share = diversity(val)["unique_trip_share"]

    cache_path = out / "eval_cache.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["path"])
    jobs = pd.read_csv(args.jobs, sep="\t")
    rows = []
    for _, j in jobs.iterrows():
        for item in str(j.outputs).split(";"):
            variant, path = item.split("=", 1)
            row = {"model": j.model, "cfg_id": j.cfg_id, "variant": variant, "params": j.params, "path": path}
            hit = cache[cache.path == path] if len(cache) else cache
            if len(hit):
                row.update({k: v for k, v in hit.iloc[0].to_dict().items() if k not in row})
            elif Path(path).exists():
                try:
                    row.update(evaluate(pd.read_csv(path), val, train))
                except Exception as exc:  # noqa: BLE001 - recorded as a failed configuration
                    row["error"] = f"{type(exc).__name__}: {exc}"
                cache = pd.concat([cache, pd.DataFrame([row])], ignore_index=True)
                cache.to_csv(cache_path, index=False)
                print(f"evaluated {j.model}/{j.cfg_id}/{variant}", flush=True)
            else:
                row["error"] = "missing output"
            rows.append(row)
    df = pd.DataFrame(rows)
    for m in RANK_METRICS:
        if m not in df:
            df[m] = np.nan
    for c in ("mnl_status", "unique_trip_share"):
        if c not in df:
            df[c] = np.nan
    df["failed"] = df[list(RANK_METRICS)].isna().any(axis=1) | (df["mnl_status"].fillna("missing") != "ok")
    df["guard_ok"] = (df["unique_trip_share"] - real_share).abs() <= args.guard

    selected, ranked = {}, []
    non_mnl = {m: h for m, h in RANK_METRICS.items() if m not in ("coef_cos", "ame_cos")}
    for model, g in df.groupby("model", sort=False):
        metrics = RANK_METRICS
        ok = g[~g.failed]
        if ok.empty:
            # Fallback: every configuration failed only because the MNL could not be estimated
            # (e.g. missing mode classes) -> rank on the five non-MNL metrics.
            ok = g[g[list(non_mnl)].notna().all(axis=1)]
            metrics = non_mnl
        pool = ok[ok.guard_ok] if ok.guard_ok.any() else ok
        if pool.empty:
            selected[model] = None
            ranked.append(g.assign(score=np.nan, used_guard=False))
            continue
        ranks = pd.concat([pool[m].rank(ascending=not hib) for m, hib in metrics.items()], axis=1)
        g = g.copy()
        g["score"] = ranks.mean(axis=1).reindex(g.index)
        g["used_guard"] = bool(ok.guard_ok.any())
        best = g.dropna(subset=["score"]).sort_values(["score", "joint_js"]).iloc[0]
        selected[model] = {"cfg_id": best.cfg_id, "variant": best.variant, "params": json.loads(best.params),
                           "score": float(best.score), "n_ranked": int(len(pool)), "guard_applied": bool(ok.guard_ok.any()),
                           "metrics_used": list(metrics)}
        ranked.append(g)
    rk = pd.concat(ranked).sort_values(["model", "score"])
    rk.to_csv(out / "ranking.csv", index=False)
    (out / "selected.json").write_text(json.dumps({"real_unique_trip_share": real_share, "selected": selected},
                                                  indent=2), encoding="utf-8")
    cols = ["model", "cfg_id", "variant", "score", "failed", "guard_ok", *RANK_METRICS, "unique_trip_share", "copy_rate"]
    print(rk[[c for c in cols if c in rk]].round(4).to_string(index=False))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
