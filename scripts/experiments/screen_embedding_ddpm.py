"""Screen Embedding-DDPM fixes on a small training subset.

Diffusion stays in the latent (embedding) space; the candidates only change the
noise schedule, the diffusion width, the parameterization, and how the
demographic condition enters the denoiser. Each config is trained once, run
through a per-timestep diagnostic, and evaluated 1:1 on a fixed held-out subset.

    python scripts/experiments/screen_embedding_ddpm.py --calibrate      # schedule check, no training
    python scripts/experiments/screen_embedding_ddpm.py                  # train + evaluate all configs
    python scripts/experiments/screen_embedding_ddpm.py --diagnose CKPT  # diagnose one checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup  # noqa: E402

setup()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import utils.test_utils  # noqa: E402
from model.EmbeddingDDPM_Net import EmbeddingDDPM, sample_embedding_ddpm, train_embedding_ddpm  # noqa: E402

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
COND_INFO = [
    {"name": "relation", "num_classes": 5},
    {"name": "sex", "num_classes": 2},
    {"name": "age_code", "num_classes": 13},
    {"name": "job_type", "num_classes": 9},
]
FEAT_COLS = [f["name"] for f in FEATURES_INFO]
COND_COLS = [c["name"] for c in COND_INFO]
ALL_COLS = COND_COLS + FEAT_COLS

# Shared R1 architecture: d_model 128, 4-layer 8-head transformer, fixed random codebook.
BASE = dict(d_model=128, backbone="transformer", nhead=8, num_layers=4, beta_schedule="cosine",
            feature_embedding_mode="fixed", fixed_codebook_type="random", decode_metric="cosine")
CONFIGS: Dict[str, Dict[str, Any]] = {
    "C0_r1_T10": dict(T=10),
    "C1_eps_T10_shift4": dict(T=10, snr_shift=4.0),
    "C2_eps_T50_shift8": dict(T=50, snr_shift=8.0),
    "C3_eps_d16_T50": dict(T=50, embed_dim=16),
    "C4_x0_d16_T50_shift2": dict(T=50, embed_dim=16, snr_shift=2.0, parameterization="x0"),
    "C5_x0_d16_T50_shift2_ctok": dict(T=50, embed_dim=16, snr_shift=2.0, parameterization="x0",
                                      cond_tokens=True, cond_dropout=0.1),
    "C7_x0_d128_T50_shift8_ctok": dict(T=50, snr_shift=8.0, parameterization="x0",
                                       cond_tokens=True, cond_dropout=0.1),
}
# Sampling-time variants that reuse another config's checkpoint.
SAMPLE_VARIANTS = {"C6_C5_cfg1": ("C5_x0_d16_T50_shift2_ctok", dict(guidance_scale=1.0))}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(cfg: Dict[str, Any]) -> EmbeddingDDPM:
    return EmbeddingDDPM(features_info=FEATURES_INFO, cond_info=COND_INFO, **{**BASE, **cfg})


def load_checkpoint(path: Path, device: torch.device) -> EmbeddingDDPM:
    ck = torch.load(path, map_location=device)
    keys = ["T", "d_model", "backbone", "nhead", "num_layers", "dropout", "mlp_hidden", "beta_schedule",
            "feature_embedding_mode", "fixed_codebook_type", "decode_metric", "x0_ce_weight", "sample_method",
            "embed_dim", "snr_shift", "parameterization", "cond_tokens", "cond_dropout", "guidance_scale"]
    kw = {k: ck[k] for k in keys if k in ck}
    m = EmbeddingDDPM(features_info=ck["features_info"], cond_info=ck["cond_info"], **kw).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def diagnose(model: EmbeddingDDPM, df: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    """Per-timestep: decode(x_t) accuracy without the network, x0-hat accuracy, condition sensitivity."""
    model.eval()
    x = torch.tensor(df[FEAT_COLS].values, dtype=torch.long, device=device)
    c = torch.tensor(df[COND_COLS].values, dtype=torch.long, device=device)
    x0 = model.encode_features(x)
    perm = torch.randperm(c.size(0), device=device)
    rows = []
    for t in range(model.T):
        tb = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        xt, _ = model.q_sample(x0, tb)
        x0p, logits = model.predict_x0(xt, c, tb)
        x0p_s, logits_s = model.predict_x0(xt, c[perm], tb)
        if logits is not None:
            pred = torch.stack([lg.argmax(-1) for lg in logits], 1)
            # condition sensitivity: total-variation between predicted categorical distributions
            sens = float(np.mean([
                0.5 * (lg.softmax(-1) - ls.softmax(-1)).abs().sum(-1).mean().item()
                for lg, ls in zip(logits, logits_s)
            ]))
        else:
            pred = model.decode_features(x0p)
            sens = float((x0p - x0p_s).norm(dim=-1).mean() / x0.norm(dim=-1).mean())
        rows.append({
            "t": t,
            "alpha_bar": float(model.alpha_bar[t]),
            "decode_xt_acc": float((model.decode_features(xt) == x).float().mean()),
            "x0hat_acc": float((pred == x).float().mean()),
            "cond_sensitivity": sens,
        })
    return {"per_t": rows}


def calibrate(df: pd.DataFrame, device: torch.device) -> None:
    """Untrained check of how informative each step's noise level is (no network involved)."""
    x = torch.tensor(df[FEAT_COLS].values, dtype=torch.long, device=device)
    for name, cfg in CONFIGS.items():
        set_seed(0)
        m = build(cfg).to(device).eval()
        x0 = m.encode_features(x)
        accs = []
        for t in range(m.T):
            tb = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            xt, _ = m.q_sample(x0, tb)
            accs.append(float((m.decode_features(xt) == x).float().mean()))
        informative = sum(0.05 < a < 0.95 for a in accs)
        q = [accs[int(i * (m.T - 1))] for i in (0, 0.25, 0.5, 0.75, 1.0)]
        logging.info("%-28s T=%3d informative_steps=%3d decode(x_t) acc at t/T=0,.25,.5,.75,1: %s",
                     name, m.T, informative, [round(a, 3) for a in q])


def evaluate(gen: pd.DataFrame, truth: pd.DataFrame, train: pd.DataFrame, seed: int) -> Dict[str, Any]:
    return utils.test_utils.evaluate_generated_trips(
        truth[FEAT_COLS].astype(int).values.tolist(),
        gen[FEAT_COLS].astype(int).values.tolist(),
        FEATURES_INFO,
        cond_info=[{"name": c} for c in COND_COLS],
        generated_df=gen[ALL_COLS],
        train_real_df=train,
        test_real_df=truth,
        random_state=seed,
    )


HEADLINE = ["joint_js", "mean_marginal_jsd", "mean_single_feature_jsd_normalized", "mean_ordinal_emd",
            "logical_validity_rate", "mnl_coef_cosine_similarity", "mnl_ame_cosine_similarity",
            "mnl_test_logloss_ratio"]


def headline(name: str, metrics: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    row = {"config": name, **extra}
    row.update({k: metrics.get(k) for k in HEADLINE})
    row.update({f"jsdn_{k}": v for k, v in metrics.get("single_feature_jsd_normalized", {}).items()})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(_REPO_ROOT.parent / "D3PM_revision/revision_exp/robustness_size/20k/seed_42/train_subset_20000.csv"))
    ap.add_argument("--full_train", default="data/train_data.csv", help="Real training data for the TRTR MNL reference.")
    ap.add_argument("--test", default="data/test_data.csv")
    ap.add_argument("--eval_n", type=int, default=100000, help="Fixed held-out subset size for 1:1 evaluation.")
    ap.add_argument("--reference_csv", default=str(_REPO_ROOT.parent / "D3PM_revision/revision_exp/robustness_size/20k/seed_42/hcd/generated_samples.csv"),
                    help="Row-aligned full-test output of D3PM-SC3T trained on the same subset (reference row).")
    ap.add_argument("--out", default="exp/embedding_ddpm_screen_20k")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--diagnose", default=None, help="Checkpoint path to diagnose only.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", encoding="utf-8")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test = pd.read_csv(args.test)
    rng = np.random.default_rng(0)
    eval_idx = np.sort(rng.choice(len(test), size=min(args.eval_n, len(test)), replace=False))
    truth = test.iloc[eval_idx].reset_index(drop=True)
    diag_df = truth.sample(2048, random_state=1)

    if args.calibrate:
        calibrate(diag_df, device)
        return
    if args.diagnose:
        res = diagnose(load_checkpoint(Path(args.diagnose), device), diag_df, device)
        for r in res["per_t"]:
            logging.info("%s", r)
        return

    train_sub = pd.read_csv(args.train)
    full_train = pd.read_csv(args.full_train)
    logging.info("train subset=%d rows, eval subset=%d rows, device=%s", len(train_sub), len(truth), device)
    summary_path = out / "summary.csv"
    rows: List[Dict[str, Any]] = []

    ref_path = Path(args.reference_csv)
    if ref_path.exists():
        ref = pd.read_csv(ref_path).iloc[eval_idx].reset_index(drop=True)
        aligned = all((ref[c].values == truth[c].values).all() for c in COND_COLS)
        if aligned:
            rows.append(headline("REF_D3PM-SC3T_20k", evaluate(ref, truth, full_train, args.seed), {"params": None}))
            logging.info("reference: %s", rows[-1])
        else:
            logging.warning("Reference CSV is not row-aligned with the test set; skipped.")

    names = args.configs or list(CONFIGS) + list(SAMPLE_VARIANTS)
    for name in names:
        cdir = out / name
        cdir.mkdir(exist_ok=True)
        metrics_path = cdir / "metrics.json"
        if metrics_path.exists() and not args.force:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(payload["headline"])
            logging.info("[%s] cached", name)
            continue

        set_seed(args.seed)
        t0 = time.time()
        if name in SAMPLE_VARIANTS:
            base, overrides = SAMPLE_VARIANTS[name]
            ckpt = out / base / "model.pth"
            if not ckpt.exists():
                logging.warning("[%s] base checkpoint %s missing; run %s first.", name, ckpt, base)
                continue
            model = load_checkpoint(ckpt, device)
            for k, v in overrides.items():
                setattr(model, k, v)
            cfg = {"base": base, **overrides}
        else:
            cfg = CONFIGS[name]
            model = build(cfg).to(device)
            model = train_embedding_ddpm(model, train_sub, FEAT_COLS, COND_COLS, epochs=args.epochs,
                                         batch_size=args.batch_size, lr=args.lr, device=device,
                                         model_save_path=str(cdir / "model.pth"))
        train_s = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diag = diagnose(model, diag_df, device)

        set_seed(args.seed)
        t1 = time.time()
        gen = sample_embedding_ddpm(model, truth, FEAT_COLS, COND_COLS, n_samples=0, device=device,
                                    batch_size=4096, match_test_one_to_one=True)
        sample_s = time.time() - t1
        gen[ALL_COLS].to_csv(cdir / "generated.csv", index=False)
        metrics = evaluate(gen, truth, full_train, args.seed)
        mid = diag["per_t"][len(diag["per_t"]) // 2]
        extra = {"params": n_params, "train_s": round(train_s, 1), "sample_s": round(sample_s, 1),
                 "mid_decode_xt_acc": mid["decode_xt_acc"], "mid_x0hat_acc": mid["x0hat_acc"],
                 "mean_cond_sensitivity": float(np.mean([r["cond_sensitivity"] for r in diag["per_t"]]))}
        row = headline(name, metrics, extra)
        metrics_path.write_text(json.dumps({"config": cfg, "headline": row, "diagnostic": diag, "evaluation": metrics},
                                           indent=2, default=float), encoding="utf-8")
        rows.append(row)
        logging.info("[%s] joint=%.4f mjsd_norm=%.5f emd=%.2f LVR=%.4f coef_cos=%s | train %.0fs sample %.0fs",
                     name, row["joint_js"], row["mean_single_feature_jsd_normalized"], row["mean_ordinal_emd"],
                     row["logical_validity_rate"], row["mnl_coef_cosine_similarity"], train_s, sample_s)
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logging.info("summary -> %s", summary_path)


if __name__ == "__main__":
    main()
