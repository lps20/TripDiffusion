"""Privacy-risk assessment: exact matches, DCR/NNDR, and NN membership inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COND = ["relation", "sex", "age_code", "job_type"]
CAT = [
    "start_type", "start_zcode_num", "act_num", "mode_num",
    "end_type", "end_zcode_num",
]
ORD = ["start_time_num_6", "trip_time_num_6"]
TRIP = CAT + ORD
MODELS = ["hcd", "tvae", "ctgan", "datgan", "tabddpm", "embedding_ddpm"]
BASELINE_TAG = {
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "datgan": "DATGAN",
    "tabddpm": "TABDDPM",
}


def extract_member(zip_path: Path, member: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as source, target.open("wb") as sink:
            sink.write(source.read())
    return target


def load_embedding_samples(
    seed: int,
    train: pd.DataFrame,
    n: int,
    cache: Path,
    zip_path: Path,
) -> pd.DataFrame:
    csv_path = cache / f"seed_{seed}" / "synthetic.csv"
    if csv_path.exists():
        cached = pd.read_csv(csv_path)
        if len(cached) >= n:
            return cached

    checkpoint = extract_member(
        zip_path,
        f"T10_cond/seed_{seed}/DDPM_TF_model.pth",
        cache / f"seed_{seed}" / "DDPM_TF_model.pth",
    )
    import torch
    from model.EmbeddingDDPM_Net import EmbeddingDDPM, sample_embedding_ddpm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device)
    model = EmbeddingDDPM(
        features_info=payload["features_info"],
        cond_info=payload["cond_info"],
        T=int(payload["T"]),
        d_model=int(payload["d_model"]),
        backbone=str(payload["backbone"]),
        nhead=int(payload.get("nhead", 8)),
        num_layers=int(payload.get("num_layers", 4)),
        mlp_hidden=payload.get("mlp_hidden"),
        dropout=float(payload.get("dropout", 0.1)),
        beta_schedule=str(payload.get("beta_schedule", "cosine")),
        feature_embedding_mode=str(payload.get("feature_embedding_mode", "learned")),
        fixed_codebook_type=str(payload.get("fixed_codebook_type", "random")),
        decode_metric=str(payload.get("decode_metric", "dot")),
        x0_ce_weight=float(payload.get("x0_ce_weight", 0.0)),
        sample_method=str(payload.get("sample_method", "ddpm")),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    cond_pool = train.sample(n=n, replace=n > len(train), random_state=seed).reset_index(drop=True)
    synthetic = sample_embedding_ddpm(
        model=model,
        test_df=cond_pool,
        feat_cols=[field["name"] for field in payload["features_info"]],
        cond_cols=[field["name"] for field in payload["cond_info"]],
        n_samples=n,
        device=device,
        batch_size=2048,
        match_test_one_to_one=True,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(csv_path, index=False)
    return synthetic


def load_synthetic(
    model: str,
    seed: int,
    train: pd.DataFrame,
    n: int,
    cache: Path,
    exp_root: Path,
) -> pd.DataFrame:
    if model in BASELINE_TAG:
        path = exp_root / "revision_baseline" / model / f"seed_{seed}" / f"{BASELINE_TAG[model]}_gene.csv"
        return pd.read_csv(path)
    if model == "hcd":
        local = exp_root / "revision_hcd_opt" / "st_cascade" / f"seed_{seed}" / "generated_samples.csv"
        if local.exists():
            return pd.read_csv(local)
        extracted = extract_member(
            exp_root / "revision_hcd_opt.zip",
            f"st_cascade/seed_{seed}/generated_samples.csv",
            cache / "hcd" / f"seed_{seed}" / "synthetic.csv",
        )
        return pd.read_csv(extracted)
    if model == "embedding_ddpm":
        return load_embedding_samples(
            seed, train, n, cache / "embedding_ddpm", exp_root / "embedding_ddpm_full.zip"
        )
    raise ValueError(model)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[TRIP].copy()
    for col in TRIP:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna().round().astype(np.int16).reset_index(drop=True)


def sample_frame(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True)


def nearest_two(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Exact mixed Gower distance: Hamming categorical + normalized ordinal L1."""
    first = np.empty(len(query), dtype=np.float32)
    second = np.empty(len(query), dtype=np.float32)
    n_features = len(CAT) + len(ORD)
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        dist = np.zeros((len(q), len(reference)), dtype=np.float32)
        for index in range(len(CAT)):
            dist += q[:, None, index] != reference[None, :, index]
        for offset in range(len(ORD)):
            index = len(CAT) + offset
            dist += np.abs(q[:, None, index] - reference[None, :, index]) / 240.0
        dist /= float(n_features)
        nearest = np.partition(dist, kth=1, axis=1)[:, :2]
        nearest.sort(axis=1)
        first[start : start + len(q)] = nearest[:, 0]
        second[start : start + len(q)] = nearest[:, 1]
    return first, second


def exact_match_rate(synthetic: pd.DataFrame, real: pd.DataFrame) -> float:
    real_hashes = set(pd.util.hash_pandas_object(real[TRIP], index=False).astype("uint64"))
    synthetic_hashes = pd.util.hash_pandas_object(synthetic[TRIP], index=False).astype("uint64")
    return float(synthetic_hashes.isin(real_hashes).mean())


def summarize_distance(prefix: str, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_dcr_mean": float(np.mean(first)),
        f"{prefix}_dcr_p01": float(np.quantile(first, 0.01)),
        f"{prefix}_dcr_p05": float(np.quantile(first, 0.05)),
        f"{prefix}_dcr_median": float(np.median(first)),
        f"{prefix}_nndr_mean": float(np.mean(first / np.maximum(second, 1e-12))),
        f"{prefix}_nndr_p05": float(np.quantile(first / np.maximum(second, 1e-12), 0.05)),
    }


def assess(
    model: str,
    seed: int,
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_size: int,
    attack_size: int,
) -> dict[str, float | int | str]:
    syn = sample_frame(clean(synthetic), sample_size, seed)
    train_clean = clean(train)
    test_clean = clean(test)
    train_ref = sample_frame(train_clean, sample_size, seed + 101)
    test_ref = sample_frame(test_clean, sample_size, seed + 202)

    syn_np = syn[TRIP].to_numpy(dtype=np.float32)
    train_np = train_ref[TRIP].to_numpy(dtype=np.float32)
    test_np = test_ref[TRIP].to_numpy(dtype=np.float32)
    d_train, n2_train = nearest_two(syn_np, train_np)
    d_test, n2_test = nearest_two(syn_np, test_np)

    members = sample_frame(train_clean, attack_size, seed + 303)[TRIP].to_numpy(dtype=np.float32)
    nonmembers = sample_frame(test_clean, attack_size, seed + 404)[TRIP].to_numpy(dtype=np.float32)
    member_d, _ = nearest_two(members, syn_np)
    nonmember_d, _ = nearest_two(nonmembers, syn_np)
    labels = np.r_[np.ones(len(member_d)), np.zeros(len(nonmember_d))]
    scores = -np.r_[member_d, nonmember_d]

    return {
        "model": model,
        "seed": seed,
        "n_synthetic_distance": len(syn),
        "exact_match_train": exact_match_rate(syn, train_clean),
        "exact_match_test": exact_match_rate(syn, test_clean),
        **summarize_distance("train", d_train, n2_train),
        **summarize_distance("test", d_test, n2_test),
        "dcr_train_minus_test": float(np.mean(d_train) - np.mean(d_test)),
        "membership_auc": float(roc_auc_score(labels, scores)),
    }


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = [col for col in rows.columns if col not in {"model", "seed"}]
    records = []
    for model, group in rows.groupby("model"):
        record: dict[str, float | int | str] = {"model": model, "n_seeds": len(group)}
        for metric in numeric:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--exp_root", type=Path, default=Path("exp"))
    parser.add_argument("--output_dir", type=Path, default=Path("revision_exp/privacy"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--attack_size", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, float | int | str]] = []
    output = args.output_dir / "per_seed.csv"
    if output.exists():
        rows = pd.read_csv(output).to_dict("records")

    completed = {(str(row["model"]), int(row["seed"])) for row in rows}
    for model in args.models:
        for seed in seeds:
            if (model, seed) in completed:
                continue
            logging.info("Assessing privacy model=%s seed=%d", model, seed)
            synthetic = load_synthetic(
                model, seed, train, args.sample_size,
                args.output_dir / "_synthetic_cache", args.exp_root,
            )
            rows.append(
                assess(model, seed, synthetic, train, test, args.sample_size, args.attack_size)
            )
            frame = pd.DataFrame(rows)
            frame.to_csv(output, index=False)
            aggregate(frame).to_csv(args.output_dir / "summary.csv", index=False)
            (args.output_dir / "summary.json").write_text(
                json.dumps(aggregate(frame).to_dict("records"), indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
