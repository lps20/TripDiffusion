"""Split full_data_without_redundant_col.csv into train/test by ID (80:20)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse

import numpy as np
import pandas as pd

from utils.data_encoding import normalize_category_values

OUTPUT_COLUMNS = [
    "relation",
    "sex",
    "age_code",
    "job_type",
    "start_type",
    "start_time_num_6",
    "start_zcode_num",
    "act_num",
    "mode_num",
    "trip_time_num_6",
    "end_type",
    "end_zcode_num",
]

OUTPUT_SCHEMA = [
    {"name": "relation", "num_classes": 5},
    {"name": "sex", "num_classes": 2},
    {"name": "age_code", "num_classes": 13},
    {"name": "job_type", "num_classes": 9},
    {"name": "start_type", "num_classes": 5},
    {"name": "start_time_num_6", "num_classes": 241},
    {"name": "start_zcode_num", "num_classes": 77},
    {"name": "act_num", "num_classes": 9},
    {"name": "mode_num", "num_classes": 9},
    {"name": "trip_time_num_6", "num_classes": 241},
    {"name": "end_type", "num_classes": 5},
    {"name": "end_zcode_num", "num_classes": 77},
]


def _normalize_output_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk[OUTPUT_COLUMNS].copy()
    for field in OUTPUT_SCHEMA:
        out[field["name"]] = normalize_category_values(
            out[field["name"]], field["num_classes"], field["name"]
        )
    return out


def collect_unique_ids(source_path: Path, id_col: str, chunksize: int) -> np.ndarray:
    unique_ids = set()
    for chunk in pd.read_csv(source_path, usecols=[id_col], chunksize=chunksize):
        unique_ids.update(chunk[id_col].unique())
    return np.array(sorted(unique_ids))


def split_ids(unique_ids: np.ndarray, test_ratio: float, seed: int) -> tuple[set, set]:
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)

    n_test = int(round(len(shuffled) * test_ratio))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return train_ids, test_ids


def write_split(
    source_path: Path,
    train_path: Path,
    test_path: Path,
    train_ids: set,
    test_ids: set,
    id_col: str,
    chunksize: int,
) -> tuple[int, int]:
    train_written = False
    test_written = False
    train_rows = 0
    test_rows = 0

    for chunk in pd.read_csv(source_path, chunksize=chunksize):
        out = _normalize_output_chunk(chunk)
        train_chunk = out[chunk[id_col].isin(train_ids)]
        test_chunk = out[chunk[id_col].isin(test_ids)]

        if not train_chunk.empty:
            train_chunk.to_csv(train_path, mode="w" if not train_written else "a", header=not train_written, index=False)
            train_written = True
            train_rows += len(train_chunk)

        if not test_chunk.empty:
            test_chunk.to_csv(test_path, mode="w" if not test_written else "a", header=not test_written, index=False)
            test_written = True
            test_rows += len(test_chunk)

    if not train_written:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(train_path, index=False)
    if not test_written:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(test_path, index=False)

    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split trip data by ID into train/test CSV files.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/full_data_without_redundant_col.csv"),
        help="Source CSV path",
    )
    parser.add_argument("--train-out", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test-out", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--id-col", type=str, default="ID", help="Column used for leakage-free split")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    unique_ids = collect_unique_ids(args.source, args.id_col, args.chunksize)
    train_ids, test_ids = split_ids(unique_ids, args.test_ratio, args.seed)

    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"ID leakage detected: {len(overlap)} overlapping IDs")

    train_rows, test_rows = write_split(
        args.source,
        args.train_out,
        args.test_out,
        train_ids,
        test_ids,
        args.id_col,
        args.chunksize,
    )

    total_rows = train_rows + test_rows
    print(f"Unique {args.id_col}: {len(unique_ids):,}")
    print(f"Train IDs: {len(train_ids):,} | Test IDs: {len(test_ids):,}")
    print(f"Train rows: {train_rows:,} ({train_rows / total_rows:.2%})")
    print(f"Test rows: {test_rows:,} ({test_rows / total_rows:.2%})")
    print(f"Saved: {args.train_out}")
    print(f"Saved: {args.test_out}")


if __name__ == "__main__":
    main()
