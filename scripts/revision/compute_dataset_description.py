"""Compute descriptive statistics for the full revision dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "full_data_without_redundant_col.csv"
COLUMNS = [
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


def main() -> None:
    counts = {column: {} for column in COLUMNS}
    rows = 0
    ids: set[int] = set()
    households: set[int] = set()
    usecols = ["ID", "sheet_code", *COLUMNS]
    for chunk in pd.read_csv(DATA, usecols=usecols, chunksize=250_000):
        rows += len(chunk)
        ids.update(chunk["ID"].dropna().astype(int).unique().tolist())
        households.update(chunk["sheet_code"].dropna().astype(int).unique().tolist())
        for column in COLUMNS:
            value_counts = chunk[column].value_counts(dropna=False)
            target = counts[column]
            for value, count in value_counts.items():
                key = "NA" if pd.isna(value) else str(int(value))
                target[key] = target.get(key, 0) + int(count)

    output: dict[str, object] = {
        "rows": rows,
        "individuals": len(ids),
        "households": len(households),
        "variables": {},
    }
    for column, raw_counts in counts.items():
        total = sum(raw_counts.values())
        ordered = sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))
        least = min(raw_counts.items(), key=lambda item: (item[1], item[0]))
        hhi = sum((count / total) ** 2 for count in raw_counts.values())
        output["variables"][column] = {
            "n_states": len(raw_counts),
            "hhi": hhi,
            "top_1": [ordered[0][0], ordered[0][1] / total],
            "top_2": [ordered[1][0], ordered[1][1] / total],
            "least": [least[0], least[1] / total],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
