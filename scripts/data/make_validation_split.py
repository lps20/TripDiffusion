"""Carve an individual-disjoint validation set out of data/train_data.csv.

train_data.csv was written by prepare_train_test_split.py in source-row order for
IDs in the train split, so the i-th train row's ID is the i-th source ID that is in
train_ids. We re-derive that mapping, hold out --val-ratio of the train IDs, and write
  data/val_data.csv            all rows of held-out IDs
  data/val_subset_50k.csv      fixed 50k-row sample used for tuning
  data/train_data_noval.csv    remaining train rows (for full-data tuning runs)
  data/val_split_ids.csv       held-out IDs
The test set is untouched.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from project_paths import setup  # noqa: E402

setup()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.data.prepare_train_test_split import split_ids  # noqa: E402

src = Path("data/full_data_without_redundant_col.csv")
ids = pd.read_csv(src, usecols=["ID"])["ID"]
train_ids, test_ids = split_ids(np.array(sorted(ids.unique())), 0.2, 42)
row_ids = ids[ids.isin(train_ids)].reset_index(drop=True)
train = pd.read_csv("data/train_data.csv")
assert len(train) == len(row_ids), (len(train), len(row_ids))

rng = np.random.default_rng(2026)
uniq = np.array(sorted(train_ids))
val_ids = set(rng.choice(uniq, size=int(round(0.05 * len(uniq))), replace=False))
is_val = row_ids.isin(val_ids).to_numpy()
val = train[is_val].reset_index(drop=True)
val.to_csv("data/val_data.csv", index=False)
train[~is_val].to_csv("data/train_data_noval.csv", index=False)
val.sample(n=50_000, random_state=2026).sort_index().to_csv("data/val_subset_50k.csv", index=False)
pd.Series(sorted(val_ids), name="ID").to_csv("data/val_split_ids.csv", index=False)
print(f"train IDs {len(uniq)} | val IDs {len(val_ids)} | val rows {len(val)} | train_noval rows {(~is_val).sum()}")
