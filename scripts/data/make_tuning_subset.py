"""Fixed 200k-row tuning subset of the training data with the validation individuals removed.

Writes data/tune_train_200k.csv (rows sampled from data/train_data_noval.csv, seed 2026).
Run scripts/data/make_validation_split.py first.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from project_paths import setup  # noqa: E402

setup()
import pandas as pd  # noqa: E402

df = pd.read_csv("data/train_data_noval.csv")
df.sample(n=200_000, random_state=2026).sort_index().to_csv("data/tune_train_200k.csv", index=False)
print("wrote data/tune_train_200k.csv")
