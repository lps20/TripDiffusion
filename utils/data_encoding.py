"""Normalize categorical column encodings for model training."""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

ONE_BASED_COND_COLUMNS = {"relation", "sex", "age_code", "job_type"}


def normalize_category_values(
    values: Union[pd.Series, np.ndarray, Iterable],
    num_classes: int,
    column_name: Optional[str] = None,
) -> np.ndarray:
    """
    Convert category codes to zero-based indices in [0, num_classes - 1].

    Source CSVs from the NTS pipeline use 1-based codes for demographics.
    Older project CSVs are already 0-based (min == 0).
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).round().astype(int).to_numpy()

    convert_one_based = column_name in ONE_BASED_COND_COLUMNS
    if not convert_one_based and arr.size > 0 and int(arr.min()) >= 1:
        convert_one_based = True

    if convert_one_based and arr.size > 0 and int(arr.min()) >= 1:
        arr = arr - 1

    return np.clip(arr, 0, num_classes - 1)


def normalize_dataframe_columns(
    df: pd.DataFrame,
    schema: Iterable[dict],
    inplace: bool = False,
) -> pd.DataFrame:
    out = df if inplace else df.copy()
    for field in schema:
        name = field["name"]
        if name not in out.columns:
            continue
        out[name] = normalize_category_values(out[name], field["num_classes"], name)
    return out
