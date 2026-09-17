"""Sequential transport generator using MNL and proportional-odds logit stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

COND_COLUMNS = ["relation", "sex", "age_code", "job_type"]
TRIP_COLUMNS = [
    "start_type",
    "start_time_num_6",
    "start_zcode_num",
    "act_num",
    "mode_num",
    "trip_time_num_6",
    "end_type",
    "end_zcode_num",
]
ALL_COLUMNS = COND_COLUMNS + TRIP_COLUMNS


@dataclass(frozen=True)
class StageSpec:
    target: str
    predictors: Sequence[str]
    kind: str = "mnl"
    numeric_predictors: Sequence[str] = ()


# Activity -> space/time -> mode, matching the paper's causal hierarchy.
STAGES = [
    StageSpec("act_num", COND_COLUMNS),
    StageSpec("start_type", [*COND_COLUMNS, "act_num"]),
    StageSpec("start_zcode_num", [*COND_COLUMNS, "act_num", "start_type"]),
    StageSpec(
        "end_type",
        [*COND_COLUMNS, "act_num", "start_type", "start_zcode_num"],
    ),
    StageSpec(
        "end_zcode_num",
        [*COND_COLUMNS, "act_num", "start_type", "start_zcode_num", "end_type"],
    ),
    StageSpec(
        "start_time_num_6",
        [*COND_COLUMNS, "act_num", "start_type", "end_type"],
        kind="ordered_logit",
    ),
    StageSpec(
        "trip_time_num_6",
        [*COND_COLUMNS, "act_num", "start_type", "end_type", "start_time_num_6"],
        kind="ordered_logit",
        numeric_predictors=("start_time_num_6",),
    ),
    StageSpec(
        "mode_num",
        [
            *COND_COLUMNS,
            "act_num",
            "start_type",
            "start_zcode_num",
            "end_type",
            "end_zcode_num",
            "start_time_num_6",
            "trip_time_num_6",
        ],
        numeric_predictors=("start_time_num_6", "trip_time_num_6"),
    ),
]


def _make_preprocessor(spec: StageSpec, dense: bool) -> ColumnTransformer:
    numeric = list(spec.numeric_predictors)
    categorical = [column for column in spec.predictors if column not in numeric]
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=not dense,
                    dtype=np.float64 if dense else np.float32,
                ),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0 if dense else 1.0)


class MNLStage:
    def __init__(self, spec: StageSpec, random_state: int):
        self.spec = spec
        self.preprocessor = _make_preprocessor(spec, dense=False)
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=400,
            random_state=random_state,
        )

    def fit(self, frame: pd.DataFrame) -> "MNLStage":
        x = self.preprocessor.fit_transform(frame[list(self.spec.predictors)])
        self.model.fit(x, frame[self.spec.target].astype(int).to_numpy())
        return self

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        x = self.preprocessor.transform(frame[list(self.spec.predictors)])
        probabilities = self.model.predict_proba(x)
        cumulative = np.cumsum(probabilities, axis=1)
        draws = rng.random(len(frame))
        indices = (cumulative < draws[:, None]).sum(axis=1)
        indices = np.minimum(indices, len(self.model.classes_) - 1)
        return self.model.classes_[indices].astype(np.int16)


class OrderedLogitStage:
    def __init__(self, spec: StageSpec):
        self.spec = spec
        self.preprocessor = _make_preprocessor(spec, dense=True)
        self.classes_: np.ndarray | None = None
        self.result = None

    def fit(self, frame: pd.DataFrame) -> "OrderedLogitStage":
        x = self.preprocessor.fit_transform(frame[list(self.spec.predictors)])
        self.classes_ = np.sort(frame[self.spec.target].astype(int).unique())
        class_to_code = {int(value): index for index, value in enumerate(self.classes_)}
        y = frame[self.spec.target].astype(int).map(class_to_code).to_numpy()
        model = OrderedModel(y, x, distr="logit")
        self.result = model.fit(method="lbfgs", maxiter=200, disp=False)
        logging.info(
            "Ordered logit %s converged=%s llf=%.2f",
            self.spec.target,
            bool(self.result.mle_retvals.get("converged", False)),
            float(self.result.llf),
        )
        return self

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if self.result is None or self.classes_ is None:
            raise RuntimeError(f"Stage {self.spec.target} has not been fitted")
        x = self.preprocessor.transform(frame[list(self.spec.predictors)])
        probabilities = np.asarray(self.result.model.predict(self.result.params, exog=x))
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / np.maximum(row_sums, 1e-12)
        cumulative = np.cumsum(probabilities, axis=1)
        draws = rng.random(len(frame))
        indices = (cumulative < draws[:, None]).sum(axis=1)
        indices = np.minimum(indices, len(self.classes_) - 1)
        return self.classes_[indices].astype(np.int16)


class SequentialEconometricGenerator:
    """Conditional one-to-one generator composed of interpretable statistical stages."""

    def __init__(self, random_state: int = 42):
        self.random_state = int(random_state)
        self.stages: Dict[str, MNLStage | OrderedLogitStage] = {}

    def fit(self, train_df: pd.DataFrame) -> "SequentialEconometricGenerator":
        frame = train_df[ALL_COLUMNS].copy()
        for column in ALL_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise").round().astype(int)
        for index, spec in enumerate(STAGES, start=1):
            logging.info(
                "Fitting stage %d/%d target=%s kind=%s predictors=%s",
                index,
                len(STAGES),
                spec.target,
                spec.kind,
                list(spec.predictors),
            )
            if spec.kind == "ordered_logit":
                stage: MNLStage | OrderedLogitStage = OrderedLogitStage(spec)
            else:
                stage = MNLStage(spec, self.random_state + index)
            self.stages[spec.target] = stage.fit(frame)
        return self

    def sample(
        self,
        condition_df: pd.DataFrame,
        batch_size: int = 10000,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        condition = condition_df[COND_COLUMNS].copy().reset_index(drop=True)
        for column in COND_COLUMNS:
            condition[column] = pd.to_numeric(condition[column], errors="raise").round().astype(int)

        batches: List[pd.DataFrame] = []
        for start in range(0, len(condition), batch_size):
            generated = condition.iloc[start : start + batch_size].copy()
            for spec in STAGES:
                generated[spec.target] = self.stages[spec.target].sample(generated, rng)
            batches.append(generated[ALL_COLUMNS])
        return pd.concat(batches, ignore_index=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "SequentialEconometricGenerator":
        return joblib.load(path)

