"""Multinomial logit (MNL) mode-choice validation for TSTR behavioral fidelity."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

MODE_CHOICE_TARGET = "mode_num"
MODE_CHOICE_FEATURES = [
    "relation",
    "sex",
    "age_code",
    "job_type",
    "trip_time_num_6",
    "start_time_num_6",
]
TRAVEL_ACTIVITY_CODE = 1
MOVING_MODE_CODES = (3, 4, 5, 6, 7, 8)
MIN_ROWS = 100
MIN_CLASS_COUNT = 20


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "mnl_status": reason,
        "mnl_behavioral_similarity": None,
        "mnl_coef_cosine_similarity": None,
        "mnl_coef_rmse": None,
        "mnl_ame_cosine_similarity": None,
        "mnl_ame_rmse": None,
        "mnl_elasticity_cosine_similarity": None,
        "mnl_elasticity_rmse": None,
        "mnl_tstr_test_logloss": None,
        "mnl_trtr_test_logloss": None,
        "mnl_test_logloss_ratio": None,
        "mnl_mode_choice": None,
        "tstr_macro_f1": None,
        "trtr_macro_f1": None,
        "tstr_accuracy": None,
        "trtr_accuracy": None,
        "tstr_trtr_f1_ratio": None,
    }


def _prepare_mode_choice_frame(df: pd.DataFrame) -> pd.DataFrame:
    needed = [MODE_CHOICE_TARGET, "act_num", *MODE_CHOICE_FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for MNL mode choice: {missing}")

    out = df.copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=needed)
    out = out[out["act_num"].round().astype(int) == TRAVEL_ACTIVITY_CODE]
    out[MODE_CHOICE_TARGET] = out[MODE_CHOICE_TARGET].round().astype(int)
    out = out[out[MODE_CHOICE_TARGET].isin(MOVING_MODE_CODES)]

    for col in MODE_CHOICE_FEATURES:
        out[col] = out[col].round().astype(int)

    return out.reset_index(drop=True)


def _class_counts(y: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _is_usable_split(y_train: np.ndarray, y_eval: np.ndarray) -> bool:
    train_counts = _class_counts(y_train)
    eval_counts = _class_counts(y_eval)
    shared = set(train_counts) & set(eval_counts)
    if len(shared) < 2:
        return False
    if any(train_counts[c] < MIN_CLASS_COUNT for c in shared):
        return False
    if any(eval_counts.get(c, 0) < 5 for c in shared):
        return False
    return True


def _fit_mnl(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> Tuple[LogisticRegression, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    model.fit(X_scaled, y)
    return model, scaler, X_scaled


def _align_coefficients(
    model_a: LogisticRegression,
    model_b: LogisticRegression,
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    classes = sorted(set(model_a.classes_).intersection(model_b.classes_))
    if not classes:
        return np.array([]), np.array([]), []

    idx_a = {int(c): i for i, c in enumerate(model_a.classes_)}
    idx_b = {int(c): i for i, c in enumerate(model_b.classes_)}
    coef_a = np.vstack([model_a.coef_[idx_a[c]] for c in classes])
    coef_b = np.vstack([model_b.coef_[idx_b[c]] for c in classes])
    return coef_a.ravel(), coef_b.ravel(), classes


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _rmse(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _predict_probabilities(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)
    return probs, model.classes_.astype(int)


def _average_marginal_effects(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
    classes: Sequence[int],
    feature_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    probs, model_classes = _predict_probabilities(model, scaler, X)
    class_to_idx = {int(c): i for i, c in enumerate(model_classes)}
    coef = model.coef_
    ame: Dict[str, Dict[str, float]] = {str(c): {} for c in classes}

    for feat_idx, feat_name in enumerate(feature_names):
        beta_full = np.array([coef[i, feat_idx] for i in range(len(model_classes))])
        weighted_beta = probs @ beta_full

        for mode in classes:
            if mode not in class_to_idx:
                continue
            col = class_to_idx[mode]
            p_k = probs[:, col]
            beta_k = coef[col, feat_idx]
            marginal = p_k * (beta_k - weighted_beta)
            ame[str(mode)][feat_name] = float(np.mean(marginal))

    return ame


def _implied_arc_elasticities(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
    classes: Sequence[int],
    feature_names: Sequence[str],
    continuous_features: Sequence[str] = ("trip_time_num_6", "start_time_num_6"),
) -> Dict[str, Dict[str, float]]:
    probs, model_classes = _predict_probabilities(model, scaler, X)
    class_to_idx = {int(c): i for i, c in enumerate(model_classes)}
    coef = model.coef_
    x_means = {name: float(np.mean(X[:, i])) for i, name in enumerate(feature_names)}
    elasticities: Dict[str, Dict[str, float]] = {str(c): {} for c in classes}

    for feat_idx, feat_name in enumerate(feature_names):
        if feat_name not in continuous_features:
            continue
        x_bar = max(x_means[feat_name], 1e-6)
        beta_full = np.array([coef[i, feat_idx] for i in range(len(model_classes))])
        weighted_beta = probs @ beta_full
        for mode in classes:
            if mode not in class_to_idx:
                continue
            col = class_to_idx[mode]
            p_k = np.mean(probs[:, col])
            beta_k = coef[col, feat_idx]
            ame_k = float(np.mean(probs[:, col] * (beta_k - weighted_beta)))
            if p_k <= 1e-8:
                elasticities[str(mode)][feat_name] = 0.0
            else:
                elasticities[str(mode)][feat_name] = float(ame_k * (x_bar / p_k))
    return elasticities


def _flatten_nested(nested: Dict[str, Dict[str, float]], classes: Sequence[int], features: Sequence[str]) -> np.ndarray:
    values = []
    for mode in classes:
        mode_key = str(mode)
        for feat in features:
            values.append(float(nested.get(mode_key, {}).get(feat, 0.0)))
    return np.asarray(values, dtype=float)


def _coef_dict(
    model: LogisticRegression,
    classes: Sequence[int],
    feature_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
    out: Dict[str, Dict[str, float]] = {}
    for mode in classes:
        if mode not in class_to_idx:
            continue
        row = model.coef_[class_to_idx[mode]]
        out[str(mode)] = {feat: float(row[i]) for i, feat in enumerate(feature_names)}
    return out


def evaluate_mnl_mode_choice_validation(
    synthetic_df: pd.DataFrame,
    train_real_df: pd.DataFrame,
    test_real_df: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Behavioral TSTR validation with a simple MNL mode-choice model.

    Fit MNL on synthetic (TSTR) and real-train (TRTR) data, then compare:
      - utility parameters
      - average marginal effects (AME)
      - implied arc elasticities for continuous covariates
      - hold-out log-loss on real test trips
    """
    try:
        syn = _prepare_mode_choice_frame(synthetic_df)
        tr = _prepare_mode_choice_frame(train_real_df)
        te = _prepare_mode_choice_frame(test_real_df)
    except ValueError as exc:
        return _empty_result(str(exc))

    if min(len(syn), len(tr), len(te)) < MIN_ROWS:
        return _empty_result("insufficient_travel_mode_rows")

    X_syn = syn[MODE_CHOICE_FEATURES].to_numpy(dtype=float)
    y_syn = syn[MODE_CHOICE_TARGET].to_numpy(dtype=int)
    X_tr = tr[MODE_CHOICE_FEATURES].to_numpy(dtype=float)
    y_tr = tr[MODE_CHOICE_TARGET].to_numpy(dtype=int)
    X_te = te[MODE_CHOICE_FEATURES].to_numpy(dtype=float)
    y_te = te[MODE_CHOICE_TARGET].to_numpy(dtype=int)

    if not _is_usable_split(y_syn, y_te) or not _is_usable_split(y_tr, y_te):
        return _empty_result("insufficient_mode_class_support")

    model_tstr, scaler_tstr, _ = _fit_mnl(X_syn, y_syn, random_state)
    model_trtr, scaler_trtr, _ = _fit_mnl(X_tr, y_tr, random_state)

    coef_tstr, coef_trtr, shared_classes = _align_coefficients(model_tstr, model_trtr, MODE_CHOICE_FEATURES)
    if coef_tstr.size == 0:
        return _empty_result("no_shared_mode_classes")

    ame_tstr = _average_marginal_effects(model_tstr, scaler_tstr, X_te, shared_classes, MODE_CHOICE_FEATURES)
    ame_trtr = _average_marginal_effects(model_trtr, scaler_trtr, X_te, shared_classes, MODE_CHOICE_FEATURES)
    elast_tstr = _implied_arc_elasticities(model_tstr, scaler_tstr, X_te, shared_classes, MODE_CHOICE_FEATURES)
    elast_trtr = _implied_arc_elasticities(model_trtr, scaler_trtr, X_te, shared_classes, MODE_CHOICE_FEATURES)

    ame_tstr_vec = _flatten_nested(ame_tstr, shared_classes, MODE_CHOICE_FEATURES)
    ame_trtr_vec = _flatten_nested(ame_trtr, shared_classes, MODE_CHOICE_FEATURES)
    elast_tstr_vec = _flatten_nested(elast_tstr, shared_classes, ("trip_time_num_6", "start_time_num_6"))
    elast_trtr_vec = _flatten_nested(elast_trtr, shared_classes, ("trip_time_num_6", "start_time_num_6"))

    tstr_probs, tstr_classes = _predict_probabilities(model_tstr, scaler_tstr, X_te)
    trtr_probs, trtr_classes = _predict_probabilities(model_trtr, scaler_trtr, X_te)
    tstr_logloss = float(log_loss(y_te, tstr_probs, labels=tstr_classes))
    trtr_logloss = float(log_loss(y_te, trtr_probs, labels=trtr_classes))

    coef_cosine = _cosine_similarity(coef_tstr, coef_trtr)
    coef_rmse = _rmse(coef_tstr, coef_trtr)
    ame_cosine = _cosine_similarity(ame_tstr_vec, ame_trtr_vec)
    ame_rmse = _rmse(ame_tstr_vec, ame_trtr_vec)
    elast_cosine = _cosine_similarity(elast_tstr_vec, elast_trtr_vec)
    elast_rmse = _rmse(elast_tstr_vec, elast_trtr_vec)

    return {
        "mnl_status": "ok",
        "mnl_behavioral_similarity": coef_cosine,
        "mnl_coef_cosine_similarity": coef_cosine,
        "mnl_coef_rmse": coef_rmse,
        "mnl_ame_cosine_similarity": ame_cosine,
        "mnl_ame_rmse": ame_rmse,
        "mnl_elasticity_cosine_similarity": elast_cosine,
        "mnl_elasticity_rmse": elast_rmse,
        "mnl_tstr_test_logloss": tstr_logloss,
        "mnl_trtr_test_logloss": trtr_logloss,
        "mnl_test_logloss_ratio": float(tstr_logloss / max(trtr_logloss, 1e-12)),
        "mnl_mode_choice": {
            "target": MODE_CHOICE_TARGET,
            "features": list(MODE_CHOICE_FEATURES),
            "travel_activity_code": TRAVEL_ACTIVITY_CODE,
            "moving_mode_codes": list(MOVING_MODE_CODES),
            "shared_mode_classes": [int(c) for c in shared_classes],
            "n_train_synthetic": int(len(syn)),
            "n_train_real": int(len(tr)),
            "n_eval_real": int(len(te)),
            "tstr_coefficients": _coef_dict(model_tstr, shared_classes, MODE_CHOICE_FEATURES),
            "trtr_coefficients": _coef_dict(model_trtr, shared_classes, MODE_CHOICE_FEATURES),
            "tstr_average_marginal_effects": ame_tstr,
            "trtr_average_marginal_effects": ame_trtr,
            "tstr_implied_elasticities": elast_tstr,
            "trtr_implied_elasticities": elast_trtr,
        },
        "tstr_macro_f1": None,
        "trtr_macro_f1": None,
        "tstr_accuracy": None,
        "trtr_accuracy": None,
        "tstr_trtr_f1_ratio": coef_cosine,
    }
