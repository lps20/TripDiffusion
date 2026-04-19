import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def compute_kl_divergence(p, q, eps=1e-10):
    """
    Compute KL divergence between two discrete distributions p and q
      KL(p||q) = sum( p * log(p/q) )
    Parameters:
      p, q: 1D numpy arrays representing probability distributions (must be normalized)
      eps: Smoothing term to avoid division by zero
    Returns:
      KL divergence value
    """
    p = np.array(p, dtype=np.float64) + eps
    q = np.array(q, dtype=np.float64) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

def compute_js_divergence(p, q, eps=1e-10):
    """
    Compute Jensen-Shannon Divergence
      JS(p||q) = 0.5*KL(p || m) + 0.5*KL(q || m) where m = 0.5*(p+q)
    """
    p = np.array(p, dtype=np.float64) + eps
    q = np.array(q, dtype=np.float64) + eps
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)

def get_feature_distribution(trips, feature_index, num_classes):
    """
    Compute the frequency distribution of a given feature_index and normalize to a probability distribution.
    Parameters:
      trips: List of trip data (e.g., lists or numpy arrays), each element being an integer value
      feature_index: The index of the feature to be analyzed (starting from 0)
      num_classes: Number of classes for this feature
    Returns:
      1D numpy array of length num_classes representing the probability distribution
    """
    counts = np.zeros(num_classes)
    for trip in trips:
        try:
            val = int(round(float(trip[feature_index])))
        except (ValueError, TypeError):
            continue
        if 0 <= val < num_classes:
            counts[val] += 1
    if counts.sum() == 0:
        return counts
    return counts / counts.sum()

def evaluate_single_feature_kl(truth_trips, generated_trips, feature_index, num_classes):
    """
    Compute KL divergence between ground truth and generated data for a single feature.
    """
    p = get_feature_distribution(truth_trips, feature_index, num_classes)
    q = get_feature_distribution(generated_trips, feature_index, num_classes)
    kl = compute_kl_divergence(p, q)
    return kl

def evaluate_all_features_kl(truth_trips, generated_trips, features_info):
    """
    Compute KL divergence for all features individually and return a dictionary 
    where keys are feature names and values are KL divergence values.
    Parameters:
      features_info: List of dictionaries, each containing "name" (feature name) and "num_classes" (number of classes);
                     The order must match the order of features in the trip data.
    """
    divergences = {}
    for idx, feat in enumerate(features_info):
        name = feat["name"]
        num_classes = feat["num_classes"]
        kl = evaluate_single_feature_kl(truth_trips, generated_trips, idx, num_classes)
        divergences[name] = kl
    return divergences

def get_joint_distribution(trips):
    """
    Compute joint distribution, result is a dictionary with keys as tuples of trip data, and values as probabilities.
    """
    joint_counts = {}
    total = len(trips)
    for trip in trips:
        key = tuple(trip)
        joint_counts[key] = joint_counts.get(key, 0) + 1
    # Normalize
    for key in joint_counts:
        joint_counts[key] /= total
    return joint_counts

def evaluate_joint_kl(truth_trips, generated_trips, smoothing=1e-10):
    """
    Compute KL divergence for joint distribution using union of keys from both distributions and apply smoothing.
    """
    truth_dist = get_joint_distribution(truth_trips)
    gen_dist = get_joint_distribution(generated_trips)
    # Get all keys from the union of the two distributions
    keys = set(truth_dist.keys()).union(set(gen_dist.keys()))
    truth_probs = []
    gen_probs = []
    for key in keys:
        truth_prob = truth_dist.get(key, 0.0) + smoothing
        gen_prob = gen_dist.get(key, 0.0) + smoothing
        truth_probs.append(truth_prob)
        gen_probs.append(gen_prob)
    truth_probs = np.array(truth_probs)
    gen_probs = np.array(gen_probs)
    truth_probs /= truth_probs.sum()
    gen_probs /= gen_probs.sum()
    return compute_kl_divergence(truth_probs, gen_probs)

def evaluate_joint_js_divergence(truth_trips, generated_trips, smoothing=1e-10):
    """
    Compute Jensen-Shannon Divergence for joint distribution with smoothing.
    """
    truth_dist = get_joint_distribution(truth_trips)
    gen_dist = get_joint_distribution(generated_trips)
    keys = set(truth_dist.keys()).union(set(gen_dist.keys()))
    truth_probs = []
    gen_probs = []
    for key in keys:
        truth_prob = truth_dist.get(key, 0.0) + smoothing
        gen_prob = gen_dist.get(key, 0.0) + smoothing
        truth_probs.append(truth_prob)
        gen_probs.append(gen_prob)
    truth_probs = np.array(truth_probs)
    gen_probs = np.array(gen_probs)
    truth_probs /= truth_probs.sum()
    gen_probs /= gen_probs.sum()
    return compute_js_divergence(truth_probs, gen_probs)

def evaluate_total_variation_distance(truth_trips, generated_trips):
    """
    Compute Total Variation Distance between ground truth and generated joint distributions,
    TVD = 0.5 * sum(|p - q|)
    """
    truth_dist = get_joint_distribution(truth_trips)
    gen_dist = get_joint_distribution(generated_trips)
    keys = set(truth_dist.keys()).union(set(gen_dist.keys()))
    total_diff = 0.0
    for key in keys:
        p = truth_dist.get(key, 0.0)
        q = gen_dist.get(key, 0.0)
        total_diff += abs(p - q)
    return 0.5 * total_diff

def evaluate_single_feature_jsd(truth_trips, generated_trips, feature_index, num_classes):
    """
    Compute JSD between ground truth and generated data for a single feature.
    """
    p = get_feature_distribution(truth_trips, feature_index, num_classes)
    q = get_feature_distribution(generated_trips, feature_index, num_classes)
    return compute_js_divergence(p, q)

def evaluate_all_features_jsd(truth_trips, generated_trips, features_info):
    """
    Compute JSD for all features individually and return a dictionary.
    """
    divergences = {}
    for idx, feat in enumerate(features_info):
        name = feat["name"]
        num_classes = feat["num_classes"]
        jsd = evaluate_single_feature_jsd(truth_trips, generated_trips, idx, num_classes)
        divergences[name] = jsd
    return divergences

def _build_samples_dataframe(samples, cond_info, features_info):
    """
    Convert sampled records from train_utils.sample_trip format into a flat DataFrame.
    Each sample item should be:
      {"condition": [...], "trip": [...]}
    """
    if samples is None or cond_info is None:
        return None

    cond_cols = [c["name"] if isinstance(c, dict) else str(c) for c in cond_info]
    trip_cols = [f["name"] if isinstance(f, dict) else str(f) for f in features_info]

    rows = []
    for item in samples:
        cond = item.get("condition", None)
        trip = item.get("trip", None)
        if cond is None or trip is None:
            continue
        if len(cond) != len(cond_cols) or len(trip) != len(trip_cols):
            continue
        row = {}
        for i, col in enumerate(cond_cols):
            row[col] = cond[i]
        for i, col in enumerate(trip_cols):
            row[col] = trip[i]
        rows.append(row)

    if len(rows) == 0:
        return None
    return pd.DataFrame(rows)

def _sanitize_int_columns(df, columns):
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).round().astype(int)
    return out

def evaluate_logical_validity_rate(generated_df):
    """
    Compute Logical Validity Rate (LVR).

    Invalid if any rule is violated:
      1) Demographics-Mode mismatch: underage driving private car.
      2) Activity-Mode mismatch.
      3) Activity-Location mismatch.
    """
    needed_cols = ["age_code", "act_num", "mode_num", "start_type"]
    if generated_df is None or any(c not in generated_df.columns for c in needed_cols):
        return {
            "logical_validity_rate": None,
            "n_total": 0,
            "n_valid": 0,
            "n_invalid": 0,
            "invalid_rule_breakdown": {
                "demographics_mode_mismatch": 0,
                "activity_mode_mismatch": 0,
                "activity_location_mismatch": 0
            }
        }

    df = _sanitize_int_columns(generated_df, needed_cols)

    age = df["age_code"]
    act = df["act_num"]
    mode = df["mode_num"]
    start_type = df["start_type"]

    # age_code is 0-based in this project (0~12), so under 18 -> codes <= 1.
    underage_driving = (age <= 1) & (mode == 5)

    stationary_acts = {0, 2, 3, 4, 5, 6, 8}
    moving_modes = {3, 4, 5, 6, 7, 8}
    stationary_modes = {0, 1, 2}
    activity_mode_mismatch = ((act == 1) & mode.isin(stationary_modes)) | (
        act.isin(stationary_acts) & mode.isin(moving_modes)
    )

    # start_type is 0-based: 0=home,1=workplace,2=school,3/4=other buckets
    activity_location_mismatch = (
        ((act == 0) & start_type.isin([1, 2])) |
        ((act == 2) & start_type.isin([0, 2])) |
        ((act == 3) & start_type.isin([0, 1]))
    )

    invalid = underage_driving | activity_mode_mismatch | activity_location_mismatch

    n_total = int(len(df))
    n_invalid = int(invalid.sum())
    n_valid = n_total - n_invalid
    lvr = 1.0 - (n_invalid / max(n_total, 1))

    return {
        "logical_validity_rate": float(lvr),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_rule_breakdown": {
            "demographics_mode_mismatch": int(underage_driving.sum()),
            "activity_mode_mismatch": int(activity_mode_mismatch.sum()),
            "activity_location_mismatch": int(activity_location_mismatch.sum())
        }
    }

def evaluate_tstr_predictive_accuracy(
    synthetic_df,
    train_real_df,
    test_real_df,
    cond_info,
    features_info,
    random_state=42
):
    """
    TSTR/TRTR utility evaluation:
      target: mode_num
      model: RandomForest
      metrics: macro F1 and accuracy
    """
    target_col = "mode_num"
    all_cols = [c["name"] for c in cond_info] + [f["name"] for f in features_info]

    if synthetic_df is None or train_real_df is None or test_real_df is None:
        return {
            "tstr_macro_f1": None,
            "trtr_macro_f1": None,
            "tstr_accuracy": None,
            "trtr_accuracy": None,
            "tstr_trtr_f1_ratio": None
        }

    required = set(all_cols)
    if not required.issubset(set(synthetic_df.columns)):
        return {
            "tstr_macro_f1": None,
            "trtr_macro_f1": None,
            "tstr_accuracy": None,
            "trtr_accuracy": None,
            "tstr_trtr_f1_ratio": None
        }
    if not required.issubset(set(train_real_df.columns)) or not required.issubset(set(test_real_df.columns)):
        return {
            "tstr_macro_f1": None,
            "trtr_macro_f1": None,
            "tstr_accuracy": None,
            "trtr_accuracy": None,
            "tstr_trtr_f1_ratio": None
        }

    feature_cols = [c for c in all_cols if c != target_col]

    syn = _sanitize_int_columns(synthetic_df, all_cols)
    tr = _sanitize_int_columns(train_real_df, all_cols)
    te = _sanitize_int_columns(test_real_df, all_cols)

    X_syn, y_syn = syn[feature_cols], syn[target_col]
    X_tr, y_tr = tr[feature_cols], tr[target_col]
    X_te, y_te = te[feature_cols], te[target_col]

    rf_kwargs = {
        "n_estimators": 200,
        "random_state": random_state,
        "n_jobs": 1,
        "class_weight": "balanced_subsample"
    }

    clf_tstr = RandomForestClassifier(**rf_kwargs)
    clf_tstr.fit(X_syn, y_syn)
    pred_tstr = clf_tstr.predict(X_te)

    clf_trtr = RandomForestClassifier(**rf_kwargs)
    clf_trtr.fit(X_tr, y_tr)
    pred_trtr = clf_trtr.predict(X_te)

    tstr_f1 = float(f1_score(y_te, pred_tstr, average="macro"))
    trtr_f1 = float(f1_score(y_te, pred_trtr, average="macro"))
    tstr_acc = float(accuracy_score(y_te, pred_tstr))
    trtr_acc = float(accuracy_score(y_te, pred_trtr))

    return {
        "tstr_macro_f1": tstr_f1,
        "trtr_macro_f1": trtr_f1,
        "tstr_accuracy": tstr_acc,
        "trtr_accuracy": trtr_acc,
        "tstr_trtr_f1_ratio": float(tstr_f1 / max(trtr_f1, 1e-12))
    }

def evaluate_generated_trips(
    truth_trips,
    generated_trips,
    features_info,
    generated_samples=None,
    cond_info=None,
    generated_df=None,
    train_real_df=None,
    test_real_df=None,
    random_state=42
):
    """
    Evaluate generated trip data by computing JSD metrics (aligned with the paper).
    
    Metrics:
      1. JSD for individual features (Single JSD)
      2. JSD for joint distribution (Joint JSD)

    Parameters:
      truth_trips: List of ground truth trip data
      generated_trips: List of generated trip data
      features_info: List of feature config dictionaries

    Returns:
      A dictionary containing values for JSD metrics.
    """
    # 1. JSD for individual features (替代原来的 Single KL)
    single_feature_jsd = evaluate_all_features_jsd(truth_trips, generated_trips, features_info)
    
    # 2. Jensen-Shannon Divergence for joint distribution (Joint JSD)
    joint_js = evaluate_joint_js_divergence(truth_trips, generated_trips)
    
    # 如果你还需要 TVD 或 Joint KL 作为参考，可以保留，但根据你的要求，这里只返回 JSD
    # joint_kl = evaluate_joint_kl(truth_trips, generated_trips)
    # tvd = evaluate_total_variation_distance(truth_trips, generated_trips)
    
    results = {
        "single_feature_jsd": single_feature_jsd,
        "joint_js": joint_js
    }

    # LVR and TSTR require condition + trip columns.
    # Priority: explicit generated_df; fallback to generated_samples list.
    if generated_df is None:
        generated_df = _build_samples_dataframe(generated_samples, cond_info, features_info)

    lvr = evaluate_logical_validity_rate(generated_df)
    results.update(lvr)

    if cond_info is not None:
        tstr = evaluate_tstr_predictive_accuracy(
            synthetic_df=generated_df,
            train_real_df=train_real_df,
            test_real_df=test_real_df,
            cond_info=cond_info,
            features_info=features_info,
            random_state=random_state
        )
        results.update(tstr)

    return results
