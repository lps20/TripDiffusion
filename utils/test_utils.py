import numpy as np
import pandas as pd

from utils.mnl_mode_choice import evaluate_mnl_mode_choice_validation

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

def compute_emd_ordinal(p, q):
    """Earth Mover Distance (Wasserstein-1) for ordinal variables."""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def normalize_jsd_entropy(jsd, num_classes):
    """Entropy-normalized JSD: JSD / log(K), comparable across cardinalities."""
    if num_classes <= 1:
        return 0.0
    denom = float(np.log(num_classes))
    return float(jsd / denom) if denom > 0 else 0.0


def normalize_jsd_vs_uniform(jsd, truth_trips, feature_index, num_classes):
    """
    Min-max style normalization against a naive uniform baseline and ideal JSD=0:
      JSD_norm = JSD(truth, gen) / JSD(truth, uniform)
    """
    p = get_feature_distribution(truth_trips, feature_index, num_classes)
    uniform = np.ones(num_classes, dtype=np.float64) / num_classes
    jsd_uniform = compute_js_divergence(p, uniform)
    if jsd_uniform <= 1e-12:
        return 0.0
    return float(jsd / jsd_uniform)


def evaluate_single_feature_emd(truth_trips, generated_trips, feature_index, num_classes):
    """Compute EMD between ground truth and generated data for one ordinal feature."""
    p = get_feature_distribution(truth_trips, feature_index, num_classes)
    q = get_feature_distribution(generated_trips, feature_index, num_classes)
    return compute_emd_ordinal(p, q)


def evaluate_all_ordinal_features_emd(truth_trips, generated_trips, features_info):
    """Compute EMD for all ordinal features."""
    emds = {}
    for idx, feat in enumerate(features_info):
        if feat.get("type") != "ordinal":
            continue
        name = feat["name"]
        num_classes = feat["num_classes"]
        emds[name] = evaluate_single_feature_emd(truth_trips, generated_trips, idx, num_classes)
    return emds


def evaluate_all_features_jsd_normalized(truth_trips, generated_trips, features_info):
    """Return entropy-normalized and uniform-baseline-normalized JSD per feature."""
    entropy_norm = {}
    uniform_norm = {}
    for idx, feat in enumerate(features_info):
        name = feat["name"]
        num_classes = feat["num_classes"]
        jsd = evaluate_single_feature_jsd(truth_trips, generated_trips, idx, num_classes)
        entropy_norm[name] = normalize_jsd_entropy(jsd, num_classes)
        uniform_norm[name] = normalize_jsd_vs_uniform(jsd, truth_trips, idx, num_classes)
    return entropy_norm, uniform_norm


def format_metric_cell(value, sci_threshold_low=1e-2, sci_threshold_high=1e2):
    """
    Uniform numeric formatting for comparison tables (e.g., Table 3).

    Uses fixed decimal precision by default; switches to scientific notation
    only when magnitude is very small or very large, with one consistent rule.
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(v):
        return "—"
    if v == 0.0:
        return "0.0000"
    av = abs(v)
    if av < sci_threshold_low or av >= sci_threshold_high:
        return f"{v:.2e}"
    return f"{v:.4f}"


def format_metrics_table_row(metrics, prefix_map=None):
    """
    Format one metrics dict into display strings for tabular export.

    prefix_map: optional {metric_key: output_prefix} for nested dict fields.
    """
    formatted = {}
    scalar_keys = [
        "joint_js",
        "joint_js_normalized",
        "mean_marginal_jsd",
        "mean_single_feature_jsd_normalized",
        "mean_ordinal_emd",
        "logical_validity_rate",
        "mnl_behavioral_similarity",
        "mnl_coef_cosine_similarity",
        "mnl_ame_cosine_similarity",
        "mnl_elasticity_cosine_similarity",
        "mnl_test_logloss_ratio",
        "tstr_trtr_f1_ratio",
    ]
    for key in scalar_keys:
        if key in metrics:
            formatted[key] = format_metric_cell(metrics[key])

    nested_specs = [
        ("single_feature_jsd", "jsd"),
        ("single_feature_jsd_normalized", "jsd_norm"),
        ("single_feature_jsd_vs_uniform", "jsd_uni"),
        ("single_feature_emd", "emd"),
    ]
    for metric_key, default_prefix in nested_specs:
        nested = metrics.get(metric_key)
        if not isinstance(nested, dict):
            continue
        prefix = (prefix_map or {}).get(metric_key, default_prefix)
        for feat_name, value in nested.items():
            formatted[f"{prefix}_{feat_name}"] = format_metric_cell(value)

    return formatted


def flatten_evaluation_metrics(
    model_name,
    metrics,
    extra_fields=None,
    include_formatted=False,
):
    """Flatten nested evaluation metrics into one CSV-friendly row."""
    row = {"model": model_name}
    if extra_fields:
        row.update(extra_fields)

    scalar_keys = [
        "joint_js",
        "joint_js_normalized",
        "mean_marginal_jsd",
        "mean_single_feature_jsd_normalized",
        "mean_ordinal_emd",
        "logical_validity_rate",
        "mnl_behavioral_similarity",
        "mnl_coef_cosine_similarity",
        "mnl_ame_cosine_similarity",
        "mnl_elasticity_cosine_similarity",
        "mnl_test_logloss_ratio",
        "tstr_trtr_f1_ratio",
    ]
    for key in scalar_keys:
        if key in metrics and metrics[key] is not None:
            row[key] = float(metrics[key])

    for feat_name, value in metrics.get("single_feature_jsd", {}).items():
        row[f"jsd_{feat_name}"] = float(value)
    for feat_name, value in metrics.get("single_feature_jsd_normalized", {}).items():
        row[f"jsd_norm_{feat_name}"] = float(value)
    for feat_name, value in metrics.get("single_feature_jsd_vs_uniform", {}).items():
        row[f"jsd_uni_{feat_name}"] = float(value)
    for feat_name, value in metrics.get("single_feature_emd", {}).items():
        row[f"emd_{feat_name}"] = float(value)

    for rule, value in metrics.get("invalid_rule_breakdown", {}).items():
        row[f"lvr_{rule}"] = value

    for count_key in ("n_total", "n_valid", "n_invalid"):
        if count_key in metrics:
            row[count_key] = metrics[count_key]

    if include_formatted:
        formatted = format_metrics_table_row(metrics)
        for key, value in formatted.items():
            row[f"fmt_{key}"] = value

    return row


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
    Behavioral TSTR validation via a simple MNL mode-choice model.

    Estimates multinomial logit on synthetic vs. real travel-mode choices and
    compares utility parameters, average marginal effects, and implied
    elasticities on held-out real trips.
    """
    del cond_info, features_info
    return evaluate_mnl_mode_choice_validation(
        synthetic_df=synthetic_df,
        train_real_df=train_real_df,
        test_real_df=test_real_df,
        random_state=random_state,
    )

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
    Evaluate generated trip data.

    Metrics:
      1. Raw JSD for individual features and joint distribution
      2. Entropy-normalized JSD (JSD / log K) and uniform-baseline-normalized JSD
      3. Earth Mover's Distance (EMD) for ordinal features
      4. Logical Validity Rate (LVR) and MNL mode-choice behavioral TSTR
    """
    single_feature_jsd = evaluate_all_features_jsd(truth_trips, generated_trips, features_info)
    single_feature_jsd_normalized, single_feature_jsd_vs_uniform = evaluate_all_features_jsd_normalized(
        truth_trips, generated_trips, features_info
    )
    single_feature_emd = evaluate_all_ordinal_features_emd(truth_trips, generated_trips, features_info)

    joint_js = evaluate_joint_js_divergence(truth_trips, generated_trips)
    joint_js_normalized = float(joint_js / np.log(2.0)) if joint_js is not None else None

    results = {
        "single_feature_jsd": single_feature_jsd,
        "single_feature_jsd_normalized": single_feature_jsd_normalized,
        "single_feature_jsd_vs_uniform": single_feature_jsd_vs_uniform,
        "single_feature_emd": single_feature_emd,
        "joint_js": joint_js,
        "joint_js_normalized": joint_js_normalized,
    }

    if single_feature_jsd_normalized:
        results["mean_single_feature_jsd_normalized"] = float(
            np.mean(list(single_feature_jsd_normalized.values()))
        )
    if single_feature_jsd:
        results["mean_marginal_jsd"] = float(np.mean(list(single_feature_jsd.values())))
    if single_feature_emd:
        results["mean_ordinal_emd"] = float(np.mean(list(single_feature_emd.values())))

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
