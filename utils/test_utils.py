import numpy as np
import pandas as pd

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
        val = trip[feature_index]
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

def evaluate_generated_trips(truth_trips, generated_trips, features_info):
    """
    Evaluate generated trip data by computing the following metrics:
      1. KL divergence for individual features
      2. KL divergence for joint distribution
      3. Jensen-Shannon Divergence for joint distribution
      4. Total Variation Distance for joint distribution

    Parameters:
      truth_trips: List of ground truth trip data, each element is a list or array of features
      generated_trips: List of generated trip data, same format as truth_trips
      features_info: List of feature config dictionaries, each containing "name" and "num_classes"

    Returns:
      A dictionary containing values for all evaluation metrics
    """
    # 1. KL divergence for individual features
    single_feature_kl = evaluate_all_features_kl(truth_trips, generated_trips, features_info)
    
    # 2. KL divergence for joint distribution
    joint_kl = evaluate_joint_kl(truth_trips, generated_trips)
    
    # 3. Jensen-Shannon Divergence for joint distribution
    joint_js = evaluate_joint_js_divergence(truth_trips, generated_trips)
    
    # 4. Total Variation Distance for joint distribution
    tvd = evaluate_total_variation_distance(truth_trips, generated_trips)
    
    return {
        "single_feature_kl": single_feature_kl,
        "joint_kl": joint_kl,
        "joint_js": joint_js,
        "tvd": tvd
    }
