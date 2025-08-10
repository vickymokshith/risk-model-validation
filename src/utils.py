import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions.

    Args:
        expected: Reference distribution values (numpy array).
        actual: Comparison distribution values (numpy array).
        buckets: Number of buckets to divide the distributions into.

    Returns:
        The PSI value. Lower values (<0.1) indicate minimal population shift; values between 0.1 and 0.25 indicate moderate shift; values above 0.25 indicate large shifts.
    """
    # Ensure inputs are numpy arrays
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    # Define breakpoints from 0 to 1 at equal intervals
    breakpoints = np.linspace(0, 1, buckets + 1)
    # Compute quantiles on the expected distribution to define bucket edges
    quantiles = np.quantile(expected, breakpoints)
    # Assign bins based on quantiles
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    # Convert counts to proportions
    expected_dist = expected_counts / expected_counts.sum()
    actual_dist = actual_counts / actual_counts.sum()
    # Replace zeros with a small value to avoid divide-by-zero and log issues
    actual_dist = np.where(actual_dist == 0, 1e-6, actual_dist)
    expected_dist = np.where(expected_dist == 0, 1e-6, expected_dist)
    psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)
    return np.sum(psi_values)


def ks_stat(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Kolmogorov–Smirnov (KS) statistic for binary classification.

    Args:
        scores: Model scores or probabilities.
        labels: Binary labels (0 for negative class, 1 for positive class).

    Returns:
        The KS statistic between 0 and 1. Higher values indicate better separation between positive and negative distributions.
    """
    # Ensure numpy arrays
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    # Sort scores and align labels accordingly
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]
    # Compute cumulative distributions
    cum_pos = np.cumsum(sorted_labels) / sorted_labels.sum()
    cum_neg = np.cumsum(1 - sorted_labels) / (len(sorted_labels) - sorted_labels.sum())
    ks_value = np.max(np.abs(cum_pos - cum_neg))
    return ks_value
