import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score


def safe_adjusted_rand_index(labels, y_true):
    """
    Compute ARI when ground-truth labels are available and compatible.

    Returns
    -------
    float
        ARI score when available.
    np.nan
        If y_true is unavailable or incompatible.
    """

    if y_true is None:
        return np.nan

    labels = np.asarray(labels)
    y_true = np.asarray(y_true)

    if labels.ndim != 1 or y_true.ndim != 1:
        return np.nan

    if len(labels) != len(y_true):
        return np.nan

    if len(labels) == 0:
        return np.nan

    return float(adjusted_rand_score(y_true, labels))


def build_cluster_quality_table(cf_modes, mode_config_fn, y_true):
    """
    Build a compact model-quality table for all comparison modes.
    """

    rows = []

    for cf_mode in cf_modes:

        (
            cluster_result,
            ranking_weights,
            cf_weights,
            ranked_features,
        ) = mode_config_fn(cf_mode)

        rows.append(
            {
                "cf_mode": cf_mode,
                "clustering_method": cluster_result.method,
                "inertia": float(cluster_result.inertia),
                "ARI": safe_adjusted_rand_index(
                    cluster_result.labels,
                    y_true,
                ),
                "n_clusters_found": int(
                    len(np.unique(cluster_result.labels))
                ),
            }
        )

    return pd.DataFrame(rows)