import numpy as np

def build_alpha_halfspaces(centers, t, weights=None, alpha=1.0):

    centers = np.asarray(centers, float)

    if weights is None:
        weights = np.ones(centers.shape[1], dtype=float) / centers.shape[1]
    #end-if

    weights = np.asarray(weights, float)

    mt = centers[t]
    halfspaces = []

    for j in range(centers.shape[0]):
        if j == t:
            continue

        mj = centers[j]

        w = weights * (mj - mt)

        b_full = 0.5 * (
            np.sum(weights * mj ** 2)
            - np.sum(weights * mt ** 2)
        )

        b_alpha = (1.0 - alpha) * np.dot(w, mt) + alpha * b_full

        halfspaces.append((w, float(b_alpha), j))

    return halfspaces

def contraction_scores_for_target_points(
    X,
    centers,
    labels,
    t,
    weights=None,
    eps=1e-12,
):
    """
    For each point assigned to target cluster t, compute the smallest alpha
    needed for that point to remain inside the homothetically contracted
    weighted Voronoi cell of t.

    score near 0 = close to centroid.
    score near 1 = close to full Voronoi boundary.
    """

    X = np.asarray(X, float)
    centers = np.asarray(centers, float)
    labels = np.asarray(labels)

    if weights is None:
        weights = np.ones(centers.shape[1], dtype=float) / centers.shape[1]

    weights = np.asarray(weights, float)

    mt = centers[t]
    X_t = X[labels == t]

    if X_t.shape[0] == 0:
        return np.asarray([], dtype=float)

    full_halfspaces = build_alpha_halfspaces(
        centers=centers,
        t=t,
        weights=weights,
        alpha=1.0,
    )

    scores = []

    for x in X_t:
        point_scores = []

        for w, b, _j in full_halfspaces:
            centre_value = float(np.dot(w, mt))
            point_value = float(np.dot(w, x))
            denom = float(b - centre_value)

            if denom <= eps:
                continue

            score = (point_value - centre_value) / denom
            point_scores.append(score)

        if point_scores:
            scores.append(max(point_scores))
        else:
            scores.append(0.0)

    scores = np.asarray(scores, dtype=float)

    return np.clip(scores, 0.0, 1.0)


def calculate_alpha_for_target_cluster(
    X,
    centers,
    labels,
    t,
    weights=None,
    retain_fraction=0.95,
    min_alpha=0.01,
):
    """
    Calculate alpha from the empirical distribution of target-cluster points.

    retain_fraction=1.00 gives the tightest contraction retaining all target points.
    retain_fraction=0.95 gives a slightly more robust contraction by allowing
    extreme boundary/outlier points to fall outside.
    """

    scores = contraction_scores_for_target_points(
        X=X,
        centers=centers,
        labels=labels,
        t=t,
        weights=weights,
    )

    if len(scores) == 0:
        return 1.0, scores

    retain_fraction = float(np.clip(retain_fraction, 0.0, 1.0))

    sorted_scores = np.sort(scores)

    if retain_fraction >= 1.0:
        alpha = sorted_scores[-1]
    else:
        idx = int(np.ceil(retain_fraction * len(sorted_scores))) - 1
        idx = int(np.clip(idx, 0, len(sorted_scores) - 1))
        alpha = sorted_scores[idx]

    alpha = float(np.clip(alpha, min_alpha, 1.0))

    return alpha, scores


def alpha_summary_for_all_clusters(
    X,
    centers,
    labels,
    weights=None,
    retain_fraction=1.0,
    min_alpha=0.0,
):
    """
    Calculate empirical contraction alpha values for every cluster.

    For retain_fraction=1.0, this gives the smallest alpha required
    to retain 100% of the data points assigned to each centroid.
    """

    X = np.asarray(X, float)
    centers = np.asarray(centers, float)
    labels = np.asarray(labels)

    if weights is None:
        weights = np.ones(centers.shape[1], dtype=float) / centers.shape[1]
    #end-if

    weights = np.asarray(weights, float)

    rows = []

    for t in range(centers.shape[0]):

        scores = contraction_scores_for_target_points(
            X=X,
            centers=centers,
            labels=labels,
            t=t,
            weights=weights,
        )

        n_target = len(scores)

        if n_target == 0:

            rows.append(
                {
                    "cluster": int(t),
                    "n_points": 0,
                    "alpha_100": np.nan,
                    "retained_points": 0,
                    "retained_pct": 0.0,
                    "min_score": np.nan,
                    "median_score": np.nan,
                    "max_score": np.nan,
                }
            )

            continue

        #end-if

        retain_fraction_clipped = float(
            np.clip(retain_fraction, 0.0, 1.0)
        )

        sorted_scores = np.sort(scores)

        if retain_fraction_clipped >= 1.0:

            alpha = sorted_scores[-1]

        else:

            idx = int(np.ceil(retain_fraction_clipped * n_target)) - 1
            idx = int(np.clip(idx, 0, n_target - 1))
            alpha = sorted_scores[idx]

        #end-if

        alpha = float(np.clip(alpha, min_alpha, 1.0))

        retained_points = int(np.sum(scores <= alpha + 1e-12))
        retained_pct = 100.0 * retained_points / n_target

        rows.append(
            {
                "cluster": int(t),
                "n_points": int(n_target),
                "alpha_100": alpha,
                "retained_points": retained_points,
                "retained_pct": retained_pct,
                "min_score": float(np.min(scores)),
                "median_score": float(np.median(scores)),
                "max_score": float(np.max(scores)),
            }
        )

    #end-for

    return rows