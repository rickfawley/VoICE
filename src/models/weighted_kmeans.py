from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ClusterResult:
    labels: np.ndarray
    centroids: np.ndarray
    weights: np.ndarray
    inertia: float
    method: str


def weighted_squared_distance(a, b, weights):
    delta = np.asarray(a, float) - np.asarray(b, float)
    weights = np.asarray(weights, float)
    return float(np.sum(weights * delta ** 2))


def assign_weighted_label(x, centroids, weights):
    x = np.asarray(x, float)
    centroids = np.asarray(centroids, float)
    weights = np.asarray(weights, float)

    dists = np.sum(
        weights[None, :] * (centroids - x[None, :]) ** 2,
        axis=1,
    )
    return int(np.argmin(dists))


def fit_voice_kmeans(
    X,
    n_clusters,
    seed=42,
    n_init=10,
    max_iter=300,
    tol=1e-4,
):
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
    ).fit(X)

    d = X.shape[1]
    weights = np.ones(d, dtype=float) / d

    return ClusterResult(
        labels=model.labels_,
        centroids=model.cluster_centers_,
        weights=weights,
        inertia=float(model.inertia_),
        method="k-means",
    )


def _weighted_distances(X, centroids, weights):
    return np.sum(
        weights[None, None, :] * (X[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )


def _single_shark_run(
    X,
    n_clusters,
    rng,
    max_iter=300,
    tol=1e-4,
    eps=1e-12,
):
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    weights = np.ones(d, dtype=float) / d

    unique_X = np.unique(X, axis=0)
    if unique_X.shape[0] < n_clusters:
        raise ValueError("Not enough unique observations to initialise clusters.")

    init_idx = rng.choice(unique_X.shape[0], size=n_clusters, replace=False)
    centroids = unique_X[init_idx].astype(float)

    prev_labels = None
    prev_centroids = centroids.copy()

    for _ in range(max_iter):
        distances = _weighted_distances(X, centroids, weights)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()

        for j in range(n_clusters):
            pts = X[labels == j]
            if len(pts) > 0:
                new_centroids[j] = pts.mean(axis=0)

        phi = np.zeros(d, dtype=float)

        for j in range(n_clusters):
            pts = X[labels == j]
            if len(pts) > 0:
                diff = pts - new_centroids[j]
                phi += np.sum(diff ** 2, axis=0)

        inv_phi = 1.0 / np.maximum(phi, eps)
        new_weights = inv_phi / np.sum(inv_phi)

        centroid_shift = np.linalg.norm(new_centroids - prev_centroids)
        same_labels = prev_labels is not None and np.array_equal(labels, prev_labels)

        centroids = new_centroids
        weights = new_weights

        if same_labels or centroid_shift <= tol:
            break

        prev_labels = labels.copy()
        prev_centroids = centroids.copy()

    distances = _weighted_distances(X, centroids, weights)
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(distances[np.arange(n), labels]))

    return labels, centroids, weights, inertia


def fit_shark(
    X,
    n_clusters,
    seed=42,
    n_init=10,
    max_iter=300,
    tol=1e-4,
):
    best = None
    rng_master = np.random.default_rng(seed)

    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        labels, centroids, weights, inertia = _single_shark_run(
            X,
            n_clusters,
            rng,
            max_iter=max_iter,
            tol=tol,
        )

        if len(np.unique(labels)) != n_clusters:
            continue

        candidate = ClusterResult(
            labels=labels,
            centroids=centroids,
            weights=weights,
            inertia=inertia,
            method="SHARK",
        )

        if best is None or candidate.inertia < best.inertia:
            best = candidate

    if best is None:
        raise RuntimeError("SHARK failed to produce a valid clustering.")

    return best


def fit_clustering(
    X,
    method,
    n_clusters,
    seed=42,
    n_init=10,
    max_iter=300,
    tol=1e-4,
):
    if method == "k-means":
        return fit_voice_kmeans(
            X,
            n_clusters,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )

    if method == "SHARK":
        return fit_shark(
            X,
            n_clusters,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )

    raise ValueError(f"Unknown clustering method: {method}")
