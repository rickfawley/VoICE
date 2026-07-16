import numpy as np
import streamlit as st

from sklearn.cluster import KMeans


@st.cache_resource
def fit_kmeans(X, n_clusters, seed, kmeans_n_init):
    return KMeans(
        n_clusters=n_clusters,
        n_init=kmeans_n_init,
        random_state=seed,
    ).fit(X)


def assign_kmeans_label(x, centroids):
    x = np.asarray(x, float)
    centers = np.asarray(centroids, float)

    squared_distances = np.sum((centers - x[None, :]) ** 2, axis=1)

    return int(np.argmin(squared_distances))


def squared_euclidean(a, b):
    delta = np.asarray(a, float) - np.asarray(b, float)
    return float(np.dot(delta, delta))