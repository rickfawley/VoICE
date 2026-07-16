import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
#from sklearn.metrics import adjusted_rand_score
from src.evaluation import safe_adjusted_rand_index

from src.config import BASE_COLOURS

def display_cluster_solution(
    X_cluster,
    X_view,
    feature_names,
    cluster_result,
    scaler,
    view_space,
    seed=0,
    a_idx=0,
    cf_mode=None,
    ranking_weights=None,
    cf_weights=None,
    y_true=None,
):
    """
    Displays the clustering solution (table + PCA Visualisation) inside
    a collapsed-by-default Streamlit expander.

    Returns
    -------
    df : pd.DataFrame
        A per-sample table including cluster assignment and distances.
    """

    #st.subheader("Input Data")
    ##st.dataframe(pd.DataFrame(X_view, columns=feature_names).head(10))
    #st.dataframe(pd.DataFrame(X_view, columns=feature_names))

    X_cluster = np.asarray(X_cluster, float)
    X_view = np.asarray(X_view, float)
    labels = cluster_result.labels
    centers = np.asarray(cluster_result.centroids, float)
    weights = np.asarray(cluster_result.weights, float)

    ari = safe_adjusted_rand_index(labels, y_true)

    if ranking_weights is None:
        ranking_weights = weights

    if cf_weights is None:
        cf_weights = weights

    ranking_weights = np.asarray(ranking_weights, float)
    cf_weights = np.asarray(cf_weights, float)

    n_clusters = centers.shape[0]
    cluster_colors = BASE_COLOURS[:n_clusters]


    # Distance to assigned centroid in the space X lives in
    d2_scaled = np.sum(
        weights[None, :] * (X_cluster - centers[labels]) ** 2,
        axis=1,
    )
    scaled_dist = np.sqrt(d2_scaled)

    # PCA for 2D Visualisation
    pca = PCA(n_components=2, random_state=seed)
    if view_space == "Scaled Metric Space":
        X_plot = X_cluster
        C_plot = centers
    else:
        X_plot = X_view
        C_plot = scaler.inverse_transform(centers)

    X_2d = pca.fit_transform(X_plot)
    C_2d = pca.transform(C_plot)

    # --- Mark Factual Point a on the PCA plot ---
    if 0 <= a_idx < X_2d.shape[0]:
        ax_x, ax_y = float(X_2d[a_idx, 0]), float(X_2d[a_idx, 1])
    else:
        ax_x, ax_y = None, None

    # Build table in the space used for clustering/optimization
    df = pd.DataFrame(X_view, columns=feature_names)
    df.insert(0, "cluster", labels.astype(int))
    df.insert(1, "scaled_dist_to_centroid", scaled_dist)

    df["method"] = cluster_result.method

    if cf_mode is not None:
        df["cf_mode"] = cf_mode

    for i, name in enumerate(feature_names):
        df[f"cluster_weight_{name}"] = weights[i]
        df[f"ranking_weight_{name}"] = ranking_weights[i]
        df[f"cf_weight_{name}"] = cf_weights[i]
    #end-for

    df["pca_1"] = X_2d[:, 0]
    df["pca_2"] = X_2d[:, 1]

    # If clustering in scaled space, optionally add original units for interpretability
    if view_space == "Original Metric Space" and scaler is not None:
        X_orig = X_view
        C_orig = scaler.inverse_transform(centers)
        d2_orig = np.sum((X_orig - C_orig[labels]) ** 2, axis=1)
        df["orig_dist_to_centroid"] = np.sqrt(d2_orig)
    #end-if

    # Plot (PCA projection)
    fig, ax = plt.subplots()

    # Points coloured by cluster (explicit colours)
    for i in range(n_clusters):
        mask = (labels == i)
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=25,
            alpha=0.6,
            color=cluster_colors[i],
            label=f"Cluster {i}",
        )

    # Centroids coloured by cluster (same colour)
    for i in range(n_clusters):
        ax.scatter(
            C_2d[i, 0],
            C_2d[i, 1],
            marker="X",
            s=200,
            edgecolor="black",
            linewidth=1.2,
            color=cluster_colors[i],
            zorder=5,
        )

    if ax_x is not None:
        ax.scatter([ax_x], [ax_y], marker="x", s=50, color="black", linewidth=3, zorder=10)
        ax.text(ax_x, ax_y, f"Factual Point {a_idx + 1}", color="black", fontsize=10, va="bottom")
    #end-if

    ax.legend(loc="upper right", frameon=True)

    ax.set_title(
        f"{cluster_result.method} clustering — PCA projection ({view_space})"
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display 

    st.subheader("Clustering Results")
    st.dataframe(df)

    if np.isfinite(ari):
        st.metric("Adjusted Rand Index (ARI)", f"{ari:.4f}")
    else:
        st.caption(
            "ARI unavailable: no ground-truth labels were provided, "
            "or labels are incompatible with the clustering result."
        )
    #end-if

    # --- Feature weight chart ---
    weight_df = pd.DataFrame({
        "feature": feature_names,
        "clustering_weight": weights,
        "ranking_weight": ranking_weights,
        "counterfactual_weight": cf_weights,
    }).sort_values("ranking_weight", ascending=False)

    st.subheader("Feature Weights")

    if cf_mode is not None:
        st.caption(
            f"Counterfactual mode: {cf_mode}. "
            "Clustering, ranking, and counterfactual weights are shown separately."
        )

    fig_w, ax_w = plt.subplots(figsize=(9, 4))

    x = np.arange(len(weight_df))
    bar_width = 0.25

    ax_w.bar(
        x - bar_width,
        weight_df["clustering_weight"],
        width=bar_width,
        label="clustering",
    )

    ax_w.bar(
        x,
        weight_df["ranking_weight"],
        width=bar_width,
        label="ranking",
    )

    ax_w.bar(
        x + bar_width,
        weight_df["counterfactual_weight"],
        width=bar_width,
        label="counterfactual",
    )

    ax_w.set_xticks(x)
    ax_w.set_xticklabels(weight_df["feature"])
    ax_w.set_xlabel("Feature")
    ax_w.set_ylabel("Weight")
    ax_w.set_title(f"Feature weights by role — {cf_mode or cluster_result.method}")
    ax_w.tick_params(axis="x", rotation=90, labelsize=8)
    ax_w.legend()

    st.pyplot(fig_w)
    st.dataframe(weight_df, use_container_width=True)





    st.subheader("PCA Cluster Visualisation")
    st.pyplot(fig)

    return df
