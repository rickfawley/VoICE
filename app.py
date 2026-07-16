# ----------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ----------------------------------------------------------------------
# Local Imports
# ----------------------------------------------------------------------

from src.config import MY_START_POINT

from src.datasets import load_dataset
from src.models.weighted_kmeans import fit_clustering

from src.results_export import (
    export_cluster_solution,
    export_model_level_evaluation_tables,
)

from src.ui import (
    render_documentation_sidebar,
    render_dataset_selector,
    render_experiment_controls,
    render_feature_controls,
)

from src.geometry import (
    calculate_alpha_for_target_cluster,
    contraction_scores_for_target_points,
    alpha_summary_for_all_clusters,
)

from src.views import (
    display_cluster_solution,
    display_bounded_voronoi_cells,
)

from src.evaluation import (
    run_model_level_evaluation,
    run_kmeans_voronoi_vs_bisector_evaluation,
)

# ======================================================================
# MAIN PROCESSING
# ======================================================================

st.set_page_config(page_title="VoICE Explainability Demo", layout="wide")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------

with st.sidebar:

    render_documentation_sidebar()

    dataset_name, default_k = render_dataset_selector()

#end-with

# ----------------------------------------------------------------------
# Load data (both representations)
# ----------------------------------------------------------------------

X_orig_all, X_scaled_all, feature_names, scaler, y_true = load_dataset(dataset_name)

# ----------------------------------------------------------------------
# Get experiment control using sidebar
# --+----1----+----2----+----3----+----4----+----5----+----6----+----7--

with st.sidebar:

    controls = render_experiment_controls(
        dataset_name=dataset_name,
        feature_names=feature_names,
        default_k=default_k,
    )

#end-with

n_clusters        = controls["n_clusters"]
seed              = controls["seed"]
cf_modes          = controls["cf_modes"]
n_init            = controls["n_init"]
max_iter          = controls["max_iter"]
tol               = controls["tol"]
view_space        = controls["view_space"]
create_timestamped_dataset = controls["create_timestamped_dataset"]


exported_solution_paths_container = controls[
    "exported_solution_paths_container"
]

# ------------------------------------------------------------------
# Clustering space (always scaled)
# ------------------------------------------------------------------

X_cluster = X_scaled_all

# ------------------------------------------------------------------
# Display / explanation space
# ------------------------------------------------------------------

if view_space == "Scaled Metric Space":
    X_view = X_scaled_all
    to_orig_fn = lambda x: scaler.inverse_transform(
        np.asarray(x, float).reshape(1, -1)
    ).ravel()
else:
    X_view = X_orig_all
#end-if

# ------------------------------------------------------------------
# Fit models in scaled space only
# ------------------------------------------------------------------

kmeans_result = fit_clustering(
    X_cluster,
    method="k-means",
    n_clusters=n_clusters,
    seed=seed,
    n_init=n_init,
    max_iter=max_iter,
    tol=tol,
)

needs_shark = any(
    mode in [
        "Ranked k-means",
        "Ranked + weighted SHARK",
    ]
    for mode in cf_modes
)

shark_result = None
if needs_shark:
    shark_result = fit_clustering(
        X_cluster,
        method="SHARK",
        n_clusters=n_clusters,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )

d = X_cluster.shape[1]
equal_weights = np.ones(d, dtype=float) / d

# ------------------------------------------------------------------
# Export clustering solutions
# ------------------------------------------------------------------
# Streamlit reruns this script on many UI interactions.
#
# Master exports are safe because they overwrite existing files.
# Timestamped backups are not safe to write on every rerun because
# they create a new file each time.
#
# The export signature below records the settings that actually affect
# the fitted clustering solutions. If the signature has not changed, we
# reuse the previously exported paths instead of writing files again.
# ------------------------------------------------------------------

exported_algorithm_names = ["k-means"]

if shark_result is not None:
    exported_algorithm_names.append("SHARK")
#end-if

export_signature = (
    "cluster_solution_export_v1",
    str(dataset_name),
    tuple(np.asarray(X_cluster).shape),
    tuple(feature_names),
    int(n_clusters),
    int(seed),
    int(n_init),
    int(max_iter),
    float(tol),
    tuple(exported_algorithm_names),
    bool(create_timestamped_dataset),
)

previous_export_signature = st.session_state.get(
    "last_cluster_export_signature"
)

if previous_export_signature != export_signature:

    kmeans_alpha_100_df = pd.DataFrame(
        alpha_summary_for_all_clusters(
            X=X_cluster,
            centers=kmeans_result.centroids,
            labels=kmeans_result.labels,
            weights=equal_weights,
            retain_fraction=1.0,
            min_alpha=0.0,
        )
    )

    kmeans_alpha_100_df.insert(0, "cf_mode", "Unweighted k-means")

    exported_solution_paths = []

    exported_solution_paths.append(
        export_cluster_solution(
            cluster_result=kmeans_result,
            X_cluster=X_cluster,
            feature_names=feature_names,
            y_true=y_true,
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            results_dir=RESULTS_DIR,
            alg_name="k-means",
            create_timestamped_dataset=create_timestamped_dataset,
            alpha_summary_df=kmeans_alpha_100_df,
        )
    )


    shark_alpha_100_df = None

    if shark_result is not None:

        shark_alpha_100_df = pd.DataFrame(
            alpha_summary_for_all_clusters(
                X=X_cluster,
                centers=shark_result.centroids,
                labels=shark_result.labels,
                weights=shark_result.weights,
                retain_fraction=1.0,
                min_alpha=0.0,
            )
        )

        shark_alpha_100_df.insert(0, "cf_mode", "Ranked + weighted SHARK")

    if shark_result is not None:
        exported_solution_paths.append(
            export_cluster_solution(
                cluster_result=shark_result,
                X_cluster=X_cluster,
                feature_names=feature_names,
                y_true=y_true,
                dataset_name=dataset_name,
                n_clusters=n_clusters,
                seed=seed,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                results_dir=RESULTS_DIR,
                alg_name="SHARK",
                create_timestamped_dataset=create_timestamped_dataset,
                alpha_summary_df=shark_alpha_100_df,
            )
        )
    #end-if

    st.session_state["last_cluster_export_signature"] = export_signature

    st.session_state["last_cluster_export_paths"] = [
        str(path)
        for path in exported_solution_paths
    ]

    st.session_state["last_cluster_export_status"] = (
        "Exported clustering solutions for the current settings."
    )

else:

    exported_solution_paths = st.session_state.get(
        "last_cluster_export_paths",
        [],
    )

    st.session_state["last_cluster_export_status"] = (
        "Reusing previously exported clustering solutions; "
        "no new files were created."
    )

#end-if

with exported_solution_paths_container:

    export_status = st.session_state.get(
        "last_cluster_export_status",
        "No clustering solutions exported yet.",
    )

    st.caption(export_status)

    if exported_solution_paths:
        for exported_solution_path in exported_solution_paths:
            st.caption(str(exported_solution_path))
    else:
        st.caption("No clustering solutions exported yet.")
    #end-if

#end-with

def get_mode_config(cf_mode):
    if cf_mode == "Unweighted k-means":
        return kmeans_result, equal_weights, equal_weights, feature_names.copy()

    if cf_mode == "Ranked k-means":
        ranking_weights = shark_result.weights
        ranked_features = [
            feature_names[i]
            for i in np.argsort(-ranking_weights)
        ]
        return kmeans_result, ranking_weights, equal_weights, ranked_features

    if cf_mode == "Ranked + weighted SHARK":
        ranking_weights = shark_result.weights
        ranked_features = [
            feature_names[i]
            for i in np.argsort(-ranking_weights)
        ]
        return shark_result, ranking_weights, shark_result.weights, ranked_features

    raise ValueError(f"Unknown counterfactual comparison mode: {cf_mode}")


def build_alpha_100_table_for_modes(cf_modes):
    rows = []

    for cf_mode in cf_modes:

        (
            mode_cluster_result,
            mode_ranking_weights,
            mode_cf_weights,
            mode_ranked_features,
        ) = get_mode_config(cf_mode)

        mode_rows = alpha_summary_for_all_clusters(
            X=X_cluster,
            centers=mode_cluster_result.centroids,
            labels=mode_cluster_result.labels,
            weights=mode_cf_weights,
            retain_fraction=1.0,
            min_alpha=0.0,
        )

        for row in mode_rows:
            row["cf_mode"] = cf_mode
            rows.append(row)

    return pd.DataFrame(rows)

def get_smallest_cluster_size(labels):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) == 0:
        return 1

    return int(
        min(
            np.sum(labels == label)
            for label in unique_labels
        )
    )


def build_comparative_sampling_summary(
    *,
    labels,
    sampling_mode,
    sample_percentage,
    sample_count_per_cluster,
    target_cluster_selection_mode,
):
    labels = np.asarray(labels)
    cluster_ids = sorted(np.unique(labels).astype(int).tolist())
    n_clusters = len(cluster_ids)

    rows = []

    for cluster_id in cluster_ids:
        cluster_size = int(np.sum(labels == cluster_id))

        if sampling_mode == "Percentage of Each Cluster Sampled":
            selected_count = int(np.ceil(cluster_size * sample_percentage / 100.0))
            selected_count = max(1, min(selected_count, cluster_size))
        else:
            # If the requested count is greater than the size of the
            # smallest cluster, the effective count used for every cluster
            # will be capped at the size of the smallest cluster. This keeps
            # the factual sampling balanced across clusters.
            smallest_cluster_size = get_smallest_cluster_size(labels)
            effective_sample_count = min(
                int(sample_count_per_cluster),
                int(smallest_cluster_size),
            )
            selected_count = int(min(effective_sample_count, cluster_size))
        #end-if

        selected_percentage = (
            100.0 * selected_count / cluster_size
            if cluster_size > 0
            else 0.0
        )

        if target_cluster_selection_mode == "Select All Target Clusters":
            target_clusters_per_factual = max(0, n_clusters - 1)
        else:
            target_clusters_per_factual = 1 if n_clusters > 1 else 0

        total_factual_target_pairs = selected_count * target_clusters_per_factual

        rows.append(
            {
                "source_cluster": cluster_id,
                "cluster_size": cluster_size,
                "selected_count": selected_count,
                "selected_percentage": selected_percentage,
                "target_clusters_per_factual": target_clusters_per_factual,
                "total_factual_target_pairs": total_factual_target_pairs,
            }
        )

    return pd.DataFrame(rows)

df_alpha_100_by_mode = build_alpha_100_table_for_modes(cf_modes)

primary_cf_mode = cf_modes[-1]

cluster_result, ranking_weights, cf_weights, ranked_features = get_mode_config(
    primary_cf_mode
)

with st.sidebar:

    with st.expander(
        "ACTIONABILITY AND FEATURE RANKING",
        expanded=True,
    ):

        ranked_features, mask = render_feature_controls(
            feature_names=feature_names,
            ranked_features=ranked_features,
            ranking_weights=ranking_weights,
            cf_mode=primary_cf_mode,
        )

labels = cluster_result.labels
centroids = cluster_result.centroids
weights = cf_weights

df_alpha_100_by_cluster = pd.DataFrame(
    alpha_summary_for_all_clusters(
        X=X_cluster,
        centers=cluster_result.centroids,
        labels=cluster_result.labels,
        weights=cf_weights,
        retain_fraction=1.0,
        min_alpha=0.0,
    )
)

df_alpha_100_by_mode = build_alpha_100_table_for_modes(cf_modes)

# ----------------------------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------------------------

st.title("Counterfactual Explanations for Weighted Clustering using Voronoi Geometry")

st.subheader("Input Data")
st.dataframe(
    pd.DataFrame(X_view, columns=feature_names),
    hide_index=False,
    )


tab_specific_cf, tab_model_eval, tab_comparative_eval = st.tabs(
    [
        "Specific Counterfactuals",
        "Model-Level Evaluation",
        "Comparative Evaluation",
    ]
)

# ----------------------------------------------------------------------
# Find Source and select Target
# ----------------------------------------------------------------------

with st.sidebar:

    with st.expander("FACTUAL POINT", expanded=True):

        st.header("Counterfactual")
        st.caption("Select a Factual Point and a counterfactual target")

        n_samples_available = len(X_cluster)
        default_a = min(MY_START_POINT, n_samples_available)

        a = st.number_input(
            "Factual Point 'a'",
            value=default_a,
            step=1,
            min_value=1,
            #max_value=len(X_opt),
            max_value=n_samples_available,
        )

        a_idx = int(a) - 1
        s = int(labels[a_idx])
        st.caption(
            f"Factual point {a} is in cluster {s}."
        )

        target_options = [i for i in range(n_clusters) if i != s]

        if not target_options:
            st.warning("No valid target clusters are available.")
            st.stop()
        #end-if

        t = st.selectbox(
            "Target cluster t",
            options=target_options,
            index=0,
        )

        max_label = n_clusters - 1
        if s > max_label or t > max_label:
            st.warning(f"s and t must be between 0 and {max_label}.")
            st.stop()
        #end-if
        if s == t:
            st.warning("Choose different source and target clusters (s != t).")
            st.stop()
        #end-if

    #end-with(expander)

    with st.expander("HOMOTHETIC CONTRACTION (α)", expanded=True):

        ALPHA_MODE_CALCULATED = "Calculated from target cluster"
        ALPHA_MODE_MANUAL = "Manual α, or off (α=1)"
        
        alpha_mode = st.radio(
            "α selection mode",
            options=[
                ALPHA_MODE_CALCULATED,
                ALPHA_MODE_MANUAL,
            ],
            index=0,
            help=(
                "Calculated computes α from the selected target-cluster geometry."
                "Manual lets you set α directly, with α=1 effectively switching "
                "Homothetic contraction off."
            ),
        )

        manual_alpha = st.slider(
            "Manual α",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            disabled=(alpha_mode != ALPHA_MODE_MANUAL),
            help=(
                "α controls how deeply the target Voronoi cell is contracted "
                "toward its centroid. α=1 uses the full target cell; smaller "
                "values require counterfactuals deeper inside the target cluster,"
                "α=0 maps only to the cluster centroid."
            ),
        )

        alpha_retain_percentage = st.slider(
            "Target-cluster retention %",
            min_value=0,
            max_value=100,
            value=100,
            step=1,
            format="%.0f%%",
            disabled=(alpha_mode != ALPHA_MODE_CALCULATED),
            help=(
                "When α is calculated, this chooses the proportion of target-cluster "
                "points that should remain inside the contracted target region."
            ),
        )

        alpha_retain_fraction = alpha_retain_percentage / 100

        # Compute contraction scores regardless of alpha mode
        alpha_scores = contraction_scores_for_target_points(
            X=X_cluster,
            centers=cluster_result.centroids,
            labels=cluster_result.labels,
            t=t,
            weights=cf_weights,
        )

        if alpha_mode == ALPHA_MODE_MANUAL:

            alpha = manual_alpha

        else:

            alpha, alpha_scores = calculate_alpha_for_target_cluster(
                X=X_cluster,
                centers=cluster_result.centroids,
                labels=cluster_result.labels,
                t=t,
                weights=cf_weights,
                retain_fraction=alpha_retain_fraction,
            )

        # Always display the active alpha
        st.metric(
            "Active α",
            f"{alpha:.3f}",
        )

        # Only show calculated-alpha explanatory text when using
        # the calculated mode
        if alpha_mode == ALPHA_MODE_CALCULATED:

            st.caption(
                f"Target retention: smallest α with at least "
                f"{alpha_retain_fraction:.0%} of points assigned "
                f"to target cluster {t}."
            )

        # Calculate actual retention statistics
        n_target = len(alpha_scores)

        n_retained = (
            int(np.sum(alpha_scores <= alpha))
            if n_target > 0
            else 0
        )

        retained_pct = (
            100 * n_retained / n_target
            if n_target > 0
            else 0
        )

        st.caption(
            f"{retained_pct:.0f}% - "
            f"{n_retained} of {n_target} items "
            f"are inside the contracted region."
        )

        if n_retained == 0:

            st.caption(
                "No actual data items are retained; "
                "only the centroid remains."
            )

        #end-if

        st.divider()

        st.caption(
            "100%-retention α by centroid. "
            "This is the smallest α required to keep all data points assigned "
            "to each cluster inside that cluster's contracted Voronoi region."
        )

        st.dataframe(
            df_alpha_100_by_cluster[
                [
                    "cluster",
                    "n_points",
                    "alpha_100",
                    "retained_points",
                    "retained_pct",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "cluster": st.column_config.NumberColumn(
                    "Cluster",
                    format="%d",
                ),
                "n_points": st.column_config.NumberColumn(
                    "Points",
                    format="%d",
                ),
                "alpha_100": st.column_config.NumberColumn(
                    "α for 100%",
                    format="%.4f",
                ),
                "retained_points": st.column_config.NumberColumn(
                    "Retained",
                    format="%d",
                ),
                "retained_pct": st.column_config.NumberColumn(
                    "Retained %",
                    format="%.1f",
                ),
            },
        )
    
    #end-with(expander)

#end-with(sidebar)

# ----------------------------------------------------------------------
# Interactive Specific Counterfactuals Tab
# ----------------------------------------------------------------------

with tab_specific_cf:

    st.write(
        f"An **interactive demonstration** showing the :\n"
        f"creation of parsimonious counterfactuals for the movement of "
        f"*Factual Point* **a = {a_idx+1}** "
        f"from *Source Cluster* **s ={s}** "
        f"to *Target Cluster* **t= {t}** "
        f"with *Actionability Mask* **M = {np.asarray(mask, dtype=int).tolist()}**.\n\n"
        f"This experiment is running using the **{dataset_name}** dataset."
    )

    st.divider()
    st.header("Specific counterfactuals for nominated factual points")

    mode_tabs = st.tabs(cf_modes)

    for mode_tab, cf_mode in zip(mode_tabs, cf_modes):

        with mode_tab:

            (
                mode_cluster_result,
                mode_ranking_weights,
                mode_cf_weights,
                mode_ranked_features,
            ) = get_mode_config(cf_mode)

            mode_s = int(mode_cluster_result.labels[a_idx])

            st.header(cf_mode)

            if mode_s == t:
                st.warning(
                    f"Skipping {cf_mode}: factual point {a_idx + 1} "
                    f"is already in target cluster {t} under this mode."
                )
                continue

            display_cluster_solution(
                X_cluster=X_cluster,
                X_view=X_view,
                feature_names=feature_names,
                cluster_result=mode_cluster_result,
                scaler=scaler,
                view_space=view_space,
                seed=seed,
                a_idx=a_idx,
                cf_mode=cf_mode,
                ranking_weights=mode_ranking_weights,
                cf_weights=mode_cf_weights,
                y_true=y_true,
            )

            display_bounded_voronoi_cells(
                X_cluster=X_cluster,
                X_view=X_view,
                feature_names=feature_names,
                cluster_result=mode_cluster_result,
                scaler=scaler,
                view_space=view_space,
                seed=seed,
                pad=0.5,
                a_idx=a_idx,
                s=mode_s,
                t=t,
                alpha=alpha,
                ranked_features=mode_ranked_features,
                mask=mask,
                cf_weights=mode_cf_weights,
                cf_mode=cf_mode,
            )

        #end-with

    #end-for

#end-with

# ----------------------------------------------------------------------
# Model Evaluation Tab
# ----------------------------------------------------------------------

with tab_model_eval:

    st.subheader("Model-Level Evaluation")

    st.caption(
        "This tab is intended for aggregate evaluation across multiple factual "
        "points, target clusters, modes, and contraction settings. "
        "Results are generated only after the button is pressed."
    )

    st.subheader("100%-Retention α by Mode and Cluster")

    st.caption(
        "These values show the smallest contraction α required to retain "
        "all points assigned to each centroid, calculated separately for "
        "each selected comparison mode."
    )

    st.dataframe(
        df_alpha_100_by_mode[
            [
                "cf_mode",
                "cluster",
                "n_points",
                "alpha_100",
                "retained_points",
                "retained_pct",
                "min_score",
                "median_score",
                "max_score",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "cf_mode": st.column_config.TextColumn(
                "Mode",
            ),
            "cluster": st.column_config.NumberColumn(
                "Cluster",
                format="%d",
            ),
            "n_points": st.column_config.NumberColumn(
                "Points",
                format="%d",
            ),
            "alpha_100": st.column_config.NumberColumn(
                "α for 100%",
                format="%.4f",
            ),
            "retained_points": st.column_config.NumberColumn(
                "Retained",
                format="%d",
            ),
            "retained_pct": st.column_config.NumberColumn(
                "Retained %",
                format="%.1f",
            ),
            "min_score": st.column_config.NumberColumn(
                "Min score",
                format="%.4f",
            ),
            "median_score": st.column_config.NumberColumn(
                "Median score",
                format="%.4f",
            ),
            "max_score": st.column_config.NumberColumn(
                "Max score",
                format="%.4f",
            ),
        },
    )

    n_model_eval_factuals = st.number_input(
        "Number of factual points to sample",
        min_value=1,
        max_value=len(X_cluster),
        value=min(50, len(X_cluster)),
        step=1,
        key="n_model_eval_factuals",
    )

    run_model_eval = st.button(
        "Run model-level evaluation",
        key="run_model_level_evaluation",
    )

    if run_model_eval:

        st.info("Running model-level evaluation...")

        df_eval_results, df_eval_summary = run_model_level_evaluation(
            X_cluster=X_cluster,
            feature_names=feature_names,
            cf_modes=cf_modes,
            mode_config_fn=get_mode_config,
            sidebar_mask=mask,
            n_factuals=int(n_model_eval_factuals),
            seed=seed,
            n_clusters=n_clusters,
            alpha_mode=alpha_mode,
            manual_alpha=manual_alpha,
            alpha_retain_fraction=alpha_retain_fraction,
        )

        model_eval_export_dir = export_model_level_evaluation_tables(
            df_eval_results=df_eval_results,
            df_eval_summary=df_eval_summary,
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            n_factuals=int(n_model_eval_factuals),
            cf_modes=cf_modes,
            alpha_mode=alpha_mode,
            manual_alpha=manual_alpha,
            alpha_retain_fraction=alpha_retain_fraction,
            sidebar_mask=mask,
            results_dir=RESULTS_DIR,
            create_timestamped_dataset=create_timestamped_dataset,
        )

        st.success(
            f"Exported model-level evaluation tables to: {model_eval_export_dir}"
        )






        st.subheader("Least-Cost Metrics")

        st.dataframe(
            df_eval_summary[
                df_eval_summary["solution_type"] == "least_cost"
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("VoICE Most-Parsimonious Metrics")

        st.dataframe(
            df_eval_summary[
                df_eval_summary["solution_type"] == "most_parsimonious"
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Model-level metric charts")

        least_cost_metrics = [
            "feasibility_rate",
            "mean_weighted_cost",
            "mean_euclidean_distance",
            "mean_changed_features",
            "mean_runtime_ms",
        ]

        most_parsimonious_metrics = [
            "feasibility_rate",
            "mean_minimal_intervention_cardinality",
            "median_minimal_intervention_cardinality",
            "single_feature_feasibility_rate",
            "mean_changed_features",
            "mean_weighted_cost",
            "mean_euclidean_distance",
            "mean_tau",
            "positive_tau_rate",
            "mean_tau_per_feature",
            "mean_runtime_ms",
        ]

        parameter_sets = (
            df_eval_summary[
                [
                    "solution_type",
                    "mask_scenario",
                    "alpha_case",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "solution_type",
                    "mask_scenario",
                    "alpha_case",
                ]
            )
        )

        metric_labels = {
            "feasibility_rate": "Feasibility Rate (%)",

            "mean_minimal_intervention_cardinality":
                "Mean Minimal Intervention Cardinality",

            "median_minimal_intervention_cardinality":
                "Median Minimal Intervention Cardinality",

            "single_feature_feasibility_rate":
                "Single-Feature Feasibility Rate (%)",

            "mean_changed_features":
                "Mean Changed Features",

            "mean_weighted_cost":
                "Mean Weighted Cost",

            "mean_euclidean_distance":
                "Mean Euclidean Distance",

            "mean_distance_per_feature":
                "Mean Distance per Changed Feature",

            "mean_tau":
                "Mean Counterfactual Tolerance τ",

            "mean_tau_per_feature":
                "Mean τ per Changed Feature",

            "mean_rho":
                "Mean Explainability Range ρ",

            "positive_tau_rate":
                "Positive τ Rate (%)",

            "mean_runtime_ms":
                "Mean Runtime (ms)",
        }

        least_cost_metrics = [
            "feasibility_rate",
            "mean_weighted_cost",
            "mean_euclidean_distance",
            "mean_runtime_ms",
        ]

        most_parsimonious_metrics = [
            "feasibility_rate",
            "mean_minimal_intervention_cardinality",
            "median_minimal_intervention_cardinality",
            "single_feature_feasibility_rate",
            "mean_changed_features",
            "mean_weighted_cost",
            "mean_euclidean_distance",
            "mean_tau",
            "positive_tau_rate",
            "mean_tau_per_feature",
            "mean_runtime_ms",
        ]

        metric_groups = {
            "Least-Cost Metrics": (
                "least_cost",
                least_cost_metrics,
            ),
            "VoICE Most-Parsimonious Metrics": (
                "most_parsimonious",
                most_parsimonious_metrics,
            ),
        }

        for group_title, (solution_type, chart_metrics) in metric_groups.items():

            st.subheader(group_title)

            df_solution = df_eval_summary[
                df_eval_summary["solution_type"] == solution_type
            ].copy()

            for metric in chart_metrics:

                if metric not in df_solution.columns:
                    continue

                st.markdown(f"### {metric_labels.get(metric, metric)}")

                mask_scenarios = (
                    df_solution["mask_scenario"]
                    .drop_duplicates()
                    .sort_values()
                )

                for mask_scenario in mask_scenarios:

                    chart_df = df_solution[
                        df_solution["mask_scenario"] == mask_scenario
                    ].copy()

                    if chart_df.empty:
                        continue

                    pivot_df = chart_df.pivot(
                        index="cf_mode",
                        columns="alpha_case",
                        values=metric,
                    )

                    pivot_df = pivot_df.dropna(how="all")

                    if pivot_df.empty:
                        continue

                    st.markdown(f"**{mask_scenario}**")
                    st.bar_chart(pivot_df)

                #end-for

            #end-for

            st.subheader("Minimal intervention cardinality distribution")

            parameter_sets = (
                df_eval_results[
                    ["mask_scenario", "alpha_case"]
                ]
                .drop_duplicates()
                .sort_values(["mask_scenario", "alpha_case"])
            )

            for _, parameter_row in parameter_sets.iterrows():

                mask_scenario = parameter_row["mask_scenario"]
                alpha_case = parameter_row["alpha_case"]

                df_param = df_eval_results[
                    (df_eval_results["solution_type"] == "most_parsimonious")
                    & (df_eval_results["mask_scenario"] == mask_scenario)
                    & (df_eval_results["alpha_case"] == alpha_case)
                    & (df_eval_results["feasible"])
                ].copy()

                if df_param.empty:
                    continue

                st.markdown(f"**{mask_scenario} | {alpha_case}**")

                rank_counts = (
                    df_param
                    .groupby(["rank_step", "cf_mode"])
                    .size()
                    .reset_index(name="count")
                    .pivot(
                        index="rank_step",
                        columns="cf_mode",
                        values="count",
                    )
                    .fillna(0)
                )

                st.bar_chart(rank_counts)


            # ---------------------------------------------------------
            # Distributional Evaluation (violin plots)
            # --------------------------------------------------------
            st.write("violin")

            def safe_key(value):
                return (
                    str(value)
                    .replace(" ", "_")
                    .replace("[", "")
                    .replace("]", "")
                    .replace(",", "_")
                    .replace("=", "")
                    .replace("|", "_")
                )

            st.header("Distributional Evaluation")

            st.subheader(
                "Least-Cost Counterfactual Distance Distribution"
            )

            df_violin = df_eval_results[
                (df_eval_results["solution_type"] == "least_cost")
                & (df_eval_results["feasible"])
            ].copy()

            for mask_scenario in (
                df_violin["mask_scenario"]
                .drop_duplicates()
                .sort_values()
            ):

                df_mask = df_violin[
                    df_violin["mask_scenario"] == mask_scenario
                ]

                if df_mask.empty:
                    continue

                st.markdown(f"**{mask_scenario}**")

                fig = px.violin(
                    df_mask,
                    x="cf_mode",
                    y="euclidean_distance",
                    color="alpha_case",
                    box=True,
                    points=False,
                )

                fig.update_layout(
                    title="Euclidean Distance Distribution",
                    xaxis_title="Mode",
                    yaxis_title="Distance",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=(
                        f"violin_EDD_"
                        f"{safe_key(mask_scenario)}_"
                        f"{safe_key(solution_type)}"
                    )
                )



            st.subheader(
                "Minimal Intervention Cardinality Distribution (Parsimony)"
            )

            df_pars = df_eval_results[
                (df_eval_results["solution_type"] == "most_parsimonious")
                & (df_eval_results["feasible"])
            ].copy()

            for mask_scenario in (
                df_pars["mask_scenario"]
                .drop_duplicates()
                .sort_values()
            ):

                df_mask = df_pars[
                    df_pars["mask_scenario"] == mask_scenario
                ]

                if df_mask.empty:
                    continue

                st.markdown(f"**{mask_scenario}**")

                fig = px.violin(
                    df_mask,
                    x="cf_mode",
                    y="rank_step",
                    color="alpha_case",
                    box=True,
                    points=False,
                )

                fig.update_layout(
                    title="Minimal Intervention Cardinality",
                    xaxis_title="Mode",
                    yaxis_title="k*",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=(
                        f"violin_MIC_"
                        f"{safe_key(mask_scenario)}_"
                        f"{safe_key(solution_type)}"
                    )
                )








            st.subheader("Detailed model-level results")

            df_eval_results_display = df_eval_results.rename(
                columns={
                    "rank_step": "minimal_intervention_cardinality",
                }
            )

            st.dataframe(
                df_eval_results_display,
                hide_index=True,
                use_container_width=True,
            )

    else:

        st.info(
            "Press the button to generate model-level evaluation results."
        )

    #end-if

#end-with

# ----------------------------------------------------------------------
# Comparative Evaluation Tab
# ----------------------------------------------------------------------

with tab_comparative_eval:

    st.subheader("Comparative Evaluation")

    st.caption(
        "Configure a k-means-only comparative experiment between "
        "Voronoi-region counterfactuals and pairwise bisecting-hyperplane "
        "counterfactuals."
    )

    st.info(
        "This comparative evaluation uses the unweighted k-means solution only. "
        "SHARK, ranked k-means, and weighted counterfactual modes are not used "
        "in this experiment."
    )

    comparative_labels = np.asarray(cluster_result.labels)
    comparative_centroids = np.asarray(cluster_result.centroids)

    n_comparative_clusters = len(np.unique(comparative_labels))

    st.subheader("Experiment Iterations")

    n_comparative_iterations = st.number_input(
        "Number of Experiment Iterations",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        key="comparative_n_iterations",
        help=(
            "Run the comparative experiment across this many distinct "
            "unweighted k-means clustering solutions."
        ),
    )

    st.caption(
        "Each iteration will generate a separate unweighted k-means clustering "
        "solution for the comparative boundary experiment. These clustering "
        "solutions must be local to this tab's evaluation runner and must not "
        "overwrite the clustering solution used by the Specific Counterfactuals "
        "or Model-Level Evaluation tabs."
    )

    if n_comparative_clusters < 2:

        st.warning(
            "Comparative evaluation requires at least two clusters."
        )

    else:

        total_observations = int(len(comparative_labels))
        smallest_cluster_size = get_smallest_cluster_size(comparative_labels)
        default_sample_count = min(50, total_observations)

        st.subheader("Factual Sampling Strategy")

        sampling_mode = st.radio(
            "Sampling mode",
            options=[
                "Percentage of Each Cluster Sampled",
                "Number from Each Cluster Sampled",
            ],
            index=0,
            key="comparative_sampling_mode",
        )

        sample_percentage = st.slider(
            "Percentage of Each Cluster Sampled",
            min_value=1,
            max_value=100,
            value=100,
            step=1,
            format="%d%%",
            disabled=(
                sampling_mode != "Percentage of Each Cluster Sampled"
            ),
            key="comparative_sample_percentage",
            help=(
                "Select this percentage of observations from each source "
                "cluster. At least one observation is selected from every "
                "non-empty cluster."
            ),
        )

        sample_count_per_cluster = st.number_input(
            "Number from Each Cluster Sampled",
            min_value=1,
            max_value=max(1, total_observations),
            value=max(1, default_sample_count),
            step=1,
            disabled=(
                sampling_mode != "Number from Each Cluster Sampled"
            ),
            key="comparative_sample_count_per_cluster",
            help=(
                "Request this number of observations from each source cluster. "
                "If the requested number is greater than the size of the "
                "smallest cluster, the smallest cluster size will be used "
                "instead."
            ),
        )

        if (
            sampling_mode == "Number from Each Cluster Sampled"
            and sample_count_per_cluster > smallest_cluster_size
        ):
            st.caption(
                f"The requested count exceeds the smallest cluster size "
                f"({smallest_cluster_size}). The effective count used for "
                f"each cluster will therefore be {smallest_cluster_size}."
            )
        #end-if

        target_cluster_selection_mode = st.radio(
            "Target cluster selection",
            options=[
                "Select All Target Clusters",
                "Select One Target Cluster at Random",
            ],
            index=0,
            key="comparative_target_cluster_selection_mode",
            help=(
                "For each sampled factual observation, either evaluate all "
                "valid target clusters or select one target cluster at random."
            ),
        )

        sampling_summary_df = build_comparative_sampling_summary(
            labels=comparative_labels,
            sampling_mode=sampling_mode,
            sample_percentage=sample_percentage,
            sample_count_per_cluster=sample_count_per_cluster,
            target_cluster_selection_mode=target_cluster_selection_mode,
        )

        total_selected = int(sampling_summary_df["selected_count"].sum())
        total_available = int(sampling_summary_df["cluster_size"].sum())
        total_factual_target_pairs = int(
            sampling_summary_df["total_factual_target_pairs"].sum()
        )

        overall_selected_percentage = (
            100.0 * total_selected / total_available
            if total_available > 0
            else 0.0
        )

        st.subheader("Sampling Preview")

        st.caption(
            f"Selected **{total_selected}** of **{total_available}** factual "
            f"observations ({overall_selected_percentage:.1f}%) per clustering "
            f"solution. This will create **{total_factual_target_pairs}** "
            f"factual-target comparisons per iteration, or approximately "
            f"**{total_factual_target_pairs * int(n_comparative_iterations)}** "
            f"comparisons across **{int(n_comparative_iterations)}** iterations."
        )

        st.dataframe(
            sampling_summary_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "source_cluster": st.column_config.NumberColumn(
                    "Source cluster",
                    format="%d",
                ),
                "cluster_size": st.column_config.NumberColumn(
                    "Cluster size",
                    format="%d",
                ),
                "selected_count": st.column_config.NumberColumn(
                    "Selected count",
                    format="%d",
                ),
                "selected_percentage": st.column_config.NumberColumn(
                    "Selected %",
                    format="%.1f%%",
                ),
                "target_clusters_per_factual": st.column_config.NumberColumn(
                    "Target clusters per factual",
                    format="%d",
                ),
                "total_factual_target_pairs": st.column_config.NumberColumn(
                    "Factual-target pairs",
                    format="%d",
                ),
            },
        )

        st.subheader("Comparison Design")

        st.markdown(
            """
            This experiment will compare two k-means counterfactual constructions:

            1. **Voronoi-region counterfactuals**  
               Counterfactuals must enter the full target Voronoi region.

            2. **Bisecting-hyperplane counterfactuals**  
               Counterfactuals are generated using the pairwise bisecting
               hyperplane between the source and target centroids.

            The implementation will be added in the comparative evaluation
            runner.
            """
        )

        run_comparative_eval = st.button(
            "Run Comparative Evaluation",
            key="run_comparative_evaluation",
        )

        if run_comparative_eval:

            st.subheader("Comparative Evaluation Results")

            with st.spinner(
                "Running k-means Voronoi-region vs Vardakas bisecting-hyperplane comparison..."
            ):
                df_comparative_results, comparative_summary = (
                    run_kmeans_voronoi_vs_bisector_evaluation(
                        X=X_cluster,
                        feature_names=feature_names,
                        dataset_name=dataset_name,
                        n_clusters=n_clusters,
                        n_iterations=int(n_comparative_iterations),
                        sampling_mode=sampling_mode,
                        sample_percentage=sample_percentage,
                        sample_count_per_cluster=sample_count_per_cluster,
                        target_cluster_selection_mode=target_cluster_selection_mode,
                        base_seed=seed,
                        n_init=1,
                        max_iter=max_iter,
                        tol=tol,
                    )
                )

            st.success("Comparative evaluation complete.")

            st.dataframe(
                df_comparative_results,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "dataset_name": st.column_config.TextColumn(
                        "Dataset",
                    ),
                    "iteration": st.column_config.NumberColumn(
                        "Iteration",
                        format="%d",
                    ),
                    "iteration_seed": st.column_config.NumberColumn(
                        "Seed",
                        format="%d",
                    ),
                    "factual_index": st.column_config.NumberColumn(
                        "Factual index",
                        format="%d",
                    ),
                    "source_cluster": st.column_config.NumberColumn(
                        "Source cluster",
                        format="%d",
                    ),
                    "target_cluster": st.column_config.NumberColumn(
                        "Target cluster",
                        format="%d",
                    ),
                    "vardakas_cost": st.column_config.NumberColumn(
                        "Vardakas cost",
                        format="%.6f",
                    ),
                    "voronoi_cost": st.column_config.NumberColumn(
                        "Voronoi cost",
                        format="%.6f",
                    ),
                    "cost_difference_voronoi_minus_vardakas": st.column_config.NumberColumn(
                        "Voronoi - Vardakas cost",
                        format="%.6f",
                    ),
                    "vardakas_inside_target_voronoi": st.column_config.CheckboxColumn(
                        "Vardakas inside target Voronoi cell",
                    ),
                    "vardakas_voronoi_status": st.column_config.TextColumn(
                        "Vardakas Voronoi status",
                    ),
                    "voronoi_status": st.column_config.TextColumn(
                        "Voronoi status",
                    ),
                    "relative_underestimation_pct": st.column_config.NumberColumn(
                        "Underestimation %",
                        format="%.2f%%",
                    ),
                    "repair_cost_if_vardakas_failed": st.column_config.NumberColumn(
                        "Repair cost",
                        format="%.6f",
                    ),
                    "voronoi_inside_target_voronoi": st.column_config.CheckboxColumn(
                        "Voronoi inside target cell",
                    ),
                    "target_cell_validity_gain": st.column_config.NumberColumn(
                        "Validity gain",
                        format="%d",
                    ),
                    "vardakas_assigned_cluster": st.column_config.NumberColumn(
                        "Vardakas assigned cluster",
                        format="%d",
                    ),
                    "voronoi_assigned_cluster": st.column_config.NumberColumn(
                        "Voronoi assigned cluster",
                        format="%d",
                    ),
                    "vardakas_assigned_to_target": st.column_config.CheckboxColumn(
                        "Vardakas assigned to target",
                    ),
                    "voronoi_assigned_to_target": st.column_config.CheckboxColumn(
                        "Voronoi assigned to target",
                    ),
                    "vardakas_num_violated_constraints": st.column_config.NumberColumn(
                        "Violated constraints",
                        format="%d",
                    ),
                    "vardakas_max_voronoi_violation": st.column_config.NumberColumn(
                        "Max violation",
                        format="%.6f",
                    ),
                    "vardakas_mean_voronoi_violation": st.column_config.NumberColumn(
                        "Mean violation",
                        format="%.6f",
                    ),
                    "valid_case_cost_gap": st.column_config.NumberColumn(
                        "Valid-case cost gap",
                        format="%.8f",
                    ),
                },
            )

            st.download_button(
                label="Download Full Comparative Results CSV",
                data=df_comparative_results.to_csv(index=False).encode("utf-8"),
                file_name="comparative_voronoi_vs_vardakas_full_results.csv",
                mime="text/csv",
                key="download_comparative_full_results",
            )


            comparative_summary_help = {
                "dataset_name": (
                    "The dataset used for this comparative evaluation run."
                ),

                "total_comparisons": (
                    "The total number of factual-target counterfactual comparisons generated. "
                    "Each comparison corresponds to one sampled factual observation and one selected target cluster."
                ),

                "vardakas_fail_count": (
                    "The number of Vardakas bisecting-hyperplane counterfactuals that fail the full target Voronoi-cell validity test. "
                    "A failure means the endpoint reached the pairwise source-target bisector but did not lie inside the intended target cluster's full Voronoi region."
                ),

                "vardakas_fail_percentage": (
                    "The percentage of all comparisons for which the Vardakas endpoint failed the full target Voronoi-cell validity test. "
                    "This is the main empirical indicator that pairwise bisector crossing does not guarantee target-cluster membership in multi-cluster settings."
                ),

                "vardakas_success_count": (
                    "The number of Vardakas counterfactuals whose endpoints satisfy all target Voronoi-cell inequalities. "
                    "These are the Vardakas cases that are geometrically valid under the same target-cell criterion used by VoICE."
                ),

                "voronoi_success_count": (
                    "The number of Voronoi counterfactuals successfully generated by projecting onto the full target Voronoi cell. "
                    "In non-degenerate cases this should equal the total number of comparisons, because the Voronoi method optimises directly over the target cell."
                ),

                "mean_vardakas_cost_successful": (
                    "The mean squared Euclidean cost of Vardakas counterfactuals, calculated only over Vardakas endpoints that are valid under the full target Voronoi-cell test. "
                    "Invalid Vardakas endpoints are excluded because their low cost is not a valid target-region counterfactual cost."
                ),

                "mean_voronoi_cost_successful": (
                    "The mean squared Euclidean cost of successfully generated Voronoi counterfactuals. "
                    "This represents the average least-cost movement required to enter the full target Voronoi cell."
                ),

                "vardakas_target_cell_validity_rate": (
                    "The percentage of Vardakas endpoints that lie inside the full target Voronoi cell. "
                    "This is the positive form of the Vardakas validity metric: higher values mean the pairwise bisector endpoint more often satisfies the complete target-region geometry."
                ),

                "voronoi_target_cell_validity_rate": (
                    "The percentage of Voronoi endpoints that lie inside the full target Voronoi cell. "
                    "This should normally be 100%, because the Voronoi method directly solves for a point inside the target cell."
                ),

                "mean_cost_underestimation_on_failures": (
                    "For Vardakas-failed cases only, this is the mean difference between the valid Voronoi cost and the invalid Vardakas cost: "
                    "Voronoi cost minus Vardakas cost. A positive value means Vardakas reports an artificially low cost by stopping before reaching the true target Voronoi region."
                ),

                "mean_relative_underestimation_pct_on_failures": (
                    "For Vardakas-failed cases only, this expresses cost underestimation as a percentage of the valid Voronoi cost: "
                    "100 × (Voronoi cost - Vardakas cost) / Voronoi cost. "
                    "It shows how optimistic the invalid Vardakas cost is relative to the nearest valid target-region counterfactual."
                ),

                "mean_repair_cost_on_failures": (
                    "For Vardakas-failed cases only, this is the mean squared distance from the invalid Vardakas endpoint to the valid Voronoi endpoint. "
                    "It measures how much additional movement is needed to repair the pairwise bisector endpoint so that it reaches the full target Voronoi region."
                ),

                "mean_violated_constraints_on_failures": (
                    "For Vardakas-failed cases only, this is the mean number of target Voronoi-cell inequalities violated by the Vardakas endpoint. "
                    "A larger value means the endpoint is not merely outside the target cell, but violates multiple competing centroid constraints."
                ),

                "mean_max_violation_on_failures": (
                    "For Vardakas-failed cases only, this is the mean of the largest target-cell violation magnitude. "
                    "It measures how badly the Vardakas endpoint violates the full target Voronoi-cell criterion."
                ),

                "mean_valid_case_cost_gap": (
                    "For Vardakas-valid cases only, this is the mean absolute difference between the Vardakas cost and the Voronoi cost. "
                    "Under the unweighted, no-alpha, least-cost setting, this should be close to zero. Non-zero values usually indicate numerical tolerance effects."
                ),
            }


            st.subheader("Summary")

            total_comparisons = comparative_summary["total_comparisons"]
            vardakas_fail_count = comparative_summary["vardakas_fail_count"]
            vardakas_fail_percentage = comparative_summary["vardakas_fail_percentage"]
            vardakas_success_count = comparative_summary["vardakas_success_count"]
            voronoi_success_count = comparative_summary["voronoi_success_count"]
            mean_vardakas_cost_successful = comparative_summary[
                "mean_vardakas_cost_successful"
            ]
            mean_voronoi_cost_successful = comparative_summary[
                "mean_voronoi_cost_successful"
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total comparisons",
                f"{total_comparisons}",
                help=comparative_summary_help["total_comparisons"],
            )

            col2.metric(
                "Vardakas fails by Voronoi criteria",
                f"{vardakas_fail_count}",
                f"{vardakas_fail_percentage:.1f}%",
                help=comparative_summary_help["vardakas_fail_count"],
            )

            col3.metric(
                "Vardakas successful counterfactuals",
                f"{vardakas_success_count}",
                help=comparative_summary_help["vardakas_success_count"],
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Voronoi successful counterfactuals",
                f"{voronoi_success_count}",
                help=comparative_summary_help["voronoi_success_count"],
            )

            col5.metric(
                "Mean Vardakas cost",
                f"{mean_vardakas_cost_successful:.6f}"
                if np.isfinite(mean_vardakas_cost_successful)
                else "N/A",
                help=comparative_summary_help["mean_vardakas_cost_successful"],
            )

            col6.metric(
                "Mean Voronoi cost",
                f"{mean_voronoi_cost_successful:.6f}"
                if np.isfinite(mean_voronoi_cost_successful)
                else "N/A",
                help=comparative_summary_help["mean_voronoi_cost_successful"],
            )

            st.subheader("Geometric Validity Metrics")

            col7, col8, col9 = st.columns(3)

            col7.metric(
                "Vardakas target-cell validity",
                f"{comparative_summary['vardakas_target_cell_validity_rate']:.1f}%",
                help=comparative_summary_help["vardakas_target_cell_validity_rate"],
            )

            col8.metric(
                "Voronoi target-cell validity",
                f"{comparative_summary['voronoi_target_cell_validity_rate']:.1f}%",
                help=comparative_summary_help["voronoi_target_cell_validity_rate"],
            )

            col9.metric(
                "Mean underestimation on Vardakas failures",
                (
                    f"{comparative_summary['mean_cost_underestimation_on_failures']:.6f}"
                    if np.isfinite(comparative_summary["mean_cost_underestimation_on_failures"])
                    else "N/A"
                ),
                help=comparative_summary_help["mean_cost_underestimation_on_failures"],
            )

            col10, col11, col12 = st.columns(3)

            col10.metric(
                "Mean relative underestimation",
                (
                    f"{comparative_summary['mean_relative_underestimation_pct_on_failures']:.2f}%"
                    if np.isfinite(comparative_summary["mean_relative_underestimation_pct_on_failures"])
                    else "N/A"
                ),
                help=comparative_summary_help["mean_relative_underestimation_pct_on_failures"],
            )

            col11.metric(
                "Mean repair cost on failures",
                (
                    f"{comparative_summary['mean_repair_cost_on_failures']:.6f}"
                    if np.isfinite(comparative_summary["mean_repair_cost_on_failures"])
                    else "N/A"
                ),
                help=comparative_summary_help["mean_repair_cost_on_failures"],
            )

            col12.metric(
                "Mean violated constraints on failures",
                (
                    f"{comparative_summary['mean_violated_constraints_on_failures']:.2f}"
                    if np.isfinite(comparative_summary["mean_violated_constraints_on_failures"])
                    else "N/A"
                ),
                help=comparative_summary_help["mean_violated_constraints_on_failures"],
            )

            summary_table = pd.DataFrame(
                [
                    {
                        "dataset_name": comparative_summary["dataset_name"],
                        "total_comparisons": comparative_summary["total_comparisons"],
                        "vardakas_target_cell_validity_rate": comparative_summary[
                            "vardakas_target_cell_validity_rate"
                        ],
                        "voronoi_target_cell_validity_rate": comparative_summary[
                            "voronoi_target_cell_validity_rate"
                        ],
                        "vardakas_fail_count": comparative_summary["vardakas_fail_count"],
                        "vardakas_fail_percentage": comparative_summary[
                            "vardakas_fail_percentage"
                        ],
                        "vardakas_success_count": comparative_summary[
                            "vardakas_success_count"
                        ],
                        "voronoi_success_count": comparative_summary[
                            "voronoi_success_count"
                        ],
                        "mean_vardakas_cost_successful": comparative_summary[
                            "mean_vardakas_cost_successful"
                        ],
                        "mean_voronoi_cost_successful": comparative_summary[
                            "mean_voronoi_cost_successful"
                        ],
                        "mean_cost_underestimation_on_failures": comparative_summary[
                            "mean_cost_underestimation_on_failures"
                        ],
                        "mean_relative_underestimation_pct_on_failures": comparative_summary[
                            "mean_relative_underestimation_pct_on_failures"
                        ],
                        "mean_repair_cost_on_failures": comparative_summary[
                            "mean_repair_cost_on_failures"
                        ],
                        "mean_violated_constraints_on_failures": comparative_summary[
                            "mean_violated_constraints_on_failures"
                        ],
                        "mean_max_violation_on_failures": comparative_summary[
                            "mean_max_violation_on_failures"
                        ],
                        "mean_valid_case_cost_gap": comparative_summary[
                            "mean_valid_case_cost_gap"
                        ],
                    }
                ]
            )

            st.subheader("Summary Table")

            st.dataframe(
                summary_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "dataset_name": st.column_config.TextColumn(
                        "Dataset",
                        help=comparative_summary_help["dataset_name"],
                    ),
                    "total_comparisons": st.column_config.NumberColumn(
                        "Total comparisons",
                        format="%d",
                        help=comparative_summary_help["total_comparisons"],
                    ),
                    "vardakas_target_cell_validity_rate": st.column_config.NumberColumn(
                        "Vardakas target-cell validity",
                        format="%.1f%%",
                        help=comparative_summary_help["vardakas_target_cell_validity_rate"],
                    ),
                    "voronoi_target_cell_validity_rate": st.column_config.NumberColumn(
                        "Voronoi target-cell validity",
                        format="%.1f%%",
                        help=comparative_summary_help["voronoi_target_cell_validity_rate"],
                    ),
                    "vardakas_fail_count": st.column_config.NumberColumn(
                        "Vardakas fail count",
                        format="%d",
                        help=comparative_summary_help["vardakas_fail_count"],
                    ),
                    "vardakas_fail_percentage": st.column_config.NumberColumn(
                        "Vardakas fail %",
                        format="%.1f%%",
                        help=comparative_summary_help["vardakas_fail_percentage"],
                    ),
                    "vardakas_success_count": st.column_config.NumberColumn(
                        "Vardakas successes",
                        format="%d",
                        help=comparative_summary_help["vardakas_success_count"],
                    ),
                    "voronoi_success_count": st.column_config.NumberColumn(
                        "Voronoi successes",
                        format="%d",
                        help=comparative_summary_help["voronoi_success_count"],
                    ),
                    "mean_vardakas_cost_successful": st.column_config.NumberColumn(
                        "Mean Vardakas cost",
                        format="%.6f",
                        help=comparative_summary_help["mean_vardakas_cost_successful"],
                    ),
                    "mean_voronoi_cost_successful": st.column_config.NumberColumn(
                        "Mean Voronoi cost",
                        format="%.6f",
                        help=comparative_summary_help["mean_voronoi_cost_successful"],
                    ),
                    "mean_cost_underestimation_on_failures": st.column_config.NumberColumn(
                        "Mean underestimation on failures",
                        format="%.6f",
                        help=comparative_summary_help["mean_cost_underestimation_on_failures"],
                    ),
                    "mean_relative_underestimation_pct_on_failures": st.column_config.NumberColumn(
                        "Mean relative underestimation",
                        format="%.2f%%",
                        help=comparative_summary_help["mean_relative_underestimation_pct_on_failures"],
                    ),
                    "mean_repair_cost_on_failures": st.column_config.NumberColumn(
                        "Mean repair cost",
                        format="%.6f",
                        help=comparative_summary_help["mean_repair_cost_on_failures"],
                    ),
                    "mean_violated_constraints_on_failures": st.column_config.NumberColumn(
                        "Mean violated constraints",
                        format="%.2f",
                        help=comparative_summary_help["mean_violated_constraints_on_failures"],
                    ),
                    "mean_max_violation_on_failures": st.column_config.NumberColumn(
                        "Mean max violation",
                        format="%.6f",
                        help=comparative_summary_help["mean_max_violation_on_failures"],
                    ),
                    "mean_valid_case_cost_gap": st.column_config.NumberColumn(
                        "Mean valid-case cost gap",
                        format="%.8f",
                        help=comparative_summary_help["mean_valid_case_cost_gap"],
                    ),
                },
            )

            st.download_button(
                label="Download Comparative Summary CSV",
                data=summary_table.to_csv(index=False).encode("utf-8"),
                file_name="comparative_voronoi_vs_vardakas_summary.csv",
                mime="text/csv",
                key="download_comparative_summary_results",
            )


            st.caption(
                "A Vardakas counterfactual is counted as FAIL when its "
                "bisecting-hyperplane endpoint does not satisfy the full "
                "target Voronoi-cell membership criteria. No alpha contraction "
                "is used in this comparative experiment."
            )

        #end-if

    #end-if

#end-with
