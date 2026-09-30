import time
import numpy as np
import pandas as pd

from src.config import MODEL_LEVEL_PUBLICATION_MODES

from src.models import fit_clustering

from src.geometry import (
    build_alpha_halfspaces,
    calculate_alpha_for_target_cluster,
)

from src.counterfactuals import (
    project_to_target_with_fixed_features,
    directional_rho,
    ranked_actionable_prefixes,
)


def random_actionability_mask(
    n_features,
    immutable_fraction,
    rng,
):
    mask = np.ones(n_features, dtype=int)

    n_immutable = int(round(n_features * immutable_fraction))
    n_immutable = min(n_immutable, n_features - 1)

    immutable_idx = rng.choice(
        n_features,
        size=n_immutable,
        replace=False,
    )

    mask[immutable_idx] = 0

    return mask


def format_mask_label(mask):
    return "[" + ",".join(str(int(v)) for v in mask) + "]"


def build_mask_scenarios(
    n_features,
    sidebar_mask,
    rng,
):
    sidebar_mask = np.asarray(sidebar_mask, dtype=int)

    mask_scenarios = {
        "All features mutable": np.ones(n_features, dtype=int),
        "Random 25% immutable": random_actionability_mask(
            n_features, 0.25, rng
        ),
        "Random 50% immutable": random_actionability_mask(
            n_features, 0.50, rng
        ),
        "Random 75% immutable": random_actionability_mask(
            n_features, 0.75, rng
        ),
    }

    if not np.all(sidebar_mask == 1):
        mask_scenarios[
            f"Sidebar selection M = {format_mask_label(sidebar_mask)}"
        ] = sidebar_mask

    return mask_scenarios


def selected_alpha_for_mode(
    X_cluster,
    cluster_result,
    target_cluster,
    weights,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
):
    if alpha_mode == "Manual α, or off (α=1)":
        return float(manual_alpha)

    alpha, _ = calculate_alpha_for_target_cluster(
        X=X_cluster,
        centers=cluster_result.centroids,
        labels=cluster_result.labels,
        t=target_cluster,
        weights=weights,
        retain_fraction=alpha_retain_fraction,
    )

    return float(alpha)


def evaluate_most_parsimonious_case(
    X_cluster,
    feature_names,
    cluster_result,
    cf_weights,
    ranked_features,
    factual_idx,
    target_cluster,
    alpha,
    mask,
    cf_mode,
    mask_scenario,
    alpha_case,
):
    a_cluster = np.asarray(X_cluster[factual_idx], dtype=float)

    x_lo = X_cluster.min(axis=0)
    x_hi = X_cluster.max(axis=0)

    halfspaces_t = build_alpha_halfspaces(
        centers=cluster_result.centroids,
        t=target_cluster,
        weights=cf_weights,
        alpha=alpha,
    )

    prefixes = ranked_actionable_prefixes(
        ranked_features=ranked_features,
        feature_names=feature_names,
        mask=mask,
    )

    start_time = time.perf_counter()

    for rank_step, (subset_names, subset_idx) in enumerate(prefixes, start=1):

        z_cf, ok, msg = project_to_target_with_fixed_features(
            a=a_cluster,
            free_idx=subset_idx,
            halfspaces=halfspaces_t,
            x_lo=x_lo,
            x_hi=x_hi,
            weights=cf_weights,
        )

        runtime_ms = 1000 * (time.perf_counter() - start_time)

        if not ok:
            continue

        u = z_cf - a_cluster

        rho, rho_diag = directional_rho(
            a=a_cluster,
            u=u,
            halfspaces=halfspaces_t,
            x_lo=x_lo,
            x_hi=x_hi,
            return_diagnostics=True,
            respect_box_bounds=True,
        )

        tau = rho - 1.0 if np.isfinite(rho) else np.inf

        changed_features = int(
            np.sum(np.abs(z_cf - a_cluster) > 1e-9)
        )

        return {
            "cf_mode": cf_mode,
            "mask_scenario": mask_scenario,
            "alpha_case": alpha_case,
            "alpha": float(alpha),
            "factual_idx": int(factual_idx),
            "source_cluster": int(cluster_result.labels[factual_idx]),
            "target_cluster": int(target_cluster),
            "feasible": True,
            "rank_step": int(rank_step),
            "changed_features": changed_features,
            "weighted_squared_distance": float(
                np.sum(cf_weights * (z_cf - a_cluster) ** 2)
            ),
            "euclidean_distance": float(
                np.linalg.norm(z_cf - a_cluster)
            ),
            "rho": float(rho),
            "tau": float(tau),
            "positive_tau": bool(tau > 0),
            "runtime_ms": float(runtime_ms),
            "solver_message": msg,
            "solution_type": "most_parsimonious",
        }

    runtime_ms = 1000 * (time.perf_counter() - start_time)

    return {
        "cf_mode": cf_mode,
        "mask_scenario": mask_scenario,
        "alpha_case": alpha_case,
        "alpha": float(alpha),
        "factual_idx": int(factual_idx),
        "source_cluster": int(cluster_result.labels[factual_idx]),
        "target_cluster": int(target_cluster),
        "feasible": False,
        "rank_step": np.nan,
        "changed_features": np.nan,
        "weighted_squared_distance": np.nan,
        "euclidean_distance": np.nan,
        "rho": np.nan,
        "tau": np.nan,
        "positive_tau": False,
        "runtime_ms": float(runtime_ms),
        "solver_message": "No feasible counterfactual found.",
        "solution_type": "most_parsimonious",
    }


def evaluate_least_cost_case(
    X_cluster,
    cluster_result,
    cf_weights,
    factual_idx,
    target_cluster,
    alpha,
    mask,
    cf_mode,
    mask_scenario,
    alpha_case,
):
    a_cluster = np.asarray(X_cluster[factual_idx], dtype=float)

    x_lo = X_cluster.min(axis=0)
    x_hi = X_cluster.max(axis=0)

    halfspaces_t = build_alpha_halfspaces(
        centers=cluster_result.centroids,
        t=target_cluster,
        weights=cf_weights,
        alpha=alpha,
    )

    free_idx = np.flatnonzero(np.asarray(mask, dtype=int) == 1)

    start_time = time.perf_counter()

    z_cf, ok, msg = project_to_target_with_fixed_features(
        a=a_cluster,
        free_idx=free_idx,
        halfspaces=halfspaces_t,
        x_lo=x_lo,
        x_hi=x_hi,
        weights=cf_weights,
    )

    runtime_ms = 1000 * (time.perf_counter() - start_time)

    if not ok:
        return {
            "solution_type": "least_cost",
            "cf_mode": cf_mode,
            "mask_scenario": mask_scenario,
            "alpha_case": alpha_case,
            "alpha": float(alpha),
            "factual_idx": int(factual_idx),
            "source_cluster": int(cluster_result.labels[factual_idx]),
            "target_cluster": int(target_cluster),
            "feasible": False,
            "rank_step": np.nan,
            "changed_features": np.nan,
            "weighted_squared_distance": np.nan,
            "euclidean_distance": np.nan,
            "rho": np.nan,
            "tau": np.nan,
            "positive_tau": False,
            "runtime_ms": float(runtime_ms),
            "solver_message": msg,
        }

    u = z_cf - a_cluster

    rho, _ = directional_rho(
        a=a_cluster,
        u=u,
        halfspaces=halfspaces_t,
        x_lo=x_lo,
        x_hi=x_hi,
        return_diagnostics=True,
        respect_box_bounds=True,
    )

    tau = rho - 1.0 if np.isfinite(rho) else np.inf

    return {
        "solution_type": "least_cost",
        "cf_mode": cf_mode,
        "mask_scenario": mask_scenario,
        "alpha_case": alpha_case,
        "alpha": float(alpha),
        "factual_idx": int(factual_idx),
        "source_cluster": int(cluster_result.labels[factual_idx]),
        "target_cluster": int(target_cluster),
        "feasible": True,
        "rank_step": np.nan,
        "changed_features": int(np.sum(np.abs(z_cf - a_cluster) > 1e-9)),
        "weighted_squared_distance": float(
            np.sum(cf_weights * (z_cf - a_cluster) ** 2)
        ),
        "euclidean_distance": float(np.linalg.norm(z_cf - a_cluster)),
        "rho": float(rho),
        "tau": float(tau),
        "positive_tau": bool(tau > 0),
        "runtime_ms": float(runtime_ms),
        "solver_message": msg,
    }

def evaluate_vardakas_baselines_placeholder(*args, **kwargs):
    """
    Placeholder for future comparative calls to the Vardakas-style algorithms.

    Expected future outputs:
    - method name
    - feasible
    - euclidean distance
    - sparsity / changed features
    - runtime
    """
    return []


def aggregate_voice_results(df_results):
    df_results = df_results.copy()

    df_results["single_feature_feasible"] = (
        df_results["feasible"] & (df_results["rank_step"] == 1)
    )

    df_results["distance_per_feature"] = np.where(
        df_results["changed_features"] > 0,
        df_results["euclidean_distance"] / df_results["changed_features"],
        np.nan,
    )

    df_results["tau_per_feature"] = np.where(
        df_results["changed_features"] > 0,
        df_results["tau"] / df_results["changed_features"],
        np.nan,
    )

    grouped = df_results.groupby(
        ["solution_type", "cf_mode", "mask_scenario", "alpha_case"],
        dropna=False,
    )

    df_summary = grouped.agg(
        attempts=("feasible", "size"),
        feasible_count=("feasible", "sum"),
        feasibility_rate=("feasible", "mean"),

        mean_minimal_intervention_cardinality=("rank_step", "mean"),
        median_minimal_intervention_cardinality=("rank_step", "median"),
        mean_changed_features=("changed_features", "mean"),
        single_feature_feasibility_rate=("single_feature_feasible", "mean"),

        mean_distance_per_feature=("distance_per_feature", "mean"),
        mean_tau_per_feature=("tau_per_feature", "mean"),
        mean_weighted_cost=("weighted_squared_distance", "mean"),
        mean_euclidean_distance=("euclidean_distance", "mean"),
        mean_rho=("rho", "mean"),
        mean_tau=("tau", "mean"),
        positive_tau_rate=("positive_tau", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
    ).reset_index()

    df_summary["feasibility_rate"] *= 100
    df_summary["positive_tau_rate"] *= 100
    df_summary["single_feature_feasibility_rate"] *= 100

    return df_summary

def build_model_level_mode_config(
    kmeans_result,
    shark_result,
    feature_names,
):
    """
    Build the mode configuration for one independently fitted
    pair of k-means and SHARK models.
    """
    feature_names = list(feature_names)

    d = len(feature_names)
    equal_weights = np.ones(d, dtype=float) / d

    def get_mode_config(cf_mode):

        if cf_mode == "Unweighted k-means":
            return (
                kmeans_result,
                equal_weights,
                equal_weights,
                feature_names.copy(),
            )

        if cf_mode == "Ranked k-means":
            ranking_weights = np.asarray(
                shark_result.weights,
                dtype=float,
            )

            ranked_features = [
                feature_names[i]
                for i in np.argsort(-ranking_weights)
            ]

            return (
                kmeans_result,
                ranking_weights,
                equal_weights,
                ranked_features,
            )

        if cf_mode == "Ranked + weighted SHARK":
            ranking_weights = np.asarray(
                shark_result.weights,
                dtype=float,
            )

            ranked_features = [
                feature_names[i]
                for i in np.argsort(-ranking_weights)
            ]

            return (
                shark_result,
                ranking_weights,
                ranking_weights,
                ranked_features,
            )

        raise ValueError(
            f"Unknown counterfactual comparison mode: {cf_mode}"
        )

    return get_mode_config

def table3_metrics_from_summary(
    df_summary,
    n_features,
):
    """
    Reduce one clustering repetition to the four metrics
    reported in Table 3.

    Table 3 uses:
      - least-cost counterfactuals;
      - all features mutable;
      - contracted target regions;
      - the mean across the three publication modes.
    """

    table3_rows = df_summary[
        (df_summary["solution_type"] == "least_cost")
        & (df_summary["mask_scenario"] == "All features mutable")
        & (df_summary["alpha_case"] == "α < 1")
        & (
            df_summary["cf_mode"].isin(
                MODEL_LEVEL_PUBLICATION_MODES
            )
        )
    ].copy()

    expected_modes = set(MODEL_LEVEL_PUBLICATION_MODES)
    present_modes = set(table3_rows["cf_mode"])

    if present_modes != expected_modes:
        missing_modes = expected_modes - present_modes

        raise RuntimeError(
            "Table 3 requires all publication modes. "
            f"Missing modes: {sorted(missing_modes)}"
        )

    if len(table3_rows) != len(MODEL_LEVEL_PUBLICATION_MODES):
        raise RuntimeError(
            "Expected exactly one Table 3 summary row per mode, "
            f"but found {len(table3_rows)} rows."
        )

    return {
        "changed_features_pct": float(
            100.0
            * table3_rows["mean_changed_features"].mean()
            / int(n_features)
        ),
        "weighted_cost": float(
            table3_rows["mean_weighted_cost"].mean()
        ),
        "mean_tau": float(
            table3_rows["mean_tau"].mean()
        ),
        "runtime_ms": float(
            table3_rows["mean_runtime_ms"].mean()
        ),
        "minimum_feasibility_rate": float(
            table3_rows["feasibility_rate"].min()
        ),
    }

def table4_metrics_from_summary(
    df_summary,
    n_features,
):
    """
    Reduce one clustering repetition to the four metrics
    reported in Table 4.

    Table 4 uses:
      - most-parsimonious counterfactuals;
      - all features mutable;
      - contracted target regions;
      - the mean across the three publication modes.
    """

    table4_rows = df_summary[
        (df_summary["solution_type"] == "most_parsimonious")
        & (df_summary["mask_scenario"] == "All features mutable")
        & (df_summary["alpha_case"] == "α < 1")
        & (
            df_summary["cf_mode"].isin(
                MODEL_LEVEL_PUBLICATION_MODES
            )
        )
    ].copy()

    expected_modes = set(MODEL_LEVEL_PUBLICATION_MODES)
    present_modes = set(table4_rows["cf_mode"])

    if present_modes != expected_modes:
        missing_modes = expected_modes - present_modes

        raise RuntimeError(
            "Table 4 requires all publication modes. "
            f"Missing modes: {sorted(missing_modes)}"
        )

    if len(table4_rows) != len(MODEL_LEVEL_PUBLICATION_MODES):
        raise RuntimeError(
            "Expected exactly one Table 4 summary row per mode, "
            f"but found {len(table4_rows)} rows."
        )

    mean_r_star = float(
        table4_rows[
            "mean_minimal_intervention_cardinality"
        ].mean()
    )

    return {
        "mean_r_star": mean_r_star,
        "r_star_over_d": float(
            mean_r_star / int(n_features)
        ),
        "single_feature_feasibility_rate": float(
            table4_rows[
                "single_feature_feasibility_rate"
            ].mean()
        ),
        "mean_tau": float(
            table4_rows["mean_tau"].mean()
        ),
        "minimum_feasibility_rate": float(
            table4_rows["feasibility_rate"].min()
        ),
    }


def run_model_level_evaluation(
    X_cluster,
    feature_names,
    cf_modes,
    mode_config_fn,
    sidebar_mask,
    n_factuals,
    seed,
    n_clusters,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
):
    rng = np.random.default_rng(seed)

    n_samples = X_cluster.shape[0]
    n_factuals = min(n_factuals, n_samples)

    factual_indices = rng.choice(
        n_samples,
        size=n_factuals,
        replace=False,
    )

    rows = []

    for cf_mode in cf_modes:

        (
            cluster_result,
            ranking_weights,
            cf_weights,
            ranked_features,
        ) = mode_config_fn(cf_mode)

        for factual_idx in factual_indices:

            source_cluster = int(cluster_result.labels[factual_idx])

            target_options = [
                c for c in range(n_clusters)
                if c != source_cluster
            ]

            if len(target_options) == 0:
                continue

            target_cluster = int(rng.choice(target_options))

            selected_alpha = selected_alpha_for_mode(
                X_cluster=X_cluster,
                cluster_result=cluster_result,
                target_cluster=target_cluster,
                weights=cf_weights,
                alpha_mode=alpha_mode,
                manual_alpha=manual_alpha,
                alpha_retain_fraction=alpha_retain_fraction,
            )

            alpha_cases = [
                ("α = 1", 1.0),
            ]

            if not np.isclose(selected_alpha, 1.0):
                alpha_cases.append(
                    ("α < 1", selected_alpha)
                )

            mask_scenarios = build_mask_scenarios(
                n_features=len(feature_names),
                sidebar_mask=sidebar_mask,
                rng=rng,
            )

            for mask_scenario, scenario_mask in mask_scenarios.items():

                for alpha_case, alpha in alpha_cases:

                    rows.append(
                        evaluate_least_cost_case(
                            X_cluster=X_cluster,
                            cluster_result=cluster_result,
                            cf_weights=cf_weights,
                            factual_idx=factual_idx,
                            target_cluster=target_cluster,
                            alpha=alpha,
                            mask=scenario_mask,
                            cf_mode=cf_mode,
                            mask_scenario=mask_scenario,
                            alpha_case=alpha_case,
                        )
                    )

                    rows.append(
                        evaluate_most_parsimonious_case(
                            X_cluster=X_cluster,
                            feature_names=feature_names,
                            cluster_result=cluster_result,
                            cf_weights=cf_weights,
                            ranked_features=ranked_features,
                            factual_idx=factual_idx,
                            target_cluster=target_cluster,
                            alpha=alpha,
                            mask=scenario_mask,
                            cf_mode=cf_mode,
                            mask_scenario=mask_scenario,
                            alpha_case=alpha_case,
                        )
                    )

                    # Placeholder for future Vardakas-style comparison.
                    # vardakas_rows = evaluate_vardakas_baselines_placeholder(...)
                    # rows.extend(vardakas_rows)

    df_results = pd.DataFrame(rows)
    df_summary = aggregate_voice_results(df_results)

    return df_results, df_summary

def run_repeated_model_level_evaluation(
    *,
    X_cluster,
    feature_names,
    sidebar_mask,
    n_factuals,
    n_clusters,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
    n_iterations=50,
    base_clustering_seed=42,
    evaluation_seed=42,
    n_init=10,
    max_iter=300,
    tol=1e-4,
):
    """
    Repeat the model-level publication evaluation across
    independently fitted clustering solutions.
    """

    repeated_summary_frames = []
    table3_iteration_rows = []
    table4_iteration_rows = []

    for iteration in range(int(n_iterations)):

        clustering_seed = (
            int(base_clustering_seed) + iteration
        )

        kmeans_result = fit_clustering(
            X_cluster,
            method="k-means",
            n_clusters=int(n_clusters),
            seed=clustering_seed,
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
        )

        shark_result = fit_clustering(
            X_cluster,
            method="SHARK",
            n_clusters=int(n_clusters),
            seed=clustering_seed,
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
        )

        mode_config_fn = build_model_level_mode_config(
            kmeans_result=kmeans_result,
            shark_result=shark_result,
            feature_names=feature_names,
        )

        _, df_summary = run_model_level_evaluation(
            X_cluster=X_cluster,
            feature_names=feature_names,
            cf_modes=MODEL_LEVEL_PUBLICATION_MODES,
            mode_config_fn=mode_config_fn,
            sidebar_mask=sidebar_mask,
            n_factuals=int(n_factuals),
            seed=int(evaluation_seed),
            n_clusters=int(n_clusters),
            alpha_mode=alpha_mode,
            manual_alpha=manual_alpha,
            alpha_retain_fraction=alpha_retain_fraction,
        )

        table3_metrics = table3_metrics_from_summary(
            df_summary=df_summary,
            n_features=len(feature_names),
        )

        table3_iteration_rows.append(
            {
                "iteration": iteration + 1,
                "clustering_seed": clustering_seed,
                "evaluation_seed": int(evaluation_seed),
                **table3_metrics,
            }
        )

        table4_metrics = table4_metrics_from_summary(
            df_summary=df_summary,
            n_features=len(feature_names),
        )

        table4_iteration_rows.append(
            {
                "iteration": iteration + 1,
                "clustering_seed": clustering_seed,
                "evaluation_seed": int(evaluation_seed),
                **table4_metrics,
            }
        )

        df_summary = df_summary.copy()

        df_summary.insert(
            0,
            "iteration",
            iteration + 1,
        )
        df_summary.insert(
            1,
            "clustering_seed",
            clustering_seed,
        )
        df_summary.insert(
            2,
            "evaluation_seed",
            int(evaluation_seed),
        )

        repeated_summary_frames.append(df_summary)

    df_repeated_summary = pd.concat(
        repeated_summary_frames,
        ignore_index=True,
    )

    df_table3_iterations = pd.DataFrame(
        table3_iteration_rows
    )

    df_table4_iterations = pd.DataFrame(
        table4_iteration_rows
    )

    if len(df_table3_iterations) < 2:
        changed_features_pct_std = np.nan
        weighted_cost_std = np.nan
        mean_tau_std = np.nan
        runtime_ms_std = np.nan
    else:
        changed_features_pct_std = float(
            df_table3_iterations[
                "changed_features_pct"
            ].std(ddof=1)
        )

        weighted_cost_std = float(
            df_table3_iterations[
                "weighted_cost"
            ].std(ddof=1)
        )

        mean_tau_std = float(
            df_table3_iterations[
                "mean_tau"
            ].std(ddof=1)
        )

        runtime_ms_std = float(
            df_table3_iterations[
                "runtime_ms"
            ].std(ddof=1)
        )

    table3_standard_deviations = {
        "n_iterations": int(len(df_table3_iterations)),
        "base_clustering_seed": int(base_clustering_seed),
        "evaluation_seed": int(evaluation_seed),
        "n_init": int(n_init),
        "sd_ddof": 1,
        "changed_features_pct_std": changed_features_pct_std,
        "weighted_cost_std": weighted_cost_std,
        "mean_tau_std": mean_tau_std,
        "runtime_ms_std": runtime_ms_std,
        "minimum_feasibility_rate": float(
            df_table3_iterations[
                "minimum_feasibility_rate"
            ].min()
        ),
    }

    if len(df_table4_iterations) < 2:
        mean_r_star_std = np.nan
        r_star_over_d_std = np.nan
        single_feature_feasibility_rate_std = np.nan
        table4_mean_tau_std = np.nan
    else:
        mean_r_star_std = float(
            df_table4_iterations[
                "mean_r_star"
            ].std(ddof=1)
        )

        r_star_over_d_std = float(
            df_table4_iterations[
                "r_star_over_d"
            ].std(ddof=1)
        )

        single_feature_feasibility_rate_std = float(
            df_table4_iterations[
                "single_feature_feasibility_rate"
            ].std(ddof=1)
        )

        table4_mean_tau_std = float(
            df_table4_iterations[
                "mean_tau"
            ].std(ddof=1)
        )

    table4_standard_deviations = {
        "n_iterations": int(len(df_table4_iterations)),
        "base_clustering_seed": int(base_clustering_seed),
        "evaluation_seed": int(evaluation_seed),
        "n_init": int(n_init),
        "sd_ddof": 1,
        "mean_r_star_std": mean_r_star_std,
        "r_star_over_d_std": r_star_over_d_std,
        "single_feature_feasibility_rate_std": (
            single_feature_feasibility_rate_std
        ),
        "mean_tau_std": table4_mean_tau_std,
        "minimum_feasibility_rate": float(
            df_table4_iterations[
                "minimum_feasibility_rate"
            ].min()
        ),
    }

    return (
        df_repeated_summary,
        df_table3_iterations,
        table3_standard_deviations,
        df_table4_iterations,
        table4_standard_deviations,
    )