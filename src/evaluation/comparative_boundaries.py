import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans


def squared_euclidean_cost(a, z):
    a = np.asarray(a, dtype=float)
    z = np.asarray(z, dtype=float)
    return float(np.sum((z - a) ** 2))


def get_voronoi_halfspaces(centroids, target_cluster):
    """
    Build the half-space representation of the unweighted k-means
    target Voronoi cell.

    The target Voronoi cell is:

        ||x - c_t||^2 <= ||x - c_j||^2 for all j != t

    which is equivalent to:

        2(c_j - c_t)^T x <= ||c_j||^2 - ||c_t||^2.
    """
    centroids = np.asarray(centroids, dtype=float)
    c_t = centroids[target_cluster]

    A_rows = []
    b_values = []

    for j, c_j in enumerate(centroids):
        if j == target_cluster:
            continue

        A_rows.append(2.0 * (c_j - c_t))
        b_values.append(float(np.dot(c_j, c_j) - np.dot(c_t, c_t)))

    return np.asarray(A_rows, dtype=float), np.asarray(b_values, dtype=float)


def is_inside_target_voronoi_cell(x, centroids, target_cluster, tolerance=1e-8):
    """
    Check whether x lies inside the full target Voronoi cell.

    This is the Voronoi validity criterion used to test whether a
    Vardakas-style bisecting-hyperplane counterfactual actually enters
    the intended target cluster region.
    """
    x = np.asarray(x, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    distances = np.sum((centroids - x) ** 2, axis=1)
    target_distance = distances[target_cluster]

    return bool(np.all(target_distance <= distances + tolerance))

def assigned_cluster(x, centroids):
    x = np.asarray(x, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    distances = np.sum((centroids - x) ** 2, axis=1)
    return int(np.argmin(distances))


def target_voronoi_violations(x, centroids, target_cluster, tolerance=1e-8):
    """
    Return the number and magnitude of target Voronoi-cell violations.

    A violation occurs when the point is farther from the target centroid
    than from some competing centroid.
    """
    x = np.asarray(x, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    distances = np.sum((centroids - x) ** 2, axis=1)
    target_distance = distances[target_cluster]

    violation_values = []

    for j, distance_j in enumerate(distances):
        if j == target_cluster:
            continue

        violation = float(target_distance - distance_j)

        if violation > tolerance:
            violation_values.append(violation)

    if len(violation_values) == 0:
        return 0, 0.0, np.nan

    return (
        int(len(violation_values)),
        float(max(violation_values)),
        float(np.mean(violation_values)),
    )


def safe_relative_underestimation(voronoi_cost, vardakas_cost):
    if not np.isfinite(voronoi_cost) or not np.isfinite(vardakas_cost):
        return np.nan

    if abs(voronoi_cost) <= 1e-14:
        return np.nan

    return float(
        100.0 * (voronoi_cost - vardakas_cost) / voronoi_cost
    )





def project_onto_bisecting_hyperplane(a, centroids, source_cluster, target_cluster):
    """
    Vardakas-style least-cost k-means counterfactual.

    This projects the factual point onto the bisecting hyperplane between
    the source centroid and target centroid.

    No alpha contraction is used.
    """
    a = np.asarray(a, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    c_s = centroids[source_cluster]
    c_t = centroids[target_cluster]

    normal = 2.0 * (c_s - c_t)
    rhs = float(np.dot(c_s, c_s) - np.dot(c_t, c_t))

    denom = float(np.dot(normal, normal))

    if denom <= 1e-14:
        return None, False, "Degenerate source-target centroid pair."

    signed_distance_numerator = float(np.dot(normal, a) - rhs)

    z = a - (signed_distance_numerator / denom) * normal

    return z, True, None


def project_onto_target_voronoi_cell(a, centroids, target_cluster):
    """
    Least-cost Voronoi counterfactual.

    This projects the factual point onto the full unweighted target
    Voronoi cell.

    No alpha contraction is used.
    No SHARK weights are used.
    """
    a = np.asarray(a, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    A, b = get_voronoi_halfspaces(
        centroids=centroids,
        target_cluster=target_cluster,
    )

    if is_inside_target_voronoi_cell(
        x=a,
        centroids=centroids,
        target_cluster=target_cluster,
    ):
        return a.copy(), True, None

    constraints = []

    for row, rhs in zip(A, b):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, row=row, rhs=rhs: rhs - np.dot(row, x),
                "jac": lambda x, row=row, rhs=rhs: -row,
            }
        )

    def objective(x):
        return 0.5 * np.sum((x - a) ** 2)

    def gradient(x):
        return x - a

    result = minimize(
        objective,
        x0=a.copy(),
        jac=gradient,
        constraints=constraints,
        method="SLSQP",
        options={
            "ftol": 1e-10,
            "maxiter": 500,
            "disp": False,
        },
    )

    if not result.success:
        # The target centroid is always inside its own Voronoi cell unless
        # centroids are degenerate. Retry from the target centroid as a
        # feasible starting point.
        result = minimize(
            objective,
            x0=centroids[target_cluster].copy(),
            jac=gradient,
            constraints=constraints,
            method="SLSQP",
            options={
                "ftol": 1e-10,
                "maxiter": 500,
                "disp": False,
            },
        )

    if not result.success:
        return None, False, result.message

    z = np.asarray(result.x, dtype=float)

    if not is_inside_target_voronoi_cell(
        x=z,
        centroids=centroids,
        target_cluster=target_cluster,
        tolerance=1e-6,
    ):
        return z, False, "Optimiser returned a point outside the target Voronoi cell."

    return z, True, None


def get_effective_sample_count(
    *,
    cluster_size,
    smallest_cluster_size,
    sampling_mode,
    sample_percentage,
    sample_count_per_cluster,
):
    if sampling_mode == "Percentage of Each Cluster Sampled":
        selected_count = int(np.ceil(cluster_size * sample_percentage / 100.0))
        return max(1, min(selected_count, cluster_size))

    # If the requested count is greater than the size of the smallest
    # cluster, the effective count used for every cluster is capped at
    # the size of the smallest cluster. This keeps factual sampling
    # balanced across clusters.
    effective_count = min(
        int(sample_count_per_cluster),
        int(smallest_cluster_size),
    )

    return int(min(effective_count, cluster_size))


def sample_factual_indices_by_cluster(
    *,
    labels,
    sampling_mode,
    sample_percentage,
    sample_count_per_cluster,
    rng,
):
    labels = np.asarray(labels)
    cluster_ids = sorted(np.unique(labels).astype(int).tolist())

    cluster_sizes = {
        cluster_id: int(np.sum(labels == cluster_id))
        for cluster_id in cluster_ids
    }

    smallest_cluster_size = min(cluster_sizes.values())

    sampled = {}

    for cluster_id in cluster_ids:
        cluster_indices = np.flatnonzero(labels == cluster_id)
        cluster_size = int(len(cluster_indices))

        n_to_sample = get_effective_sample_count(
            cluster_size=cluster_size,
            smallest_cluster_size=smallest_cluster_size,
            sampling_mode=sampling_mode,
            sample_percentage=sample_percentage,
            sample_count_per_cluster=sample_count_per_cluster,
        )

        sampled_indices = rng.choice(
            cluster_indices,
            size=n_to_sample,
            replace=False,
        )

        sampled[cluster_id] = sampled_indices.tolist()

    return sampled


def choose_target_clusters(
    *,
    source_cluster,
    all_cluster_ids,
    target_cluster_selection_mode,
    rng,
):
    valid_targets = [
        cluster_id
        for cluster_id in all_cluster_ids
        if cluster_id != source_cluster
    ]

    if target_cluster_selection_mode == "Select All Target Clusters":
        return valid_targets

    if len(valid_targets) == 0:
        return []

    return [int(rng.choice(valid_targets))]


def summarise_comparative_results(df_results, dataset_name=None):
    total_comparisons = int(len(df_results))

    if total_comparisons == 0:
        return {
            "dataset_name": dataset_name,
            "total_comparisons": 0,
            "vardakas_fail_count": 0,
            "vardakas_fail_percentage": 0.0,
            "vardakas_success_count": 0,
            "voronoi_success_count": 0,
            "mean_vardakas_cost_successful": np.nan,
            "mean_voronoi_cost_successful": np.nan,

            "vardakas_target_cell_validity_rate": 0.0,
            "voronoi_target_cell_validity_rate": 0.0,

            "mean_cost_underestimation_on_failures": np.nan,
            "mean_relative_underestimation_pct_on_failures": np.nan,
            "mean_repair_cost_on_failures": np.nan,
            "mean_violated_constraints_on_failures": np.nan,
            "mean_max_violation_on_failures": np.nan,
            "mean_valid_case_cost_gap": np.nan,
        }

    vardakas_fail_count = int((df_results["vardakas_voronoi_status"] == "FAIL").sum())
    vardakas_success_count = int((df_results["vardakas_voronoi_status"] == "SUCCESS").sum())
    voronoi_success_count = int((df_results["voronoi_status"] == "SUCCESS").sum())

    voronoi_target_cell_valid_count = int(
        df_results["voronoi_inside_target_voronoi"].sum()
    )

    vardakas_target_cell_valid_count = int(
        df_results["vardakas_inside_target_voronoi"].sum()
    )

    failed_vardakas_df = df_results[
        df_results["vardakas_voronoi_status"] == "FAIL"
    ]

    vardakas_success_df = df_results[
        df_results["vardakas_voronoi_status"] == "SUCCESS"
    ]

    voronoi_success_df = df_results[
        df_results["voronoi_status"] == "SUCCESS"
    ]

    mean_vardakas_cost_successful = (
        float(vardakas_success_df["vardakas_cost"].mean())
        if len(vardakas_success_df) > 0
        else np.nan
    )

    mean_voronoi_cost_successful = (
        float(voronoi_success_df["voronoi_cost"].mean())
        if len(voronoi_success_df) > 0
        else np.nan
    )

    return {
        "dataset_name": dataset_name,
        "total_comparisons": total_comparisons,
        "vardakas_fail_count": vardakas_fail_count,
        "vardakas_fail_percentage": 100.0 * vardakas_fail_count / total_comparisons,
        "vardakas_success_count": vardakas_success_count,
        "voronoi_success_count": voronoi_success_count,
        "mean_vardakas_cost_successful": mean_vardakas_cost_successful,
        "mean_voronoi_cost_successful": mean_voronoi_cost_successful,
        "vardakas_target_cell_validity_rate": (
            100.0 * vardakas_target_cell_valid_count / total_comparisons
        ),
        "voronoi_target_cell_validity_rate": (
            100.0 * voronoi_target_cell_valid_count / total_comparisons
        ),
        "mean_cost_underestimation_on_failures": (
            float(failed_vardakas_df["cost_difference_voronoi_minus_vardakas"].mean())
            if len(failed_vardakas_df) > 0
            else np.nan
        ),
        "mean_relative_underestimation_pct_on_failures": (
            float(failed_vardakas_df["relative_underestimation_pct"].mean())
            if len(failed_vardakas_df) > 0
            else np.nan
        ),
        "mean_repair_cost_on_failures": (
            float(failed_vardakas_df["repair_cost_if_vardakas_failed"].mean())
            if len(failed_vardakas_df) > 0
            else np.nan
        ),
        "mean_violated_constraints_on_failures": (
            float(failed_vardakas_df["vardakas_num_violated_constraints"].mean())
            if len(failed_vardakas_df) > 0
            else np.nan
        ),
        "mean_max_violation_on_failures": (
            float(failed_vardakas_df["vardakas_max_voronoi_violation"].mean())
            if len(failed_vardakas_df) > 0
            else np.nan
        ),
        "mean_valid_case_cost_gap": (
            float(df_results["valid_case_cost_gap"].mean())
            if df_results["valid_case_cost_gap"].notna().any()
            else np.nan
        ),
    }

def summarise_comparative_results_by_iteration(
    df_results,
    dataset_name=None,
):
    """
    Reduce the comparative experiment to one summary row per
    independently fitted k-means iteration.

    These iteration-level values are the source of the standard
    deviations reported in Table 2.
    """
    columns = [
        "dataset_name",
        "iteration",
        "iteration_seed",
        "total_comparisons",
        "vardakas_fail_count",
        "vardakas_target_cell_validity_rate",
        "voronoi_target_cell_validity_rate",
        "table2_vardakas_repair_cost",
        "table2_voronoi_repair_cost",
    ]

    if df_results.empty:
        return pd.DataFrame(columns=columns)

    iteration_rows = []

    for (iteration, iteration_seed), group in df_results.groupby(
        ["iteration", "iteration_seed"],
        sort=True,
    ):
        run_summary = summarise_comparative_results(
            group,
            dataset_name=dataset_name,
        )

        # Table 2 reports repair cost as zero when no repair is required.
        if run_summary["vardakas_fail_count"] == 0:
            vardakas_repair_cost = 0.0
        else:
            vardakas_repair_cost = run_summary[
                "mean_repair_cost_on_failures"
            ]

        iteration_rows.append(
            {
                "dataset_name": dataset_name,
                "iteration": int(iteration),
                "iteration_seed": int(iteration_seed),
                "total_comparisons": run_summary["total_comparisons"],
                "vardakas_fail_count": run_summary["vardakas_fail_count"],
                "vardakas_target_cell_validity_rate": run_summary[
                    "vardakas_target_cell_validity_rate"
                ],
                "voronoi_target_cell_validity_rate": run_summary[
                    "voronoi_target_cell_validity_rate"
                ],
                "table2_vardakas_repair_cost": vardakas_repair_cost,
                "table2_voronoi_repair_cost": 0.0,
            }
        )

    return pd.DataFrame(iteration_rows)

def add_comparative_iteration_standard_deviations(
    summary,
    iteration_summary,
):
    """
    Add Table 2 run-to-run standard deviations to the existing
    pooled comparative summary.

    The existing pooled values are retained unchanged. Standard
    deviations are sample standard deviations across independently
    fitted k-means iterations (ddof=1).
    """
    summary = dict(summary)

    n_iterations = int(len(iteration_summary))
    summary["n_iterations"] = n_iterations

    if n_iterations < 2:
        summary["vardakas_target_cell_validity_rate_std"] = np.nan
        summary["voronoi_target_cell_validity_rate_std"] = np.nan
        summary["table2_vardakas_repair_cost_std"] = np.nan
        summary["table2_voronoi_repair_cost_std"] = np.nan

    else:
        summary["vardakas_target_cell_validity_rate_std"] = float(
            iteration_summary[
                "vardakas_target_cell_validity_rate"
            ].std(ddof=1)
        )

        summary["voronoi_target_cell_validity_rate_std"] = float(
            iteration_summary[
                "voronoi_target_cell_validity_rate"
            ].std(ddof=1)
        )

        summary["table2_vardakas_repair_cost_std"] = float(
            iteration_summary[
                "table2_vardakas_repair_cost"
            ].std(ddof=1)
        )

        summary["table2_voronoi_repair_cost_std"] = float(
            iteration_summary[
                "table2_voronoi_repair_cost"
            ].std(ddof=1)
        )

    # Preserve the existing pooled Table 2 value before the slash.
    if summary["vardakas_fail_count"] == 0:
        summary["table2_vardakas_repair_cost"] = 0.0
    else:
        summary["table2_vardakas_repair_cost"] = summary[
            "mean_repair_cost_on_failures"
        ]

    # VoICE requires no repair under the Table 2 criterion.
    summary["table2_voronoi_repair_cost"] = 0.0

    return summary

def run_kmeans_voronoi_vs_bisector_evaluation(
    *,
    X,
    feature_names,
    dataset_name,
    n_clusters,
    n_iterations,
    sampling_mode,
    sample_percentage,
    sample_count_per_cluster,
    target_cluster_selection_mode,
    base_seed,
    n_init=1,
    max_iter=300,
    tol=1e-4,
):
    """
    Compare unweighted k-means Voronoi-region counterfactuals against
    Vardakas-style bisecting-hyperplane counterfactuals.

    Important:
    This function deliberately fits new local k-means objects inside the
    comparative evaluation loop. It does not mutate cluster_result,
    shark_result, Streamlit session state, or any objects used by the other
    app tabs.

    No alpha contraction is used.
    No SHARK weighting is used.
    """
    X = np.asarray(X, dtype=float)
    feature_names = list(feature_names)

    rows = []

    for iteration in range(int(n_iterations)):
        iteration_seed = int(base_seed) + iteration

        # Local k-means model for this comparative iteration only.
        # Do not assign this object to cluster_result or st.session_state.
        comparative_kmeans = KMeans(
            n_clusters=int(n_clusters),
            random_state=iteration_seed,
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
        )

        labels = comparative_kmeans.fit_predict(X)
        centroids = comparative_kmeans.cluster_centers_

        all_cluster_ids = sorted(np.unique(labels).astype(int).tolist())

        rng = np.random.default_rng(iteration_seed + 10_000)

        sampled_by_cluster = sample_factual_indices_by_cluster(
            labels=labels,
            sampling_mode=sampling_mode,
            sample_percentage=sample_percentage,
            sample_count_per_cluster=sample_count_per_cluster,
            rng=rng,
        )

        for source_cluster, factual_indices in sampled_by_cluster.items():
            for factual_index in factual_indices:
                a = X[factual_index]

                target_clusters = choose_target_clusters(
                    source_cluster=source_cluster,
                    all_cluster_ids=all_cluster_ids,
                    target_cluster_selection_mode=target_cluster_selection_mode,
                    rng=rng,
                )

                for target_cluster in target_clusters:
                    vardakas_z, vardakas_ok, vardakas_error = (
                        project_onto_bisecting_hyperplane(
                            a=a,
                            centroids=centroids,
                            source_cluster=source_cluster,
                            target_cluster=target_cluster,
                        )
                    )

                    if vardakas_ok:
                        vardakas_cost = squared_euclidean_cost(a, vardakas_z)

                        vardakas_inside_voronoi = is_inside_target_voronoi_cell(
                            x=vardakas_z,
                            centroids=centroids,
                            target_cluster=target_cluster,
                        )

                        vardakas_status = (
                            "SUCCESS"
                            if vardakas_inside_voronoi
                            else "FAIL"
                        )

                        vardakas_assigned_cluster = assigned_cluster(
                            x=vardakas_z,
                            centroids=centroids,
                        )

                        (
                            vardakas_num_violated_constraints,
                            vardakas_max_violation,
                            vardakas_mean_violation,
                        ) = target_voronoi_violations(
                            x=vardakas_z,
                            centroids=centroids,
                            target_cluster=target_cluster,
                        )

                    else:
                        vardakas_cost = np.nan
                        vardakas_inside_voronoi = False
                        vardakas_status = "FAIL"
                        vardakas_assigned_cluster = np.nan
                        vardakas_num_violated_constraints = np.nan
                        vardakas_max_violation = np.nan
                        vardakas_mean_violation = np.nan

                    voronoi_z, voronoi_ok, voronoi_error = (
                        project_onto_target_voronoi_cell(
                            a=a,
                            centroids=centroids,
                            target_cluster=target_cluster,
                        )
                    )

                    if voronoi_ok:
                        voronoi_cost = squared_euclidean_cost(a, voronoi_z)
                        voronoi_status = "SUCCESS"

                        voronoi_inside_voronoi = is_inside_target_voronoi_cell(
                            x=voronoi_z,
                            centroids=centroids,
                            target_cluster=target_cluster,
                            tolerance=1e-6,
                        )

                        voronoi_assigned_cluster = assigned_cluster(
                            x=voronoi_z,
                            centroids=centroids,
                        )

                    else:
                        voronoi_cost = np.nan
                        voronoi_status = "FAIL"
                        voronoi_inside_voronoi = False
                        voronoi_assigned_cluster = np.nan

                    #end-if

                    cost_difference = (
                        voronoi_cost - vardakas_cost
                        if np.isfinite(voronoi_cost) and np.isfinite(vardakas_cost)
                        else np.nan
                    )

                    relative_underestimation_pct = safe_relative_underestimation(
                        voronoi_cost=voronoi_cost,
                        vardakas_cost=vardakas_cost,
                    )

                    if (
                        vardakas_ok
                        and voronoi_ok
                        and not vardakas_inside_voronoi
                    ):
                        repair_cost = squared_euclidean_cost(vardakas_z, voronoi_z)
                    else:
                        repair_cost = np.nan

                    if (
                        vardakas_ok
                        and voronoi_ok
                        and vardakas_inside_voronoi
                    ):
                        valid_case_cost_gap = abs(voronoi_cost - vardakas_cost)
                    else:
                        valid_case_cost_gap = np.nan

                    target_cell_validity_gain = (
                        int(voronoi_inside_voronoi) - int(vardakas_inside_voronoi)
                    )


                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "iteration": iteration + 1,
                            "iteration_seed": iteration_seed,
                            "factual_index": int(factual_index),
                            "source_cluster": int(source_cluster),
                            "target_cluster": int(target_cluster),

                            "vardakas_cost": vardakas_cost,
                            "voronoi_cost": voronoi_cost,
                            "cost_difference_voronoi_minus_vardakas": cost_difference,

                            "relative_underestimation_pct": (
                                relative_underestimation_pct
                                if vardakas_status == "FAIL"
                                else np.nan
                            ),

                            "repair_cost_if_vardakas_failed": repair_cost,

                            "vardakas_inside_target_voronoi": bool(vardakas_inside_voronoi),
                            "voronoi_inside_target_voronoi": bool(voronoi_inside_voronoi),

                            "target_cell_validity_gain": int(target_cell_validity_gain),

                            "vardakas_voronoi_status": vardakas_status,
                            "voronoi_status": voronoi_status,

                            "vardakas_assigned_cluster": vardakas_assigned_cluster,
                            "voronoi_assigned_cluster": voronoi_assigned_cluster,

                            "vardakas_assigned_to_target": (
                                bool(vardakas_assigned_cluster == target_cluster)
                                if np.isfinite(vardakas_assigned_cluster)
                                else False
                            ),

                            "voronoi_assigned_to_target": (
                                bool(voronoi_assigned_cluster == target_cluster)
                                if np.isfinite(voronoi_assigned_cluster)
                                else False
                            ),

                            "vardakas_num_violated_constraints": vardakas_num_violated_constraints,
                            "vardakas_max_voronoi_violation": vardakas_max_violation,
                            "vardakas_mean_voronoi_violation": vardakas_mean_violation,

                            "valid_case_cost_gap": valid_case_cost_gap,

                            "vardakas_error": vardakas_error,
                            "voronoi_error": voronoi_error,
                        }
                    )

    df_results = pd.DataFrame(rows)

    summary = summarise_comparative_results(
        df_results,
        dataset_name=dataset_name,
    )

    iteration_summary = summarise_comparative_results_by_iteration(
        df_results,
        dataset_name=dataset_name,
    )

    summary = add_comparative_iteration_standard_deviations(
        summary,
        iteration_summary,
    )

    return df_results, summary, iteration_summary