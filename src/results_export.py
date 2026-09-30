import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_LEVEL_PUBLICATION_MODES
from src.evaluation import safe_adjusted_rand_index

def safe_path_name(value):
    """
    Return a file-system safe version of a user-facing name.
    """

    value = str(value)
    value = re.sub(r'[<>:"/\\|?*]', "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def format_ari(ari):
    """
    Format ARI for use in result folder names.
    """

    if ari is None or not np.isfinite(ari):
        return "NA"

    return f"{ari:.4f}"

def make_json_serializable(value):
    """
    Convert NumPy and Path values into JSON-serializable Python values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return value

def build_result_folder_name(alg_name, dataset_name, ari):
    """
    Build the required result folder name:

        [alg] [dataset] (ARI=xxx)
    """

    alg_name = safe_path_name(alg_name)
    dataset_name = safe_path_name(dataset_name)
    return f"{alg_name} {dataset_name} (ARI={format_ari(ari)})"


def build_assignments_dataframe(
    *,
    X_cluster,
    feature_names,
    labels,
    y_true=None,
):
    """
    Build the row-level exported dataset containing features, cluster labels,
    and ground-truth labels where available.
    """

    labels = np.asarray(labels)

    df_assignments = pd.DataFrame(
        X_cluster,
        columns=feature_names,
    )

    df_assignments.insert(0, "cluster", labels)
    df_assignments.insert(0, "row_id", np.arange(len(labels)))

    if y_true is not None:
        y_true = np.asarray(y_true)

        if len(y_true) == len(labels):
            df_assignments.insert(2, "y_true", y_true)

    return df_assignments

def write_csv_export(
    *,
    df,
    experiment_dir,
    summary_dir,
    history_dir,
    master_stem,
    history_stem,
    timestamp=None,
):
    """
    Write the current/master CSV to both:

        results/experiments/[experiment folder]/
        results/summary/

    If timestamp is provided, also write a timestamped backup to:

        results/history/
    """

    experiment_path = experiment_dir / f"{master_stem}.csv"
    summary_path = summary_dir / f"{master_stem}.csv"

    df.to_csv(experiment_path, index=False)
    df.to_csv(summary_path, index=False)

    history_path = None

    if timestamp is not None:
        history_path = history_dir / f"{history_stem}_{timestamp}.csv"
        df.to_csv(history_path, index=False)

    return experiment_path, summary_path, history_path


def write_json_export(
    *,
    payload,
    experiment_dir,
    summary_dir,
    history_dir,
    master_stem,
    history_stem,
    timestamp=None,
):
    """
    Write the current/master JSON to both:

        results/experiments/[experiment folder]/
        results/summary/

    If timestamp is provided, also write a timestamped backup to:

        results/history/
    """

    experiment_path = experiment_dir / f"{master_stem}.json"
    summary_path = summary_dir / f"{master_stem}.json"

    payload = make_json_serializable(payload)

    with open(experiment_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    history_path = None

    if timestamp is not None:
        history_path = history_dir / f"{history_stem}_{timestamp}.json"

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return experiment_path, summary_path, history_path


def export_cluster_solution(
    *,
    cluster_result,
    X_cluster,
    feature_names,
    y_true,
    dataset_name,
    n_clusters,
    seed,
    n_init,
    max_iter,
    tol,
    results_dir="results",
    alg_name=None,
    create_timestamped_dataset=False,
    alpha_summary_df=None,
):
    """
    Export one fitted clustering solution.

    The export folder is named:

        results/[alg] [dataset] (ARI=xxx)/

    Master files are always created or overwritten:

        dataset.csv
        centroids.csv
        feature_weights.csv
        metadata.json

    If create_timestamped_dataset is True, timestamped backups are also created:

        dataset_YYYYmmdd_HHMMSS_microseconds.csv
        centroids_YYYYmmdd_HHMMSS_microseconds.csv
        feature_weights_YYYYmmdd_HHMMSS_microseconds.csv
        metadata_YYYYmmdd_HHMMSS_microseconds.json
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = results_dir / "experiments"
    summary_dir = results_dir / "summary"
    history_dir = results_dir / "history"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    alg = alg_name or cluster_result.method
    labels = np.asarray(cluster_result.labels)
    ari = safe_adjusted_rand_index(labels, y_true)
    file_prefix = safe_path_name(f"{alg} {dataset_name}")

    folder_name = build_result_folder_name(
        alg_name=alg,
        dataset_name=dataset_name,
        ari=ari,
    )

    history_prefix = safe_path_name(folder_name)

    experiment_dir = experiments_dir / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    timestamp = None

    if create_timestamped_dataset:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    df_assignments = build_assignments_dataframe(
        X_cluster=X_cluster,
        feature_names=feature_names,
        labels=labels,
        y_true=y_true,
    )

    df_assignments.insert(1, "dataset", dataset_name)
    df_assignments.insert(2, "algorithm", alg)

    dataset_experiment_path, dataset_summary_path, dataset_history_path = (
        write_csv_export(
            df=df_assignments,
            experiment_dir=experiment_dir,
            summary_dir=summary_dir,
            history_dir=history_dir,
            master_stem=f"{file_prefix} dataset",
            history_stem=f"{history_prefix} dataset",
            timestamp=timestamp,
        )
    )

    df_centroids = pd.DataFrame(
        cluster_result.centroids,
        columns=feature_names,
    )

    df_centroids.insert(0, "cluster", np.arange(len(df_centroids)))
    df_centroids.insert(1, "dataset", dataset_name)
    df_centroids.insert(2, "algorithm", alg)

    centroids_experiment_path, centroids_summary_path, centroids_history_path = (
        write_csv_export(
            df=df_centroids,
            experiment_dir=experiment_dir,
            summary_dir=summary_dir,
            history_dir=history_dir,
            master_stem=f"{file_prefix} centroids",
            history_stem=f"{history_prefix} centroids",
            timestamp=timestamp,
        )
    )

    weights = np.asarray(cluster_result.weights, dtype=float)

    df_weights = pd.DataFrame(
        {
            "feature": feature_names,
            "weight": weights,
        }
    ).sort_values("weight", ascending=False)

    df_weights.insert(0, "rank", np.arange(1, len(df_weights) + 1))
    df_weights.insert(1, "dataset", dataset_name)
    df_weights.insert(2, "algorithm", alg)

    feature_weights_experiment_path, feature_weights_summary_path, feature_weights_history_path = (
        write_csv_export(
            df=df_weights,
            experiment_dir=experiment_dir,
            summary_dir=summary_dir,
            history_dir=history_dir,
            master_stem=f"{file_prefix} feature_weights",
            history_stem=f"{history_prefix} feature_weights",
            timestamp=timestamp,
        )
    )

    # --------------------------------------------------------------
    # Alpha 100%-retention diagnostics export
    # --------------------------------------------------------------

    alpha_experiment_path = None
    alpha_summary_path = None
    alpha_history_path = None

    if alpha_summary_df is not None:

        df_alpha = alpha_summary_df.copy()

        if "dataset" not in df_alpha.columns:
            df_alpha.insert(0, "dataset", dataset_name)

        if "algorithm" not in df_alpha.columns:
            df_alpha.insert(1, "algorithm", alg)

        alpha_experiment_path, alpha_summary_path, alpha_history_path = (
            write_csv_export(
                df=df_alpha,
                experiment_dir=experiment_dir,
                summary_dir=summary_dir,
                history_dir=history_dir,
                master_stem=f"{file_prefix} alpha_100_by_cluster",
                history_stem=f"{history_prefix} alpha_100_by_cluster",
                timestamp=timestamp,
            )
        )




    metadata_master_stem = f"{file_prefix} metadata"
    metadata_history_stem = f"{history_prefix} metadata"

    metadata_experiment_path = experiment_dir / f"{metadata_master_stem}.json"
    metadata_summary_path = summary_dir / f"{metadata_master_stem}.json"

    metadata_history_path = None

    if timestamp is not None:
        metadata_history_path = (
            history_dir / f"{metadata_history_stem}_{timestamp}.json"
        )

    metadata_master_stem = f"{file_prefix} metadata"
    metadata_history_stem = f"{history_prefix} metadata"

    metadata = {
        "algorithm": alg,
        "dataset": dataset_name,
        "folder_name": folder_name,
        "experiment_dir": str(experiment_dir),
        "summary_dir": str(summary_dir),
        "history_dir": str(history_dir),
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "inertia": float(cluster_result.inertia),
        "ARI": None if not np.isfinite(ari) else float(ari),
        "n_observations": int(np.asarray(X_cluster).shape[0]),
        "n_features": int(np.asarray(X_cluster).shape[1]),
        "feature_names": list(feature_names),
        "create_timestamped_dataset": bool(create_timestamped_dataset),
        "timestamp": timestamp,
        "paths": {
            "experiment": {
                "dataset": str(dataset_experiment_path),
                "centroids": str(centroids_experiment_path),
                "feature_weights": str(feature_weights_experiment_path),
                "alpha_100_by_cluster": (
                    None
                    if alpha_experiment_path is None
                    else str(alpha_experiment_path)
                ),
                "metadata": str(metadata_experiment_path),
            },
            "summary": {
                "dataset": str(dataset_summary_path),
                "centroids": str(centroids_summary_path),
                "feature_weights": str(feature_weights_summary_path),
                "alpha_100_by_cluster": (
                    None
                    if alpha_summary_path is None
                    else str(alpha_summary_path)
                ),
                "metadata": str(metadata_summary_path),
            },
            "history": {
                "dataset": (
                    None
                    if dataset_history_path is None
                    else str(dataset_history_path)
                ),
                "centroids": (
                    None
                    if centroids_history_path is None
                    else str(centroids_history_path)
                ),
                "feature_weights": (
                    None
                    if feature_weights_history_path is None
                    else str(feature_weights_history_path)
                ),
                "alpha_100_by_cluster": (
                    None
                    if alpha_history_path is None
                    else str(alpha_history_path)
                ),
                "metadata": (
                    None
                    if metadata_history_path is None
                    else str(metadata_history_path)
                ),
            },
        },
    }

    metadata_experiment_path, metadata_summary_path, metadata_history_path = (
        write_json_export(
            payload=metadata,
            experiment_dir=experiment_dir,
            summary_dir=summary_dir,
            history_dir=history_dir,
            master_stem=metadata_master_stem,
            history_stem=metadata_history_stem,
            timestamp=timestamp,
        )
    )

    return experiment_dir

def export_table3_repeated_results(
    *,
    df_table3_iterations,
    table3_standard_deviations,
    dataset_name,
    n_clusters,
    seed,
    n_init,
    max_iter,
    tol,
    n_factuals,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
    sidebar_mask,
    results_dir="results",
    create_timestamped_dataset=False,
):
    """
    Export the repetition-level values and standard deviations
    used to populate Table 3.
    """

    results_dir = Path(results_dir)

    experiments_dir = results_dir / "experiments"
    summary_dir = results_dir / "summary"
    history_dir = results_dir / "history"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    export_name = build_model_level_export_name(
        dataset_name=dataset_name,
        n_clusters=n_clusters,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        n_factuals=n_factuals,
        cf_modes=MODEL_LEVEL_PUBLICATION_MODES,
        alpha_mode=alpha_mode,
        manual_alpha=manual_alpha,
        alpha_retain_fraction=alpha_retain_fraction,
        sidebar_mask=sidebar_mask,
    )

    experiment_dir = experiments_dir / export_name
    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = None

    if create_timestamped_dataset:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

    df_iterations = df_table3_iterations.copy()

    if "dataset" not in df_iterations.columns:
        df_iterations.insert(
            0,
            "dataset",
            dataset_name,
        )

    df_std = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_clusters": int(n_clusters),
                "n_factuals": int(n_factuals),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "alpha_mode": alpha_mode,
                "manual_alpha": float(manual_alpha),
                "alpha_retain_fraction": float(
                    alpha_retain_fraction
                ),
                **table3_standard_deviations,
            }
        ]
    )

    write_csv_export(
        df=df_iterations,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=f"{export_name} table3_iterations",
        history_stem=f"{export_name} table3_iterations",
        timestamp=timestamp,
    )

    write_csv_export(
        df=df_std,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=(
            f"{export_name} table3_standard_deviations"
        ),
        history_stem=(
            f"{export_name} table3_standard_deviations"
        ),
        timestamp=timestamp,
    )

    return experiment_dir

def export_table4_repeated_results(
    *,
    df_table4_iterations,
    table4_standard_deviations,
    dataset_name,
    n_clusters,
    seed,
    n_init,
    max_iter,
    tol,
    n_factuals,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
    sidebar_mask,
    results_dir="results",
    create_timestamped_dataset=False,
):
    """
    Export the repetition-level values and standard deviations
    used to populate Table 4.
    """

    results_dir = Path(results_dir)

    experiments_dir = results_dir / "experiments"
    summary_dir = results_dir / "summary"
    history_dir = results_dir / "history"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    export_name = build_model_level_export_name(
        dataset_name=dataset_name,
        n_clusters=n_clusters,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        n_factuals=n_factuals,
        cf_modes=MODEL_LEVEL_PUBLICATION_MODES,
        alpha_mode=alpha_mode,
        manual_alpha=manual_alpha,
        alpha_retain_fraction=alpha_retain_fraction,
        sidebar_mask=sidebar_mask,
    )

    experiment_dir = experiments_dir / export_name
    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = None

    if create_timestamped_dataset:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

    df_iterations = df_table4_iterations.copy()

    if "dataset" not in df_iterations.columns:
        df_iterations.insert(
            0,
            "dataset",
            dataset_name,
        )

    df_std = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_clusters": int(n_clusters),
                "n_factuals": int(n_factuals),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "alpha_mode": alpha_mode,
                "manual_alpha": float(manual_alpha),
                "alpha_retain_fraction": float(
                    alpha_retain_fraction
                ),
                **table4_standard_deviations,
            }
        ]
    )

    write_csv_export(
        df=df_iterations,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=f"{export_name} table4_iterations",
        history_stem=f"{export_name} table4_iterations",
        timestamp=timestamp,
    )

    write_csv_export(
        df=df_std,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=(
            f"{export_name} table4_standard_deviations"
        ),
        history_stem=(
            f"{export_name} table4_standard_deviations"
        ),
        timestamp=timestamp,
    )

    return experiment_dir


def build_model_level_export_name(
    *,
    dataset_name,
    n_clusters,
    seed,
    n_init,
    max_iter,
    tol,
    n_factuals,
    cf_modes,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
    sidebar_mask,
):
    """
    Build a standard model-level export name.

    Format:

        Model Level (arg1, arg2, ...) Dataset [dataset]
    """

    mode_aliases = {
        "Unweighted k-means": "Unweighted",
        "Ranked k-means": "Ranked",
        "Ranked + weighted SHARK": "SHARK",
    }

    mode_text = "+".join(
        mode_aliases.get(str(mode), safe_path_name(str(mode)))
        for mode in cf_modes
    )

    mask_text = "".join(
        str(int(value))
        for value in np.asarray(sidebar_mask, dtype=int).tolist()
    )

    alpha_mode_text = str(alpha_mode).replace(" ", "_")

    args_text = (
        f"k={int(n_clusters)}, "
        f"seed={int(seed)}, "
        f"n={int(n_factuals)}, "
        f"n_init={int(n_init)}, "
        f"max_iter={int(max_iter)}, "
        f"tol={float(tol):.6g}, "
        f"modes={mode_text}, "
        f"alpha={alpha_mode_text}, "
        f"manual_alpha={float(manual_alpha):.3f}, "
        f"retain={float(alpha_retain_fraction):.2f}, "
        f"mask={mask_text}"
    )

    #return safe_path_name(
    #    f"Model Level ({args_text}) Dataset {dataset_name}"
    #)

    return safe_path_name(
        f"{mode_text} {dataset_name}"
    )

def export_model_level_evaluation_tables(
    *,
    df_eval_results,
    df_eval_summary,
    dataset_name,
    n_clusters,
    seed,
    n_init,
    max_iter,
    tol,
    n_factuals,
    cf_modes,
    alpha_mode,
    manual_alpha,
    alpha_retain_fraction,
    sidebar_mask,
    results_dir="results",
    create_timestamped_dataset=False,
):
    """
    Export model-level evaluation tables using the project results protocol.

    Current/master files are written to both:

        results/experiments/[model-level folder]/
        results/summary/

    Timestamped backup files are written to:

        results/history/

    when create_timestamped_dataset is True.
    """

    results_dir = Path(results_dir)

    experiments_dir = results_dir / "experiments"
    summary_dir = results_dir / "summary"
    history_dir = results_dir / "history"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    export_name = build_model_level_export_name(
        dataset_name=dataset_name,
        n_clusters=n_clusters,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        n_factuals=n_factuals,
        cf_modes=cf_modes,
        alpha_mode=alpha_mode,
        manual_alpha=manual_alpha,
        alpha_retain_fraction=alpha_retain_fraction,
        sidebar_mask=sidebar_mask,
    )

    folder_name = export_name
    file_prefix = export_name
    history_prefix = export_name

    experiment_dir = experiments_dir / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    timestamp = None

    if create_timestamped_dataset:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # --------------------------------------------------------------
    # Split summary metrics into paper-facing least-cost and
    # parsimonious tables
    # --------------------------------------------------------------

    df_least_cost_metrics = df_eval_summary[
        df_eval_summary["solution_type"] == "least_cost"
    ].copy()

    df_parsimonious_metrics = df_eval_summary[
        df_eval_summary["solution_type"] == "most_parsimonious"
    ].copy()

    # --------------------------------------------------------------
    # Split detailed row-level results too
    # --------------------------------------------------------------

    df_least_cost_results = df_eval_results[
        df_eval_results["solution_type"] == "least_cost"
    ].copy()

    df_parsimonious_results = df_eval_results[
        df_eval_results["solution_type"] == "most_parsimonious"
    ].copy()

    export_context = {
        "dataset": dataset_name,
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "n_factuals": int(n_factuals),
    }

    def add_export_context(df):
        df = df.copy()

        for column_name, value in reversed(list(export_context.items())):
            if column_name not in df.columns:
                df.insert(0, column_name, value)

        return df

    df_least_cost_metrics = add_export_context(df_least_cost_metrics)
    df_parsimonious_metrics = add_export_context(df_parsimonious_metrics)
    df_least_cost_results = add_export_context(df_least_cost_results)
    df_parsimonious_results = add_export_context(df_parsimonious_results)

    # --------------------------------------------------------------
    # Export displayed metric tables
    # --------------------------------------------------------------

    (
        least_cost_metrics_experiment_path,
        least_cost_metrics_summary_path,
        least_cost_metrics_history_path,
    ) = write_csv_export(
        df=df_least_cost_metrics,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=f"{file_prefix} least_cost_metrics",
        history_stem=f"{history_prefix} least_cost_metrics",
        timestamp=timestamp,
    )

    (
        parsimonious_metrics_experiment_path,
        parsimonious_metrics_summary_path,
        parsimonious_metrics_history_path,
    ) = write_csv_export(
        df=df_parsimonious_metrics,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=f"{file_prefix} parsimonious_metrics",
        history_stem=f"{history_prefix} parsimonious_metrics",
        timestamp=timestamp,
    )

    # --------------------------------------------------------------
    # Export detailed result tables
    # --------------------------------------------------------------

    (
        least_cost_results_experiment_path,
        least_cost_results_summary_path,
        least_cost_results_history_path,
    ) = write_csv_export(
        df=df_least_cost_results,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=f"{file_prefix} least_cost_results",
        history_stem=f"{history_prefix} least_cost_results",
        timestamp=timestamp,
    )

    (
        parsimonious_results_experiment_path,
        parsimonious_results_summary_path,
        parsimonious_results_history_path,
    ) = write_csv_export(
        df=df_parsimonious_results,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=f"{file_prefix} parsimonious_results",
        history_stem=f"{history_prefix} parsimonious_results",
        timestamp=timestamp,
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata_master_stem = f"{file_prefix} metadata"
    metadata_history_stem = f"{history_prefix} metadata"

    metadata_experiment_path = experiment_dir / f"{metadata_master_stem}.json"
    metadata_summary_path = summary_dir / f"{metadata_master_stem}.json"

    metadata_history_path = None

    if timestamp is not None:
        metadata_history_path = (
            history_dir / f"{metadata_history_stem}_{timestamp}.json"
        )

    metadata = {
        "export_type": "model_level_evaluation",
        "dataset": dataset_name,
        "folder_name": folder_name,
        "experiment_dir": str(experiment_dir),
        "summary_dir": str(summary_dir),
        "history_dir": str(history_dir),
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "n_factuals": int(n_factuals),
        "cf_modes": list(cf_modes),
        "alpha_mode": alpha_mode,
        "manual_alpha": float(manual_alpha),
        "alpha_retain_fraction": float(alpha_retain_fraction),
        "sidebar_mask": np.asarray(sidebar_mask, dtype=int).tolist(),
        "create_timestamped_dataset": bool(create_timestamped_dataset),
        "timestamp": timestamp,
        "paths": {
            "experiment": {
                "least_cost_metrics": str(
                    least_cost_metrics_experiment_path
                ),
                "parsimonious_metrics": str(
                    parsimonious_metrics_experiment_path
                ),
                "least_cost_results": str(
                    least_cost_results_experiment_path
                ),
                "parsimonious_results": str(
                    parsimonious_results_experiment_path
                ),
                "metadata": str(metadata_experiment_path),
            },
            "summary": {
                "least_cost_metrics": str(
                    least_cost_metrics_summary_path
                ),
                "parsimonious_metrics": str(
                    parsimonious_metrics_summary_path
                ),
                "least_cost_results": str(
                    least_cost_results_summary_path
                ),
                "parsimonious_results": str(
                    parsimonious_results_summary_path
                ),
                "metadata": str(metadata_summary_path),
            },
            "history": {
                "least_cost_metrics": (
                    None
                    if least_cost_metrics_history_path is None
                    else str(least_cost_metrics_history_path)
                ),
                "parsimonious_metrics": (
                    None
                    if parsimonious_metrics_history_path is None
                    else str(parsimonious_metrics_history_path)
                ),
                "least_cost_results": (
                    None
                    if least_cost_results_history_path is None
                    else str(least_cost_results_history_path)
                ),
                "parsimonious_results": (
                    None
                    if parsimonious_results_history_path is None
                    else str(parsimonious_results_history_path)
                ),
                "metadata": (
                    None
                    if metadata_history_path is None
                    else str(metadata_history_path)
                ),
            },
        },
    }

    write_json_export(
        payload=metadata,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        history_dir=history_dir,
        master_stem=metadata_master_stem,
        history_stem=metadata_history_stem,
        timestamp=timestamp,
    )

    return experiment_dir
