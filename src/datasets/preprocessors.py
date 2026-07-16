import numpy as np
import pandas as pd

from src.config import DATASET_INFORMATION


def preprocess_dataset(df: pd.DataFrame, dataset_name: str):
    info = DATASET_INFORMATION[dataset_name]

    df = df.copy()

    df.columns = df.columns.astype(str).str.strip()

    # Remove common CSV artifact columns, e.g. Unnamed: 32
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")

    ground_truth_col = info.get("ground_truth")
    feature_exclusions = info.get("feature_exclusions", [])
    categorical_encoding = info.get("categorical_encoding", "none")
    missing_strategy = info.get("missing_strategy", "error")

    y_true = None

    if ground_truth_col and ground_truth_col in df.columns:
        y_true = (
            df[ground_truth_col]
            .astype("category")
            .cat.codes
            .to_numpy()
        )

    drop_cols = []

    if ground_truth_col and ground_truth_col in df.columns:
        drop_cols.append(ground_truth_col)

    for col in feature_exclusions:
        if col in df.columns and col not in drop_cols:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    if categorical_encoding == "onehot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        if categorical_cols:
            df = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=True,
            )

    elif categorical_encoding == "ordinal":
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes

    elif categorical_encoding == "none":
        pass

    else:
        raise ValueError(
            f"Unknown categorical_encoding={categorical_encoding!r} "
            f"for dataset {dataset_name}"
        )

    if df.isna().any().any():

        if missing_strategy == "drop_rows":

            keep_mask = ~df.isna().any(axis=1)

            df = df.loc[keep_mask].copy()

            if y_true is not None:
                y_true = y_true[keep_mask.to_numpy()]

        elif missing_strategy == "mean_impute":

            df = df.fillna(df.mean(numeric_only=True))

        elif missing_strategy == "error":

            missing_report = (
                df.isna()
                .sum()
                .loc[lambda s: s > 0]
                .sort_values(ascending=False)
            )

            raise ValueError(
                f"Dataset {dataset_name} contains missing values after preprocessing:\n"
                f"{missing_report.to_string()}"
            )

        else:

            raise ValueError(
                f"Unknown missing_strategy={missing_strategy!r} "
                f"for dataset {dataset_name}"
            )

    if y_true is not None and len(y_true) != len(df):
        raise ValueError(
            f"Length mismatch after preprocessing: "
            f"X has {len(df)} rows but y_true has {len(y_true)} labels."
        )

    X_orig = df.astype(float).to_numpy()
    feature_names = df.columns.tolist()

    return X_orig, feature_names, y_true