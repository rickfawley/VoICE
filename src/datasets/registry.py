import streamlit as st

from sklearn.preprocessing import StandardScaler

from .loaders import load_local_csv_dataset
from .preprocessors import preprocess_dataset


@st.cache_data
def load_dataset(dataset_name: str):

    df = load_local_csv_dataset(dataset_name)

    X_orig, feature_names, y_true = preprocess_dataset(
        df,
        dataset_name,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)

    return X_orig, X_scaled, feature_names, scaler, y_true