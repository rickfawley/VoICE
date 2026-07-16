import pandas as pd
import streamlit as st

from src.config import DATASET_INFORMATION


def get_dataset_path(dataset_name: str):
    return DATASET_INFORMATION[dataset_name]["local_path"]


@st.cache_data
def load_local_csv_dataset(dataset_name: str):
    info = DATASET_INFORMATION[dataset_name]
    local_path = info["local_path"]
    sep = info.get("sep", ",")

    if not local_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found at {local_path}"
        )

    return pd.read_csv(local_path, sep=sep)