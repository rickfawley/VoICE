import streamlit as st

from src.config import DATASET_INFORMATION


def render_dataset_selector():

    available_datasets = [
        name
        for name, info in DATASET_INFORMATION.items()
        if info.get("include_in_experiments", True)
    ]

    with st.expander("SELECT DATASET FOR EXPERIMENT", expanded=True):

        with st.expander("Dataset Information", expanded=False):

            table_md = (
                "| **Dataset** | URL |\n"
                "|------------|------|\n"
            )

            for name in available_datasets:
                info = DATASET_INFORMATION[name]
                url = info.get("url", "")

                table_md += f"| {name} | {url} |\n"

            st.markdown(table_md)

            st.caption("Local downloads have been used for this utility.")

        st.caption("Select the dataset for the experiment.")

        dataset_name = st.radio(
            "Dataset",
            options=available_datasets,
            index=0,
            horizontal=False,
            help=(
                "Select the dataset used for the experiment.\n\n"
                "Descriptions can be found in the **Dataset Information** expander\n"
            ),
        )

        default_k = DATASET_INFORMATION.get(
            dataset_name,
            {},
        ).get("default_k", 2)

    return dataset_name, default_k