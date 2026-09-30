import streamlit as st

from src.config import MODEL_LEVEL_PUBLICATION_MODES as CF_MODES

def render_experiment_controls(
    dataset_name,
    feature_names,
    default_k,
):

    with st.expander("CLUSTERING SETTINGS", expanded=True):

        n_clusters = st.slider("Number of clusters", 2, 6, default_k)

        st.caption(
            f"This is the value normally used for the **{dataset_name}** dataset."
        )

        # Three independent checkboxes
        cf_mode_1 = st.checkbox(
            CF_MODES[0],
            value=True,
            help=(
                "Unweighted k-means uses equal feature weights and original feature order. "
            ),
        )

        cf_mode_2 = st.checkbox(
            CF_MODES[1],
            value=False,
            help=(
                "Ranked k-means uses SHARK-derived feature ranks but keeps unweighted k-means geometry. "
            ),
        )

        cf_mode_3 = st.checkbox(
            CF_MODES[2],
            value=False,
            help=(
                "Ranked + weighted SHARK uses both SHARK-derived ranks and SHARK-weighted geometry."
            ),
        )

        # Collect selected modes into a list
        cf_modes = []

        if cf_mode_1:
            cf_modes.append(CF_MODES[0])

        if cf_mode_2:
            cf_modes.append(CF_MODES[1])

        if cf_mode_3:
            cf_modes.append(CF_MODES[2])

        # Ensure at least one mode is selected
        if len(cf_modes) == 0:
            st.warning("Select at least one counterfactual comparison mode.")
            cf_modes = [CF_MODES[0]]

        with st.expander("Runtime Hyperparameters", expanded=False):
            n_init = st.number_input("n_init", value=10, step=1)
            max_iter = st.number_input("max_iter", value=300, step=1)
            tol = st.number_input("tol", value=1e-4, format="%.6f")
            seed = st.number_input("Random seed", value=42, step=1)

        with st.expander("Export Settings", expanded=False):

            export_dataset_mode = st.radio(
                "Dataset export mode",
                options=[
                    "Create/Overwrite Master Only",
                    "Create/Overwrite Master & Create Timestamped Backups",
                ],
                index=0,
                horizontal=False,
                help=(
                    "The master dataset.csv file is always created or overwritten. "
                    "The timestamped option additionally creates a preserved backup "
                    "copy for each run."
                ),
            )

            create_timestamped_dataset = (
                export_dataset_mode
                == "Create/Overwrite Master & Create Timestamped Backups"
            )

            st.divider()
            st.caption("Exported clustering solutions")

            exported_solution_paths_container = st.container()

        #end-with

    with st.expander("COUNTERFACTUAL EXPLAINABILITY SETTINGS", expanded=True):

        st.caption(
            "This controls display and explanation units only. "
            "Counterfactual optimisation is always performed in scaled clustering space."
        )

        view_space = st.radio(
            "Display / Explanation space",
            options=["Original Metric Space", "Scaled Metric Space"],
            index=1,
            horizontal=False,
            help=(
                "This controls only how points, rays, and deltas are displayed.\n\n"
                "Counterfactual optimisation is always performed in the scaled clustering space. "
                "The selected comparison mode controls whether the solver uses equal weights or SHARK-derived weights.\n\n"
                "• Original Metric Space: display coordinates and deltas in raw feature units.\n"
                "• Scaled Metric Space: display coordinates and deltas in scaled units.\n\n"
                "PCA is used only for visualisation."
            ),
        )

        st.markdown("<hr>", unsafe_allow_html=True)

    return {
        "n_clusters": n_clusters,
        "seed": seed,
        "cf_modes": cf_modes,
        "n_init": n_init,
        "max_iter": max_iter,
        "tol": tol,
        "view_space": view_space,
        "create_timestamped_dataset": create_timestamped_dataset,
        "export_dataset_mode": export_dataset_mode,
        "exported_solution_paths_container": exported_solution_paths_container,
    }
