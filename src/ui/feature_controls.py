import numpy as np
import pandas as pd
import streamlit as st


def render_feature_controls(
    feature_names,
    ranked_features,
    ranking_weights,
    cf_mode,
):
    ranking_weights = np.asarray(ranking_weights, float)
    weight_lookup = dict(zip(feature_names, ranking_weights))

    st.header("Actionability Mask M")

    st.caption(
        "Select which features are allowed to change when generating "
        "counterfactual explanations. Unticked features are fixed at their "
        "factual values."
    )

    st.caption(
        f"Features are shown in the active importance order for \n\n"
        f"**{cf_mode}**."
    )

    mask_by_name = {}

    for rank, name in enumerate(ranked_features, start=1):

        weight = weight_lookup.get(name, np.nan)

        pct = weight * 100

        if 0 < abs(pct) < 0.01:
            pct_str = f"{pct:g}"
        else:
            pct_str = f"{pct:.2f}".rstrip("0").rstrip(".")

        if cf_mode == "1. Unweighted k-means":
            label = f"{name}"
        elif cf_mode == "2. Ranked k-means":
            label = f"{name}\n\n(Rank {rank})"
        elif cf_mode == "3. Ranked + weighted SHARK":
            label = f"{name}\n\n(Rank {rank} - {pct_str}%)"
        else:
            label = f"{name}\n\n(Rank {rank} - {pct_str}%)"
        #end-if

        mask_by_name[name] = st.checkbox(
            label,
            value=True,
            key=f"actionable_{cf_mode}_{name}",
        )

    mask_original_order = [
        mask_by_name[name]
        for name in feature_names
    ]

    mask_array = np.asarray(mask_original_order, dtype=int)

    st.subheader("Selected Actionability Mask")

    st.code(
        f"M = {mask_array.tolist()}",
        language="text",
    )

    n_actionable = int(mask_array.sum())
    n_immutable = int(len(mask_array) - n_actionable)

    st.caption(
        f"{n_actionable} actionable feature(s), "
        f"{n_immutable} immutable feature(s)."
    )

    if n_actionable == 0:
        st.error(
            "At least one feature must remain actionable. "
            "Please tick at least one feature before continuing."
        )
        st.stop()

    st.caption(
        "Internally, the mask is converted back to the original feature order "
        "so that solver indexing remains correct."
    )

    st.markdown("---")

    st.header("Feature Importance Ranking")

    st.caption(
        "This ranking controls the parsimonious counterfactual search. "
        "Minimal intervention cardinality 1 allows only the highest-ranked actionable feature to vary; "
        "Minimal intervention cardinality 2 allows the top two actionable features; and so on."
    )

    st.caption(
        "Only the feature order is used for ranked k-means. "
        "The counterfactual geometry remains unweighted in that mode."
    )

    ranking_df = pd.DataFrame({
        "rank": range(1, len(ranked_features) + 1),
        "feature": ranked_features,
        "ranking_weight": [
            weight_lookup.get(name, np.nan)
            for name in ranked_features
        ],
    })

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
    )

    return ranked_features, mask_original_order