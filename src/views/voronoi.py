import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay, ConvexHull

from src.config import BASE_COLOURS
from src.geometry import (
    halfspace_polygon_2d,
    build_alpha_halfspaces,
)
from src.counterfactuals import (
    project_to_target_with_fixed_features,
    directional_rho,
    ranked_actionable_prefixes,
)

def display_bounded_voronoi_cells(
    X_cluster,
    X_view,
    feature_names,
    cluster_result,
    scaler,
    view_space,
    seed=0,
    pad=0.5,
    a_idx=0,
    s=None,
    t=None,
    alpha=1.0,
    ranked_features=None,
    mask=None,
    cf_weights=None,
    cf_mode=None,
):
    """
    Build explicit bounded Voronoi cell polygons (one per centroid) in PCA-2D.

    IMPORTANT: This is a Visualisation-space (PCA) truncation.
    Use separate bounds in the counterfactual solver in the full feature space.
    """
    X_cluster = np.asarray(X_cluster, float)
    X_view = np.asarray(X_view, float)

    centers = np.asarray(cluster_result.centroids, float)
    cluster_weights = np.asarray(cluster_result.weights, float)

    if cf_weights is None:
        cf_weights = cluster_weights
    #end-if
    weights = np.asarray(cf_weights, float)

    labels = cluster_result.labels
    n_clusters = centers.shape[0]
    cluster_colors = BASE_COLOURS[:n_clusters]

    # 2D PCA projection for Visualisation
    if view_space == "Scaled Metric Space":
        X_plot = X_cluster
        C_plot = centers
    else:
        X_plot = X_view
        C_plot = scaler.inverse_transform(centers)

    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X_plot)
    C2 = pca.transform(C_plot)
    # --- Mark observation a in this PCA space ---
    if 0 <= a_idx < X2.shape[0]:
        obs_x, obs_y = float(X2[a_idx, 0]), float(X2[a_idx, 1])
    else:
        obs_x, obs_y = None, None

    # Observation window in PCA plane (box)
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
    box = (x_min, x_max, y_min, y_max)

    # For equations display
    cell_inequalities = []  # list of dicts per cell

    # Plot setup
    fig, ax = plt.subplots()

    #ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=18, alpha=0.25)
    #ax.scatter(C2[:, 0], C2[:, 1], marker="X", s=220, edgecolor="k")

    for i in range(n_clusters):
        cluster_mask = (labels == i)
        ax.scatter(
            X2[cluster_mask, 0],
            X2[cluster_mask, 1],
            s=18,
            alpha=0.35,
            color=cluster_colors[i],
            label=f"Cluster {i}",
        )

    for i in range(n_clusters):

        # Make source/target centroids slightly larger
        size = 350 if i in [s, t] else 250

        ax.scatter(
            C2[i, 0],
            C2[i, 1],
            marker="X",
            s=size,
            edgecolor="black",
            linewidth=1.5,
            color=cluster_colors[i],
            zorder=6,
        )

        # Add labels for source and target
        if i == s:
            ax.text(
                C2[i, 0],
                C2[i, 1],
                "  Source (s)",
                fontsize=11,
                weight="bold",
                color="black",
                verticalalignment="center",
            )

        if i == t:
            ax.text(
                C2[i, 0],
                C2[i, 1],
                "  Target (t)",
                fontsize=11,
                weight="bold",
                color="black",
                verticalalignment="center",
            )


    # Build each cell polygon
    for i in range(n_clusters):
        ci = C2[i]

        halfspaces = []
        ineqs_for_text_pca = []
        ineqs_for_text_cluster = []

        for j in range(n_clusters):
            if j == i:
                continue

            cj = C2[j]
            v = cj - ci
            c = 0.5 * (np.dot(cj, cj) - np.dot(ci, ci))

            a, b = float(v[0]), float(v[1])
            halfspaces.append((a, b, float(c)))
            ineqs_for_text_pca.append((a, b, float(c), j))

            # --- Original-space inequality ---
            mi = centers[i]
            mj = centers[j]

            w = weights * (mj - mi)

            b_cluster = 0.5 * (
                np.sum(weights * mj ** 2)
                - np.sum(weights * mi ** 2)
            )

            ineqs_for_text_cluster.append((w.copy(), float(b_cluster), j))

        poly = halfspace_polygon_2d(halfspaces, box)
        if poly is None:
            continue

        # Draw polygon (closed)
        poly_closed = np.vstack([poly, poly[0]])

        # First draw the ordinary Voronoi cell boundary
        line_styles = ["-", "--", "-.", ":"]
        style = line_styles[i % len(line_styles)]

        ax.plot(
            poly_closed[:, 0],
            poly_closed[:, 1],
            linewidth=2,
            linestyle=style,
            color=cluster_colors[i],
            zorder=4,
        )

        # Then, for the target only, draw the α-contracted boundary on top
        if i == t and alpha < 1.0:
            centroid = ci
            poly_alpha = centroid + alpha * (poly - centroid)
            poly_alpha_closed = np.vstack([poly_alpha, poly_alpha[0]])

            ax.plot(
                poly_alpha_closed[:, 0],
                poly_alpha_closed[:, 1],
                color="black",
                linestyle=":",
                linewidth=3.0,
                zorder=12,
                label="α-contraction boundary",
            )

            # label slightly offset from centroid
            ax.text(
                centroid[0] + 0.03,
                centroid[1] + 0.03,
                f"α={alpha:.2f}",
                fontsize=9,
                color="black",
                zorder=13,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
            )

            ax.text(
                centroid[0],
                centroid[1],
                f"α-boundary (α={alpha:.2f})",
                fontsize=9,
                color="black",
            )

        # Cycle linestyle per cluster for visual separation
        line_styles = ["-", "--", "-.", ":"]
        style = line_styles[i % len(line_styles)]

        #ax.plot(
        #    poly_closed[:, 0],
        #    poly_closed[:, 1],
        #    linewidth=2,
        #    linestyle=style,
        #    color=cluster_colors[i],
        #)

        # Label cell near centroid (if centroid lies outside box, clamp visually)
        cx = float(np.clip(ci[0], x_min, x_max))
        cy = float(np.clip(ci[1], y_min, y_max))
        ax.text(cx, cy, f"Cell {i}", fontsize=10)

        cell_inequalities.append(
            {
                "cell": i,
                "ineqs_pca": ineqs_for_text_pca,
                "ineqs_cluster": ineqs_for_text_cluster,
            }
        )



    if obs_x is not None:
        ax.scatter([obs_x], [obs_y], marker="x", s=50, color="black", linewidth=3, zorder=20)
        ax.text(obs_x, obs_y, f"Factual Point {a_idx + 1}", color="black", fontsize=10, va="bottom")

    ax.set_title("Bounded Voronoi cells (PCA-2D), truncated to observation window")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right", frameon=True)

    st.subheader(
        f"Bounded Voronoi cells (PCA-2D, {view_space}), truncated to observation window"
    )
    if cf_mode is not None:
        st.caption(
            f"Counterfactual mode: {cf_mode}. "
            "Cluster labels and centroids come from the selected clustering result; "
            "Voronoi feasibility and solver cost use the counterfactual weights for this mode."
        )
    st.pyplot(fig)

    st.write(
        "Each polygon is a display-space PCA projection of the weighted clustering Voronoi geometry. "
        "The clustering model, counterfactual feasibility, α-contraction, and Voronoi membership "
        "are computed in scaled space. The selected view controls only how the result is displayed."
    )

    st.subheader(f"Cell inequalities (PCA-2D display space: {view_space})")
    st.latex(rf"x_{{\min}}={x_min:.3g},\ x_{{\max}}={x_max:.3g},\ y_{{\min}}={y_min:.3g},\ y_{{\max}}={y_max:.3g}")

    for cell in cell_inequalities:
        i = cell["cell"]
        st.markdown(f"**Cell {i}**")
        for a, b, c, j in cell["ineqs_pca"]:
            st.latex(
                rf"({a:.4g})\,u_1 + ({b:.4g})\,u_2 \le {c:.4g}\quad(\text{{vs. centroid }}{j})"
            )
        st.latex(rf"x_{{\min}} \le u_1 \le x_{{\max}},\quad y_{{\min}} \le u_2 \le y_{{\max}}")
        st.markdown("---")

    st.subheader("Cell inequalities (scaled clustering space)")
    st.write(
        "These are the Voronoi half-space inequalities used by the selected clustering backend. "
        "They remain in scaled space even when the display view is set to original metric space."
    )

    for cell in cell_inequalities:
        i = cell["cell"]
        st.markdown(f"**Cell {i}**")

        for w, b_cluster, j in cell["ineqs_cluster"]:
            lhs_terms = []
            for k, wk in enumerate(w):
                name = feature_names[k] if k < len(feature_names) else f"x_{k}"
                lhs_terms.append(rf"({wk:.4g})\,\mathrm{{{name}}}")
            lhs = " + ".join(lhs_terms)
            st.latex(rf"{lhs} \le {b_cluster:.4g}\quad(\text{{vs. centroid }}{j})")

        st.markdown("---")


    # ------------------------------------------------------------
    # Ranked actionable rays from Factual Point a to target boundary
    # ------------------------------------------------------------
    if ranked_features is not None and mask is not None and t is not None and obs_x is not None:

        st.subheader("Ranked actionable counterfactual rays (PCA-2D)")

        a_cluster = np.asarray(X_cluster[a_idx], float)
        x_lo = X_cluster.min(axis=0)
        x_hi = X_cluster.max(axis=0)

        # Use alpha-contracted target region if alpha < 1, else full target cell
        halfspaces_t = build_alpha_halfspaces(
            centers,
            t=t,
            weights=weights,
            alpha=alpha,
        )

        prefixes = ranked_actionable_prefixes(
            ranked_features=ranked_features,
            feature_names=feature_names,
            mask=mask,
        )

        if len(prefixes) == 0:
            st.info(
                "No counterfactual rays generated because all features are immutable."
            )
        else:

            if len(prefixes) > 1:
                mode_key = str(cf_mode).replace(" ", "_").replace("+", "plus").replace(".", "")

                max_rays_to_show = st.slider(
                    "Max ranked rays to display",
                    min_value=1,
                    max_value=max(1, len(prefixes)),
                    value=max(1, len(prefixes)),
                    step=1,
                    key=f"max_rays_to_show_{mode_key}",
                )
            else:
                max_rays_to_show = 1
            #end-if

            show_all_rays_faded = st.checkbox(
                "Show non-selected rays faded in background",
                value=True,
                key=f"show_all_rays_faded_{mode_key}",
            )

            prefixes_to_plot = prefixes[:max_rays_to_show]

            selected_ray_idx = st.selectbox(
                "Highlight ranked ray",
                options=list(range(len(prefixes_to_plot))),
                index=0,
                format_func=lambda i: f"Rank {i+1}: {', '.join(prefixes_to_plot[i][0])}",
                key=f"selected_ray_idx_{mode_key}",
            )

            fig_rays, ax_rays = plt.subplots()

            # Background points
            for i in range(n_clusters):
                mask_i = (labels == i)
                ax_rays.scatter(
                    X2[mask_i, 0],
                    X2[mask_i, 1],
                    s=12,
                    alpha=0.12,
                    color=cluster_colors[i],
)

            # Draw cell outlines for context, and overlay the α-boundary on the target cell
            for i in range(n_clusters):
                ci = C2[i]
                halfspaces = []
                for j in range(n_clusters):
                    if j == i:
                        continue
                    cj = C2[j]
                    v = cj - ci
                    c = 0.5 * (np.dot(cj, cj) - np.dot(ci, ci))
                    halfspaces.append((float(v[0]), float(v[1]), float(c)))

                poly = halfspace_polygon_2d(halfspaces, box)
                if poly is None:
                    continue

                poly_closed = np.vstack([poly, poly[0]])
                lw = 2.5 if i == t else 1.0
                alpha_poly = 0.85 if i == t else 0.18

                # full Voronoi cell outline
                ax_rays.plot(
                    poly_closed[:, 0],
                    poly_closed[:, 1],
                    color=cluster_colors[i],
                    linewidth=lw,
                    alpha=alpha_poly,
                    zorder=3,
                )

                # α-contracted boundary for the target cell
                if i == t and alpha < 1.0:
                    centroid = ci
                    poly_alpha = centroid + alpha * (poly - centroid)
                    poly_alpha_closed = np.vstack([poly_alpha, poly_alpha[0]])

                    ax_rays.plot(
                        poly_alpha_closed[:, 0],
                        poly_alpha_closed[:, 1],
                        color="black",
                        linestyle=":",
                        linewidth=3.0,
                        zorder=8,
                        label="α-contraction boundary",
                    )

                    ax_rays.text(
                        centroid[0] + 0.03,
                        centroid[1] + 0.03,
                        f"α={alpha:.2f}",
                        fontsize=9,
                        color="black",
                        zorder=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
                    )
                #end-if

                ax_rays.scatter(
                    C2[i, 0],
                    C2[i, 1],
                    marker="X",
                    s=220 if i not in [s, t] else 300,
                    edgecolor="black",
                    linewidth=1.2,
                    color=cluster_colors[i],
                    zorder=9,
                )

                label = f"Cluster {i}"
                if i == s:
                    label += " (source)"
                elif i == t:
                    label += " (target)"

                ax_rays.text(
                    C2[i, 0] + 0.04,
                    C2[i, 1] + 0.04,
                    label,
                    fontsize=9,
                    weight="bold" if i in [s, t] else "normal",
                    color="black",
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
                    zorder=10,
                )



            # Factual Point a
            ax_rays.scatter(
                [obs_x], [obs_y],
                marker="x",
                s=60,
                color="black",
                linewidth=2.5,
                zorder=10,
            )
            ax_rays.text(obs_x, obs_y, f"a={a_idx+1}", color="black", fontsize=10, va="bottom")

            ray_rows = []

            for z_idx, (subset_names, subset_idx) in enumerate(prefixes_to_plot, start=1):

                z_cf, ok, msg = project_to_target_with_fixed_features(
                    a=a_cluster,
                    free_idx=subset_idx,
                    halfspaces=halfspaces_t,
                    x_lo=x_lo,
                    x_hi=x_hi,
                    weights=weights,
                )

                if not ok:
                    ray_rows.append({
                        "cf_mode": cf_mode,
                        "rank_step": z_idx,
                        "minimal_intervention_cardinality": z_idx,
                        "mutable_features": ", ".join(subset_names),
                        "feasible": False,
                        "weighted_squared_distance": np.nan,
                        "euclidean_distance": np.nan,
                        "Explainability Range [1,ρ]": "",
                        "τ": np.nan,
                        **{f"Δ {name}": np.nan for name in ranked_features},
                    })
                    continue

                if view_space == "Scaled Metric Space":
                    z_plot = z_cf
                    a_display = a_cluster
                    z_display = z_cf
                else:
                    z_plot = scaler.inverse_transform(z_cf.reshape(1, -1))[0]
                    a_display = scaler.inverse_transform(a_cluster.reshape(1, -1))[0]
                    z_display = scaler.inverse_transform(z_cf.reshape(1, -1))[0]

                z2 = pca.transform(z_plot.reshape(1, -1))[0]
                delta_display = z_display - a_display

                is_selected = (z_idx - 1) == selected_ray_idx

                if is_selected:
                    ray_alpha = 1.0
                    ray_lw = 3.2
                    ray_ms = 6
                    ray_zorder = 12
                elif show_all_rays_faded:
                    ray_alpha = 0.22
                    ray_lw = 1.4
                    ray_ms = 3
                    ray_zorder = 6
                else:
                    continue

                ax_rays.plot(
                    [obs_x, z2[0]],
                    [obs_y, z2[1]],
                    linewidth=ray_lw,
                    linestyle="-",
                    marker="o",
                    markersize=ray_ms,
                    alpha=ray_alpha,
                    zorder=ray_zorder,
                )

                if is_selected:
                    short_label = ", ".join(subset_names[:2])
                    if len(subset_names) > 2:
                        short_label += ", ..."
                    ax_rays.text(
                        z2[0],
                        z2[1],
                        f"z{z_idx}: {short_label}",
                        fontsize=9,
                        color="black",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
                    )
                else:
                    ax_rays.text(
                        z2[0],
                        z2[1],
                        f"{z_idx}",
                        fontsize=8,
                        color="black",
                        alpha=0.55,
                    )

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

                tau = (
                    rho - 1.0
                    if np.isfinite(rho)
                    else np.inf
                )

                raw_rho = rho_diag.get("raw_rho", np.nan)
                raw_tau = rho_diag.get("tau_raw", np.nan)
                was_clipped = rho_diag.get("was_clipped", False)
                active_bound = rho_diag.get("active_bound") or {}
                active_feature_idx = active_bound.get("feature_index", None)

                if active_feature_idx is not None:
                    active_feature = feature_names[int(active_feature_idx)]
                else:
                    active_feature = ""
                #end-if

                row = {
                    "cf_mode": cf_mode,
                    "rank_step": z_idx,
                    "mutable_features": ", ".join(subset_names),
                    "feasible": True,
                    "weighted_squared_distance": float(
                        np.sum(weights * (z_cf - a_cluster) ** 2)
                    ),
                    "euclidean_distance": float(
                        np.linalg.norm(z_cf - a_cluster)
                    ),

                    "Explainability Range [1,ρ]":
                        f"[1, {rho:.3f}]",

                    "τ":
                        float(tau),

                    # Diagnostics for the feasible range
                    "ρ_raw": raw_rho,
                    "τ_raw": raw_tau,
                    "ρ_clipped": rho,
                    "ρ_was_clipped": was_clipped,
                    "ρ_active_source": active_bound.get("source", ""),
                    "ρ_active_numer": active_bound.get("numer", np.nan),
                    "ρ_active_denom": active_bound.get("denom", np.nan),
                    "ρ_active_feature": active_feature,
                }

                for name in ranked_features:
                    i_name = feature_names.index(name)
                    row[f"Δ {name}"] = float(delta_display[i_name])
                #end-for

                ray_rows.append(row)

            ax_rays.set_title(
                f"PCA Visualistion of Counterfactual rays from observation to target boundary ({view_space})"
            )
            ax_rays.set_xlabel("PCA 1")
            ax_rays.set_ylabel("PCA 2")
            ax_rays.set_xlim(x_min, x_max)
            ax_rays.set_ylim(y_min, y_max)
            ax_rays.legend(loc="best", fontsize=8, frameon=True)

            st.pyplot(fig_rays)

            st.write(
                "Euclidean distance measures ordinary geometric displacement in scaled space. "
                "Weighted squared distance measures the optimisation cost used by the counterfactual solver. "
                "The comparison mode controls whether equal k-means weights or SHARK-derived weights "
                "are used in the optimisation geometry."
            )

            df_rays = pd.DataFrame(ray_rows)

            base_cols = [
                col for col in [
                    "cf_mode",
                    "rank_step",
                    "mutable_features",
                    "feasible",
                    "weighted_squared_distance",
                    "euclidean_distance",
                    "Explainability Range [1,ρ]",
                    "τ",
                    "ρ_raw",
                    "τ_raw",
                    "ρ_clipped",
                    "ρ_was_clipped",
                    "ρ_active_source",
                    "ρ_active_numer",
                    "ρ_active_denom",
                    "ρ_active_feature",
                ]
                if col in df_rays.columns
            ]

            delta_cols = [
                f"Δ {name}"
                for name in ranked_features
                if f"Δ {name}" in df_rays.columns
            ]

            df_rays = df_rays[base_cols + delta_cols]

            st.subheader("Parsimonious Explainability vectors")

            st.caption(
                "Explainability Range [1,ρ] shows the admissible intervention interval "
                "along the counterfactual direction within the contracted target region. "
                "τ = ρ - 1 is the normalised counterfactual tolerance. "
                "Larger values indicate greater intervention flexibility and robustness."
            )

            st.dataframe(df_rays)


            # ------------------------------------------------------------
            # Ranked step vs distance / cost chart
            # ------------------------------------------------------------
            plot_cols = [
                col for col in [
                    "weighted_squared_distance",
                    "euclidean_distance",
                    "weighted_norm",
                ]
                if col in df_rays.columns
            ]

            df_plot = df_rays[df_rays["feasible"] == True].copy()

            if len(df_plot) > 0 and len(plot_cols) > 0:
                st.subheader("Ranked step vs counterfactual distance")

                fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))

                for col in plot_cols:
                    ax_metrics.plot(
                        df_plot["rank_step"],
                        df_plot[col],
                        marker="o",
                        linewidth=2,
                        label=col.replace("_", " "),
                    )
                #end-for

                ax_metrics.set_xlabel("Ranked feature-prefix step")
                ax_metrics.xaxis.set_major_locator(MaxNLocator(integer=True))

                x_max = int(df_plot["rank_step"].max())
                ax_metrics.set_xlim(left=1, right=x_max)
                ax_metrics.set_xticks(range(1, x_max + 1))

                ax_metrics.set_ylabel("Value")
                ax_metrics.set_title("Counterfactual cost and distance by ranked step")
                ax_metrics.legend()
                ax_metrics.grid(True, alpha=0.25)

                st.pyplot(fig_metrics)
            else:
                st.info(
                    "No feasible ranked counterfactuals available for the distance chart."
                )
            #end-if

            # ------------------------------------------------------------
            # Ranked step vs counterfactual tolerance
            # ------------------------------------------------------------
            if "τ" in df_plot.columns:
                df_tau = df_plot[
                    np.isfinite(df_plot["τ"])
                ].copy()

                if len(df_tau) > 0:
                    st.subheader("Ranked step vs counterfactual tolerance")

                    fig_tau, ax_tau = plt.subplots(figsize=(8, 4))

                    ax_tau.plot(
                        df_tau["rank_step"],
                        df_tau["τ"],
                        marker="o",
                        linewidth=2,
                        label="τ = ρ - 1",
                    )

                    ax_tau.set_xlabel("Ranked feature-prefix step")
                    ax_tau.xaxis.set_major_locator(MaxNLocator(integer=True))

                    x_max = int(df_tau["rank_step"].max())
                    ax_tau.set_xlim(left=1, right=x_max)
                    ax_tau.set_xticks(range(1, x_max + 1))

                    ax_tau.set_ylabel("Normalised counterfactual tolerance τ")
                    ax_tau.set_title("Counterfactual tolerance by ranked step")
                    ax_tau.legend()
                    ax_tau.grid(True, alpha=0.25)

                    st.pyplot(fig_tau)



def display_voronoi_delaunay_boundaries(
    X,
    feature_names,
    cluster_result,
    seed=0,
):
    """
    Visualise Voronoi–Delaunay boundaries between k-means clusters.

    - Uses PCA(2D) projection to draw a readable diagram.
    - Uses Delaunay triangulation on the projected centroids to decide which
      cluster pairs are "neighbors" (Voronoi-adjacent).
    - For each neighbor pair (i, j), draws the perpendicular bisector line in 2D
      and labels it H_{i,j}.
    - Prints equations:
        (A) Original-space hyperplane equation (true Voronoi bisector for k-means)
        (B) PCA-2D line equation (what is drawn)
    """

    X = np.asarray(X, float)
    centers = np.asarray(cluster_result.centroids, float)
    weights = np.asarray(cluster_result.weights, float)
    labels = cluster_result.labels
    n_clusters = centers.shape[0]
    # Observation bounds in the true clustering space
    x_lo = X.min(axis=0)
    x_hi = X.max(axis=0)

    cmap = cm.get_cmap("tab10", n_clusters)
    cluster_colors = BASE_COLOURS[:n_clusters]

    # ---------- 2D projection for diagram ----------
    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X)
    C2 = pca.transform(centers)

    hull = ConvexHull(X2)
    hull_pts = X2[hull.vertices]

    # ---------- Delaunay adjacency (on 2D centroids) ----------
    # If n_clusters is small or degenerate in 2D, Delaunay can fail.
    # We'll fall back to "all pairs" if needed.
    neighbor_pairs = set()
    try:
        tri = Delaunay(C2)
        # Each simplex is a triangle (in 2D); take all edges
        for simplex in tri.simplices:
            simplex = list(simplex)
            for a in range(len(simplex)):
                for b in range(a + 1, len(simplex)):
                    i, j = simplex[a], simplex[b]
                    neighbor_pairs.add(tuple(sorted((int(i), int(j)))))
    except Exception:
        # Fallback: show all pairwise boundaries
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                neighbor_pairs.add((i, j))

    neighbor_pairs = sorted(neighbor_pairs)

    # ---------- Plot ----------
    fig, ax = plt.subplots()

    # scatter the points (in PCA plane) for context, colored by cluster

    #ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=18, alpha=0.35)
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        ax.scatter(
            X2[cluster_mask, 0],
            X2[cluster_mask, 1],
            s=18,
            alpha=0.35,
            color=cluster_colors[i],
            label=f"Cluster {i}",
        )

    # plot centroids
    #ax.scatter(C2[:, 0], C2[:, 1], marker="X", s=220, edgecolor="k")
    for i in range(n_clusters):
        ax.scatter(
            C2[i, 0],
            C2[i, 1],
            marker="X",
            s=250,
            edgecolor="black",
            linewidth=1.2,
            color=cluster_colors[i],
            zorder=5,
        )

    # Determine plotting bounds
    pad = 0.5
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad

    # store equations for display
    eq_rows = []

    for (i, j) in neighbor_pairs:
        ci2 = C2[i]
        cj2 = C2[j]
        v2 = cj2 - ci2  # normal vector in 2D for bisector line

        # If centroids are identical in projection (unlikely), skip
        if np.allclose(v2, 0):
            continue

        # 2D bisector line: v2^T x = (||cj||^2 - ||ci||^2)/2
        b2 = 0.5 * (np.dot(cj2, cj2) - np.dot(ci2, ci2))

        # Draw the line across the plot:
        # If v2[1] != 0 -> y = (b2 - v2[0]*x) / v2[1]
        # else -> vertical line x = b2 / v2[0]
        if abs(v2[1]) > 1e-12:
            xs = np.array([x_min, x_max])
            ys = (b2 - v2[0] * xs) / v2[1]
            ax.plot(xs, ys, linewidth=1.2)
            # label near midpoint of segment in plot-range
            xm = 0.5 * (x_min + x_max)
            ym = (b2 - v2[0] * xm) / v2[1]
        else:
            x_vert = b2 / v2[0]
            ax.plot([x_vert, x_vert], [y_min, y_max], linewidth=1.2)
            xm = x_vert
            ym = 0.5 * (y_min + y_max)

        ax.text(xm, ym, f"H_{i},{j}", fontsize=9)

        # ---------- True ORIGINAL-space hyperplane ----------
        # Hyperplane separating Voronoi cells of centroids m_i and m_j:
        # (m_j - m_i)^T x = (||m_j||^2 - ||m_i||^2)/2
        mi = centers[i]
        mj = centers[j]
        w = weights * (mj - mi)

        b = 0.5 * (
            np.sum(weights * mj ** 2)
            - np.sum(weights * mi ** 2)
        )

        eq_rows.append(
            {
                "pair": (i, j),
                "w_orig": w,
                "b_orig": b,
                "w_2d": v2,
                "b_2d": b2,
            }
        )

    ax.set_title("Voronoi–Delaunay boundaries between k-means clusters (PCA 2D view)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ---------- Streamlit display (collapsed expander) ----------
    with st.expander("Voronoi–Delaunay boundaries (diagram + equations)", expanded=False):

        st.subheader("Boundary equations")

        st.write(
            "For each neighboring cluster pair $(i,j)$ we show:\n"
            "- **Original-space hyperplane** (true k-means Voronoi boundary): "
            "$(m_j - m_i)^T x = \\frac{\\|m_j\\|^2 - \\|m_i\\|^2}{2}$\n"
            "- **PCA-2D line** (what’s drawn in the diagram): "
            "$(c_j - c_i)^T u = \\frac{\\|c_j\\|^2 - \\|c_i\\|^2}{2}$"
        )

        for row in eq_rows:
            i, j = row["pair"]
            w = row["w_orig"]
            b = row["b_orig"]
            v2 = row["w_2d"]
            b2 = row["b_2d"]

            st.markdown(f"**Boundary** $H_{{{i},{j}}}$")

            # Original-space equation expanded
            # sum_k w_k x_k = b
            lhs_terms = []
            for k, wk in enumerate(w):
                # show short feature name: x_0.. or use feature_names
                name = feature_names[k] if k < len(feature_names) else f"x_{k}"
                lhs_terms.append(f"({wk:.4g})\\,{name}")
            lhs = " + ".join(lhs_terms)
            st.latex(rf"{lhs} = {b:.4g}")

            # PCA-2D equation
            st.latex(rf"({v2[0]:.4g})\,u_1 + ({v2[1]:.4g})\,u_2 = {b2:.4g}")

            st.markdown("---")

        st.subheader("Diagram (PCA 2D)")
        st.pyplot(fig)
        st.write(
            "This diagram shows a 2D PCA projection of the k-means Voronoi boundaries. "
            "The true decision regions exist in the original feature space (ℝ^d), "
            "where each centroid defines exactly one convex Voronoi cell. "
            "Here, those high-dimensional separating hyperplanes are projected into 2D, "
            "where they appear as straight lines. "
            "Because projection does not preserve the full geometry, the intersecting "
            "lines may visually form more regions than there are clusters. "
            "These extra regions are Visualisation artifacts — not additional clusters."
        )