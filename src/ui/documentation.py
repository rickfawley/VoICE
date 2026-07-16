from pathlib import Path
import streamlit as st

def render_documentation_sidebar():

    with st.expander("DOCUMENTATION",expanded=False):

        st.header("Academic references")

        paper_reference = "Vardakas, 2025"
        paper_citation = (
                    f'G. Vardakas, A. Karra, E. Pitoura, and A. Likas,'
                    f'“Counterfactual explanations for k-means and gaussian clustering,” '
                    f'in Proceedings of the IEEE International Conference on Tools with Artificial Intelligence, 2025.'
        )
        paper_path = Path("references/2501.10234v1.pdf")

        if paper_path.exists():
            st.download_button(
                paper_reference,
                data=paper_path.read_bytes(),
                file_name=paper_path.name,
                mime="application/pdf",
                help = paper_citation,
            )
        else:
            st.caption(f"{paper_reference} PDF not found - expected at {paper_path}.")
        #end-if

        st.header("Wikipedia (common algorithms)")
        st.markdown(
            """
        - [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
        - [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
        - [Expectation–maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
        - [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
        - [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier)
        - [Root-finding algorithms](https://en.wikipedia.org/wiki/Root-finding_algorithms)
        - [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)
        - [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
        - [Homothetic contraction](https://en.wikipedia.org/wiki/Homothetic_center)
        - [Lipschitz Continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity)
        
            """
        )

        repo_root = Path(".").resolve()
        repo_url = f"file:///{repo_root}"
        st.markdown(
            f"""
            <a href="{repo_url}" target="_blank" rel="noopener noreferrer"
            style="text-decoration:none;">
                <button style="
                    padding:0.5em 1em;
                    font-size:1em;
                    cursor:pointer;
                ">
                    📂 Code repository
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )

    #end-with(expander)
