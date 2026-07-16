# VoICE: Voronoi-Induced Counterfactual Explainability

VoICE is a research implementation for generating counterfactual explanations for feature-weighted clustering. It explains how an observation can be changed so that it belongs to a chosen target cluster while respecting feature actionability, empirical data bounds, and the geometry of the complete target Voronoi region.

The repository accompanies the manuscript **“Counterfactuals for Feature-Weighted Clustering”** and combines:

- ordinary k-means clustering;
- SHARK feature-weighted k-means;
- least-cost and parsimonious counterfactual explanations;
- weighted Voronoi geometry;
- immutable/actionable feature constraints;
- homothetic contraction of target regions; and
- model-level and comparative evaluation.

> **Research status:** This repository is a research prototype. It includes the Streamlit application, benchmark data, reference papers, and generated experiment outputs. It does not yet include an automated test suite or a standalone scripted reproduction pipeline.

## Contents

- [Method overview](#method-overview)
- [Counterfactual modes](#counterfactual-modes)
- [Application workflow](#application-workflow)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [Datasets](#datasets)
- [Results and exports](#results-and-exports)
- [Repository structure](#repository-structure)
- [References](#references)
- [Limitations](#limitations)
- [Citation](#citation)
- [Licence](#licence)

## Method overview

Let an observation be \(a \in \mathbb{R}^d\), let \(m_t\) be the centroid of a requested target cluster \(t\), and let \(q \in \mathbb{R}_{\ge 0}^d\) be a vector of feature weights.

VoICE uses the weighted squared distance

\[
D_q(x,m)=\sum_{r=1}^{d} q_r(x_r-m_r)^2.
\]

Rather than moving the factual point only to the pairwise boundary between its source and target centroids, VoICE requires the counterfactual to lie inside the **complete weighted Voronoi cell** of the target cluster. For every competing centroid \(m_j\), target membership can be represented by a linear inequality:

\[
q \odot (m_j-m_t)^\top z
\le
\frac{1}{2}\left(
\sum_r q_r m_{jr}^2-
\sum_r q_r m_{tr}^2
\right).
\]

The least-cost counterfactual solves a constrained projection problem of the form

\[
\min_z \sum_r q_r(z_r-a_r)^2
\]

subject to:

- membership of the target Voronoi region;
- immutable features remaining equal to their factual values; and
- optional lower and upper bounds derived from the observed data.

The implementation solves this problem with SciPy’s SLSQP optimiser.

### Parsimonious explanations

For a parsimonious explanation, features are ranked and enabled progressively. The first feasible prefix gives the smallest intervention cardinality found under that ranking. SHARK weights are used as the feature ranking in the ranked modes.

### Homothetic contraction

A contraction parameter \(\alpha\) moves each target-cell boundary towards its centroid:

- \(\alpha=1\): the complete Voronoi cell;
- smaller \(\alpha\): a more conservative region nearer the target centroid.

The application can use a manually selected \(\alpha\) or derive a value from the empirical distribution of points assigned to the target cluster.

### Directional range

After finding a minimum counterfactual, VoICE can extend its direction of travel until the ray leaves the feasible region. The resulting quantity \(\rho\) measures the maximum feasible ray multiplier, while

\[
\tau=\rho-1
\]

measures the additional feasible travel beyond the minimum counterfactual.

## Counterfactual modes

The application supports three comparison modes.

| Mode | Clustering geometry | Feature ranking | Counterfactual cost |
|---|---|---|---|
| **Unweighted k-means** | Ordinary k-means | Original feature order | Equal feature weights |
| **Ranked k-means** | Ordinary k-means | SHARK-derived ranking | Equal feature weights |
| **Ranked + weighted SHARK** | SHARK weighted clustering | SHARK-derived ranking | SHARK feature weights |

Clustering and counterfactual optimisation are always performed in standardised feature space. Results can be displayed in either scaled space or transformed back into original feature units.

## Application workflow

The Streamlit application provides three main tabs.

### Specific Counterfactuals

Use this tab to:

1. choose a factual observation;
2. select a target cluster;
3. mark features as actionable or immutable;
4. choose a contraction strategy;
5. compare the selected counterfactual modes; and
6. inspect cluster, Voronoi, least-cost, and parsimonious visualisations.

PCA is used only for visualisation when the data has more than two dimensions. Optimisation remains in the complete scaled feature space.

### Model-Level Evaluation

This tab evaluates counterfactual behaviour across multiple factual points, target clusters, actionability masks, comparison modes, and contraction settings. It reports metrics including:

- feasibility rate;
- weighted counterfactual cost;
- changed-feature count;
- minimal intervention cardinality;
- single-feature feasibility;
- directional \(\rho\) and \(\tau\); and
- runtime.

### Comparative Evaluation

This tab compares two unweighted k-means constructions:

1. projection into the complete target Voronoi region; and
2. projection onto the pairwise source–target bisecting hyperplane associated with the Vardakas/CFCLUST approach.

The comparison checks whether the pairwise-boundary solution is actually assigned to the requested target cluster when all centroids are considered.

## Installation

### Requirements

The checked-in environment identifies Python **3.9** as the original interpreter. A fresh virtual environment should be created rather than reusing the repository’s bundled `.venv` directory.

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

The current `requirements.txt` is encoded as UTF-16. Convert it to UTF-8 once before installing dependencies:

```bash
python -c "from pathlib import Path; p=Path('requirements.txt'); p.write_text(p.read_text(encoding='utf-16'), encoding='utf-8')"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Principal dependencies include NumPy, pandas, SciPy, scikit-learn, Plotly, Matplotlib, and Streamlit.

## Running the application

From the repository root:

```bash
python -m streamlit run app.py
```

Streamlit will print the local application URL, normally:

```text
http://localhost:8501
```

The application reads local files from `data/`, writes experiment exports under `results/`, and expects reference PDFs under `references/`. Run it from the repository root so those relative paths resolve correctly.

## Datasets

The application currently registers ten local benchmark datasets.

| Dataset | Default clusters | Ground-truth field | Notable preprocessing |
|---|---:|---|---|
| Iris | 3 | `species` | Numeric features |
| Wine | 3 | `class` | Numeric features |
| Palmer Penguins | 3 | `species` | Excludes island, sex, and year; drops incomplete rows |
| Breast Cancer Wisconsin Diagnostic | 2 | `diagnosis` | Excludes ID |
| Wholesale Customers | 3 | `Region` | Excludes Channel and Region from clustering |
| Diabetes Health Indicators | 3 | `Diabetes_012` | Excludes BMI |
| Obesity Levels | 7 | `NObeyesdad` | Excludes Weight; one-hot encodes categorical features |
| German Credit | 2 | `default` | One-hot encodes categorical features |
| Heart Failure Clinical Records | 2 | `DEATH_EVENT` | Numeric features |
| Student Performance | 5 | `derived_pseudo_classifier` | Excludes G1, G2, and G3; one-hot encodes categorical features |

Ground-truth fields are used only for evaluation, including adjusted Rand index calculations. They are removed from the feature matrix before clustering.

All retained features are converted to numeric values and standardised with `StandardScaler` before clustering.

> **Known configuration issue:** the current cluster-count slider permits values from 2 to 6, while the Obesity dataset is configured with a default of 7. The slider range should be increased or the configured default adjusted before using that dataset unchanged.

## Results and exports

The application exports clustering solutions and model-level evaluation tables beneath `results/`.

### Clustering exports

For each fitted solution, the exporter can create:

- `dataset.csv` — row IDs, cluster assignments, optional ground truth, and scaled features;
- `centroids.csv` — fitted centroid coordinates;
- `feature_weights.csv` — equal or SHARK-derived feature weights;
- `alpha_100_by_cluster.csv` — empirical contraction values retaining all assigned observations; and
- `metadata.json` — dataset, algorithm, hyperparameter, path, and quality metadata.

Current/master files are written to both:

```text
results/experiments/<experiment folder>/
results/summary/
```

Optional timestamped copies are written to:

```text
results/history/
```

### Model-level exports

Model-level runs can produce:

- least-cost row-level results and aggregate metrics;
- parsimonious row-level results and aggregate metrics; and
- metadata describing the evaluation settings.

### Comparative exports

Comparative Voronoi-versus-bisector summaries are stored in:

```text
results/compare/
```

The included results are generated research outputs, not an automated reproducibility guarantee. A future release should add experiment configuration files, a non-interactive runner, frozen reference outputs, and verification scripts.

## Repository structure

```text
VoICE/
├── app.py                         # Streamlit entry point
├── requirements.txt              # Pinned Python environment
├── tracker.txt                   # Informal development notes
├── data/                         # Local benchmark datasets
├── references/                   # SHARK, CFCLUST, and VoICE papers
├── results/                      # Generated experiment outputs
└── src/
    ├── config.py                 # Dataset registration and UI constants
    ├── results_export.py         # CSV and JSON export helpers
    ├── counterfactuals/
    │   ├── ranking.py            # Ranked actionable feature prefixes
    │   └── solver.py             # SLSQP projection and directional range
    ├── datasets/
    │   ├── loaders.py            # Local dataset loading
    │   ├── preprocessors.py      # Encoding, filtering, and missing values
    │   └── registry.py           # Scaling and dataset access
    ├── evaluation/
    │   ├── cluster_quality.py    # Adjusted Rand index summaries
    │   ├── comparative_boundaries.py
    │   └── model_level.py
    ├── geometry/
    │   ├── halfspaces.py         # Two-dimensional halfspace utilities
    │   └── voronoi.py            # Weighted and contracted Voronoi cells
    ├── models/
    │   ├── execute_kmeans.py
    │   └── weighted_kmeans.py    # Ordinary k-means and SHARK
    ├── ui/                       # Streamlit controls and documentation
    └── views/                    # Cluster and Voronoi visualisations
```

## References

The `references/` directory contains the three papers most directly connected to the implementation.

1. **Richard J. Fawley and Renato Cordeiro de Amorim.** “Shapley-inspired feature weighting in k-means with no additional hyperparameters.” *Expert Systems with Applications*, 331, 133406, 2026. DOI: `10.1016/j.eswa.2026.133406`.

2. **Georgios Vardakas, Antonia Karra, Evaggelia Pitoura, and Aristidis Likas.** “Counterfactual Explanations for k-means and Gaussian Clustering.” arXiv:2501.10234v1, 17 January 2025.

3. **Richard J. Fawley and Renato Cordeiro de Amorim.** “Counterfactuals for Feature-Weighted Clustering.” Draft manuscript accompanying VoICE.

The papers serve different roles:

- the SHARK paper defines the feature-weighted clustering method;
- the Vardakas paper supplies the principal pairwise-boundary baseline; and
- the VoICE manuscript defines the complete Voronoi-region approach implemented here.

Before distributing the repository publicly, verify the redistribution terms for every included PDF. The project’s source-code licence does not automatically apply to publisher PDFs, preprints, or datasets.

## Limitations

- A geometrically feasible counterfactual is not necessarily causally valid, semantically plausible, or safe to act upon.
- Data-derived feature bounds do not guarantee that every combination of values corresponds to a realistic observation.
- One-hot encoded variables are currently treated as separate numerical dimensions unless additional grouped constraints are introduced.
- Results depend on feature scaling, cluster count, initialisation, feature weights, contraction, and optimiser tolerances.
- Cluster identifiers are arbitrary and do not necessarily correspond to meaningful real-world classes.
- Adjusted Rand index measures agreement with supplied labels; it does not establish that the clustering is intrinsically correct.
- SLSQP may fail or return tolerance-sensitive boundary solutions for difficult constraint sets.
- The comparative evaluator implements an unweighted k-means boundary comparison, not the full Gaussian-clustering method described by Vardakas et al.
- The repository currently has no automated test suite.
- Generated result folders may include machine-specific metadata from the environment in which they were produced.

VoICE should be treated as research software, not as a decision-making system for clinical, financial, educational, employment, or other high-impact applications.

## Citation

Until a final archival citation is available, cite the accompanying manuscript and identify the software version or Git commit used in the experiment.

```bibtex
@article{fawley_voice,
  title   = {Counterfactuals for Feature-Weighted Clustering},
  author  = {Fawley, Richard J. and de Amorim, Renato Cordeiro},
  note    = {Manuscript and VoICE research software},
  url     = {https://github.com/rickfawley/VoICE}
}
```

The SHARK clustering method should additionally be cited as:

```bibtex
@article{fawley2026shark,
  title   = {Shapley-inspired feature weighting in k-means with no additional hyperparameters},
  author  = {Fawley, Richard J. and de Amorim, Renato Cordeiro},
  journal = {Expert Systems with Applications},
  volume  = {331},
  pages   = {133406},
  year    = {2026},
  doi     = {10.1016/j.eswa.2026.133406}
}
```

## Licence

Unless otherwise stated, the project source code and documentation are licensed under the **Apache License 2.0**. See `LICENSE` for the complete terms.

Datasets, academic papers, and other third-party materials are not automatically covered by the project licence and remain subject to their respective terms. A public release should include a `THIRD_PARTY_NOTICES.md` file describing those materials.

## Generated results and omitted files

To keep the repository reasonably sized and within GitHub's file-size limits, some generated experiment artefacts are intentionally excluded from version control.

The following files are not included:

* `results/experiments/` — complete per-run experiment outputs, including detailed counterfactual result tables and dataset snapshots.
* `results/summary/* dataset.csv` — generated copies of datasets used in individual clustering runs.

These files are outputs rather than source data. The underlying benchmark datasets remain available under `data/`, while smaller derived artefacts—including centroids, feature weights, metadata, evaluation metrics, and comparison summaries—are retained under `results/summary/` and `results/compare/`.

The omitted files can be recreated locally by rerunning the corresponding clustering and counterfactual experiments through the Streamlit application. Generated outputs are ignored by Git to prevent large or duplicated files from being committed accidentally.

Results may vary slightly between environments because of differences in Python package versions, numerical optimisation, and random initialisation.


---

Software repository: [github.com/rickfawley/VoICE](https://github.com/rickfawley/VoICE)
