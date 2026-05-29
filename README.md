# Dimensionality Reduction and Manifold Learning for Forced Jet Flows

Bachelor Thesis (TFG) in Data Science and Engineering.

Author: Alejandro Fonseca Ortés

Tutors: Eduardo García Portugués and Marco Raiola

This repository contains the computational work for a thesis on the
characterization and classification of forced turbulent jet flows from particle
image velocimetry (PIV) data. The central question is whether the
high-dimensional velocity snapshots produced by PIV concentrate near a
lower-dimensional structure, and whether reduced coordinates preserve the
physical organization of the flow.

The current thesis pipeline combines:

- intrinsic-dimension estimation;
- principal component analysis (PCA), interpreted as proper orthogonal
  decomposition (POD);
- sparse PCA;
- spectral proper orthogonal decomposition (SPOD);
- nonlinear manifold embeddings with t-SNE and UMAP;
- Random Forest prediction tasks for phase, actuation frequency, and
  generalization to unseen frequencies.

## Data Policy

The `.mat` and `.npz` files are large, restricted research-group artifacts. They
are not public repository assets and must not be redistributed, uploaded, or
shared without explicit permission from the research group. This applies both to
raw PIV files and to derived `.npz` caches, because the derived files still
contain experimental data or data products computed from it.

Do not overwrite, delete, or modify any stored `.mat` or `.npz` data files. All
transformations should be saved as new derived artifacts. The repository
`.gitignore` excludes these file types to keep confidential and heavy data out of
version control.

Prefer the compressed `.npz` files in `code/compressed_data/` for ordinary
analysis. The raw `.mat` files in `code/data/` are much larger and should only
be used when a variable is not available in the compressed files, such as
phase-averaged fields, mean fields, Reynolds stresses, or stored POD outputs.

Important snapshot-count convention:

- Raw `RunX_PIV.mat` files store 2031 time steps.
- Most compressed files store 2030 snapshots because the first time step is
  discarded during preprocessing.
- `RUN1_PIV_compressed.npz` is an exception and currently stores 2031
  snapshots.

## Dataset

The experiment consists of 25 runs of a turbulent jet at Reynolds number
`Re = 10000`. Runs differ by actuation Strouhal number `St_act`, listed in
`code/data/info.txt`. Run 1 is the unforced/natural jet with `St_act = 0`; Runs
2-25 are forced runs from `St_act = 0.05` to `St_act = 1.20`.

Each velocity snapshot is defined on a `269 x 319` grid with two fluctuating
velocity components, so one full snapshot has
`2 x 269 x 319 = 171622` velocity coordinates.

Access to the data files requires authorization from the research group. A fresh
clone of the repository should not be expected to include the heavy `.mat` and
`.npz` artifacts unless they have been provided separately through an approved
channel.

| Artifact | Location | Main contents |
| --- | --- | --- |
| Raw PIV data | `code/data/RunX_PIV.mat` | Full experimental data, including mean fields, instantaneous fields, phase averages, POD arrays, and Reynolds stresses |
| Compressed PIV data | `code/compressed_data/RUNX_PIV_compressed.npz` | Preferred data for most analyses: `X`, `Y`, `u`, and `v` |
| Spatial grid | `code/spatial_grid.npz` | Shared grid arrays |
| Phase averages | `code/phase_avg_data/` | Extracted phase-averaged and mean fields, currently for RUN2 |
| PCA scores | `code/pca_data/` | Per-run PCA scores and explained variance ratios |
| SPOD results | `code/SPOD_data/` | Per-run SPOD frequencies, eigenvalues, and complex spatial modes |
| Sparse PCA results | `code/SPCA_data/` | Sparse modes, scores, masks, and reconstruction diagnostics |
| Cluster labels | `code/clustering_data/` | Earlier clustering labels used in phase-correspondence experiments |
| t-SNE results | `code/tsne_data/` | Global, single-run, out-of-sample, and original-space t-SNE artifacts |
| UMAP results | `code/umap_data/` | Global and per-run 3D UMAP embeddings |
| ML results | `code/ml_results/` | CSV metrics for supervised prediction experiments |
| Figures | `code/figures/` | Exploratory figures, experiment figures, and thesis-ready outputs |

## SPOD Frequency Convention

The SPOD frequencies stored in `code/SPOD_data/RUNX_PIV_SPOD.npz` are in cycles
per snapshot because the SPOD computation used `dt = 1`. Forced runs were
sampled at 20 snapshots per actuation cycle, so the actuation frequency in
snapshot units is `0.05` cycles per snapshot.

When comparing SPOD spectra across runs, convert stored frequencies to physical
Strouhal number with

```text
St = freqs * (St_act / 0.05)
```

This scaling is required for thesis figures and cross-run spectral comparisons.

## Repository Structure

```text
TFG_datos/
|-- README.md
|-- .gitignore
|-- .gitattributes
|-- .vscode/
`-- code/
    |-- data_utils.py
    |-- spod_utils.py
    |-- ml_experiment_utils.py
    |-- spatial_grid.npz
    |
    |-- data/
    |-- compressed_data/
    |-- phase_avg_data/
    |-- pca_data/
    |-- SPOD_data/
    |-- SPCA_data/
    |-- clustering_data/
    |-- tsne_data/
    |-- umap_data/
    |-- ml_results/
    |-- centroid_plots/
    |-- comparisons/
    |
    |-- figures/
    |   |-- thesis/
    |   |-- tsne_experiments/
    |   |-- spca_experiments/
    |   `-- phase_correspondence/
    |
    |-- data_visualization.ipynb
    |-- PCA.ipynb
    |-- intrinsic_dimension.ipynb
    |-- intrinsic_dimension_corrected.ipynb
    |-- SPOD_Run1.ipynb
    |-- SPOD_comparison.ipynb
    |-- spod_cross_run_harmonics.ipynb
    |-- manifold_analysis.ipynb
    |-- cross_run_analysis.ipynb
    |-- phase_correspondence.ipynb
    |-- centroid_phase_comparison.ipynb
    |-- cross_run_tsne.ipynb
    |-- global_cross_run_tsne.ipynb
    |-- global_cross_run_umap.ipynb
    |-- individual_run_umap.ipynb
    |-- tsne_compare_hyperparameters.ipynb
    |-- tsne_knn_experiment.ipynb
    |-- tsne_out_of_sample_test.ipynb
    |-- ml_classification_experiment.ipynb
    |-- ml_spca_kpca_experiment.ipynb
    |-- ml_tsne_oos_experiment.ipynb
    |-- ml_tsne_oos_pca50_grad2_experiment.ipynb
    |-- spca_run2_experiments.ipynb
    |-- tsne_run2_original_space_2d.ipynb
    |-- global_tsne_original_space_3d.ipynb
    `-- thesis_figures.ipynb
```

## Main Workflows

### Data loading and visualization

- `code/data_utils.py` contains utilities for loading MATLAB PIV data and
  plotting velocity fields.
- `code/data_visualization.ipynb` is the exploratory notebook for inspecting raw
  fields, mean fields, and fluctuations.
- `code/thesis_figures.ipynb` is the main notebook for thesis-ready figures.
  Figures intended for the manuscript are saved in `code/figures/thesis/`.

### Intrinsic dimension

- `code/intrinsic_dimension.ipynb` is the original exploratory notebook.
- `code/intrinsic_dimension_corrected.ipynb` is the corrected notebook for
  thesis-ready intrinsic-dimension results.

The corrected workflow applies estimators such as Two-NN, MLE, MiND ML, DANCo,
and local PCA diagnostics to RUN2 after spatial downsampling.

### Modal and spectral decompositions

- `code/PCA.ipynb` contains the older PCA, Sparse PCA, t-SNE, and clustering
  exploration.
- `code/SPOD_Run1.ipynb` and `code/SPOD_comparison.ipynb` cover early SPOD
  computation and PCA/SPOD comparison.
- `code/spod_cross_run_harmonics.ipynb` analyzes SPOD harmonic structure across
  forced runs.
- `code/spca_run2_experiments.ipynb` stores corrected Sparse PCA experiments for
  RUN2.
- `code/spod_utils.py` contains reusable SPOD routines.

### Phase correspondence and clustering history

- `code/phase_correspondence.ipynb` compares SPOD modes and clustering outputs
  with phase-averaged reference fields.
- `code/centroid_phase_comparison.ipynb` compares earlier cluster centroids with
  phase-averaged fields.
- `code/centroid_plots/`, `code/comparisons/`, and
  `code/figures/phase_correspondence/` store outputs from these historical
  validation experiments.

### Nonlinear embeddings

- `code/manifold_analysis.ipynb` contains early t-SNE and manifold-analysis
  experiments.
- `code/global_cross_run_tsne.ipynb` computes the global 3D t-SNE embedding from
  PCA-preprocessed data.
- `code/cross_run_tsne.ipynb`, `code/tsne_compare_hyperparameters.ipynb`,
  `code/tsne_knn_experiment.ipynb`, and `code/tsne_out_of_sample_test.ipynb`
  contain t-SNE experiments and out-of-sample tests.
- `code/tsne_run2_original_space_2d.ipynb` computes a PCA-free RUN2 t-SNE
  experiment from original velocity-space distances.
- `code/global_tsne_original_space_3d.ipynb` is the corresponding PCA-free
  global t-SNE experiment, using a guarded subsampling strategy by default.
- `code/global_cross_run_umap.ipynb` and `code/individual_run_umap.ipynb`
  contain global and per-run UMAP experiments.

### Supervised learning

- `code/ml_experiment_utils.py` contains shared utilities for supervised
  experiments.
- `code/ml_classification_experiment.ipynb` compares PCA and original t-SNE
  representations.
- `code/ml_spca_kpca_experiment.ipynb` adds Sparse PCA and Kernel PCA.
- `code/ml_tsne_oos_experiment.ipynb` evaluates corrected out-of-sample t-SNE
  based on 500 PCA input coordinates.
- `code/ml_tsne_oos_pca50_grad2_experiment.ipynb` repeats the corrected
  out-of-sample t-SNE comparison using 50 PCA input coordinates before t-SNE.

The supervised tasks include:

- phase classification;
- actuation Strouhal-number classification;
- joint phase and frequency classification;
- actuation-frequency regression on unseen held-out runs;
- phase transfer to unseen held-out runs.

## Current Thesis Findings

The current thesis version supports the following main conclusions:

- The PIV snapshots occupy a much lower-dimensional structure than the ambient
  `171622`-coordinate velocity space. Intrinsic-dimension estimates vary by
  method and scale, but remain far below the ambient dimension.
- PCA/POD provides a strong linear baseline. For RUN2, the first ten modes retain
  most of the fluctuation energy, while 500 modes retain almost all of it.
- Sparse PCA produces more localized spatial modes while preserving predictive
  information comparable to PCA.
- SPOD isolates structures organized by actuation frequency and harmonics, which
  PCA spreads across multiple time-domain modes.
- t-SNE reveals phase organization within individual forced runs and separates
  forcing frequencies in a global embedding.
- Random Forest experiments show that actuation frequency is more consistently
  encoded than actuation phase, especially when testing on frequencies excluded
  from training.

## Dependencies

The project was developed in a Python/Anaconda environment. Core packages used
across scripts and notebooks include:

```text
numpy
scipy
matplotlib
pandas
seaborn
h5py
scikit-learn
scikit-dimension
umap-learn
tqdm
```

Some notebooks are memory-intensive. Global analyses over all forced runs stream
or cache intermediate results where possible, but many operations still require
substantial RAM and disk space.

## Reproducibility Notes

- The repository is provided primarily for review and traceability of the thesis
  results, not for external modification.
- The notebooks document the computational history of the project. Some are
  exploratory or historical, while others produced thesis-ready figures and
  tables.
- Prefer cached artifacts in `pca_data/`, `SPOD_data/`, `SPCA_data/`,
  `tsne_data/`, `umap_data/`, and `ml_results/` when reproducing thesis figures.
- Do not rerun expensive notebooks unless the intended output filenames are new
  or the old outputs are known to be obsolete.
- Treat notebooks that produced thesis figures or result tables as part of the
  experimental record.
- Keep thesis-specific figures in `code/figures/thesis/` and avoid plot titles
  inside saved figures; the thesis caption provides the interpretation.
