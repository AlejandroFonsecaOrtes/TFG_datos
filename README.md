# Pattern Recognition in Velocimetric Data of Turbulent Jet Flows

Bachelor's Thesis (TFG) — Statistics and Data Science

## Overview

This project applies unsupervised machine learning and spectral decomposition techniques to high-dimensional Particle Image Velocimetry (PIV) data from turbulent jet flows. The central argument is that the dynamics of periodically forced jets lie on a low-dimensional manifold, and that data-driven algorithms can autonomously recover the phase-locked coherent structures without external hardware triggers.

The computational pipeline chains together linear filtering (PCA), sparse localization (Sparse PCA), non-linear embedding (t-SNE) and unsupervised clustering (hierarchical Ward linkage) to map the intrinsic topology of the flow.

## Dataset

The experimental data consists of **25 runs** of a turbulent jet at Re = 10,000, each forced at a different Strouhal number (see `code/data/info.txt`). Each run contains **2,030 snapshots** on a **269 × 319** spatial grid with two velocity components (u, v). All quantities are non-dimensionalized: spatial coordinates as X = x/D and velocities as U = u/U∞.

| Data source | Location | Size per run | Description |
|---|---|---|---|
| Raw PIV | `code/data/RunX_PIV.mat` | ~8 GB | Full dataset including phase-averages, POD modes and Reynolds stresses |
| Compressed | `code/compressed_data/RUNX_PIV_compressed.npz` | ~2.6 GB | Fluctuating velocity fields (u, v) and spatial grid. Preferred for most analyses |
| SPOD results | `code/SPOD_data/RUNX_PIV_SPOD.npz` | ~1.3 GB | Pre-computed frequencies, eigenvalues and complex spatial modes |
| Sparse PCA | `code/SPCA_data/RUNX_PIV_SPCA.npz` | ~50 MB | Sparse components, temporal scores and sparsity metrics |
| Cluster labels | `code/clustering_data/RUNX_PIV_labels.npz` | ~8 KB | Pre-computed cluster assignments per snapshot |
| Spatial grid | `code/spatial_grid.npz` | ~700 KB | Shared X, Y coordinate meshgrids |

## Project Structure

```
TFG_datos/
├── AGENTS.md                  # Guidelines for AI agent contributors
├── README.md                  # This file
├── influentia_info.pdf        # Initial project brief (outdated)
│
├── papers/                    # Reference literature (11 PDFs)
│   ├── Guide to Spectral Proper Orthogonal Decomposition.pdf
│   ├── Sparse Principal Component Analysis.pdf
│   ├── Van der Maaten and Hinton- Visualizing Data using t-SNE.pdf
│   ├── Manifold Dimension Estimation - An Empirical Study.pdf
│   └── ...

└── code/
    ├── TODO.txt               # Detailed task breakdown for implementation
    ├── data_utils.py          # Data loading and velocity field plotting helpers
    ├── spod_utils.py          # SPOD computation utilities
    │
    ├── data/                  # Raw .mat files and run metadata
    ├── compressed_data/       # Preferred .npz velocity fields
    ├── SPOD_data/             # Pre-computed SPOD decompositions
    ├── SPCA_data/             # Pre-computed Sparse PCA results
    ├── clustering_data/       # Pre-computed cluster labels
    ├── figures/               # Exported publication-quality plots
    ├── centroid_plots/        # Per-cluster centroid velocity fields
    ├── comparisons/           # POD vs SPOD comparison plots
    │
    ├── data_visualization.ipynb       # Exploratory visualization of raw PIV fields
    ├── PCA.ipynb                      # POD, Sparse PCA, t-SNE and clustering pipeline
    ├── intrinsic_dimension.ipynb      # Intrinsic dimension estimation (TwoNN, MLE, etc.)
    ├── SPOD_Run1.ipynb                # SPOD computation and storage
    ├── SPOD_comparison.ipynb          # POD vs SPOD spectral analysis
    ├── phase_correspondence.ipynb     # Cluster centroids vs phase-averages validation
    ├── centroid_phase_comparison.ipynb # Centroid vs phase-average comparison (K=20)
    ├── manifold_analysis.ipynb        # t-SNE topology, SPCA PSD, 1D projection
    └── cross_run_analysis.ipynb       # Multi-run clustering invariance test
```

## Notebooks

Each notebook follows a strict **Markdown (math) → Code → Markdown (interpretation)** workflow.

| Notebook | Purpose |
|---|---|
| `data_visualization.ipynb` | Exploratory plots of raw velocity fields and mean flows |
| `PCA.ipynb` | Core pipeline: POD energy truncation → Sparse PCA → t-SNE embedding → hierarchical clustering → centroid visualization |
| `intrinsic_dimension.ipynb` | Estimates the intrinsic dimension of the flow manifold using TwoNN, MLE and local PCA methods |
| `SPOD_Run1.ipynb` | Computes and saves Spectral POD results for individual runs |
| `SPOD_comparison.ipynb` | Compares POD and SPOD eigenspectra to identify tonal vs broadband energy |
| `phase_correspondence.ipynb` | Validates unsupervised clusters against hardware-triggered phase-averages via cosine similarity |
| `centroid_phase_comparison.ipynb` | Compares K=20 cluster centroids with phase-averaged fields using cosine similarity matrices |
| `manifold_analysis.ipynb` | t-SNE on PCA vs SPCA scores, PSD of sparse temporal coefficients, 1D manifold projection |
| `cross_run_analysis.ipynb` | Merges snapshots from multiple runs (St=0 and St=0.05) and tests topological invariance of the clustering |

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
h5py
seaborn
```

All notebooks were developed with Python 3.x and standard Anaconda distributions.

## Key Results

- **Intrinsic dimension d ≈ 1** for the forced jet: the t-SNE embedding forms a continuous circular loop, and its 1D projection yields a monotonic sawtooth waveform ordered by phase.
- **Sparse PCA separates physics from noise**: modes localized in the shear layer exhibit sharp tonal peaks at the forcing frequency, while modes in the far field show broadband turbulent spectra.
- **Blind phase recovery**: hierarchical clustering centroids achieve cosine similarity > 0.95 with hardware-triggered phase-averages, proving that the phase-locked structures can be extracted without external triggers.
- **Spectral validation**: the leading SPOD mode at St_act matches the spatial topology of the physical phase-average.
- **Cross-run invariance**: the manifold geometry of the coherent structures is preserved across different forcing frequencies.

## References

- Zou, Hastie & Tibshirani (2006). Sparse Principal Component Analysis.
- Van der Maaten & Hinton (2008). Visualizing Data using t-SNE.
- Schmidt & Colonius (2020). Guide to Spectral Proper Orthogonal Decomposition.
- Navarro-González et al. (2022). A dual-time implementation of the SPOD.
- Bi & Lafaye de Micheaux (2025). Manifold Dimension Estimation: An Empirical Study.

See `papers/` for the full collection.
