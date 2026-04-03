# AI Agent Instructions for TFG (Pattern Recognition in Velocimetric Data)

Welcome. You are an AI agent assisting with a Bachelor's Thesis (TFG) on statistics and data science focused on the identification of velocity and pressure field patterns in jet flows. Please meticulously follow the rules below to ensure your contributions align with the project's rigorous academic standards.

## 1. Project Context and Understanding
- **Familiarize First:** Before completing any task, you must carefully study the existing code in the repository to fully understand the project context, data structures, and the mathematical implementation. Pay less attention to `influentia_info.pdf` as the project has evolved past its initial conception.
- **Literature and References:** The theoretical foundation of this work is documented in the `papers` folder. Agents must actively read and refer to the PDFs inside this directory to understand the implementation. When referencing a paper in the chat or notebooks, a simple mention of the author or title is sufficient.

## 2. Dataset Structure
The project contains 25 experimental runs of a turbulent jet flow acquired via Particle Image Velocimetry (PIV). Each run corresponds to a different actuation Strouhal number (see `code/data/info.txt` for the full list). All data is non-dimensionalized: spatial coordinates as `X = x/D`, velocities as `U = u/U_inf`, with `Re = 10000`. Each run contains 2030 snapshots on a spatial grid of 269×319 points.

**Important:** Always prefer the pre-processed `.npz` files over the raw `.mat` files unless you specifically need a variable that only exists in the `.mat` (e.g., phase-averages, POD modes, Reynolds stresses). The `.npz` files are significantly smaller and faster to load.

### 2.1 Raw Data — `code/data/RunX_PIV.mat`
Loaded via `data_utils.load_piv_data()`. Each `.mat` file (~8 GB) contains the complete experimental dataset for one run:

| Variable | Shape | Description |
|----------|-------|-------------|
| `X`, `Y` | (269, 319) | Spatial coordinate meshgrids (non-dimensional `x/D`, `y/D`) |
| `U`, `V` | (2031, 269, 319) | Instantaneous velocity components (horizontal, vertical) |
| `Um`, `Vm` | (269, 319) | Time-averaged mean velocity components |
| `u`, `v` | (2031, 269, 319) | Fluctuating velocity components (`u = U - Um`) |
| `Uph`, `Vph` | (20, 269×319) | Phase-averaged velocity (20 phases of the actuation cycle) |
| `Phi` | (2031, 171622) | POD spatial modes (171622 = 269×319×2 components) |
| `Psi` | (2031, 2031) | POD temporal coefficients |
| `Sig` | (1, 2031) | POD singular values (energy per mode) |
| `Uc` | scalar | Characteristic velocity for non-dimensionalization |
| `Res` | scalar | Resolution (jet diameter D) |
| `tke` | (269, 319) | Turbulent kinetic energy field |
| `u2`, `v2` | (269, 319) | Reynolds normal stresses ⟨uu⟩, ⟨vv⟩ |
| `uv` | (269, 319) | Reynolds shear stress ⟨uv⟩ |

### 2.2 Compressed Data — `code/compressed_data/RUNX_PIV_compressed.npz`
The preferred dataset for most analyses. Contains only the fluctuating velocity fields (~2.6 GB):

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `X` | (269, 319) | float32 | Spatial grid X |
| `Y` | (269, 319) | float32 | Spatial grid Y |
| `u` | (2030, 269, 319) | float64 | Fluctuating horizontal velocity |
| `v` | (2030, 269, 319) | float64 | Fluctuating vertical velocity |

### 2.3 Spatial Grid — `code/spatial_grid.npz`
A lightweight file containing only the spatial coordinates, shared across all runs:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `X_grid` | (269, 319) | float32 | Spatial grid X |
| `Y_grid` | (269, 319) | float32 | Spatial grid Y |

### 2.4 SPOD Results — `code/SPOD_data/RUNX_PIV_SPOD.npz`
Pre-computed Spectral Proper Orthogonal Decomposition results (~1.3 GB per run):

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `freqs` | (129,) | float64 | Frequency bins (in units of Hz/dt, since dt=1) |
| `eigvals` | (129, 14) | float64 | SPOD eigenvalues per frequency (energy of each mode) |
| `eigvecs` | (129, 14, 171622) | complex64 | SPOD spatial modes (complex-valued; 171622 = 2×269×319) |
| `X` | (269, 319) | float32 | Spatial grid X |
| `Y` | (269, 319) | float32 | Spatial grid Y |

### 2.5 Sparse PCA Results — `code/SPCA_data/RUNX_PIV_SPCA.npz`
Pre-computed Sparse PCA decomposition (currently only RUN2):

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `components` | (30, 171622) | float32 | Sparse spatial modes |
| `scores` | (2030, 30) | float32 | Temporal coefficients |
| `feature_mask` | (171622,) | bool | Mask of active features |
| `sparsity_modes` | scalar | float64 | Sparsity percentage of the modes |
| `FVE` | scalar | float64 | Fraction of Variance Explained |
| `time_sec` | scalar | float64 | Computation time in seconds |

### 2.6 Clustering Results — `code/clustering_data/RUNX_PIV_labels.npz`
Pre-computed cluster assignments (currently only RUN2):

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `labels` | (2030,) | int32 | Cluster label for each snapshot |

### 2.7 Run Information — `code/data/info.txt`
Maps each run number to its actuation Strouhal number `St_act = f_act * D / U_inf`. Runs marked with `*` may have special significance. Run 1 has `St_act = 0` (natural, unforced jet).

## 3. Figures and Visualization
Every plot produced in this project may end up directly in the thesis document. Figures must be **publication-quality** from the start—do not leave formatting for later.

- **Labels and titles:** Every axis must have a label with units or non-dimensional notation (e.g., `$x/D$`, `$u/U_{\infty}$`). Every figure must have a descriptive title. Use LaTeX rendering in matplotlib (`plt.rc('text', usetex=False)` with `$...$` math notation).
- **Font sizes:** Axis labels, titles, and legends must be legible. Use `fontsize=12` or larger as a minimum for labels. Tick labels should not be smaller than 10pt.
- **Colormaps:** Use perceptually uniform colormaps (`RdBu_r` for diverging fields centered on zero, `viridis` or `inferno` for sequential data). Avoid `jet` for new plots (it distorts perception of magnitudes).
- **Colorbars:** Every contour/heatmap plot must include a colorbar with a descriptive label.
- **Aspect ratio:** Velocity field plots must use `ax.set_aspect('equal')` to preserve physical proportions.
- **Layout:** Use `plt.tight_layout()` or `constrained_layout=True` to prevent label clipping. Multi-panel figures should use `plt.subplots()` with shared axes where appropriate.
- **Consistency:** Maintain the same colormap, arrow scale, and grid subsampling across comparable plots within a notebook so that visual differences reflect data differences, not formatting choices.
- **Saving figures:** Save every important figure to disk as a high-resolution PNG or PDF using `plt.savefig('filename.png', dpi=300, bbox_inches='tight')`. This makes it easy to include them in the thesis later.

## 4. Code Quality and Interpretability
- **Simplicity First:** The codebase will serve as the foundation for an academic thesis. The code must be highly intuitive, simple, and prioritize interpretability over hyper-optimization. 
- **Comment Density:** Aim for approximately 50% of your code lines to be comments. Explain the "why" and the underlying mathematical operations explicitly using a mix of detailed function docstrings and step-by-step inline comments.
- **Library Usage:** Primarily stick to the libraries already utilized in the existing codebase (e.g., `numpy`, `scipy`, `matplotlib`, `scikit-learn`). However, if an essential functionality is missing, other libraries may be used.
- **Auxiliary Files:** If a piece of code is too complex for a notebook and is better suited as an auxiliary function, proactively create a `.py` file (like `data_utils.py` or `spod_utils.py`) to hold it. You do not need explicit permission to structure the project logically.

## 5. Jupyter Notebook Workflow
When authoring or modifying Jupyter notebooks, you must strictly follow this structure for every major logical block:
1. **Mathematical Explanation (Markdown Cell):** A rigorous explanation of the mathematical concepts that the subsequent code will execute and how does it help the objective of the thesis. Use clean LaTeX formatting.
2. **Implementation (Code Cell):** The clearly commented, simple Python code that executes the math.
3. **Interpretation (Markdown Cell):** A rigorous interpretation of the results obtained from the code execution.

The **Interpretation** cell is critical. It must not be a superficial summary of what the plot shows. It must:
- State the **observed pattern** (e.g., "The first SPOD mode at $St = 0.05$ exhibits a cosine similarity of 0.97 with phase-average $P_8$").
- Provide a **physical or statistical explanation** for why that pattern occurs (e.g., "This confirms that the leading spectral mode at the forcing frequency captures the same coherent vortex structure as the phase-triggered average").
- Draw a **conclusion** relevant to the thesis argument (e.g., "SPOD can therefore replace hardware-triggered phase-averaging for identifying dominant coherent structures in forced jets").

**CRITICAL: You must NEVER write an interpretation or conclusion cell before executing the code and analyzing the actual outputs and plots.** Pre-writing conclusions based on what you *expect* the results to show is not how science or engineering works. The workflow is strictly:
1. Write the math explanation cell.
2. Write and **execute** the code cell.
3. **Observe** the actual numerical outputs, printed values, and generated plots.
4. **Only then** write the interpretation cell, based exclusively on what the data shows — not on prior assumptions.

If the results contradict your expectations, report what you observe honestly. Unexpected results are often more scientifically valuable than confirmations.

## 6. Communication and Output Tone
- **Academic Rigor:** All explanations, interpretations, and text must be rigorously written. Absolutely avoid the "AI touch" (e.g., overly enthusiastic tone, repetitive transitional phrases like "In summary," or "Let's dive in!"). Use a formal, objective, academic voice.
- **Proactive Questions:** If you encounter blockers, ambiguities, or issues that compromise the quality of your output, halt your execution and ask the user for clarification.
- **Comprehensive File Analysis:** If asked to analyze all files, you must go *file by file*. Analyze the code, explicitly evaluate whether it satisfies these project requirements, output the report directly in the chat, and offer a plan for improvements.

---

*Note: This file is a living document. The user may update these rules as the thesis evolves.*
