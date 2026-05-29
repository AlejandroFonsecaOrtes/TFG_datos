"""
Utility functions for the ML classification experiment.

This module provides the infrastructure for comparing dimensionality reduction
approaches (PCA at various ranks vs. 3D t-SNE) for classifying actuation phase
and Strouhal number from PIV velocity snapshots.

The experiment uses Random Forest models at three complexity levels and tests
both standard classification (Tasks A, B, C) and generalization to unseen
actuation frequencies (Task D).
"""

import numpy as np
import os
import gc
import time
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, confusion_matrix
)

# ============================================================================
# 1. CONSTANTS AND CONFIGURATION
# ============================================================================

# Actuation Strouhal numbers for each run (from info.txt)
ST_INFO = {
    1: 0.0, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20, 6: 0.25, 7: 0.30,
    8: 0.35, 9: 0.40, 10: 0.45, 11: 0.50, 12: 0.55, 13: 0.60,
    14: 0.65, 15: 0.70, 16: 0.75, 17: 0.80, 18: 0.85, 19: 0.90,
    20: 0.95, 21: 1.00, 22: 1.05, 23: 1.10, 24: 1.15, 25: 1.20
}

# Forced runs only (exclude Run 1 which is unforced, no phase)
FORCED_RUNS = list(range(2, 26))

# Runs to hold out for generalization testing (Task D)
# These are spaced across the St range: 0.15, 0.45, 0.75, 1.05
HELD_OUT_RUNS = [4, 10, 16, 22]

# Number of snapshots per run (after discarding first frame)
N_SNAPSHOTS = 2030

# Number of phases per actuation cycle
N_PHASES = 20

# Number of PCA components in the global IncrementalPCA
N_PCA_COMPONENTS = 500

# Random state for reproducibility
RANDOM_STATE = 42

# Random Forest complexity levels
COMPLEXITY_LEVELS = {
    'Low':    {'n_estimators': 10,  'max_depth': 5},
    'Medium': {'n_estimators': 100, 'max_depth': 20},
    'High':   {'n_estimators': 500, 'max_depth': None},
}

# PCA dimensionalities to sweep
PCA_DIMS = [10, 50, 100, 200, 500]


# ============================================================================
# 2. DATA LOADING AND GLOBAL PCA
# ============================================================================

def load_or_compute_global_pca(compressed_data_dir, output_path,
                               n_components=N_PCA_COMPONENTS,
                               forced_runs=FORCED_RUNS):
    """
    Load pre-computed global PCA scores or compute them via IncrementalPCA.

    The IncrementalPCA streams through one run at a time to avoid loading
    all 24 runs (~60 GB) into memory simultaneously.

    Parameters
    ----------
    compressed_data_dir : str
        Path to directory containing RUNX_PIV_compressed.npz files.
    output_path : str
        Path to save/load the global PCA scores (.npz).
    n_components : int
        Number of PCA components (default 500).
    forced_runs : list of int
        Run indices to include (default: 2-25, forced runs only).

    Returns
    -------
    pca_scores : ndarray of shape (n_total_snapshots, n_components)
        Global PCA scores for all forced runs concatenated.
    run_labels : ndarray of shape (n_total_snapshots,)
        Run index for each snapshot.
    st_labels : ndarray of shape (n_total_snapshots,)
        Strouhal number for each snapshot.
    phase_labels : ndarray of shape (n_total_snapshots,)
        Phase label (0-19) for each snapshot.
    """
    # Check if pre-computed scores exist
    if os.path.exists(output_path):
        print(f"Loading pre-computed global PCA scores from {output_path}...")
        data = np.load(output_path)
        return (data['pca_scores'], data['run_labels'],
                data['st_labels'], data['phase_labels'])

    print(f"Computing global IncrementalPCA ({n_components} components)...")

    # ------ PASS 1: Fit IncrementalPCA ------
    print("--- PASS 1: Fitting IncrementalPCA ---")
    ipca = IncrementalPCA(n_components=n_components)

    for run_idx in forced_runs:
        data_path = os.path.join(
            compressed_data_dir, f"RUN{run_idx}_PIV_compressed.npz"
        )
        if not os.path.exists(data_path):
            print(f"  RUN {run_idx}: Data not found, skipping.")
            continue

        print(f"  Fitting RUN {run_idx}...")
        d = np.load(data_path)
        u, v = d['u'], d['v']
        nt = u.shape[0]

        # Flatten (u, v) into a single feature vector per snapshot
        X_full = np.empty((nt, 2 * u.shape[1] * u.shape[2]), dtype=np.float32)
        X_full[:, :u.shape[1]*u.shape[2]] = u.reshape(nt, -1)
        X_full[:, u.shape[1]*u.shape[2]:] = v.reshape(nt, -1)

        del u, v, d
        gc.collect()

        # Partial fit in chunks to control memory
        chunk_size = 800
        for start_idx in range(0, nt, chunk_size):
            end_idx = min(start_idx + chunk_size, nt)
            ipca.partial_fit(X_full[start_idx:end_idx])
            gc.collect()

        del X_full
        gc.collect()

    var_explained = np.sum(ipca.explained_variance_ratio_)
    print(f"\nTotal variance explained by {n_components} components: "
          f"{var_explained:.2%}")

    # ------ PASS 2: Transform to PCA scores ------
    print("\n--- PASS 2: Extracting PCA Scores ---")

    all_scores = []
    all_run_labels = []
    all_st_labels = []
    all_phase_labels = []

    for run_idx in forced_runs:
        data_path = os.path.join(
            compressed_data_dir, f"RUN{run_idx}_PIV_compressed.npz"
        )
        if not os.path.exists(data_path):
            continue

        print(f"  Transforming RUN {run_idx}...")
        d = np.load(data_path)
        u, v = d['u'], d['v']
        nt = u.shape[0]

        X_full = np.empty((nt, 2 * u.shape[1] * u.shape[2]), dtype=np.float32)
        X_full[:, :u.shape[1]*u.shape[2]] = u.reshape(nt, -1)
        X_full[:, u.shape[1]*u.shape[2]:] = v.reshape(nt, -1)

        del u, v, d
        gc.collect()

        # Transform in chunks
        scores_list = []
        chunk_size = 800
        for start_idx in range(0, nt, chunk_size):
            end_idx = min(start_idx + chunk_size, nt)
            scores_list.append(ipca.transform(X_full[start_idx:end_idx]))

        all_scores.append(np.vstack(scores_list).astype(np.float32))

        # Labels for this run
        all_run_labels.append(np.full(nt, run_idx, dtype=np.int32))
        all_st_labels.append(
            np.full(nt, ST_INFO[run_idx], dtype=np.float32)
        )
        all_phase_labels.append(
            (np.arange(nt) % N_PHASES).astype(np.int32)
        )

        del X_full, scores_list
        gc.collect()

    # Concatenate all runs
    pca_scores = np.vstack(all_scores)
    run_labels = np.concatenate(all_run_labels)
    st_labels = np.concatenate(all_st_labels)
    phase_labels = np.concatenate(all_phase_labels)

    print(f"\nGlobal PCA matrix shape: {pca_scores.shape}")
    print(f"Size in memory: {pca_scores.nbytes / 1e6:.1f} MB")

    # Save for future use
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        pca_scores=pca_scores,
        run_labels=run_labels,
        st_labels=st_labels,
        phase_labels=phase_labels,
        var_explained=var_explained
    )

    return pca_scores, run_labels, st_labels, phase_labels


# ============================================================================
# 3. DATA SPLITTING
# ============================================================================

def create_standard_split(pca_scores, phase_labels, st_labels,
                          test_size=0.2, random_state=RANDOM_STATE):
    """
    Create a stratified 80/20 train/test split across all snapshots.

    Stratification uses a combined (St_act, phase) label to ensure
    proportional representation of all class combinations.

    Returns
    -------
    train_idx, test_idx : arrays of indices into the global arrays.
    """
    # Create a combined label for stratification
    # Encode as: st_label_index * N_PHASES + phase_label
    unique_st = np.unique(st_labels)
    st_to_idx = {s: i for i, s in enumerate(unique_st)}
    combined_labels = np.array([
        st_to_idx[st_labels[i]] * N_PHASES + phase_labels[i]
        for i in range(len(phase_labels))
    ])

    indices = np.arange(len(pca_scores))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=combined_labels
    )
    return train_idx, test_idx


def create_leave_freq_out_split(run_labels, held_out_runs=HELD_OUT_RUNS):
    """
    Create a leave-frequencies-out split for generalization testing.

    Training set: all snapshots from runs NOT in held_out_runs.
    Test set: all snapshots from held_out_runs.

    Returns
    -------
    train_idx, test_idx : arrays of indices into the global arrays.
    """
    test_mask = np.isin(run_labels, held_out_runs)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


# ============================================================================
# 4. t-SNE EMBEDDING (sklearn + out-of-sample projections)
# ============================================================================

def _calibrate_sigma(distances, target_perplexity, tol=1e-5, max_iter=200):
    """
    Binary search for the Gaussian kernel bandwidth sigma that yields
    the desired perplexity for a given set of neighbor distances.

    Perplexity = 2^H(P), where H is the Shannon entropy of the
    conditional probability distribution P_{j|i}.
    """
    lo, hi = 1e-10, 1e4

    for _ in range(max_iter):
        sigma = (lo + hi) / 2.0

        # Compute conditional probabilities with this sigma
        p = np.exp(-distances**2 / (2.0 * sigma**2))
        sum_p = p.sum()

        # Degenerate case: all probabilities collapsed to zero
        if sum_p < 1e-12:
            lo = sigma
            continue

        p = p / sum_p

        # Compute Shannon entropy in bits
        mask = p > 1e-12
        entropy = -np.sum(p[mask] * np.log2(p[mask]))
        perplexity = 2.0 ** entropy

        if np.abs(perplexity - target_perplexity) < tol:
            break
        if perplexity > target_perplexity:
            hi = sigma
        else:
            lo = sigma

    return sigma


def fit_tsne_embedding(X_train, n_components=3, perplexity=50,
                       random_state=RANDOM_STATE):
    """
    Fit sklearn t-SNE on the training set only.

    This helper keeps the t-SNE fitting step separate from the out-of-sample
    projection step, so the same training embedding can be reused for both
    KL-based and k-NN-based test projections without leaking test information
    into the nonlinear embedding.
    """
    from sklearn.manifold import TSNE

    print(f"Fitting sklearn TSNE (n_components={n_components}, "
          f"perplexity={perplexity}) on {X_train.shape[0]} samples "
          f"with {X_train.shape[1]} features...")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate='auto',
        init='pca',
        random_state=random_state,
        n_jobs=-1
    )

    t0 = time.time()
    train_embedding = tsne.fit_transform(X_train)
    fit_time = time.time() - t0
    print(f"  Fitting completed in {fit_time/60:.1f} minutes.")

    return train_embedding, fit_time


def _embed_new_points(X_train_hd, train_embedding, X_test_hd,
                      k=50, perplexity=30):
    """
    Embed new (test) points into an existing t-SNE embedding by fixing
    all training positions and optimizing each test point's position
    individually via L-BFGS-B.

    For each test point:
      1. Find its k nearest neighbors in the high-dimensional space
      2. Compute Gaussian affinities with perplexity-calibrated bandwidth
      3. Initialize position as the affinity-weighted centroid of neighbors
      4. Minimize the KL divergence between high-dimensional affinities (p)
         and low-dimensional Student-t affinities (q), keeping training
         positions fixed.

    The low-dimensional q distribution is globally normalized over all
    training points, not only over the k neighbors. This global denominator is
    the repulsive part of the t-SNE objective. Without it, test points can move
    arbitrarily far away because the local neighbor-only q distribution becomes
    nearly uniform at infinity.

    Parameters
    ----------
    X_train_hd : ndarray (n_train, n_features)
        Training data in the original high-dimensional space.
    train_embedding : ndarray (n_train, n_components)
        Fixed training positions in the t-SNE embedding.
    X_test_hd : ndarray (n_test, n_features)
        Test data in the original high-dimensional space.
    k : int
        Number of nearest neighbors to consider (default 50).
    perplexity : float
        Target perplexity for the Gaussian kernel (default 30).

    Returns
    -------
    test_embedding : ndarray (n_test, n_components)
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.optimize import minimize

    n_test = X_test_hd.shape[0]
    n_comp = train_embedding.shape[1]

    print(f"Finding {k}-NN for {n_test} test points...")

    # Step 1: Find k nearest training neighbors for every test point
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_train_hd)
    distances, indices = nn.kneighbors(X_test_hd)

    # Clamp target perplexity to be at most k-1 (can't exceed #neighbors)
    target_perp = min(perplexity, k - 1)

    test_embedding = np.zeros((n_test, n_comp))

    for i in range(n_test):
        dists_i = distances[i]                        # (k,)
        nbr_idx = indices[i]                          # (k,)
        nbr_pos = train_embedding[nbr_idx]            # (k, n_comp)

        # Step 2: Calibrate sigma and compute high-dim affinities
        sigma = _calibrate_sigma(dists_i, target_perp)
        p = np.exp(-dists_i**2 / (2.0 * sigma**2))
        p /= p.sum() + 1e-12

        # Step 3: Initialize at affinity-weighted centroid
        y0 = (p[:, None] * nbr_pos).sum(axis=0)

        # Step 4: Optimize the KL divergence with global t-SNE normalization.
        def objective_and_grad(y):
            # q is normalized over all fixed training locations. This preserves
            # the global repulsive force that prevents the previous exploding
            # test-coordinate artifact.
            diff_all = y[None, :] - train_embedding
            sq_dist_all = np.sum(diff_all**2, axis=1)
            q_unnorm_all = 1.0 / (1.0 + sq_dist_all)
            z_all = q_unnorm_all.sum() + 1e-12
            q_all = q_unnorm_all / z_all

            q_k = q_all[nbr_idx]
            q_unnorm_k = q_unnorm_all[nbr_idx]

            # KL divergence: sum p * log(p/q)
            kl = np.sum(p * np.log((p + 1e-12) / (q_k + 1e-12)))

            # Attractive term: pull y toward the high-dimensional neighbors.
            attr_factor = p * q_unnorm_k
            grad_attr = np.sum(
                attr_factor[:, None] * (y[None, :] - nbr_pos),
                axis=0
            )

            # Repulsive term: push y away from all fixed training points.
            rep_factor = q_all * q_unnorm_all
            grad_rep = np.sum(rep_factor[:, None] * diff_all, axis=0)

            # For the single-point out-of-sample objective
            # C(y) = sum_j p_j log(p_j / q_j), the derivative is
            # 2 * (attraction - repulsion). The familiar factor 4 in standard
            # t-SNE comes from the symmetric full-embedding objective.
            grad = 2.0 * (grad_attr - grad_rep)

            return kl, grad

        result = minimize(
            objective_and_grad, y0, jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        test_embedding[i] = result.x

        if (i + 1) % 2000 == 0:
            print(f"    Embedded {i+1}/{n_test} test points...")

    return test_embedding


def embed_new_points_knn(X_train_hd, train_embedding, X_test_hd, k=10):
    """
    Embed test points by inverse-distance-weighted k-NN interpolation.

    This is a deterministic, inexpensive baseline for t-SNE out-of-sample
    projection. It does not optimize a t-SNE objective; it places each test
    snapshot at the weighted average of its nearest training coordinates in
    the already fitted t-SNE space.
    """
    from sklearn.neighbors import NearestNeighbors

    print(f"Finding {k}-NN for {X_test_hd.shape[0]} test points "
          "in high-dimensional space...")
    t0 = time.time()

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_train_hd)
    distances, indices = nn.kneighbors(X_test_hd)

    # Inverse-distance weights emphasize the nearest physically similar
    # snapshots while the small epsilon avoids division by zero.
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)

    test_embedding = np.zeros(
        (X_test_hd.shape[0], train_embedding.shape[1]),
        dtype=train_embedding.dtype
    )
    for i in range(X_test_hd.shape[0]):
        test_embedding[i] = np.average(
            train_embedding[indices[i]],
            axis=0,
            weights=weights[i]
        )

    print(f"  k-NN interpolation completed in {time.time() - t0:.1f} seconds.")
    return test_embedding


def embed_new_points_kl(X_train_hd, train_embedding, X_test_hd,
                        k=50, perplexity=30):
    """
    Public wrapper for the globally normalized KL out-of-sample t-SNE method.

    Keeping this wrapper separate from the private implementation makes
    notebooks clearer: the method name states the mathematical objective being
    used, while the implementation remains available for backward-compatible
    calls through fit_and_transform_tsne().
    """
    return _embed_new_points(
        X_train_hd,
        train_embedding,
        X_test_hd,
        k=k,
        perplexity=perplexity
    )


def fit_and_transform_tsne(X_train, X_test, n_components=3, perplexity=50,
                           k_neighbors=50, random_state=RANDOM_STATE,
                           transform_method='kl'):
    """
    Fit sklearn t-SNE on training data, then embed test points by
    projecting them into the fixed training embedding.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, n_features)
        Training PCA scores.
    X_test : ndarray of shape (n_test, n_features)
        Test PCA scores (out-of-sample).
    n_components : int
        Dimensionality of the t-SNE embedding (default 3).
    perplexity : float
        t-SNE perplexity parameter for fitting (default 50).
    k_neighbors : int
        Number of nearest neighbors for out-of-sample embedding (default 50).
    transform_method : {'kl', 'knn'}
        Out-of-sample projection method. 'kl' uses the globally normalized
        KL-divergence objective; 'knn' uses inverse-distance interpolation.

    Returns
    -------
    train_embedding : ndarray of shape (n_train, n_components)
    test_embedding : ndarray of shape (n_test, n_components)
    fit_time : float
        Time to fit the embedding on training data (seconds).
    transform_time : float
        Time to embed test data (seconds).
    """
    train_embedding, fit_time = fit_tsne_embedding(
        X_train,
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )

    # Embed test data while keeping the training positions fixed.
    print(f"Embedding {X_test.shape[0]} test points with "
          f"{transform_method!r} projection (k={k_neighbors})...")
    t0 = time.time()
    if transform_method == 'kl':
        test_embedding = embed_new_points_kl(
            X_train, train_embedding, X_test,
            k=k_neighbors, perplexity=min(perplexity, k_neighbors - 1)
        )
    elif transform_method == 'knn':
        test_embedding = embed_new_points_knn(
            X_train, train_embedding, X_test, k=k_neighbors
        )
    else:
        raise ValueError("transform_method must be either 'kl' or 'knn'.")

    transform_time = time.time() - t0
    print(f"  Embedding completed in {transform_time:.1f} seconds.")

    return train_embedding, test_embedding, fit_time, transform_time


# ============================================================================
# 5. MODEL TRAINING AND EVALUATION
# ============================================================================

def count_total_nodes(model):
    """
    Count the total number of nodes across all trees in a Random Forest.

    This serves as a proxy for model complexity — more informative than
    just counting n_estimators, because it captures depth and branching.
    """
    total = 0
    for estimator in model.estimators_:
        # For multi-output, each estimator contains sub-estimators
        if hasattr(estimator, 'tree_'):
            total += estimator.tree_.node_count
        else:
            # Multi-output case: estimator is itself a tree
            total += estimator.tree_.node_count
    return total


def run_classification(X_train, X_test, y_train, y_test,
                       n_estimators, max_depth):
    """
    Train a RandomForestClassifier and return metrics.

    Returns
    -------
    dict with keys: accuracy, f1_macro, train_time, inference_time,
                    total_nodes, predictions
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    total_nodes = count_total_nodes(model)

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'train_time': train_time,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(y_test),
        'total_nodes': total_nodes,
        'predictions': y_pred,
    }


def run_multi_output_classification(X_train, X_test, y_train, y_test,
                                    n_estimators, max_depth):
    """
    Train a multi-output RandomForestClassifier for joint (phase, St_act)
    prediction.

    y_train, y_test should be arrays of shape (n_samples, 2) where
    column 0 = phase, column 1 = St_act class.

    Returns
    -------
    dict with per-output metrics and exact match ratio.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train (sklearn RF natively supports multi-output)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    # Per-output metrics
    acc_phase = accuracy_score(y_test[:, 0], y_pred[:, 0])
    acc_st = accuracy_score(y_test[:, 1], y_pred[:, 1])
    f1_phase = f1_score(y_test[:, 0], y_pred[:, 0],
                        average='macro', zero_division=0)
    f1_st = f1_score(y_test[:, 1], y_pred[:, 1],
                     average='macro', zero_division=0)

    # Exact match ratio: both predictions correct
    exact_match = np.mean(
        (y_pred[:, 0] == y_test[:, 0]) & (y_pred[:, 1] == y_test[:, 1])
    )

    total_nodes = count_total_nodes(model)

    return {
        'accuracy_phase': acc_phase,
        'accuracy_st': acc_st,
        'f1_phase': f1_phase,
        'f1_st': f1_st,
        'exact_match': exact_match,
        'train_time': train_time,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(y_test),
        'total_nodes': total_nodes,
        'predictions': y_pred,
    }


def run_regression(X_train, X_test, y_train, y_test,
                   n_estimators, max_depth):
    """
    Train a RandomForestRegressor for St_act interpolation.

    Returns
    -------
    dict with regression metrics: MAE, RMSE, R2.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    total_nodes = sum(est.tree_.node_count for est in model.estimators_)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_time': train_time,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(y_test),
        'total_nodes': total_nodes,
        'predictions': y_pred,
    }


# ============================================================================
# 6. EXPERIMENT RUNNERS
# ============================================================================

def build_representations(pca_scores_train, pca_scores_test,
                          tsne_train=None, tsne_test=None):
    """
    Build a dictionary of (name -> (X_train, X_test)) representations.

    Includes PCA at various truncation levels and (optionally) t-SNE.
    """
    reps = {}
    for k in PCA_DIMS:
        reps[f'PCA-{k}'] = (pca_scores_train[:, :k],
                            pca_scores_test[:, :k])

    if tsne_train is not None and tsne_test is not None:
        reps['t-SNE-3D'] = (tsne_train, tsne_test)

    return reps


def run_all_standard_experiments(representations, phase_train, phase_test,
                                 st_train, st_test):
    """
    Run Tasks A (phase), B (St_act), C (joint) for all representations
    and complexity levels.

    Returns
    -------
    results : list of dicts, each containing task, representation,
              complexity level, and all metrics.
    """
    results = []

    # Convert float St_act labels to strings for classification
    # (sklearn classifiers reject continuous float labels)
    st_train_cls = np.array([f'{s:.2f}' for s in st_train])
    st_test_cls = np.array([f'{s:.2f}' for s in st_test])

    for rep_name, (X_train, X_test) in representations.items():
        for level_name, params in COMPLEXITY_LEVELS.items():
            print(f"\n  [{rep_name} | {level_name}]", end="", flush=True)
            n_est = params['n_estimators']
            max_d = params['max_depth']

            # --- Task A: Phase classification ---
            res_a = run_classification(
                X_train, X_test, phase_train, phase_test,
                n_est, max_d
            )
            results.append({
                'task': 'A (Phase)',
                'representation': rep_name,
                'complexity': level_name,
                **{k: v for k, v in res_a.items() if k != 'predictions'}
            })
            print(f"  A:{res_a['accuracy']:.3f}", end="", flush=True)

            # --- Task B: St_act classification ---
            res_b = run_classification(
                X_train, X_test, st_train_cls, st_test_cls,
                n_est, max_d
            )
            results.append({
                'task': 'B (St_act)',
                'representation': rep_name,
                'complexity': level_name,
                **{k: v for k, v in res_b.items() if k != 'predictions'}
            })
            print(f"  B:{res_b['accuracy']:.3f}", end="", flush=True)

            # --- Task C: Joint classification ---
            y_train_joint = np.column_stack([phase_train, st_train_cls])
            y_test_joint = np.column_stack([phase_test, st_test_cls])
            res_c = run_multi_output_classification(
                X_train, X_test, y_train_joint, y_test_joint,
                n_est, max_d
            )
            results.append({
                'task': 'C (Joint)',
                'representation': rep_name,
                'complexity': level_name,
                **{k: v for k, v in res_c.items() if k != 'predictions'}
            })
            print(f"  C:{res_c['exact_match']:.3f}", end="", flush=True)

    print("\nDone.")
    return results


def run_all_generalization_experiments(representations,
                                       phase_train, phase_test,
                                       st_train_continuous,
                                       st_test_continuous):
    """
    Run Task D (generalization) for all representations and complexity levels.

    D.1: St_act regression (continuous target)
    D.2: Phase classification (transfer to unseen frequencies)

    Returns
    -------
    results : list of dicts with task, representation, complexity, metrics.
    """
    results = []

    for rep_name, (X_train, X_test) in representations.items():
        for level_name, params in COMPLEXITY_LEVELS.items():
            print(f"\n  [{rep_name} | {level_name}]", end="", flush=True)
            n_est = params['n_estimators']
            max_d = params['max_depth']

            # --- Task D.1: St_act regression ---
            res_d1 = run_regression(
                X_train, X_test,
                st_train_continuous, st_test_continuous,
                n_est, max_d
            )
            results.append({
                'task': 'D.1 (St regression)',
                'representation': rep_name,
                'complexity': level_name,
                **{k: v for k, v in res_d1.items() if k != 'predictions'}
            })
            print(f"  D1-R2:{res_d1['r2']:.3f}", end="", flush=True)

            # --- Task D.2: Phase classification (transfer) ---
            res_d2 = run_classification(
                X_train, X_test, phase_train, phase_test,
                n_est, max_d
            )
            results.append({
                'task': 'D.2 (Phase transfer)',
                'representation': rep_name,
                'complexity': level_name,
                **{k: v for k, v in res_d2.items() if k != 'predictions'}
            })
            print(f"  D2:{res_d2['accuracy']:.3f}", end="", flush=True)

    print("\nDone.")
    return results
