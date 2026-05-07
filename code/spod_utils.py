import numpy as np
from numpy.fft import rfft
from scipy.signal import get_window
from scipy.linalg import eigh
import re

def compute_spod(U, V, n_fft=256, overlap=None, window_type='hamming', n_modes=None):
    """
    SPOD using method of snapshots for memory efficiency (explained in Schmidt and Colonius, 2020)

    Parameters:
        U, V: arrays of shape (nt, nx, ny)
        n_fft: number of samples per block (e.g., 256)
        overlap: overlap between blocks (default: n_fft // 2)
        window_type: string (e.g., 'hamming')
        n_modes: int or None - number of leading modes to return

    Returns:
        freqs: ndarray of shape (n_freq,)
        eigvals: ndarray (n_freq, n_modes)
        eigvecs: ndarray (n_freq, n_modes, spatial_dof)
    """
    if overlap is None:
        overlap = n_fft // 2

    nt, nx, ny = U.shape
    spatial_dof = 2 * nx * ny # spatial degrees of freedom (called M in the paper)
    step = n_fft - overlap # how many realizations we advance per block
    n_blocks = (nt - overlap) // step
    n_freq = n_fft // 2 + 1

    if n_blocks < 2:
        raise ValueError("Not enough time blocks.")

    # Check that the wanted number of modes is valid (max modes = n_blocks)
    if n_modes is not None and n_modes > n_blocks:
      print(f"Warning: Requested n_modes={n_modes} exceeds number of blocks={n_blocks}. Reducing n_modes to {n_blocks}.")
      n_modes = n_blocks

    # Create window and normalization factor
    window = get_window(window_type, n_fft)
    window = window / np.sqrt(np.sum(window**2) / n_fft)  # normalize energy

    # Prepare array for Fourier realizations: shape (n_freq, n_blocks, spatial_dof)
    Q_hat = np.empty((n_freq, n_blocks, spatial_dof), dtype=np.complex64)

    # For each block, we flatten the spatial dimensions compute the FFT of each
    # realization of the block and store it in the Q_hat matrix
    for i in range(n_blocks):
        start = i * step
        end = start + n_fft
        u_block = U[start:end].reshape(n_fft, -1)
        v_block = V[start:end].reshape(n_fft, -1)
        block = np.concatenate((u_block, v_block), axis=1)
        block *= window[:, None]
        # Since the data we are using is real, we will use rfft() which will only output positive frequencies
        fft_block = rfft(block, axis=0)
        Q_hat[:, i, :] = fft_block  # shape (n_freq, spatial_dof)

    # Prepare arrays for eigenvalues and eigenvectors
    eigvals = np.zeros((n_freq, n_blocks if n_modes is None else n_modes))
    eigvecs = np.zeros((n_freq, n_blocks if n_modes is None else n_modes, spatial_dof), dtype=np.complex64)

    # Solution to the eigenvalue problem
    for f in range(n_freq):
        Qf = Q_hat[f]  # (n_blocks, spatial_dof)
        # (n_blocks - 1) for the unbiased estimate
        C = Qf @ Qf.conj().T / (n_blocks - 1)  # shape (n_blocks, n_blocks)

        # Eigen-decomposition
        lam, psi = eigh(C)
        idx = np.argsort(lam)[::-1]
        lam = lam[idx]
        psi = psi[:, idx]

        if n_modes is not None:
            lam = lam[:n_modes]
            psi = psi[:, :n_modes]

        eigvals[f, :len(lam)] = lam.real
        # project the eigenvectors from snapshot space into the full physical
        # space, obtaining the actual SPOD modes.
        eigvecs[f, :len(lam)] = (psi.T @ Qf) / np.sqrt(lam[:, None])

    # d is the sampling spacing (time between snapshots). At the moment I don't know it, so we will leave it as 1 for now and use Hz/dt
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    return freqs, eigvals, eigvecs


def load_spod_npz(path):
    """Load minimal arrays needed for spectra-only comparison."""
    d = np.load(path, allow_pickle=False)
    freqs = d["freqs"]        # shape: (n_freq,)
    eigvals = d["eigvals"]    # shape: (n_freq, n_modes)
    return freqs, eigvals

def normalize_spectrum(eigvals):
    """Return (eigvals_norm, total_energy_by_freq)."""
    tot = eigvals.sum(axis=1, keepdims=True)  # (n_freq,1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(tot > 0, eigvals / tot, 0.0)
    return norm, tot.squeeze(-1)

def summarize_run(freqs, eigvals, k_energy=3):
    """Compute lightweight summary dict for spectra-only comparisons."""
    eigvals_norm, total_energy = normalize_spectrum(eigvals)
    k = min(k_energy, eigvals.shape[1])
    cum_energy_first_k = eigvals_norm[:, :k].sum(axis=1)  # (n_freq,)
    return {
        "freqs": freqs.astype(np.float32),
        "eigvals_norm": eigvals_norm.astype(np.float32),      # (n_freq, n_modes)
        "total_energy": total_energy.astype(np.float64),      # (n_freq,)
        "cum_energy_first_k": cum_energy_first_k.astype(np.float32),
        "k_energy": int(k),
        "n_modes_total": int(eigvals.shape[1]),
        "n_freq_total": int(eigvals.shape[0]),
    }

def run_sort_key(r):
    # Extract the number between "RUN" and "_"
    m = re.search(r"RUN(\d+)", r["name"])
    return int(m.group(1)) if m else r["name"]


