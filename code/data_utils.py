import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_piv_data(mat_path):
    """Load PIV data from .mat file"""
    data = {}
    with h5py.File(mat_path, 'r') as f:
        # Load key variables
        data['Phi'] = f['Phi'][:] # POD spatial modes (2031 modes × 171622 flattened spatial points(269x319x2components))
        data['Psi'] = f['Psi'][:] # POD temporal coefficients (2031 snapshots × 2031 modes)
        data['Res'] = f['Res'][0,0] # resolution (jet diameter D)
        data['Sig'] = f['Sig'][:] # POD singular values/eigenvalues (energy of each mode) (1x2031)
        data['U'] = f['U'][:]  # instantaneous horizontal velocity component (2031 snapshots × 269×319 grid)
        data['Uc'] = f['Uc'][0,0] # characteristic velocity for non-dimensionalizing (scalar)
        data['Um'] = f['Um'][:]  # time-averaged horizontal velocity component (269x319)
        data['Uph'] = f['Uph'][:] # phase-averaged horizontal velocity component (20 phases × spatial grid)
        data['V'] = f['V'][:]  # instantaneous vertical velocity component (2031 snapshots × 269×319 grid)
        data['Vm'] = f['Vm'][:] # time-averaged vertical velocity component (269x319)
        data['Vph'] = f['Vph'][:] # phase-averaged vertical velocity component (20 phases × spatial grid)
        data['X'] = f['X'][:] # spatial coordinate meshgrid (269x319) (non-dimensional X=x/D)
        data['Y'] = f['Y'][:] # spatial coordinate meshgrid (269x319) (non-dimensional Y=y/D)
        data['tke'] = f['tke'][:] # turbulent kinetic energy (269x319)
        data['u'] = f['u'][:] # fluctuating horizontal velocity component, u = U-Um (2031 snapshots × 269×319 grid)
        data['u2'] = f['u2'][:] # reynolds normal horizontal stress <uu> (269x319)
        data['uv'] = f['uv'][:] # reynolds shear stress <uv> (269x319)
        data['v'] = f['v'][:] # fluctuating vertical velocity component, v = V-Vm (2031 snapshots × 269×319 grid)
        data['v2'] = f['v2'][:] # reynolds normal vertical stress <vv> (269x319)
        
    
    return data


def plot_spod_mode(X, Y, mode, title=None, scale=0.1, quiver_step=10, cmap="RdBu_r"):
    """
    Visualizes a 2D SPOD mode using a quiver plot.

    Parameters:
        X, Y: arrays of shape (nx, ny) - spatial coordinates
        mode: array of shape (2, nx, ny) - real part of SPOD mode [u_mode, v_mode]
        title: optional string - plot title
        scale: float - scale factor for quiver arrows
        quiver_step: int - step size for downsampling quiver grid
        cmap: str - colormap for background magnitude
    """
    u, v = mode[0], mode[1]
    mag = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(10, 6))
    cont = ax.contourf(X, Y, mag, cmap=cmap, levels=50)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label("Mode Magnitude", fontsize=12)

    ax.quiver(
        X[::quiver_step, ::quiver_step],
        Y[::quiver_step, ::quiver_step],
        u[::quiver_step, ::quiver_step],
        v[::quiver_step, ::quiver_step],
        scale=scale, color="k", alpha=0.7
    )

    ax.set_title(title or "SPOD Mode", fontsize=14)
    ax.set_xlabel("$x/D$", fontsize=12)
    ax.set_ylabel("$y/D$", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_velocity_field(X, Y, uc, vc, title=None, scale=20, quiver_step=5, cmap="RdBu_r"):
        # magnitude of centroid velocity
    mag = np.sqrt(uc**2 + vc**2)

    fig, ax = plt.subplots(figsize=(10, 6))

    # filled contour of magnitude
    cont = ax.contourf(X, Y, mag, cmap=cmap, levels=50)
    cbar = plt.colorbar(cont, ax=ax)
    cbar.set_label("Velocity magnitude", fontsize=12)

    # black arrows on top
    ax.quiver(
        X[::quiver_step, ::quiver_step],
        Y[::quiver_step, ::quiver_step],
        uc[::quiver_step, ::quiver_step],
        vc[::quiver_step, ::quiver_step],
        scale=scale, color="k", alpha=0.7
    )

    ax.set_title(title or "Velocity Field", fontsize=14)
    ax.set_xlabel("$x/D$", fontsize=12)
    ax.set_ylabel("$y/D$", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()