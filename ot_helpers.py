import numpy as np
import ot
from scipy.fftpack import dct, idct, dst, idst
from scipy.ndimage import map_coordinates, zoom

# OT HELPERS
def _prepare_density(density):
    """Epsilon floor + normalise to mean=1."""
    rho = density.copy()
    rho += 1e-6 * rho.mean()
    return rho / rho.mean()

def _downsample_to_ot_grid(rho_full, Nx, Ny, Nx_ot, Ny_ot):
    """Block-average to coarser OT grid; return source/uniform-target marginals."""
    rho_ot = zoom(rho_full, (Nx_ot / Nx, Ny_ot / Ny), order=1)
    rho_ot = rho_ot / rho_ot.mean()
    rho_ot = np.clip(rho_ot, 1e-10, None)
    a = rho_ot.ravel()
    a = a / a.sum()
    b = np.ones(Nx_ot * Ny_ot) / (Nx_ot * Ny_ot)
    return a, b

def _build_cost_matrix(Nx_ot, Ny_ot):
    """Pairwise Euclidean cost matrix on the OT grid, normalised to [0,1]."""
    x_ot = np.linspace(0, 1, Nx_ot)
    y_ot = np.linspace(0, 1, Ny_ot)
    X_ot, Y_ot = np.meshgrid(x_ot, y_ot)
    grid_pts = np.vstack([X_ot.ravel(), Y_ot.ravel()]).T   # (Nx_ot*Ny_ot, 2)
    M = ot.dist(grid_pts, grid_pts, metric='euclidean')
    M /= M.max()
    return M, grid_pts, X_ot, Y_ot

def _solve_ot(a, b, M, reg, numItermax=5000):
    """Sinkhorn regularised OT; returns transport plan G."""
    return ot.sinkhorn(a, b, M, reg=reg, numItermax=numItermax, stopThr=1e-9)

def _barycentric_projection(G, a, grid_pts, Nx_ot, Ny_ot, X_ot, Y_ot):
    """Barycentric projection: T[i] = sum_j G[i,j]*pts[j] / a[i]; returns dX, dY on OT grid."""
    eps = a.max() * 1e-6
    T = np.empty_like(grid_pts)
    mask = a > eps
    T[mask]  = (G[mask] @ grid_pts) / a[mask, None]
    T[~mask] = grid_pts[~mask]
    Tx_ot = T[:, 0].reshape(Ny_ot, Nx_ot)
    Ty_ot = T[:, 1].reshape(Ny_ot, Nx_ot)
    return Tx_ot - X_ot, Ty_ot - Y_ot

def _upsample_displacement(dX_ot, dY_ot, Nx, Ny, Nx_ot, Ny_ot):
    """Bilinear upsampling of coarse OT displacement to full resolution in pixel units."""
    scale_up = (Ny / Ny_ot, Nx / Nx_ot)
    dX = zoom(dX_ot, scale_up, order=1) * (Nx - 1)
    dY = zoom(dY_ot, scale_up, order=1) * (Ny - 1)
    return dX, dY

def run_ot_pipeline(density_shirley, density_orig, Nx, Ny, Nx_ot, Ny_ot,
                    M, grid_pts, X_ot, Y_ot, reg=0.005):
    """Full OT pipeline: prepare → downsample → Sinkhorn → barycentric proj → upsample."""
    rho = _prepare_density(density_shirley)
    a, b = _downsample_to_ot_grid(rho, Nx, Ny, Nx_ot, Ny_ot)
    G = _solve_ot(a, b, M, reg=reg)
    dX_ot, dY_ot = _barycentric_projection(G, a, grid_pts, Nx_ot, Ny_ot, X_ot, Y_ot)
    dX, dY = _upsample_displacement(dX_ot, dY_ot, Nx, Ny, Nx_ot, Ny_ot)
    # OT transports source to uniform by construction; output density is analytically known
    ot_density = np.ones((Ny, Nx)) * density_orig.mean()
    return dX, dY, ot_density