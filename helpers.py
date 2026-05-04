import numpy as np
import ot
from scipy.fftpack import dct, idct, dst, idst
from scipy.ndimage import map_coordinates, zoom

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# HELPERS
def initial_meteosat_density():
    resolution = 0.5
    lon_min, lon_max = -50, 50
    lat_min, lat_max = -50, 50

    # Calculate number of grid cells
    nx = int(abs(lon_max - lon_min) / resolution)
    ny = int(abs(lat_max - lat_min) / resolution)

    # Create coordinate arrays (cell centers)
    lons = np.linspace(lon_min + resolution/2, lon_max - resolution/2, nx)
    lats = np.linspace(lat_min + resolution/2, lat_max - resolution/2, ny)

    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')

    # Calculate distance from center (0, 0) for each grid cell
    distance = np.sqrt(lon_grid**2 + lat_grid**2)

    # Meteosat FOV radius (approximately 80 degrees from nadir)
    fov_radius = 60

    # Density model: peaks at 6 in center, decreases to 0 at edge
    max_density = 4.5
    density = np.zeros_like(distance)

    # Only within the disc FOV
    mask = distance <= fov_radius

    # Cosine-based falloff (smoother than linear)
    normalized_dist = distance[mask] / fov_radius
    density[mask] = max_density * np.cos(normalized_dist * np.pi / 2)**2

    # Ensure no zeros for cartogram algorithm
    min_density = 0.0
    density[density == 0] = min_density
    return density, lons, lats

def double_gaussian() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two Gaussian bumps on a uniform background, matching Meteosat grid style."""
    resolution = 0.5
    lon_min, lon_max = -50, 50
    lat_min, lat_max = -50, 50

    nx = int(abs(lon_max - lon_min) / resolution)  # 200
    ny = int(abs(lat_max - lat_min) / resolution)  # 200

    lons = np.linspace(lon_min + resolution / 2, lon_max - resolution / 2, nx)
    lats = np.linspace(lat_min + resolution / 2, lat_max - resolution / 2, ny)

    xx, yy = np.meshgrid(lons, lats, indexing='ij')

    # Gaussian centers at (-20, 0) and (20, 0), sigma^2 scaled to new domain
    sigma2 = 500.0  # 0.05 * 100^2, preserves relative width from unit domain
    density = (0.0
               + 3.0 * np.exp(-((xx + 20)**2 + yy**2) / sigma2)
               + 3.0 * np.exp(-((xx - 20)**2 + yy**2) / sigma2))

    return density, lons, lats

def scale_density(
    density: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 4.0,
) -> np.ndarray:
    """Min-max scale density to [target_min, target_max]."""
    d_min, d_max = density.min(), density.max()
    return target_min + (density - d_min) / (d_max - d_min) * (target_max - target_min)

def shirley_square_to_disk(u, v):
    """
    Shirley-Chiu concentric square-to-disk mapping.
    u, v in [-1, 1] -> x, y in unit disk.
    Vectorized for NumPy arrays.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    x = np.zeros_like(u)
    y = np.zeros_like(v)

    mask = (u != 0) | (v != 0)
    um = u[mask]
    vm = v[mask]

    a = um
    b = vm
    abs_a = np.abs(a)
    abs_b = np.abs(b)

    r = np.empty_like(a)
    phi = np.empty_like(a)

    cond = abs_a > abs_b
    r[cond] = a[cond]
    phi[cond] = (np.pi / 4.0) * (b[cond] / a[cond])

    r[~cond] = b[~cond]
    phi[~cond] = (np.pi / 2.0) - (np.pi / 4.0) * (a[~cond] / b[~cond])

    x[mask] = r * np.cos(phi)
    y[mask] = r * np.sin(phi)

    return x, y

def shirley_map_density_to_square(density):
    """
    Map a circular (disk) density field inside a square grid to a full square
    using the Shirley-Chiu mapping. Returns a new density of same shape.
    """
    density = np.asarray(density, dtype=float)
    nx, ny = density.shape

    # Target square grid in [-1, 1]
    u = np.linspace(-1.0, 1.0, nx, endpoint=False) + (1.0 / nx)
    v = np.linspace(-1.0, 1.0, ny, endpoint=False) + (1.0 / ny)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    # Map square -> disk
    xd, yd = shirley_square_to_disk(uu, vv)

    # Disk coords -> source grid indices
    ix = (xd + 1.0) * 0.5 * (nx - 1)
    iy = (yd + 1.0) * 0.5 * (ny - 1)

    # Sample original density
    mapped = map_coordinates(density, [ix, iy], order=1, mode="constant", cval=0.0)
    return mapped

def density_from_displacement(u0, dX, dY):
    # u0: initial density of shape (x,y)
    # dX: displacement in x-direction of shape (x,y) (from gn_c)
    # dY: displacement in y-direction of shape (x,y) (from gn_c)

    # apply x and y displacement to u0
    # Calculate the actual final density using the Jacobian determinant
    # This transforms the original density based on area changes from displacement
    dx_dlon = np.gradient(dX, axis=0)
    dx_dlat = np.gradient(dX, axis=1)
    dy_dlon = np.gradient(dY, axis=0)
    dy_dlat = np.gradient(dY, axis=1)

    jacobian = (1 + dx_dlon) * (1 + dy_dlat) - dx_dlat * dy_dlon
    u_final = u0 / np.abs(jacobian) + 1e-6
    u_final[jacobian <= 0] = 0
    return u_final

def spectral_gaussian_blur(density, blur_sigma=3.0, floor_eps=1e-3):
    """
    Spectral Gaussian blur + positive floor, consistent with gn_codex / gn_py.

    Applies the blur in DCT space (Neumann BCs, matching the solver) then clamps
    any non-positive values to floor_eps.  Returns the processed spatial density,
    ready to pass straight into apply_heat_eq_adaptive_dct_dst or apply_heat_eq_claude.

    Parameters
    ----------
    density   : 2-D array, raw input density
    blur_sigma: Gaussian sigma in the same units as the grid index (pixels).
                Use 0 or None to skip blurring.
    floor_eps : replace values <= 0 with this after blurring.
                Use 0 or None to skip the floor.
    """
    rho = np.asarray(density, dtype=float)

    if blur_sigma:
        lx, ly = rho.shape
        rho_ft = dct(dct(rho, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')
        i_arr = np.arange(lx, dtype=float)[:, None]
        j_arr = np.arange(ly, dtype=float)[None, :]
        gauss = np.exp(-0.5 * blur_sigma**2 * np.pi**2 *
                       ((i_arr / lx)**2 + (j_arr / ly)**2))
        rho_ft *= gauss
        rho = idct(idct(rho_ft, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')

    if floor_eps:
        rho = np.where(rho <= 0, float(floor_eps), rho)

    return rho

def compute_jacobian(dX, dY):
    J = ((1 + np.gradient(dX, axis=0)) * (1 + np.gradient(dY, axis=1))
         - np.gradient(dX, axis=1) * np.gradient(dY, axis=0))
    return J

def displacement_jacobian(dispX, dispY, dx=1.0, dy=1.0):
    """
    Compute the Jacobian determinant of a 2D displacement field.

    The full transformation is T(x,y) = (x + u, y + v), so the
    Jacobian matrix is:
        J = | 1 + du/dx   du/dy |
            | dv/dx       1 + dv/dy |

    Parameters
    ----------
    dispX : np.ndarray (H, W)
        Displacement in x/lon direction.
    dispY : np.ndarray (H, W)
        Displacement in y/lat direction.
    dx : float or np.ndarray
        Grid spacing in x direction (scalar or 1D array of length W).
    dy : float or np.ndarray
        Grid spacing in y direction (scalar or 1D array of length H).

    Returns
    -------
    det_J : np.ndarray (H, W)
        Jacobian determinant at each grid point.
        det_J > 0 everywhere: fold-free, orientation-preserving.
        det_J = 0: degenerate (collision).
        det_J < 0: fold (orientation flip).
    """
    # Partial derivatives of u
    du_dy, du_dx = np.gradient(dispX, dy, dx)
    # Partial derivatives of v
    dv_dy, dv_dx = np.gradient(dispY, dy, dx)

    # det(J) = (1 + du/dx)(1 + dv/dy) - (du/dy)(dv/dx)
    det_J = (1.0 + du_dx) * (1.0 + dv_dy) - (du_dy * dv_dx)

    return det_J

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


# PLOTTING
def plot_density_contour(
    density: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    label_levels: list[float] = [1, 2, 3, 4],
    fov_radius: float = 75.0,
    cmap: str = "Blues",
    line_color = 'black',
    font_color ='black',
    fontsize = 12,
    figsize: tuple = (7, 6.5),
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    Z = density.T  # (nx, ny) → (ny, nx) for matplotlib
    max_density = float(density.max())

    def _isoline_radius(level):
        # Clip to [0,1] to guard against floating-point overshoot
        arg = np.clip(level / max_density, 0.0, 1.0)
        return fov_radius * (2 / np.pi) * np.arccos(np.sqrt(arg))

    # Clamp label positions to stay safely inside the data domain
    lon_margin = (lons[-1] - lons[0]) * 0.05
    lat_margin = (lats[-1] - lats[0]) * 0.05
    lon_max_safe = lons[-1] - lon_margin
    lat_max_safe = lats[-1] - lat_margin

    def _label_pos(level):
        r = _isoline_radius(level) / np.sqrt(2)
        return (min(r, lon_max_safe), min(r, lat_max_safe))

    manual_locations = [_label_pos(lv) for lv in label_levels]

    fig, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(lons, lats, Z,
                     levels=np.linspace(0, max_density, 256),
                     cmap=cmap, vmin=0, vmax=max_density)

    cs = ax.contour(lons, lats, Z,
                    levels=label_levels,
                    colors=line_color, linewidths=0.9,
                    linestyles="dashed", alpha=0.85)

    clabels = ax.clabel(cs, levels=label_levels,
                        inline=True, inline_spacing=4,
                        fontsize=fontsize, fmt="%g",
                        manual=False)
    seen = set()
    for lbl in clabels:
        txt = lbl.get_text()
        if txt in seen:
            lbl.set_visible(False)
        else:
            seen.add(txt)
            lbl.set_color(line_color)
            lbl.set_fontweight("bold")
            lbl.set_backgroundcolor((1, 1, 1, 0.0))

    ax.add_patch(Circle((0, 0), fov_radius, fill=False,
                         linestyle="--", linewidth=0.9,
                         edgecolor="white", alpha=0.5, zorder=3))

    #cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    #cbar.set_label("Observations per grid cell", fontsize=10)
    #cbar.set_ticks(label_levels)
    #cbar.ax.tick_params(labelsize=9)

    ax.set_xlim(lons[0], lons[-1])
    ax.set_ylim(lats[0], lats[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, color="white", linewidth=0.35, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig, ax

def streamplot_displacements(X, Y, dX, dY, title: str = None, cmap: str = "Blues", colorbar: bool = True):
    magnitude = np.sqrt(dX**2 + dY**2)

    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    fig, ax = plt.subplots(figsize=(8, 8 * y_range / x_range))

    im = ax.pcolormesh(X, Y, magnitude, cmap=cmap, shading='auto')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g°'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g°'))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    ax.streamplot(X, Y, dX.T, dY.T,
                  color='white', linewidth=0.8,
                  density=1.5, arrowsize=1.2)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect('equal')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label='Displacement magnitude')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_field(
    data: np.ndarray,
    lons: np.ndarray = None,
    lats: np.ndarray = None,
    title: str = None,
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    colorbar: bool = True,
    histogram: bool = False,
    bins: int = 50,
    figsize: tuple = (6, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D array with optional colorbar and histogram, all at matching height.

    Parameters
    ----------
    data      : 2D array to display with imshow
    lons      : 1D longitude array — when provided (with lats), axes show degree labels
    lats      : 1D latitude array
    title     : axes title
    cmap      : colormap name
    vmin/vmax : color scale limits (None = data min/max)
    colorbar  : attach a colorbar to the right of the image
    histogram : attach a value histogram to the right
    bins      : number of histogram bins
    figsize   : (width, height) of the figure
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=figsize)

    finite = data.ravel()
    finite = finite[np.isfinite(finite)]
    if vmin is None and finite.size > 0:
        vmin = float(np.percentile(finite, 2))
    if vmax is None and finite.size > 0:
        vmax = float(np.percentile(finite, 98))

    use_geo = lons is not None and lats is not None
    if use_geo:
        extent = [lons[0], lons[-1], lats[0], lats[-1]]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect='equal')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g°'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g°'))
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)

    if colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax)

    if histogram:
        hax = divider.append_axes("right", size="55%", pad=0.15 if not colorbar else 0.8)
        if finite.size > 0 and np.unique(finite).size > 1:
            hax.hist(finite, bins=bins, color='steelblue', edgecolor='none')
            hax.yaxis.set_label_position("right")
            hax.yaxis.tick_right()
            hax.set_xlabel("Value")
            hax.set_ylabel("Count")
        else:
            msg = "no finite data" if finite.size == 0 else f"constant\n{finite[0]:.4g}"
            hax.text(0.5, 0.5, msg, ha='center', va='center', transform=hax.transAxes)
            hax.set_axis_off()

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    return fig, ax