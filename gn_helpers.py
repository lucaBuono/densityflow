import numpy as np
from scipy.fftpack import dct, idct, dst, idst

def solve_heat_eq(initial_density, Nx, Ny, eps=1e-10,
                        min_t=1e3, max_t=1e12, max_iter=10000, verbose=True):
    """       
    Adaptive time stepping version matching the C code's diff_integrate
    Using DCT and DST instead of FFT.

    Parameters
    ----------
    initial_density : np.ndarray (Nx, Ny)
        Input density field (must be positive everywhere; apply blur + floor first).
    Nx, Ny : int
        Grid dimensions (number of cells in x/lon and y/lat direction).
    alpha : float
        Diffusion coefficient scaling the heat decay rate. Higher values
        diffuse faster per unit time (rarely needs tuning; default 1.0).
    eps : float
        Small floor added to density in the velocity computation v = -∇u/u
        to avoid division by zero in near-zero regions.
    min_t : float
        Minimum integration time before convergence is checked. The solver
        runs at least until t >= min_t regardless of max_change.
    max_t : float
        Hard upper limit on integration time. The solver stops at t >= max_t
        even if convergence has not been reached.
    max_iter : int
        Maximum number of outer time steps before the loop is force-stopped.
    verbose : bool
        Print iteration progress (t, delta_t, max_change) every 10 steps.
    """
    u0 = initial_density

    u_hat0 = dct(dct(u0, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')

    kx = np.pi * np.arange(Nx) / Nx
    ky = np.pi * np.arange(Ny) / Ny
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2

    ABS_TOL = min(Nx, Ny) * 1e-6
    INC_AFTER_ACC = 1.1
    DEC_AFTER_NOT_ACC = 0.75
    CONV_MAX_CHANGE = min(Nx, Ny) * 1e-9
    MIN_DELTA_T = 1e-14

    x_init = np.arange(0.5, Nx)
    y_init = np.arange(0.5, Ny)
    X_init, Y_init = np.meshgrid(x_init, y_init, indexing='ij')
    pos_x = X_init.copy()
    pos_y = Y_init.copy()

    t = 0.0
    delta_t = 1e-2
    iter_count = 0

    def compute_velocity(t_val):
        u_hat = u_hat0 * np.exp(-K2 * t_val)

        u = idct(idct(u_hat, axis=0, type=2, norm='ortho'),
                 axis=1, type=2, norm='ortho')

        kx_shifted = KX[1:, :]
        vx_hat = -kx_shifted * u_hat[1:, :]
        vx_hat = np.pad(vx_hat, ((0, 1), (0, 0)), mode='constant')

        ky_shifted = KY[:, 1:]
        vy_hat = -ky_shifted * u_hat[:, 1:]
        vy_hat = np.pad(vy_hat, ((0, 0), (0, 1)), mode='constant')

        ux = idst(idct(vx_hat, axis=1, type=2, norm='ortho'),
                  axis=0, type=2, norm='ortho')
        uy = idct(idst(vy_hat, axis=1, type=2, norm='ortho'),
                  axis=0, type=2, norm='ortho')

        vx = -ux / (u + eps)
        vy = -uy / (u + eps)

        vx[0, :] = 0.0;  vx[-1, :] = 0.0
        vy[:, 0] = 0.0;  vy[:, -1] = 0.0

        return vx, vy, u

    def interpolate_velocity(vx_grid, vy_grid, px, py):
        px = np.clip(px, 0, Nx - 1.001)
        py = np.clip(py, 0, Ny - 1.001)
        ix = np.floor(px).astype(int)
        iy = np.floor(py).astype(int)
        fx = px - ix
        fy = py - iy
        ix1, iy1 = ix + 1, iy + 1
        v = lambda g: (g[ix, iy] * (1-fx)*(1-fy) + g[ix1, iy] * fx*(1-fy) +
                       g[ix, iy1] * (1-fx)*fy   + g[ix1, iy1] * fx*fy)
        return v(vx_grid), v(vy_grid)

    while iter_count < max_iter:
        vx_grid, vy_grid, _ = compute_velocity(t)
        vx_interp, vy_interp = interpolate_velocity(vx_grid, vy_grid, pos_x, pos_y)

        accept = False
        while not accept:
            pos_x_euler = pos_x + vx_interp * delta_t
            pos_y_euler = pos_y + vy_interp * delta_t

            pos_x_mid_check = pos_x + 0.5 * delta_t * vx_interp
            pos_y_mid_check = pos_y + 0.5 * delta_t * vy_interp

            if (np.any(pos_x_mid_check < 0) or np.any(pos_x_mid_check > Nx) or
                    np.any(pos_y_mid_check < 0) or np.any(pos_y_mid_check > Ny)):
                delta_t *= DEC_AFTER_NOT_ACC
                if delta_t < MIN_DELTA_T:
                    if verbose:
                        print(f"WARNING: delta_t collapsed (out-of-bounds) at iter {iter_count}. "
                              f"Forcing acceptance.")
                    pos_x_mid = np.clip(pos_x_euler, 0.0, Nx)
                    pos_y_mid = np.clip(pos_y_euler, 0.0, Ny)
                    delta_t = MIN_DELTA_T
                    accept = True
                continue

            vx_grid_mid, vy_grid_mid, _ = compute_velocity(t + 0.5 * delta_t)
            vx_interp_mid, vy_interp_mid = interpolate_velocity(
                vx_grid_mid, vy_grid_mid, pos_x_mid_check, pos_y_mid_check)

            pos_x_mid = pos_x + vx_interp_mid * delta_t
            pos_y_mid = pos_y + vy_interp_mid * delta_t

            error_sq = (pos_x_mid - pos_x_euler)**2 + (pos_y_mid - pos_y_euler)**2

            if np.any(error_sq > ABS_TOL):
                delta_t *= DEC_AFTER_NOT_ACC
                if delta_t < MIN_DELTA_T:
                    if verbose:
                        print(f"WARNING: delta_t collapsed at iter {iter_count}. "
                              f"Forcing acceptance.")
                    delta_t = MIN_DELTA_T
                    accept = True
            else:
                accept = True

        max_change = np.max((pos_x_mid - pos_x)**2 + (pos_y_mid - pos_y)**2)
        pos_x = pos_x_mid
        pos_y = pos_y_mid
        t += delta_t
        iter_count += 1
        delta_t *= INC_AFTER_ACC

        if verbose and iter_count % 10 == 0:
            print(f"iter = {iter_count}, t = {t:.3e}, delta_t = {delta_t:.3e}, "
                  f"max_change = {max_change:.3e}")

        if t >= min_t and (max_change <= CONV_MAX_CHANGE or t >= max_t):
            break

    vx_final, vy_final, u_final = compute_velocity(t)
    dX = pos_x - X_init
    dY = pos_y - Y_init

    print(f"Final: t = {t:.3e}, iterations = {iter_count}")
    return u_final, dX, dY, vx_final, vy_final