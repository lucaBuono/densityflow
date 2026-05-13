# %%
from gn import solve_heat_eq
from helpers import *
from helpers import _build_cost_matrix
import ot

# %%
# initialize simulated densities
u0_sat, lons_sat, lats_sat = initial_meteosat_density()
print(f"u0_sat.min: {u0_sat.min()}")
print(f"u0_sat.max: {u0_sat.max()}")
print(f"u0_sat.mean: {u0_sat.mean()}")
plot_density_contour(u0_sat.T, lons_sat, lats_sat,
                               fontsize=12,
                               font_color='black',
                               line_color='black',
                               cmap='Blues')
plt.show()

u0_doublegauss, lons, lats = double_gaussian()
u0_doublegauss = scale_density(u0_doublegauss, target_min=0.0, target_max=4.5)
print(f"u0_doublegauss.min: {u0_doublegauss.min()}")
print(f"u0_doublegauss.max: {u0_doublegauss.max()}")
print(f"u0_doublegauss.mean: {u0_doublegauss.mean()}")
plot_density_contour(u0_doublegauss, lons_sat, lats_sat,
                               fontsize=12,
                               font_color='black',
                               line_color='black',
                               cmap='Blues')
# %%
# lets apply the OT pipeline to equalise the densities
Nx, Ny = u0_sat.shape   # 200 x 200
Nx_ot, Ny_ot = 64, 64
sinkhorn_reg = 0.005
# %%
# Build cost matrix once (shared for both densities)
# 64×64 instead of 200×200: exact OT (ot.emd) is O(n³) and Sinkhorn on 40k×40k is still prohibitive; 
# the coarse displacement is bilinearly upsampled afterward.
print("Building OT cost matrix...")
M_ot, grid_pts_ot, X_ot_grid, Y_ot_grid = _build_cost_matrix(Nx_ot, Ny_ot)
# %%
# Shirley mapping: disk FOV → filled square (same first step as SatFile)
# As it's the case with GN, OT won't move anything to zero-regions, 
# therefore Shirley-Chiu mapping and flooring prepares the density to contain only non-zero bins/pixels
shirley_sat   = shirley_map_density_to_square(u0_sat)
print(f"Sat. density after shirley: {np.mean(shirley_sat)}")
plot_density_contour(shirley_sat, lons, lats)
shirley_gauss = shirley_map_density_to_square(u0_doublegauss)
print(f"Doublegauss density after shirley: {np.mean(shirley_gauss)}")
plot_density_contour(shirley_gauss, lons, lats)
# %%
# with the densities prepared, lets run the OT pipeline to equalise the densities
print("\nRunning OT on Meteosat density...")
dX_sat, dY_sat, ot_density_sat = run_ot_pipeline(shirley_sat, u0_sat, Nx, Ny, Nx_ot, Ny_ot,
                                                    M_ot, grid_pts_ot, X_ot_grid, Y_ot_grid, reg=sinkhorn_reg)
print(f"Satellite density mean after OT: {ot_density_sat.mean()}")

print("\nRunning OT on double-Gaussian density...")
dX_doublegauss, dY_doublegauss, ot_density_doublegauss = run_ot_pipeline(shirley_gauss, u0_doublegauss, Nx, Ny, Nx_ot, Ny_ot,
                                                        M_ot, grid_pts_ot, X_ot_grid, Y_ot_grid, reg=sinkhorn_reg)
print(f"Doublegauss density mean after OT: {ot_density_doublegauss.mean()}")
# no need to plot the densities, as they become a constant value (per definition of OT)
# %%
streamplot_displacements(lons, lats, dX_sat, dY_sat, title="Satellite density displacements")
streamplot_displacements(lons, lats, dX_doublegauss, dY_doublegauss, title="Doublegauss density displacements")
# %%
# now lets look at the jacobian and the determinant jacobian of the displacements field to determine if folds were created
displacement_jacobian_sat = compute_jacobian(dX_sat, dY_sat)
det_J_sat = displacement_jacobian(dX_sat, dY_sat)
#plot_field(displacement_jacobian_sat, cmap='RdBu_r', colorbar=True, lons=lons, lats=lats, title="Jacobian (satellite)")
plot_field(det_J_sat, cmap='RdBu_r', colorbar=True, lons=lons, lats=lats, title="Determinant jacobian (satellite)")

displacement_jacobian_doublegauss = compute_jacobian(dX_doublegauss, dY_doublegauss)
det_J_doublegauss = displacement_jacobian(dX_doublegauss, dY_doublegauss)
#plot_field(displacement_jacobian_doublegauss, cmap='RdBu_r', colorbar=True, lons=lons, lats=lats, title="Jacobian (doublegauss)")
plot_field(det_J_doublegauss, cmap='RdBu_r', colorbar=True, lons=lons, lats=lats, title="Determinant jacobian (doublegauss)")
# %%
# looking at the values of the jacobian determinants, we expect no negative values to validate that the warp is fold-free
print(f"sat:   min={det_J_sat.min():.4f}, max={det_J_sat.max():.4f}, mean={det_J_sat.mean():.4f}, folds={(det_J_sat<=0).sum()}")
print(f"gauss: min={det_J_doublegauss.min():.4f}, max={det_J_doublegauss.max():.4f}, mean={det_J_doublegauss.mean():.4f}, folds={(det_J_doublegauss<=0).sum()}")
# %%
