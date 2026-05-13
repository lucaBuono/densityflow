# %%
from gn_helpers import solve_heat_eq
from helpers import *
# %%
# initialize simulated densities
u0_sat, lons_sat, lats_sat = initial_meteosat_density()
print(f"u0_sat.min: {u0_sat.min()}")
print(f"u0_sat.max: {u0_sat.max()}")
print(f"Initial sat. density mean: {u0_sat.mean()}")
fig, ax = plot_density_contour(u0_sat.T, lons_sat, lats_sat,
                               fontsize=12,
                               font_color='black',
                               line_color='black',
                               cmap='Blues')
plt.show()

u0_doublegauss, lons, lats = double_gaussian()
u0_doublegauss = scale_density(u0_doublegauss, target_min=0.0, target_max=4.5)
print(f"u0_doublegauss.min: {u0_doublegauss.min()}")
print(f"u0_doublegauss.max: {u0_doublegauss.max()}")
print(f"Initial doublegauss density mean: {u0_doublegauss.mean()}")

fig, ax = plot_density_contour(u0_doublegauss, lons_sat, lats_sat,
                               fontsize=12,
                               font_color='black',
                               line_color='black',
                               cmap='Blues')
# %%
# apply the shirley mapping to move values to fill empty/zero-regions
# Note: GN and OT would otherwise not move anything to zero-density regions, the regions you want to fill must therefore contain very small, non-zero density values!
u0_sat_shirley = shirley_map_density_to_square(u0_sat)
print(f"u0_sat_shirley.min: {u0_sat_shirley.min()}")
print(f"u0_sat_shirley.max: {u0_sat_shirley.max()}")
print(f"Sat. density mean after Shirley-Chiu mapping: {u0_sat_shirley.mean()}")
plot_density_contour(u0_sat_shirley, lons, lats)

u0_doublegauss_shirley = shirley_map_density_to_square(u0_doublegauss)
print(f"u0_doublegauss_shirley.min: {u0_doublegauss_shirley.min()}")
print(f"u0_doublegauss_shirley.max: {u0_doublegauss_shirley.max()}")
print(f"Doublegauss density mean after Shirley-Chiu mapping: {u0_doublegauss_shirley.mean()}")
plot_density_contour(u0_doublegauss_shirley, lons, lats)
# %%
u0_sat_shirley_blur = spectral_gaussian_blur(u0_sat_shirley, blur_sigma=3.0, floor_eps=1e-3)
print(f"u0_sat_shirley_blur.min: {u0_sat_shirley_blur.min()}")
print(f"u0_sat_shirley_blur.max: {u0_sat_shirley_blur.max()}")
print(f"Sat. density mean after Shirley-Chiu mapping and spectral blur: {u0_sat_shirley_blur.mean()}")
plot_density_contour(u0_sat_shirley_blur, lons, lats)


u0_doublegauss_shirley_blur = spectral_gaussian_blur(u0_doublegauss_shirley, blur_sigma=3.0, floor_eps=1e-3)
print(f"u0_doublegauss_shirley_blur.min: {u0_doublegauss_shirley_blur.min()}")
print(f"u0_doublegauss_shirley_blur.max: {u0_doublegauss_shirley_blur.max()}")
print(f"Doublegauss density mean after Shirley-Chiu mappingand spectral blur: {u0_doublegauss_shirley_blur.mean()}")
plot_density_contour(u0_doublegauss_shirley_blur, lons, lats)

# %%
# now after preparing the density, lets apply the Gastner-Newman algorithm to solve the heat equation for density equalisation
# Nx, Ny = 200, 200
Nx = u0_sat.shape[0]
Ny = u0_sat.shape[1]

u_sat, dX_sat, dY_sat, vx_final_sat, vy_final_sat = solve_heat_eq(u0_sat_shirley_blur, 
                                                                  Nx=Nx, Ny=Ny,
                                                                  eps=1e-6,
                                                                  min_t=1e3, max_t=1e15,
                                                                  max_iter=50000,
                                                                  verbose=True)
print(f"GN-equalized satellite density mean: {u_sat.mean()}")


u_doublegauss, dX_doublegauss, dY_doublegauss, vx_final_doublegauss, vy_final_doublegauss = solve_heat_eq(u0_doublegauss_shirley_blur, 
                                                                                                          Nx=Nx, Ny=Ny,
                                                                                                          eps=1e-6, 
                                                                                                          min_t=1e3, max_t=1e15,
                                                                                                          max_iter=50000, 
                                                                                                          verbose=True)
print(f"GN-equalized doublegauss density mean: {u_doublegauss.mean()}")
# %%
# lets now visualize the displacements fields
streamplot_displacements(lons, lats, dX_sat, dY_sat, title="Satellite density displacements")
streamplot_displacements(lons, lats, dX_doublegauss, dY_doublegauss, title="Doublegauss density displacements")
# %%
# now lets look at the jacobian and the determinant jacobian of the displacements field to determine if folds were created
displacement_jacobian_sat = compute_jacobian(dX_sat, dY_sat)
det_J_sat = displacement_jacobian(dX_sat, dY_sat)
plot_field(det_J_sat, cmap='RdBu_r', colorbar=True, lons=lons, lats=lats, title="Determinant jacobian (satellite)")

displacement_jacobian_doublegauss = compute_jacobian(dX_doublegauss, dY_doublegauss)
det_J_doublegauss = displacement_jacobian(dX_doublegauss, dY_doublegauss)
plot_field(det_J_doublegauss, cmap='RdBu_r', colorbar=True, lons=lons, lats=lats, title="Determinant jacobian (doublegauss)")
# %%
