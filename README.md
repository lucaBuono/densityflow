# densityflow

Simulated pipelines for **density equalisation** via two methods: the Gastner–Newman (GN) diffusion algorithm and discrete Optimal Transport (OT) with Sinkhorn regularisation. Developed as part of PhD research on observation density redistribution for satellite data (Meteosat FOV).

## Problem

Given a non-uniform 2D observation density (e.g. a circular satellite field-of-view with a peaked distribution), compute a displacement field `(dX, dY)` that transports the density to a spatially uniform one. The displacement field must be **fold-free**: `det(J) > 0` everywhere, where `J = I + ∇d` is the Jacobian of the map `T(x) = x + d(x)`.

## Methods

### Gastner–Newman (`gn.py`)
Solves the heat equation on the density field using DCT/DST spectral methods. Particle positions are advected by the velocity field `v = -∇u / u` with an adaptive midpoint integrator until the density flattens. Returns the final uniform density, displacement field, and velocity field.

### Optimal Transport (`ot_simulated_pipeline.py`)
Solves a regularised OT problem (Sinkhorn) on a coarsened `64×64` grid, then bilinearly upsamples the displacement to the full `200×200` resolution. The output density is analytically `mean(u0)` everywhere (mass-preserving by construction).

## Preprocessing

Both pipelines require the source density to be strictly positive over the full domain — GN and OT will not move mass into zero regions. Two preprocessing steps handle this:

1. **Shirley–Chiu mapping** (`shirley_map_density_to_square`): remaps a circular FOV to a filled square, eliminating zero corners.
2. **Spectral Gaussian blur** (`spectral_gaussian_blur`): floors near-zero values to `floor_eps` so no bin is exactly zero (GN pipeline only).

## Repo structure

| File | Purpose |
|---|---|
| `gn.py` | GN heat-equation solver |
| `helpers.py` | Density generators, preprocessing, OT internals, Jacobian utilities, plotting |
| `gn_simulated_pipeline.py` | End-to-end GN demo on two test densities |
| `ot_simulated_pipeline.py` | End-to-end OT demo on two test densities |
| `requirements.txt` | Python dependencies |

## Diagnostics

The Jacobian determinant is the primary quality metric:

```python
from helpers import displacement_jacobian
det_J = displacement_jacobian(dX, dY)
print(f"min={det_J.min():.4f}, max={det_J.max():.4f}, folds={(det_J<=0).sum()}")
```

- `det(J) > 0` everywhere: fold-free
- `det(J) ≈ 1` on average: mass-preserving (spatial mean ~1 is expected; ~0.8 in practice due to boundary truncation and discrete approximation)
- `det(J) ≤ 0`: fold — the map is locally invalid

## Dependencies

```
scipy numpy matplotlib mpl-tools pot  # pot = Python Optimal Transport
```
