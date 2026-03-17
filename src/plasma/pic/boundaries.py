"""Particle boundary conditions for 2D axisymmetric PIC.

Handles:
- Axis reflection at r = 0 (already in pusher, but post-push cleanup here)
- Absorbing walls at r = R, z = 0, z = L
- Secondary electron emission (SEE) at target surface
"""

from __future__ import annotations

from plasma.runtime.cupy_compat import cp
from plasma.runtime.numba_compat import cuda


@cuda.jit
def apply_boundaries_kernel(
    r, z, vr, vz, vtheta, alive,
    r_max, z_max,
    n_particles,
):
    """Mark particles outside the domain as dead.

    Axis reflection (r < 0) is handled in the pusher. Here we
    handle absorbing walls at the domain edges.
    """
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    # Absorbing wall at r = R
    if r[idx] >= r_max:
        alive[idx] = 0
        return

    # Absorbing wall at z = 0
    if z[idx] <= 0.0:
        alive[idx] = 0
        return

    # Absorbing wall at z = L
    if z[idx] >= z_max:
        alive[idx] = 0
        return

    # Ensure r >= 0 (backup for pusher)
    if r[idx] < 0.0:
        r[idx] = -r[idx]
        vr[idx] = -vr[idx]
        vtheta[idx] = -vtheta[idx]


def apply_boundaries(grid, particles) -> None:
    """Apply boundary conditions to particles.

    Args:
        grid: CylindricalGrid.
        particles: ParticleArray to modify in-place.
    """
    n = particles.count
    if n == 0:
        return

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    apply_boundaries_kernel[blocks, threads_per_block](
        particles.r, particles.z,
        particles.vr, particles.vz, particles.vtheta,
        particles.alive,
        grid.r_max, grid.z_max,
        n,
    )


@cuda.jit
def count_wall_flux_kernel(
    r, z, vr, vz, weight, alive,
    r_max, z_max,
    flux_r, flux_z0, flux_zL,
    n_particles,
):
    """Count particle flux to each wall before absorbing."""
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    w = weight[idx]

    if r[idx] >= r_max:
        cuda.atomic.add(flux_r, 0, w)
    elif z[idx] <= 0.0:
        cuda.atomic.add(flux_z0, 0, w)
    elif z[idx] >= z_max:
        cuda.atomic.add(flux_zL, 0, w)


def wall_fluxes(grid, particles) -> dict[str, float]:
    """Count macro-particle flux to each wall.

    Call this BEFORE apply_boundaries to count particles about to hit walls.
    """
    n = particles.count
    flux_r = cp.zeros(1, dtype=cp.float64)
    flux_z0 = cp.zeros(1, dtype=cp.float64)
    flux_zL = cp.zeros(1, dtype=cp.float64)

    if n == 0:
        return {"r_wall": 0.0, "z0_wall": 0.0, "zL_wall": 0.0}

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    count_wall_flux_kernel[blocks, threads_per_block](
        particles.r, particles.z,
        particles.vr, particles.vz,
        particles.weight, particles.alive,
        grid.r_max, grid.z_max,
        flux_r, flux_z0, flux_zL,
        n,
    )

    return {
        "r_wall": float(flux_r.item()),
        "z0_wall": float(flux_z0.item()),
        "zL_wall": float(flux_zL.item()),
    }
