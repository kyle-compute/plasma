"""Field interpolation from grid nodes to particle positions (CIC gather).

From Taccogna 2023 Eq. 4:
    E_p = sum_g E_g * W(r_g - r_p)

This is the adjoint of the charge deposition operation: we use the same
CIC (bilinear) weighting function W to interpolate fields from the four
surrounding nodes to each particle position.

Consistency between deposit and gather is critical — using the same shape
function ensures momentum conservation (Birdsall & Langdon, Ch. 8).
"""

from __future__ import annotations

from plasma.runtime.cupy_compat import cp
from plasma.runtime.numba_compat import cuda


@cuda.jit(fastmath=True)
def gather_field_kernel(
    field_r, field_z,
    Er_at_p, Ez_at_p,
    r, z, alive,
    dr, dz, nr, nz,
    n_particles,
):
    """Interpolate 2D vector field from grid nodes to particle positions.

    Args:
        field_r, field_z: Field components on nodes, shape (nr+1, nz+1).
        Er_at_p, Ez_at_p: Output field at particles, shape (n,).
        r, z: Particle positions [m].
        alive: Alive flags.
        dr, dz: Grid spacing.
        nr, nz: Cell counts.
        n_particles: Number of particles.
    """
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    # Fractional grid position
    ri = r[idx] / dr
    zi = z[idx] / dz

    # Lower-left node
    i = int(ri)
    j = int(zi)

    # Clamp
    if i < 0:
        i = 0
    if i >= nr:
        i = nr - 1
    if j < 0:
        j = 0
    if j >= nz:
        j = nz - 1

    # CIC weights
    wr = ri - i
    wz = zi - j

    # Bilinear interpolation
    Er_at_p[idx] = (
        field_r[i, j] * (1.0 - wr) * (1.0 - wz)
        + field_r[i + 1, j] * wr * (1.0 - wz)
        + field_r[i, j + 1] * (1.0 - wr) * wz
        + field_r[i + 1, j + 1] * wr * wz
    )

    Ez_at_p[idx] = (
        field_z[i, j] * (1.0 - wr) * (1.0 - wz)
        + field_z[i + 1, j] * wr * (1.0 - wz)
        + field_z[i, j + 1] * (1.0 - wr) * wz
        + field_z[i + 1, j + 1] * wr * wz
    )


def gather_electric_field(
    grid,
    Er_grid: cp.ndarray,
    Ez_grid: cp.ndarray,
    particles,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Interpolate electric field from grid to particle positions.

    Args:
        grid: CylindricalGrid.
        Er_grid, Ez_grid: E-field on nodes, shape (nr+1, nz+1).
        particles: ParticleArray.

    Returns:
        (Er_at_particles, Ez_at_particles): Arrays of shape (count,).
    """
    n = particles.count
    Er_p = cp.zeros(n, dtype=cp.float64)
    Ez_p = cp.zeros(n, dtype=cp.float64)

    if n == 0:
        return Er_p, Ez_p

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    gather_field_kernel[blocks, threads_per_block](
        Er_grid, Ez_grid,
        Er_p, Ez_p,
        particles.r, particles.z, particles.alive,
        grid.dr, grid.dz, grid.nr, grid.nz,
        n,
    )

    return Er_p, Ez_p


def gather_magnetic_field(
    grid,
    Br_grid: cp.ndarray,
    Bz_grid: cp.ndarray,
    particles,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Interpolate magnetic field from grid to particle positions.

    Same CIC interpolation as electric field.
    """
    n = particles.count
    Br_p = cp.zeros(n, dtype=cp.float64)
    Bz_p = cp.zeros(n, dtype=cp.float64)

    if n == 0:
        return Br_p, Bz_p

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    gather_field_kernel[blocks, threads_per_block](
        Br_grid, Bz_grid,
        Br_p, Bz_p,
        particles.r, particles.z, particles.alive,
        grid.dr, grid.dz, grid.nr, grid.nz,
        n,
    )

    return Br_p, Bz_p


@cuda.jit(fastmath=True)
def gather_scalar_kernel(
    field,
    val_at_p,
    r, z, alive,
    dr, dz, nr, nz,
    n_particles,
):
    """Interpolate scalar field to particle positions."""
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    ri = r[idx] / dr
    zi = z[idx] / dz

    i = int(ri)
    j = int(zi)
    if i < 0:
        i = 0
    if i >= nr:
        i = nr - 1
    if j < 0:
        j = 0
    if j >= nz:
        j = nz - 1

    wr = ri - i
    wz = zi - j

    val_at_p[idx] = (
        field[i, j] * (1.0 - wr) * (1.0 - wz)
        + field[i + 1, j] * wr * (1.0 - wz)
        + field[i, j + 1] * (1.0 - wr) * wz
        + field[i + 1, j + 1] * wr * wz
    )


def gather_scalar(grid, field: cp.ndarray, particles) -> cp.ndarray:
    """Interpolate a scalar field (e.g., potential) to particle positions."""
    n = particles.count
    val_p = cp.zeros(n, dtype=cp.float64)

    if n == 0:
        return val_p

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    gather_scalar_kernel[blocks, threads_per_block](
        field, val_p,
        particles.r, particles.z, particles.alive,
        grid.dr, grid.dz, grid.nr, grid.nz,
        n,
    )
    return val_p
