"""Charge deposition onto cylindrical (r, z) grid using CIC weighting.

Cloud-in-Cell (CIC) = bilinear interpolation of particle charge onto the
four surrounding grid nodes. In cylindrical coordinates, each node's
contribution is weighted by the volume overlap.

From Taccogna 2023 Eq. 6:
    rho_g = (1/V_g) * sum_p(q_p * w_p * W(r_g - r_p))

where W is the CIC shape function (tent function in each dimension).

For axisymmetric geometry, charge density rho(r,z) is deposited on a
node-centered grid. The density must account for the 2*pi*r factor:
particles closer to the axis represent a smaller physical ring of charge.
"""

from __future__ import annotations

from plasma.runtime.cupy_compat import cp
from plasma.runtime.numba_compat import cuda


@cuda.jit
def deposit_charge_kernel(
    rho,
    r, z, weight, alive, charge,
    dr, dz, nr, nz,
    n_particles,
):
    """Deposit particle charge onto node-centered grid using CIC.

    Args:
        rho: Charge density array [C/m^3], shape (nr+1, nz+1). Modified in-place.
        r, z: Particle positions [m].
        weight: Macro-particle weights.
        alive: Alive flags.
        charge: Species charge [C] (scalar).
        dr, dz: Grid spacing [m].
        nr, nz: Number of cells.
        n_particles: Total particles.
    """
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    # Particle grid coordinates (fractional index)
    ri = r[idx] / dr
    zi = z[idx] / dz

    # Lower-left node indices
    i = int(ri)
    j = int(zi)

    # Clamp to valid range
    if i < 0:
        i = 0
    if i >= nr:
        i = nr - 1
    if j < 0:
        j = 0
    if j >= nz:
        j = nz - 1

    # CIC weights (fractional distances to lower-left node)
    wr = ri - i
    wz = zi - j

    # Charge contribution = q * w_macro
    q_contribution = charge * weight[idx]

    # Bilinear distribution to 4 surrounding nodes
    # Using atomic adds for thread safety
    cuda.atomic.add(rho, (i, j), q_contribution * (1.0 - wr) * (1.0 - wz))
    cuda.atomic.add(rho, (i + 1, j), q_contribution * wr * (1.0 - wz))
    cuda.atomic.add(rho, (i, j + 1), q_contribution * (1.0 - wr) * wz)
    cuda.atomic.add(rho, (i + 1, j + 1), q_contribution * wr * wz)


def deposit_charge(
    grid,
    particles_list: list,
) -> cp.ndarray:
    """Deposit charge from all species onto the grid.

    Args:
        grid: CylindricalGrid instance.
        particles_list: List of ParticleArray instances.

    Returns:
        rho: Charge density [C/m^3] on nodes, shape (nr+1, nz+1).
    """
    nr, nz = grid.nr, grid.nz

    # Raw charge accumulator (not yet divided by volume)
    rho_raw = cp.zeros((nr + 1, nz + 1), dtype=cp.float64)

    threads_per_block = 256

    for particles in particles_list:
        n = particles.count
        if n == 0:
            continue
        blocks = (n + threads_per_block - 1) // threads_per_block
        deposit_charge_kernel[blocks, threads_per_block](
            rho_raw,
            particles.r, particles.z,
            particles.weight, particles.alive,
            particles.species.charge,
            grid.dr, grid.dz, nr, nz,
            n,
        )

    # Convert from accumulated charge to charge density
    # rho [C/m^3] = raw_charge / node_volume
    node_vol = cp.asarray(grid.node_volumes())
    # Avoid division by zero at nodes with zero volume
    node_vol_safe = cp.maximum(node_vol, 1e-30)
    rho = rho_raw / node_vol_safe

    return rho


def deposit_number_density(
    grid,
    particles,
) -> cp.ndarray:
    """Deposit number density [m^-3] for a single species."""
    nr, nz = grid.nr, grid.nz
    n_raw = cp.zeros((nr + 1, nz + 1), dtype=cp.float64)

    n_part = particles.count
    if n_part == 0:
        return n_raw

    threads_per_block = 256
    blocks = (n_part + threads_per_block - 1) // threads_per_block

    # Deposit with unit charge to get weighted particle count
    deposit_charge_kernel[blocks, threads_per_block](
        n_raw,
        particles.r, particles.z,
        particles.weight, particles.alive,
        1.0,  # Unit charge → gives sum of weights
        grid.dr, grid.dz, nr, nz,
        n_part,
    )

    node_vol = cp.asarray(grid.node_volumes())
    node_vol_safe = cp.maximum(node_vol, 1e-30)
    return n_raw / node_vol_safe
