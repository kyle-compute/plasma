"""Macro-particle weight management for PIC simulations.

In cylindrical geometry, particles near the axis represent smaller physical
volumes (2*pi*r*dr*dz) than those near the outer wall. The macro-particle
weight w_p determines how many physical particles each macro-particle represents.

This module provides:
- Weight initialization based on target density and grid cell volume
- Weight splitting (split heavy particles to reduce noise near axis)
- Weight merging (merge light particles far from axis to save memory)
"""

from __future__ import annotations

import cupy as cp
import numpy as np


def compute_weight(
    n_target: float,
    volume: float,
    n_macroparticles: int,
) -> float:
    """Compute macro-particle weight for given target density.

    w = n_target * volume / n_macroparticles

    Args:
        n_target: Target number density [m^-3].
        volume: Volume of region to fill [m^3].
        n_macroparticles: Number of macro-particles.

    Returns:
        Weight (number of physical particles per macro-particle).
    """
    if n_macroparticles <= 0:
        return 0.0
    return n_target * volume / n_macroparticles


def compute_cell_weight(
    grid,
    n_target: float,
    ppc: int,
) -> cp.ndarray:
    """Compute per-cell macro-particle weight.

    In cylindrical geometry, cells near the axis have smaller volume,
    so each macro-particle there represents fewer physical particles.

    Args:
        grid: CylindricalGrid.
        n_target: Target number density [m^-3].
        ppc: Particles per cell.

    Returns:
        weights: Array of shape (nr,) — one weight per radial cell.
    """
    return cp.asarray(n_target * grid.cell_volumes / ppc)


def initialize_particles_uniform(
    grid,
    species,
    n_density: float,
    temperature_ev: float,
    ppc: int = 50,
) -> dict[str, cp.ndarray]:
    """Generate uniformly distributed particles with Maxwellian velocities.

    Args:
        grid: CylindricalGrid.
        species: Species definition (has mass, charge attributes).
        n_density: Number density [m^-3].
        temperature_ev: Temperature [eV].
        ppc: Macro-particles per cell.

    Returns:
        Dict with keys: r, z, vr, vz, vtheta, weight.
    """
    from plasma.core.constants import E_CHARGE

    nr, nz = grid.nr, grid.nz
    n_total = nr * nz * ppc

    # Thermal speed: v_th = sqrt(kT/m) where kT = T_eV * e
    v_th = float(np.sqrt(E_CHARGE * temperature_ev / species.mass))

    # Allocate arrays
    r_arr = np.empty(n_total, dtype=np.float64)
    z_arr = np.empty(n_total, dtype=np.float64)
    vr_arr = np.empty(n_total, dtype=np.float64)
    vz_arr = np.empty(n_total, dtype=np.float64)
    vt_arr = np.empty(n_total, dtype=np.float64)
    w_arr = np.empty(n_total, dtype=np.float64)

    rng = np.random.default_rng(42)
    idx = 0

    for i in range(nr):
        r_lo = grid.r_edges[i]
        r_hi = grid.r_edges[i + 1]
        # Weight per macro-particle in this radial ring:
        w = n_density * grid.cell_volumes[i] / ppc

        for j in range(nz):
            z_lo = grid.z_edges[j]
            z_hi = grid.z_edges[j + 1]

            for _ in range(ppc):
                # Random position in cell (uniform in r^2 for cylindrical)
                r_rand = np.sqrt(rng.uniform(r_lo**2, r_hi**2))
                z_rand = rng.uniform(z_lo, z_hi)

                r_arr[idx] = r_rand
                z_arr[idx] = z_rand
                vr_arr[idx] = rng.normal(0.0, v_th)
                vz_arr[idx] = rng.normal(0.0, v_th)
                vt_arr[idx] = rng.normal(0.0, v_th)
                w_arr[idx] = w
                idx += 1

    return {
        "r": cp.asarray(r_arr[:idx]),
        "z": cp.asarray(z_arr[:idx]),
        "vr": cp.asarray(vr_arr[:idx]),
        "vz": cp.asarray(vz_arr[:idx]),
        "vtheta": cp.asarray(vt_arr[:idx]),
        "weight": cp.asarray(w_arr[:idx]),
    }
