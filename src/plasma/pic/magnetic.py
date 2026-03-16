"""Static magnetic field for magnetron sputtering geometry.

Provides:
- Analytical magnetic field from circular current loops (Biot-Savart)
- Loading pre-computed field maps from .npz files
- Uniform field for benchmarks

In HiPIMS, the magnetron's permanent magnets create a static B-field
that traps electrons near the target surface. The field is predominantly
radial (Br) near the target and axial (Bz) further away.

Typical race-track field strength: B_rt ~ 30-100 mT (0.03-0.1 T).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipe, ellipk


def uniform_field(
    grid,
    Br: float = 0.0,
    Bz: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Create uniform magnetic field on grid nodes.

    Useful for benchmarks (e.g., cyclotron frequency test).

    Returns:
        (Br_grid, Bz_grid): Arrays of shape (nr+1, nz+1).
    """
    Br_grid = np.full((grid.n_nodes_r, grid.n_nodes_z), Br)
    Bz_grid = np.full((grid.n_nodes_r, grid.n_nodes_z), Bz)
    return Br_grid, Bz_grid


def current_loop_field(
    r_obs: float,
    z_obs: float,
    r_loop: float,
    z_loop: float,
    current: float,
) -> tuple[float, float]:
    """Magnetic field from a single circular current loop.

    Uses elliptic integral formulation (Jackson, Ch. 5).

    Args:
        r_obs, z_obs: Observation point [m].
        r_loop: Loop radius [m].
        z_loop: Loop axial position [m].
        current: Loop current [A] (positive = counterclockwise when viewed from +z).

    Returns:
        (Br, Bz) at the observation point [T].
    """
    mu0 = 4.0e-7 * np.pi
    dz = z_obs - z_loop
    r = max(r_obs, 1e-12)  # Avoid singularity at r=0

    alpha2 = r_loop**2 + r**2 + dz**2 - 2.0 * r_loop * r
    beta2 = r_loop**2 + r**2 + dz**2 + 2.0 * r_loop * r
    beta = np.sqrt(beta2)

    k2 = 1.0 - alpha2 / beta2
    k2 = min(k2, 1.0 - 1e-15)  # Clamp for numerical safety

    K = float(ellipk(k2))
    E = float(ellipe(k2))

    coeff = mu0 * current / (2.0 * np.pi)

    # Br component
    Br = (coeff * dz / (r * beta)) * (
        -K + E * (r_loop**2 + r**2 + dz**2) / alpha2
    )

    # Bz component
    Bz = (coeff / beta) * (
        K - E * (r_loop**2 - r**2 - dz**2) / alpha2
    )

    return float(Br), float(Bz)


def magnetron_field(
    grid,
    inner_loop_r: float,
    outer_loop_r: float,
    loop_z: float = -0.005,
    current_inner: float = 1000.0,
    current_outer: float = -600.0,
) -> tuple[NDArray, NDArray]:
    """Compute magnetic field from simplified magnetron geometry.

    Models the magnetron as two concentric current loops representing
    the inner and outer magnet poles. The field is computed at every
    grid node using the elliptic integral formulation.

    Args:
        grid: CylindricalGrid.
        inner_loop_r: Inner magnet pole radius [m].
        outer_loop_r: Outer magnet pole radius [m].
        loop_z: Axial position of magnet assembly [m] (negative = behind target).
        current_inner: Equivalent current for inner pole [A].
        current_outer: Equivalent current for outer pole [A] (opposite sign).

    Returns:
        (Br_grid, Bz_grid): Field on nodes, shape (nr+1, nz+1).
    """
    nr1, nz1 = grid.n_nodes_r, grid.n_nodes_z
    Br_grid = np.zeros((nr1, nz1))
    Bz_grid = np.zeros((nr1, nz1))

    for i in range(nr1):
        r = grid.r_edges[i]
        for j in range(nz1):
            z = grid.z_edges[j]

            # Inner pole contribution
            br1, bz1 = current_loop_field(r, z, inner_loop_r, loop_z, current_inner)
            # Outer pole contribution
            br2, bz2 = current_loop_field(r, z, outer_loop_r, loop_z, current_outer)

            Br_grid[i, j] = br1 + br2
            Bz_grid[i, j] = bz1 + bz2

    return Br_grid, Bz_grid


def load_field(path: str | Path) -> tuple[NDArray, NDArray]:
    """Load pre-computed magnetic field from .npz file.

    Expected keys: 'Br', 'Bz' with shape (nr+1, nz+1).
    """
    data = np.load(path)
    return data["Br"], data["Bz"]


def save_field(
    path: str | Path,
    Br: NDArray,
    Bz: NDArray,
    r_edges: NDArray | None = None,
    z_edges: NDArray | None = None,
) -> None:
    """Save magnetic field to .npz file."""
    arrays = {"Br": Br, "Bz": Bz}
    if r_edges is not None:
        arrays["r_edges"] = r_edges
    if z_edges is not None:
        arrays["z_edges"] = z_edges
    np.savez(path, **arrays)
