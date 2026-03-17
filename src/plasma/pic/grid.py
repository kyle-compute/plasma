"""Cylindrical (r, z) grid for 2D axisymmetric PIC simulations.

The grid stores field quantities at cell centers and edges.
Cell volumes account for the cylindrical geometry: V_cell = 2*pi*r*dr*dz.
The axis at r=0 requires special treatment to avoid singularities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plasma.core.constants import E_CHARGE, EPSILON_0, M_ELECTRON


@dataclass
class CylindricalGrid:
    """2D axisymmetric grid in (r, z) coordinates.

    Grid layout (cell-centered fields, edge-defined potential):

        z_edge[0]  z_edge[1]  z_edge[2]   ...  z_edge[nz]
          |           |           |                |
          |  (0,0)    |  (0,1)    |                |
    r=0 --+-----------+-----------+--- ... --------+
          |  (1,0)    |  (1,1)    |                |
          +-----------+-----------+--- ... --------+
          |           |           |                |
         ...
          +-----------+-----------+--- ... --------+
          |  (nr-1,0) |           |                |
    r=R --+-----------+-----------+--- ... --------+

    Attributes:
        nr: Number of cells in radial direction.
        nz: Number of cells in axial direction.
        r_max: Outer radius [m].
        z_max: Axial length [m].
        dr: Radial cell size [m].
        dz: Axial cell size [m].
    """

    nr: int
    nz: int
    r_max: float
    z_max: float

    def __post_init__(self) -> None:
        self.dr = self.r_max / self.nr
        self.dz = self.z_max / self.nz

        # Cell-center coordinates
        self.r_centers = np.linspace(
            0.5 * self.dr, self.r_max - 0.5 * self.dr, self.nr
        )
        self.z_centers = np.linspace(
            0.5 * self.dz, self.z_max - 0.5 * self.dz, self.nz
        )

        # Cell-edge (node) coordinates — for potential / field solve
        self.r_edges = np.linspace(0.0, self.r_max, self.nr + 1)
        self.z_edges = np.linspace(0.0, self.z_max, self.nz + 1)

        # Cell volumes: V_i = 2*pi*r_i*dr*dz (annular ring)
        # At the axis, use the exact volume of a disc: pi*dr^2/4*dz * 2pi
        # For cell i: V = pi*(r_edge[i+1]^2 - r_edge[i]^2) * dz
        r_lo = self.r_edges[:-1]
        r_hi = self.r_edges[1:]
        self.cell_volumes = np.pi * (r_hi**2 - r_lo**2) * self.dz  # (nr,)

        # Inverse volumes for charge deposition (avoid divide-by-zero at r=0)
        self.inv_cell_volumes = np.where(
            self.cell_volumes > 0, 1.0 / self.cell_volumes, 0.0
        )

    @property
    def n_nodes_r(self) -> int:
        """Number of grid nodes in r (nr + 1)."""
        return self.nr + 1

    @property
    def n_nodes_z(self) -> int:
        """Number of grid nodes in z (nz + 1)."""
        return self.nz + 1

    @property
    def n_nodes(self) -> int:
        """Total number of grid nodes."""
        return self.n_nodes_r * self.n_nodes_z

    @property
    def n_cells(self) -> int:
        return self.nr * self.nz

    def node_volumes(self) -> NDArray[np.float64]:
        """Compute volumes associated with each grid node (for node-centered quantities).

        Uses the dual-mesh approach: each node owns the volume from the
        midpoints of its neighboring cells.  Result is cached after first call.

        Returns:
            Array of shape (nr+1, nz+1) with node volumes [m^3].
        """
        if hasattr(self, "_node_volumes_cache"):
            return self._node_volumes_cache

        vol = np.zeros((self.n_nodes_r, self.n_nodes_z))

        # Vectorised radial bounds
        r_lo = np.empty(self.n_nodes_r)
        r_hi = np.empty(self.n_nodes_r)
        r_lo[0] = 0.0
        r_hi[0] = 0.5 * self.dr
        r_lo[self.nr] = self.r_max - 0.5 * self.dr
        r_hi[self.nr] = self.r_max
        interior = slice(1, self.nr)
        r_lo[interior] = self.r_edges[interior] - 0.5 * self.dr
        r_hi[interior] = self.r_edges[interior] + 0.5 * self.dr

        ring_area = np.pi * (r_hi**2 - r_lo**2)  # (n_nodes_r,)

        # Axial extents
        dz_nodes = np.full(self.n_nodes_z, self.dz)
        dz_nodes[0] = 0.5 * self.dz
        dz_nodes[self.nz] = 0.5 * self.dz

        vol = ring_area[:, None] * dz_nodes[None, :]
        self._node_volumes_cache = vol
        return vol

    def node_volumes_gpu(self):
        """Return node volumes as a cached GPU (CuPy) array."""
        from plasma.runtime.cupy_compat import cp

        if not hasattr(self, "_node_volumes_gpu_cache"):
            self._node_volumes_gpu_cache = cp.asarray(self.node_volumes())
        return self._node_volumes_gpu_cache

    def debye_length(self, n_e: float, te_ev: float) -> float:
        """Electron Debye length [m].

        lambda_De = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))
        """
        if n_e <= 0 or te_ev <= 0:
            return float("inf")
        te_j = te_ev * E_CHARGE
        return float(np.sqrt(EPSILON_0 * te_j / (n_e * E_CHARGE**2)))

    def plasma_frequency(self, n_e: float) -> float:
        """Electron plasma frequency [rad/s].

        omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
        """
        if n_e <= 0:
            return 0.0
        return float(np.sqrt(n_e * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON)))

    def check_constraints(
        self,
        n_e: float,
        te_ev: float,
        dt: float,
        zeta: float = 1.0,
    ) -> dict[str, bool]:
        """Check PIC numerical constraints.

        Args:
            n_e: Electron density [m^-3].
            te_ev: Electron temperature [eV].
            dt: Timestep [s].
            zeta: Debye length resolution factor (dr <= zeta * lambda_De).

        Returns:
            Dict of constraint name → satisfied (True/False).
        """
        lam_de = self.debye_length(n_e, te_ev)
        omega_pe = self.plasma_frequency(n_e)

        # Thermal velocity of electrons
        v_th_e = np.sqrt(E_CHARGE * te_ev / M_ELECTRON)

        results = {}

        # Debye length resolution
        results["debye_r"] = self.dr <= zeta * lam_de
        results["debye_z"] = self.dz <= zeta * lam_de

        # CFL: v_e * dt <= min(dr, dz)
        dx_min = min(self.dr, self.dz)
        results["cfl"] = v_th_e * dt <= dx_min

        # Plasma frequency: omega_pe * dt < 2 (stability for leapfrog)
        # Practical limit: omega_pe * dt < 0.2 for accuracy
        results["plasma_freq"] = omega_pe * dt < 0.2

        return results

    def r_to_index(self, r: float) -> float:
        """Convert radial position to fractional grid index."""
        return r / self.dr

    def z_to_index(self, z: float) -> float:
        """Convert axial position to fractional grid index."""
        return z / self.dz

    def contains(self, r: float, z: float) -> bool:
        """Check if position (r, z) is inside the grid."""
        return 0.0 <= r <= self.r_max and 0.0 <= z <= self.z_max
