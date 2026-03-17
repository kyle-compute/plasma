"""Poisson solver for electrostatic field in cylindrical (r, z) coordinates.

Solves: kappa * epsilon_0 * nabla^2(phi) = -rho

where kappa is the artificial permittivity factor (1.0 = physical),
and nabla^2 in cylindrical coordinates is:

    (1/r) d/dr(r dphi/dr) + d^2phi/dz^2

At r=0 (axis), L'Hopital gives: 2 * d^2phi/dr^2 + d^2phi/dz^2

Boundary conditions:
    - r=0 (axis): Neumann (dphi/dr = 0, by symmetry)
    - r=R (wall): Dirichlet (phi = phi_wall) or Neumann
    - z=0, z=L: Dirichlet or Neumann depending on geometry

The equation is discretized using second-order finite differences on the
node-centered grid and solved as a sparse linear system Ax = b.

References:
    - Birdsall & Langdon, Ch. 6.
    - Taccogna et al. (2023), Eq. 7-9.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from plasma.core.constants import EPSILON_0
from plasma.runtime.cupy_compat import cp, gpu_csr_matrix, gpu_spsolve


class PoissonSolverCylindrical:
    """Finite-difference Poisson solver on a cylindrical (r, z) grid.

    Pre-computes the sparse coefficient matrix on construction so that
    each solve only requires a sparse LU back-substitution.

    Usage:
        solver = PoissonSolverCylindrical(grid, bc_r_max="dirichlet")
        phi = solver.solve(rho, phi_wall=0.0)
        Er, Ez = solver.electric_field(phi)
    """

    def __init__(
        self,
        grid,
        bc_r_max: str = "dirichlet",
        bc_z_min: str = "dirichlet",
        bc_z_max: str = "dirichlet",
        permittivity_factor: float = 1.0,
    ):
        """Initialize solver and build coefficient matrix.

        Args:
            grid: CylindricalGrid instance.
            bc_r_max: "dirichlet" or "neumann" at outer radial wall.
            bc_z_min: "dirichlet" or "neumann" at z=0.
            bc_z_max: "dirichlet" or "neumann" at z=L.
            permittivity_factor: Artificial permittivity kappa (>1 inflates Debye length).
        """
        self.grid = grid
        self.nr = grid.nr
        self.nz = grid.nz
        self.dr = grid.dr
        self.dz = grid.dz
        self.kappa = permittivity_factor

        self.bc_r_max = bc_r_max
        self.bc_z_min = bc_z_min
        self.bc_z_max = bc_z_max

        # Total unknowns = (nr+1) * (nz+1) grid nodes
        self.n_r = grid.n_nodes_r
        self.n_z = grid.n_nodes_z
        self.n_total = self.n_r * self.n_z

        # Build and cache sparse matrix
        self._build_matrix()

    def _node_index(self, i: int, j: int) -> int:
        """Flat index for node (i, j) in row-major order."""
        return i * self.n_z + j

    def _build_matrix(self) -> None:
        """Build the sparse coefficient matrix for the Poisson equation.

        Uses a 5-point stencil in cylindrical coordinates.
        """
        n = self.n_total
        A = lil_matrix((n, n), dtype=np.float64)

        dr = self.dr
        dz = self.dz
        dr2 = dr * dr
        dz2 = dz * dz

        # Interior mask: which nodes are Dirichlet boundary
        self._is_bc = np.zeros(n, dtype=bool)

        for i in range(self.n_r):
            r_i = self.grid.r_edges[i]

            for j in range(self.n_z):
                k = self._node_index(i, j)

                # Check if this is a boundary node
                is_boundary = False

                # Outer radial wall (i = nr)
                if i == self.n_r - 1 and self.bc_r_max == "dirichlet":
                    A[k, k] = 1.0
                    self._is_bc[k] = True
                    is_boundary = True

                # z = 0 boundary
                if j == 0 and not is_boundary and self.bc_z_min == "dirichlet":
                    A[k, k] = 1.0
                    self._is_bc[k] = True
                    is_boundary = True

                # z = L boundary
                if j == self.n_z - 1 and not is_boundary and self.bc_z_max == "dirichlet":
                    A[k, k] = 1.0
                    self._is_bc[k] = True
                    is_boundary = True

                if is_boundary:
                    continue

                # Interior nodes (including axis)
                if i == 0:
                    # Axis (r = 0): use L'Hopital's rule
                    # nabla^2 phi = 2 * d2phi/dr2 + d2phi/dz2
                    # d2phi/dr2 ≈ (phi[1,j] - phi[0,j]) * 2 / dr^2
                    # (using Neumann: phi[-1,j] = phi[1,j])
                    coeff_r = 4.0 / dr2  # Factor 2 from L'Hopital * 2 from symmetry
                    A[k, k] = -(coeff_r + 2.0 / dz2)
                    A[k, self._node_index(1, j)] = coeff_r
                else:
                    # General interior: (1/r)d/dr(r*dphi/dr) + d2phi/dz2
                    r_plus_half = r_i + 0.5 * dr
                    r_minus_half = r_i - 0.5 * dr

                    coeff_plus = r_plus_half / (r_i * dr2)
                    coeff_minus = r_minus_half / (r_i * dr2)
                    coeff_center_r = -(coeff_plus + coeff_minus)

                    A[k, k] = coeff_center_r - 2.0 / dz2
                    A[k, self._node_index(i + 1, j)] = coeff_plus
                    A[k, self._node_index(i - 1, j)] = coeff_minus

                # z second derivative (same for axis and interior)
                if j > 0:
                    A[k, self._node_index(i, j - 1)] = 1.0 / dz2
                else:
                    # Neumann BC at z=0: phi[i,-1] = phi[i,1]
                    A[k, self._node_index(i, 1)] += 1.0 / dz2

                if j < self.n_z - 1:
                    A[k, self._node_index(i, j + 1)] = 1.0 / dz2
                else:
                    # Neumann BC at z=L: phi[i,nz+1] = phi[i,nz-1]
                    A[k, self._node_index(i, self.n_z - 2)] += 1.0 / dz2

        # Convert to CSR for efficient solve
        A_csr = csr_matrix(A)
        self._A_gpu = gpu_csr_matrix(A_csr)

    def solve(
        self,
        rho: cp.ndarray,
        phi_wall_r: float = 0.0,
        phi_wall_z0: float = 0.0,
        phi_wall_zL: float = 0.0,
    ) -> cp.ndarray:
        """Solve Poisson equation for given charge density.

        Args:
            rho: Charge density [C/m^3] on nodes, shape (nr+1, nz+1).
            phi_wall_r: Potential at r=R wall [V].
            phi_wall_z0: Potential at z=0 [V].
            phi_wall_zL: Potential at z=L [V].

        Returns:
            phi: Electrostatic potential [V] on nodes, shape (nr+1, nz+1).
        """
        # Build RHS: b = -rho / (kappa * epsilon_0)
        b = -rho / (self.kappa * EPSILON_0)
        b_flat = b.ravel()

        # Apply Dirichlet BCs
        for i in range(self.n_r):
            for j in range(self.n_z):
                k = self._node_index(i, j)
                if i == self.n_r - 1 and self.bc_r_max == "dirichlet":
                    b_flat[k] = phi_wall_r
                elif j == 0 and self.bc_z_min == "dirichlet":
                    b_flat[k] = phi_wall_z0
                elif j == self.n_z - 1 and self.bc_z_max == "dirichlet":
                    b_flat[k] = phi_wall_zL

        # Solve sparse system
        phi_flat = gpu_spsolve(self._A_gpu, b_flat)

        return phi_flat.reshape(self.n_r, self.n_z)

    def electric_field(self, phi: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute E = -grad(phi) using central differences.

        Args:
            phi: Potential on nodes, shape (nr+1, nz+1).

        Returns:
            (Er, Ez): Electric field components on nodes [V/m].
        """
        Er = cp.zeros_like(phi)
        Ez = cp.zeros_like(phi)

        # Central differences in r (one-sided at boundaries)
        Er[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * self.dr)
        Er[0, :] = -(phi[1, :] - phi[0, :]) / self.dr  # Forward at axis
        Er[-1, :] = -(phi[-1, :] - phi[-2, :]) / self.dr  # Backward at wall

        # Central differences in z
        Ez[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dz)
        Ez[:, 0] = -(phi[:, 1] - phi[:, 0]) / self.dz
        Ez[:, -1] = -(phi[:, -1] - phi[:, -2]) / self.dz

        return Er, Ez
