"""Poisson solver validation using the Method of Manufactured Solutions (MMS).

We choose a known analytical potential phi(r, z), compute the corresponding
charge density rho = -epsilon_0 * nabla^2(phi), feed rho to the solver,
and verify the computed phi matches the analytical solution.

For cylindrical coordinates:
    nabla^2(phi) = (1/r) d/dr(r dphi/dr) + d^2phi/dz^2

A good test function that satisfies Dirichlet BCs (phi=0 at walls):
    phi(r, z) = sin(pi*r/R) * sin(pi*z/L) * r

This automatically satisfies phi(r=0)=0 (axis) and can be evaluated for
the exact Laplacian.
"""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import EPSILON_0
from plasma.pic.grid import CylindricalGrid
from plasma.pic.poisson import PoissonSolverCylindrical


class TestPoissonMMS:
    """Method of Manufactured Solutions for the Poisson solver."""

    def test_uniform_charge_density(self):
        """Solve nabla^2(phi) = const with Dirichlet BCs.

        For a uniform charge density between grounded walls, the solution
        should be parabolic.
        """
        grid = CylindricalGrid(nr=20, nz=20, r_max=0.05, z_max=0.1)
        solver = PoissonSolverCylindrical(
            grid,
            bc_r_max="dirichlet",
            bc_z_min="dirichlet",
            bc_z_max="dirichlet",
        )

        # Uniform charge density
        n_e = 1e16  # m^-3
        rho = cp.full((grid.n_nodes_r, grid.n_nodes_z), -n_e * 1.6e-19)

        phi = solver.solve(rho, phi_wall_r=0.0, phi_wall_z0=0.0, phi_wall_zL=0.0)
        phi_np = cp.asnumpy(phi)

        # Basic checks
        assert phi_np.shape == (grid.n_nodes_r, grid.n_nodes_z)
        assert np.all(np.isfinite(phi_np))

        # With negative rho (electrons), phi should be non-zero in interior
        # Sign depends on boundary conditions and geometry
        assert abs(phi_np[grid.nr // 2, grid.nz // 2]) > 0

        # BCs: phi should be ~0 at walls
        assert phi_np[-1, :] == pytest.approx(0.0, abs=1e-3)
        assert phi_np[:, 0] == pytest.approx(0.0, abs=1e-3)
        assert phi_np[:, -1] == pytest.approx(0.0, abs=1e-3)

    def test_quadratic_manufactured_solution(self):
        """Manufactured solution: phi(r,z) = A * (R^2 - r^2) * z * (L - z).

        This satisfies phi=0 at r=R, z=0, z=L (Dirichlet).
        At r=0 (axis), dphi/dr = 0 by symmetry (the r^2 term).

        The Laplacian in cylindrical coords:
        nabla^2(phi) = (1/r) d/dr(r * d/dr[A(R^2 - r^2)z(L-z)])
                     + d^2/dz^2[A(R^2 - r^2)z(L-z)]

        dphi/dr = A * (-2r) * z(L-z)
        r * dphi/dr = -2Ar^2 * z(L-z)
        d/dr(r dphi/dr) = -4Ar * z(L-z)
        (1/r)*d/dr(r dphi/dr) = -4A * z(L-z)

        d^2phi/dz^2 = A * (R^2 - r^2) * (-2)

        So: nabla^2(phi) = -4A*z*(L-z) - 2A*(R^2 - r^2)
        And: rho = -epsilon_0 * nabla^2(phi) = epsilon_0 * [4A*z*(L-z) + 2A*(R^2-r^2)]
        """
        R = 0.05
        L = 0.1
        A = 1e6  # Scale factor for non-trivial potential

        grid = CylindricalGrid(nr=30, nz=30, r_max=R, z_max=L)
        solver = PoissonSolverCylindrical(
            grid,
            bc_r_max="dirichlet",
            bc_z_min="dirichlet",
            bc_z_max="dirichlet",
        )

        # Compute exact rho at each node
        rho_np = np.zeros((grid.n_nodes_r, grid.n_nodes_z))
        phi_exact_np = np.zeros((grid.n_nodes_r, grid.n_nodes_z))

        for i in range(grid.n_nodes_r):
            r = grid.r_edges[i]
            for j in range(grid.n_nodes_z):
                z = grid.z_edges[j]
                phi_exact_np[i, j] = A * (R**2 - r**2) * z * (L - z)
                laplacian = -4.0 * A * z * (L - z) - 2.0 * A * (R**2 - r**2)
                rho_np[i, j] = -EPSILON_0 * laplacian

        rho = cp.asarray(rho_np)
        phi_computed = solver.solve(rho, phi_wall_r=0.0, phi_wall_z0=0.0, phi_wall_zL=0.0)
        phi_comp_np = cp.asnumpy(phi_computed)

        # Compare interior points (exclude boundaries)
        interior = phi_exact_np[1:-1, 1:-1]
        computed = phi_comp_np[1:-1, 1:-1]

        # Relative error should be small (limited by discretization ~ h^2)
        # With 30 cells and h ~ 2mm, expect ~1-5% error
        rel_error = np.abs(computed - interior) / (np.abs(interior) + 1e-30)
        max_rel_error = np.max(rel_error[interior != 0])

        assert max_rel_error < 0.1, f"Max relative error {max_rel_error:.4f} > 10%"

    def test_electric_field_from_linear_potential(self):
        """If phi varies linearly, E should be constant."""
        grid = CylindricalGrid(nr=10, nz=10, r_max=0.05, z_max=0.1)
        solver = PoissonSolverCylindrical(grid)

        # Linear potential in z: phi = V0 * (1 - z/L)
        V0 = 100.0
        phi = cp.zeros((grid.n_nodes_r, grid.n_nodes_z), dtype=cp.float64)
        for j in range(grid.n_nodes_z):
            z = grid.z_edges[j]
            phi[:, j] = V0 * (1.0 - z / grid.z_max)

        Er, Ez = solver.electric_field(phi)
        Er_np = cp.asnumpy(Er)
        Ez_np = cp.asnumpy(Ez)

        # Er should be ~0 everywhere
        assert np.max(np.abs(Er_np)) < 1e-6

        # Ez should be ~V0/L everywhere (constant)
        expected_Ez = V0 / grid.z_max
        # Check interior (boundaries have one-sided differences)
        assert Ez_np[5, 5] == pytest.approx(expected_Ez, rel=0.01)


class TestPoissonGrid:
    """Grid-related tests for the Poisson solver."""

    def test_solver_builds_for_small_grid(self):
        """Solver should initialize without error for small grids."""
        grid = CylindricalGrid(nr=5, nz=5, r_max=0.01, z_max=0.02)
        solver = PoissonSolverCylindrical(grid)
        assert solver.n_total == 36  # 6 * 6 nodes

    def test_zero_rho_gives_zero_phi(self):
        """Zero charge density with grounded walls should give phi=0."""
        grid = CylindricalGrid(nr=10, nz=10, r_max=0.05, z_max=0.1)
        solver = PoissonSolverCylindrical(grid)

        rho = cp.zeros((grid.n_nodes_r, grid.n_nodes_z), dtype=cp.float64)
        phi = solver.solve(rho)
        phi_np = cp.asnumpy(phi)

        assert np.max(np.abs(phi_np)) < 1e-10
