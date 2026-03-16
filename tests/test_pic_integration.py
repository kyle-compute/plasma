"""Integration tests for the full PIC loop.

These tests verify that all PIC components work together correctly:
- Charge deposition
- Poisson solve
- Field gather
- Particle push
- Boundary conditions
"""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, M_AR, M_ELECTRON
from plasma.pic.deposit import deposit_charge, deposit_number_density
from plasma.pic.grid import CylindricalGrid
from plasma.pic.loop import pic_step, run_pic
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.poisson import PoissonSolverCylindrical
from plasma.pic.weighting import initialize_particles_uniform


@pytest.fixture
def small_grid():
    return CylindricalGrid(nr=10, nz=20, r_max=0.02, z_max=0.04)


@pytest.fixture
def electron():
    return Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)


@pytest.fixture
def ion():
    return Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)


class TestChargeDeposition:
    def test_single_particle_deposits_nonzero(self, small_grid, electron):
        """A single particle should deposit non-zero charge density."""
        p = ParticleArray(species=electron)
        p.allocate(1)
        p.add_particles(
            r=np.array([0.01]),
            z=np.array([0.02]),
            vr=np.array([0.0]),
            vz=np.array([0.0]),
            vtheta=np.array([0.0]),
            weight=np.array([1e10]),  # Large weight for visible density
        )

        rho = deposit_charge(small_grid, [p])
        rho_np = cp.asnumpy(rho)

        # Charge density should be non-zero somewhere
        assert np.any(rho_np != 0)
        # Electron charge is negative
        assert np.min(rho_np) < 0

    def test_quasineutral_gives_zero_rho(self, small_grid, electron, ion):
        """Equal electron and ion densities should give ~zero net charge."""
        n0 = 1e16
        ppc = 10

        e_data = initialize_particles_uniform(
            small_grid, electron, n0, temperature_ev=3.0, ppc=ppc,
        )
        i_data = initialize_particles_uniform(
            small_grid, ion, n0, temperature_ev=0.1, ppc=ppc,
        )

        electrons = ParticleArray(species=electron)
        electrons.allocate(len(e_data["r"]))
        electrons.add_particles(**e_data)

        ions = ParticleArray(species=ion)
        ions.allocate(len(i_data["r"]))
        ions.add_particles(**i_data)

        rho = deposit_charge(small_grid, [electrons, ions])
        rho_np = cp.asnumpy(rho)

        # Net charge should be nearly zero (not exactly due to random positions)
        mean_rho = np.mean(np.abs(rho_np))
        max_rho = n0 * E_CHARGE  # Scale for comparison
        assert mean_rho / max_rho < 0.3  # Within 30% of neutral

    def test_number_density_positive(self, small_grid, electron):
        """Number density should be non-negative."""
        p = ParticleArray(species=electron)
        p.allocate(100)
        n_data = initialize_particles_uniform(
            small_grid, electron, 1e16, temperature_ev=3.0, ppc=5,
        )
        p.add_particles(**n_data)

        n_e = deposit_number_density(small_grid, p)
        n_np = cp.asnumpy(n_e)
        assert np.all(n_np >= 0)


class TestFullPICStep:
    def test_pic_step_runs(self, small_grid, electron, ion):
        """A single PIC step should execute without error."""
        solver = PoissonSolverCylindrical(small_grid)
        n0 = 1e15
        ppc = 5

        e_data = initialize_particles_uniform(
            small_grid, electron, n0, 3.0, ppc,
        )
        i_data = initialize_particles_uniform(
            small_grid, ion, n0, 0.1, ppc,
        )

        electrons = ParticleArray(species=electron)
        electrons.allocate(len(e_data["r"]) * 2)
        electrons.add_particles(**e_data)

        ions = ParticleArray(species=ion)
        ions.allocate(len(i_data["r"]) * 2)
        ions.add_particles(**i_data)

        dt = 1e-11

        phi, stats = pic_step(
            small_grid,
            [electrons, ions],
            solver,
            dt,
        )

        phi_np = cp.asnumpy(phi)
        assert isinstance(stats, dict)
        assert phi_np.shape == (small_grid.n_nodes_r, small_grid.n_nodes_z)
        assert np.all(np.isfinite(phi_np))

    def test_pic_loop_conserves_particles_approx(self, small_grid, electron, ion):
        """Over a few steps, particle count should decrease only from wall losses."""
        solver = PoissonSolverCylindrical(small_grid)
        n0 = 1e15
        ppc = 5

        e_data = initialize_particles_uniform(
            small_grid, electron, n0, 1.0, ppc,
        )
        i_data = initialize_particles_uniform(
            small_grid, ion, n0, 0.05, ppc,
        )

        electrons = ParticleArray(species=electron)
        electrons.allocate(len(e_data["r"]) * 2)
        electrons.add_particles(**e_data)

        ions = ParticleArray(species=ion)
        ions.allocate(len(i_data["r"]) * 2)
        ions.add_particles(**i_data)

        n_i_before = ions.n_alive

        dt = 1e-12  # Very small dt → few wall losses
        n_steps = 5

        diag = run_pic(
            small_grid,
            [electrons, ions],
            solver,
            dt,
            n_steps=n_steps,
            diag_interval=1,
        )

        # Ions should barely move (heavy, low T)
        assert ions.n_alive >= n_i_before * 0.9

        # Diagnostics should have entries
        assert len(diag.time) > 0
        assert len(diag.field_energy) > 0


class TestGridConstraints:
    def test_debye_length(self):
        grid = CylindricalGrid(nr=100, nz=200, r_max=0.05, z_max=0.1)
        lam = grid.debye_length(n_e=1e16, te_ev=3.0)
        # For n_e=1e16, T_e=3eV: lambda_De ~ 1.3e-4 m
        assert 1e-5 < lam < 1e-3

    def test_plasma_frequency(self):
        grid = CylindricalGrid(nr=10, nz=10, r_max=0.05, z_max=0.1)
        omega = grid.plasma_frequency(n_e=1e16)
        # omega_pe = sqrt(n_e * e^2 / (eps0 * me)) ~ 5.6e9 rad/s for 1e16
        assert 1e9 < omega < 1e10

    def test_check_constraints_reports(self):
        grid = CylindricalGrid(nr=100, nz=200, r_max=0.05, z_max=0.1)
        result = grid.check_constraints(n_e=1e16, te_ev=3.0, dt=1e-11)
        assert isinstance(result, dict)
        assert "debye_r" in result
        assert "cfl" in result
        assert "plasma_freq" in result
