"""Tests for diagnostics: IEDF, EEDF, spatial profiles, collectors."""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, M_AR, M_ELECTRON
from plasma.diagnostics.collectors import CollisionTracker, SubstrateCollector
from plasma.diagnostics.distributions import (
    compute_eedf,
    compute_iedf,
    compute_velocity_histogram,
)
from plasma.diagnostics.spatial import (
    electron_density_profile,
    electron_temperature_profile,
    potential_snapshot,
)
from plasma.pic.grid import CylindricalGrid
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.weighting import initialize_particles_uniform


@pytest.fixture
def grid():
    return CylindricalGrid(nr=10, nz=20, r_max=0.05, z_max=0.1)


@pytest.fixture
def electron_species():
    return Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)


@pytest.fixture
def ion_species():
    return Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)


def _make_particles(grid, species, n0=1e15, te_ev=3.0, ppc=5):
    data = initialize_particles_uniform(grid, species, n0, te_ev, ppc)
    p = ParticleArray(species=species)
    p.allocate(len(data["r"]) * 2)
    p.add_particles(**data)
    return p


class TestIEDF:
    def test_iedf_known_energy(self, ion_species):
        """Monoenergetic ions at 100 eV should peak near 100 eV."""
        ions = ParticleArray(species=ion_species)
        ions.allocate(500)
        n = 200
        v_100ev = np.sqrt(2.0 * 100.0 * E_CHARGE / M_AR)
        ions.add_particles(
            r=np.full(n, 0.02),
            z=np.full(n, 0.05),
            vr=np.zeros(n),
            vz=np.full(n, v_100ev),
            vtheta=np.zeros(n),
            weight=np.full(n, 1e10),
        )

        energy, counts = compute_iedf(ions, z_plane=0.05, dz_capture=0.01, n_bins=50, e_max_ev=200)
        peak_idx = np.argmax(counts)
        peak_energy = energy[peak_idx]
        assert 80.0 < peak_energy < 120.0

    def test_iedf_empty(self, ion_species):
        ions = ParticleArray(species=ion_species)
        ions.allocate(10)
        energy, counts = compute_iedf(ions, z_plane=0.05, dz_capture=0.01)
        assert np.all(counts == 0)


class TestEEDF:
    def test_eedf_maxwellian_shape(self, grid, electron_species):
        """Maxwellian electrons should produce monotonically decreasing EEDF at high E."""
        electrons = _make_particles(grid, electron_species, n0=1e16, te_ev=3.0, ppc=50)
        energy, f_e = compute_eedf(electrons, n_bins=50, e_max_ev=30.0)

        # EEDF should be positive where there are particles
        assert np.any(f_e > 0)

        # For Maxwellian, f(E) peaks near T_e/2 then decays
        peak_idx = np.argmax(f_e)
        # Above the peak, it should generally decrease
        high_e_fe = f_e[peak_idx + 5:]
        if len(high_e_fe) > 2:
            # Allow some noise but trend should be downward
            assert high_e_fe[-1] < high_e_fe[0] * 10  # Not increasing wildly

    def test_eedf_empty(self, electron_species):
        electrons = ParticleArray(species=electron_species)
        electrons.allocate(10)
        energy, f_e = compute_eedf(electrons)
        assert np.all(f_e == 0)


class TestVelocityHistogram:
    def test_velocity_histogram_shape(self, grid, electron_species):
        electrons = _make_particles(grid, electron_species, n0=1e15, te_ev=3.0)
        v_bins, counts = compute_velocity_histogram(electrons, component="vz")
        assert len(v_bins) == len(counts)
        assert np.any(counts > 0)


class TestSpatialProfiles:
    def test_density_profile_positive(self, grid, electron_species):
        electrons = _make_particles(grid, electron_species, n0=1e16, te_ev=3.0)
        n_e = electron_density_profile(grid, electrons)
        assert n_e.shape == (grid.n_nodes_r, grid.n_nodes_z)
        assert np.all(n_e >= 0)

    def test_temperature_order_of_magnitude(self, grid, electron_species):
        """3 eV input electrons should give ~3 eV temperature."""
        electrons = _make_particles(grid, electron_species, n0=1e16, te_ev=3.0, ppc=50)
        te = electron_temperature_profile(grid, electrons)
        assert te.shape == (grid.n_nodes_r, grid.n_nodes_z)

        # Average temperature in interior (away from boundaries)
        interior = te[2:-2, 2:-2]
        interior_nonzero = interior[interior > 0]
        if len(interior_nonzero) > 0:
            mean_te = np.mean(interior_nonzero)
            assert 0.5 < mean_te < 20.0  # Should be order-of-magnitude correct

    def test_potential_snapshot(self):
        phi_gpu = cp.ones((5, 10), dtype=cp.float64) * 42.0
        phi_cpu = potential_snapshot(phi_gpu)
        assert isinstance(phi_cpu, np.ndarray)
        np.testing.assert_allclose(phi_cpu, 42.0)


class TestSubstrateCollector:
    def test_accumulates(self, ion_species):
        collector = SubstrateCollector(z_plane=0.09, dz_capture=0.01)
        ions = ParticleArray(species=ion_species)
        ions.allocate(100)

        # Add ions near substrate
        n = 10
        ions.add_particles(
            r=np.full(n, 0.02),
            z=np.full(n, 0.089),
            vr=np.zeros(n), vz=np.full(n, 1e4),
            vtheta=np.zeros(n), weight=np.full(n, 1e10),
        )

        c1 = collector.record_absorbed(ions, t=0.0)
        c2 = collector.record_absorbed(ions, t=1e-9)

        assert c1 > 0
        assert collector.total_count == c1 + c2
        assert collector.latest_count() == c2
        assert collector.latest_mean_energy_ev() > 0.0
        assert collector.mean_energy_ev() > 0.0

        energy, counts = collector.iedf(n_bins=50, e_max_ev=100)
        assert np.any(counts > 0)
        radial = collector.radial_flux_profile(np.linspace(0.0, 0.05, 6))
        assert radial.shape == (5,)
        assert np.any(radial > 0.0)

    def test_merges_species_records_with_same_time(self, ion_species):
        collector = SubstrateCollector(z_plane=0.09, dz_capture=0.01)
        speed = np.sqrt(2.0 * 50.0 * E_CHARGE / ion_species.mass)

        count_a = collector.record_particle_data(
            r=np.array([0.01, 0.02]),
            vr=np.zeros(2),
            vz=np.full(2, speed),
            vtheta=np.zeros(2),
            weight=np.ones(2),
            mass_kg=ion_species.mass,
            t=1.0e-9,
            species_name="Ar+",
        )
        count_b = collector.record_particle_data(
            r=np.array([0.03]),
            vr=np.zeros(1),
            vz=np.full(1, speed),
            vtheta=np.zeros(1),
            weight=np.ones(1),
            mass_kg=ion_species.mass,
            t=1.0e-9,
            species_name="Cu+",
        )

        assert collector.total_count == count_a + count_b
        assert collector.latest_count() == count_a + count_b
        assert collector.species_totals()["Ar+"] == count_a
        assert collector.species_totals()["Cu+"] == count_b


class TestCollisionTracker:
    def test_records(self):
        tracker = CollisionTracker()
        tracker.record({"elastic": 10, "ionization": 2})
        tracker.record({"elastic": 5, "ionization": 1, "null": 50})

        totals = tracker.totals()
        assert totals["elastic"] == 15
        assert totals["ionization"] == 3
        assert totals["null"] == 50
        assert tracker.n_steps == 2
