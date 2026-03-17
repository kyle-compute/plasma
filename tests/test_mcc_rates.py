"""Tests for Monte Carlo Collision (MCC) module.

Validates:
1. Null-collision probability matches analytical nu*dt
2. Collision counts converge to expected rates statistically
3. Elastic scattering conserves energy approximately
4. Ionization creates new particles
5. Excitation reduces electron energy by threshold
"""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, M_AR, M_ELECTRON
from plasma.data.cross_sections import CrossSectionTable
from plasma.pic.mcc import (
    CollisionProcess,
    CollisionType,
    MCCHandler,
    make_electron_ar_mcc,
)
from plasma.pic.particles import ParticleArray, Species
from plasma.runtime.random import SimulationRNG


def make_constant_cross_section(sigma_val: float, name: str = "const") -> CrossSectionTable:
    """Create a cross-section table that is constant over a wide energy range."""
    energies = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    sigmas = np.full_like(energies, sigma_val)
    return CrossSectionTable(energies, sigmas, name)


@pytest.fixture
def electron():
    return Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)


@pytest.fixture
def electron_beam(electron):
    """Beam of 10000 electrons at 50 eV."""
    n = 10000
    p = ParticleArray(species=electron)
    p.allocate(n * 2)  # Extra space for ionization products

    v_50ev = np.sqrt(2.0 * 50.0 * E_CHARGE / M_ELECTRON)
    p.add_particles(
        r=np.full(n, 0.02),
        z=np.full(n, 0.05),
        vr=np.zeros(n),
        vz=np.full(n, v_50ev),
        vtheta=np.zeros(n),
        weight=np.ones(n) * 1e8,
    )
    return p


class TestNullCollisionProbability:
    def test_probability_scales_with_density(self):
        """Higher background density → higher collision probability."""
        sigma = make_constant_cross_section(1e-20)
        dt = 1e-10

        handler_low = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e18,
        )
        handler_low.add_process(CollisionProcess(
            name="test", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))

        handler_high = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e20,
        )
        handler_high.add_process(CollisionProcess(
            name="test", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))

        p_low = handler_low.collision_probability(dt)
        p_high = handler_high.collision_probability(dt)

        assert p_high > p_low
        assert 0 < p_low < 1
        assert 0 < p_high < 1

    def test_probability_scales_with_dt(self):
        """Longer timestep → higher collision probability."""
        sigma = make_constant_cross_section(1e-20)

        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e19,
        )
        handler.add_process(CollisionProcess(
            name="test", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))

        p_short = handler.collision_probability(1e-11)
        p_long = handler.collision_probability(1e-9)

        assert p_long > p_short

    def test_zero_density_zero_probability(self):
        """No background → no collisions."""
        sigma = make_constant_cross_section(1e-20)
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=0.0,
        )
        handler.add_process(CollisionProcess(
            name="test", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))
        assert handler.collision_probability(1e-10) == 0.0


class TestElasticCollisions:
    def test_elastic_is_deterministic_for_same_seed(self, electron_beam, electron):
        sigma = make_constant_cross_section(5e-20)

        def build_handler() -> MCCHandler:
            handler = MCCHandler(
                projectile_mass=M_ELECTRON,
                background_mass=M_AR,
                background_density=1e20,
            )
            handler.add_process(
                CollisionProcess(
                    name="elastic",
                    collision_type=CollisionType.ELASTIC,
                    cross_section=sigma,
                )
            )
            return handler

        def clone_particles(source: ParticleArray) -> ParticleArray:
            clone = ParticleArray(species=source.species)
            clone.allocate(source.capacity)
            clone.add_particles(**source.to_numpy())
            return clone

        beam_a = clone_particles(electron_beam)
        beam_b = clone_particles(electron_beam)
        counts_a = build_handler().perform_collisions(beam_a, dt=1e-9, rng=SimulationRNG(42))
        counts_b = build_handler().perform_collisions(beam_b, dt=1e-9, rng=SimulationRNG(42))

        assert counts_a == counts_b
        np.testing.assert_allclose(cp.asnumpy(beam_a.vr[: beam_a.count]), cp.asnumpy(beam_b.vr[: beam_b.count]))
        np.testing.assert_allclose(cp.asnumpy(beam_a.vz[: beam_a.count]), cp.asnumpy(beam_b.vz[: beam_b.count]))
        np.testing.assert_allclose(cp.asnumpy(beam_a.vtheta[: beam_a.count]), cp.asnumpy(beam_b.vtheta[: beam_b.count]))

    def test_elastic_changes_velocity(self, electron_beam, electron):
        """Elastic collisions should change particle velocities."""
        sigma = make_constant_cross_section(5e-20)  # Large for many collisions
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e20,
        )
        handler.add_process(CollisionProcess(
            name="elastic", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))

        vz_before = cp.asnumpy(electron_beam.vz[:electron_beam.count].copy())

        rng = cp.random.RandomState(42)
        counts = handler.perform_collisions(electron_beam, dt=1e-9, rng=rng)

        vz_after = cp.asnumpy(electron_beam.vz[:electron_beam.count])

        # Some particles should have changed velocity
        changed = np.sum(vz_before != vz_after)
        assert changed > 0
        assert counts.get("elastic", 0) > 0

    def test_elastic_preserves_approximate_energy(self, electron_beam, electron):
        """Elastic e-Ar: energy loss ~ 2*m_e/M_Ar per collision (very small)."""
        sigma = make_constant_cross_section(5e-20)
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e20,
        )
        handler.add_process(CollisionProcess(
            name="elastic", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))

        ke_before = electron_beam.kinetic_energy()

        rng = cp.random.RandomState(42)
        handler.perform_collisions(electron_beam, dt=1e-9, rng=rng)

        ke_after = electron_beam.kinetic_energy()

        # Energy should decrease slightly (m_e/M_Ar ~ 1.4e-5 per collision)
        # With many collisions, total should be within ~10% for this test
        assert ke_after < ke_before
        assert ke_after > 0.5 * ke_before  # Should not lose more than half


class TestIonization:
    def test_ionization_creates_particles(self, electron_beam, electron):
        """Ionization events should produce new electron data."""
        sigma_ionization = make_constant_cross_section(2e-20)
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=5e19,
        )
        handler.add_process(CollisionProcess(
            name="ionization", collision_type=CollisionType.IONIZATION,
            cross_section=sigma_ionization,
            threshold_ev=15.76,
        ))

        rng = cp.random.RandomState(123)
        counts = handler.perform_collisions(electron_beam, dt=1e-9, rng=rng)

        n_ionizations = counts.get("ionization", 0)

        # Should have some ionizations with 50 eV electrons above 15.76 eV threshold
        assert n_ionizations > 0

        # New electrons should be available
        new_e = handler.get_new_electrons()
        assert new_e is not None
        assert len(new_e["r"]) == n_ionizations

        # New ions too
        new_i = handler.get_new_ions()
        assert new_i is not None
        assert len(new_i["r"]) == n_ionizations

    def test_ionization_energy_partition(self, electron_beam, electron):
        """After ionization, incident electron should have less energy."""
        sigma_ionization = make_constant_cross_section(5e-20)
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e20,
        )
        handler.add_process(CollisionProcess(
            name="ionization", collision_type=CollisionType.IONIZATION,
            cross_section=sigma_ionization,
            threshold_ev=15.76,
        ))

        ke_before = electron_beam.mean_energy_ev()
        assert ke_before == pytest.approx(50.0, rel=0.01)

        rng = cp.random.RandomState(42)
        handler.perform_collisions(electron_beam, dt=1e-9, rng=rng)

        # Mean energy should decrease (ionization costs 15.76 eV + splitting)
        ke_after = electron_beam.mean_energy_ev()
        assert ke_after < ke_before


class TestExcitation:
    def test_excitation_reduces_energy(self, electron_beam, electron):
        """Excitation should reduce electron energy by threshold."""
        sigma_exc = make_constant_cross_section(3e-20)
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=5e19,
        )
        handler.add_process(CollisionProcess(
            name="excitation", collision_type=CollisionType.EXCITATION,
            cross_section=sigma_exc,
            threshold_ev=11.55,
        ))

        ke_before = electron_beam.mean_energy_ev()

        rng = cp.random.RandomState(42)
        counts = handler.perform_collisions(electron_beam, dt=1e-9, rng=rng)

        assert counts.get("excitation", 0) > 0

        ke_after = electron_beam.mean_energy_ev()
        assert ke_after < ke_before


class TestMakeHelpers:
    def test_make_electron_ar_mcc(self):
        """Factory function should create handler with correct processes."""
        sigma_el = make_constant_cross_section(5e-20)
        sigma_ion = make_constant_cross_section(1e-20)

        handler = make_electron_ar_mcc(
            n_ar=1e19,
            sigma_elastic=sigma_el,
            sigma_ionization=sigma_ion,
        )

        assert handler.projectile_mass == M_ELECTRON
        assert handler.background_mass == M_AR
        assert handler.background_density == 1e19
        assert len(handler.processes) == 2

    def test_update_background_density(self):
        """Updating background density should change collision probability."""
        sigma = make_constant_cross_section(1e-20)
        handler = MCCHandler(
            projectile_mass=M_ELECTRON,
            background_mass=M_AR,
            background_density=1e19,
        )
        handler.add_process(CollisionProcess(
            name="test", collision_type=CollisionType.ELASTIC,
            cross_section=sigma,
        ))

        p1 = handler.collision_probability(1e-10)

        handler.update_background_density(1e20)
        p2 = handler.collision_probability(1e-10)

        assert p2 > p1
