"""Tests for the Boris pusher: energy and momentum conservation.

Key properties to verify:
1. In a uniform B-field with no E-field, the Boris pusher preserves |v_perp|
   exactly and keeps |v| constant — no numerical heating or cooling.
2. Cyclotron frequency matches omega_c = |q|B/m.
3. In a uniform E-field with no B-field, energy gain equals q*E*dx.
"""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, M_AR, M_ELECTRON
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.pusher import boris_push, electrostatic_push


@pytest.fixture
def electron_species():
    return Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)


@pytest.fixture
def ion_species():
    return Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)


def make_single_particle(species, r=0.01, z=0.05, vr=1e5, vz=0.0, vtheta=0.0):
    """Create a ParticleArray with one particle."""
    p = ParticleArray(species=species)
    p.allocate(1)
    p.add_particles(
        r=np.array([r]),
        z=np.array([z]),
        vr=np.array([vr]),
        vz=np.array([vz]),
        vtheta=np.array([vtheta]),
        weight=np.array([1.0]),
    )
    return p


class TestBorisConservation:
    """Boris pusher should conserve energy in a static magnetic field."""

    def test_speed_conservation_uniform_B(self, electron_species):
        """In uniform B with no E, |v| should be constant."""
        v0 = 1e6  # m/s
        p = make_single_particle(electron_species, r=0.01, z=0.05, vr=v0, vz=0.0, vtheta=0.0)

        # Uniform B_z = 0.05 T, no E
        n = p.count
        Er = cp.zeros(n, dtype=cp.float64)
        Ez = cp.zeros(n, dtype=cp.float64)
        Br = cp.zeros(n, dtype=cp.float64)
        Bz = cp.full(n, 0.05, dtype=cp.float64)

        dt = 1e-12  # 1 ps timestep
        v_initial = float(cp.sqrt(p.vr[0]**2 + p.vz[0]**2 + p.vtheta[0]**2).item())

        # Run 1000 steps
        for _ in range(1000):
            boris_push(p, Er, Ez, Br, Bz, dt)

        v_final = float(cp.sqrt(p.vr[0]**2 + p.vz[0]**2 + p.vtheta[0]**2).item())

        # Speed should be conserved to machine precision
        assert v_final == pytest.approx(v_initial, rel=1e-10)

    def test_kinetic_energy_conservation(self, electron_species):
        """Kinetic energy should be exactly conserved in pure B field."""
        p = make_single_particle(
            electron_species, r=0.01, z=0.05,
            vr=5e5, vz=3e5, vtheta=4e5,
        )

        n = p.count
        Er = cp.zeros(n, dtype=cp.float64)
        Ez = cp.zeros(n, dtype=cp.float64)
        Br = cp.full(n, 0.01, dtype=cp.float64)
        Bz = cp.full(n, 0.03, dtype=cp.float64)

        KE_initial = p.kinetic_energy()
        dt = 1e-12

        for _ in range(500):
            boris_push(p, Er, Ez, Br, Bz, dt)

        KE_final = p.kinetic_energy()
        assert KE_final == pytest.approx(KE_initial, rel=1e-10)

    def test_cyclotron_period(self, electron_species):
        """Particle should return to initial position after one cyclotron period."""
        Bz_val = 0.01  # T
        omega_c = abs(electron_species.charge) * Bz_val / electron_species.mass
        T_c = 2.0 * np.pi / omega_c  # Cyclotron period

        # Need dt << T_c for accuracy
        n_steps_per_period = 100
        dt = T_c / n_steps_per_period

        v_perp = 1e6
        p = make_single_particle(
            electron_species, r=0.05, z=0.05,
            vr=v_perp, vz=0.0, vtheta=0.0,
        )

        n = p.count
        Er = cp.zeros(n, dtype=cp.float64)
        Ez = cp.zeros(n, dtype=cp.float64)
        Br = cp.zeros(n, dtype=cp.float64)
        Bz = cp.full(n, Bz_val, dtype=cp.float64)

        vr0 = float(p.vr[0].item())
        vt0 = float(p.vtheta[0].item())

        # Push for one full period
        for _ in range(n_steps_per_period):
            boris_push(p, Er, Ez, Br, Bz, dt)

        vr1 = float(p.vr[0].item())
        vt1 = float(p.vtheta[0].item())

        # After one period, velocity should return to initial
        assert vr1 == pytest.approx(vr0, rel=0.01)
        assert vt1 == pytest.approx(vt0, abs=v_perp * 0.01)


class TestElectrostaticPush:
    """Electrostatic (no B) push tests."""

    def test_uniform_acceleration(self, ion_species):
        """Ion in uniform E-field should gain energy = q*E*d."""
        E_val = 1e4  # V/m

        p = make_single_particle(
            ion_species, r=0.01, z=0.05,
            vr=0.0, vz=0.0, vtheta=0.0,
        )

        n = p.count
        Er = cp.zeros(n, dtype=cp.float64)
        Ez = cp.full(n, E_val, dtype=cp.float64)  # Accelerate in +z

        dt = 1e-9
        n_steps = 100

        for _ in range(n_steps):
            electrostatic_push(p, Er, Ez, dt)

        # Expected: v_z = q*E*dt*n_steps / m
        expected_vz = ion_species.charge * E_val * dt * n_steps / ion_species.mass
        actual_vz = float(p.vz[0].item())
        assert actual_vz == pytest.approx(expected_vz, rel=1e-6)

    def test_axis_reflection(self, ion_species):
        """Particle moving toward r=0 should reflect."""
        p = make_single_particle(
            ion_species, r=0.001, z=0.05,
            vr=-1e5, vz=0.0, vtheta=0.0,
        )

        n = p.count
        Er = cp.zeros(n, dtype=cp.float64)
        Ez = cp.zeros(n, dtype=cp.float64)

        dt = 1e-8  # Large enough to cross r=0

        electrostatic_push(p, Er, Ez, dt)

        # Should have reflected: r > 0 and vr > 0
        assert float(p.r[0].item()) > 0
        assert float(p.vr[0].item()) > 0
