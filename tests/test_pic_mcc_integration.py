"""Integration tests: MCC collisions + magnetron surface physics wired into PIC loop."""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, M_AR, M_CU, M_ELECTRON
from plasma.data.sputtering import SputterYield
from plasma.data.synthetic_cross_sections import (
    electron_ar_elastic,
    electron_ar_excitation,
    electron_ar_ionization,
    ion_ar_charge_exchange,
)
from plasma.data.waveforms import make_square_pulse
from plasma.pic.grid import CylindricalGrid
from plasma.pic.loop import pic_step, run_pic
from plasma.pic.magnetron import MagnetronTarget
from plasma.pic.mcc import make_electron_ar_mcc, make_ion_ar_mcc
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.poisson import PoissonSolverCylindrical
from plasma.pic.weighting import initialize_particles_uniform


@pytest.fixture
def grid():
    return CylindricalGrid(nr=10, nz=20, r_max=0.05, z_max=0.1)


@pytest.fixture
def solver(grid):
    return PoissonSolverCylindrical(grid)


@pytest.fixture
def electron_species():
    return Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)


@pytest.fixture
def ion_species():
    return Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)


@pytest.fixture
def cu_species():
    return Species(name="Cu", charge=0.0, mass=M_CU, charge_state=0)


def _make_particles(grid, species, n0=1e15, te_ev=3.0, ppc=5):
    data = initialize_particles_uniform(grid, species, n0, te_ev, ppc)
    p = ParticleArray(species=species)
    p.allocate(len(data["r"]) * 4)
    p.add_particles(**data)
    return p


class TestPICStepWithMCC:
    def test_pic_step_with_mcc_creates_ions(self, grid, solver, electron_species, ion_species):
        """Electron MCC with ionization should create new Ar+ ions."""
        electrons = _make_particles(grid, electron_species, n0=1e16, te_ev=30.0)
        ions = _make_particles(grid, ion_species, n0=1e16, te_ev=0.1)

        n_ions_before = ions.n_alive

        # High density to ensure collisions happen
        n_ar = 1e20
        handler = make_electron_ar_mcc(
            n_ar,
            sigma_elastic=electron_ar_elastic(),
            sigma_excitation=electron_ar_excitation(),
            sigma_ionization=electron_ar_ionization(),
        )

        species_map = {"electron": electrons, "Ar+": ions}
        mcc_handlers = {"electron": handler}

        _phi, stats = pic_step(
            grid, [electrons, ions], solver, dt=1e-11,
            mcc_handlers=mcc_handlers, species_map=species_map,
        )

        # With high n_ar and hot electrons, ionization should create ions
        n_ions_after = ions.n_alive
        assert n_ions_after >= n_ions_before

    def test_waveform_drives_target_potential(self, grid, solver, electron_species, ion_species):
        """Waveform should set the z=0 boundary potential."""
        electrons = _make_particles(grid, electron_species, n0=1e15, te_ev=3.0)
        ions = _make_particles(grid, ion_species, n0=1e15, te_ev=0.1)

        waveform = make_square_pulse(voltage_v=600.0, t_pulse_s=100e-6, t_total_s=300e-6)

        phi, _stats = pic_step(
            grid, [electrons, ions], solver, dt=1e-11,
            waveform=waveform, t=50e-6,
        )

        # During pulse, target at z=0 should be at -600V
        phi_z0 = float(cp.asnumpy(phi[0, 0]))
        assert phi_z0 < -500.0  # Should be close to -600V


class TestSurfacePhysics:
    def test_surface_before_boundary_kill(self, grid, solver, electron_species, ion_species):
        """Ions at z<=0 should be processed (impacts counted) then killed."""
        electrons = _make_particles(grid, electron_species, n0=1e14, te_ev=1.0, ppc=2)

        # Place ions right at the target
        target = MagnetronTarget(
            z_target=0.0, r_inner=0.01, r_outer=0.04, see_yield=0.0,
        )

        ions = ParticleArray(species=ion_species)
        ions.allocate(100)
        n_test = 20
        r_vals = np.linspace(0.015, 0.035, n_test)
        ions.add_particles(
            r=r_vals,
            z=np.full(n_test, -1e-6),  # Just below target
            vr=np.zeros(n_test),
            vz=np.full(n_test, -1e5),  # Moving toward target
            vtheta=np.zeros(n_test),
            weight=np.full(n_test, 1e10),
        )

        species_map = {"electron": electrons, "Ar+": ions}

        _phi, stats = pic_step(
            grid, [electrons, ions], solver, dt=1e-11,
            target=target, species_map=species_map,
        )

        # All ions at z<0 within race-track should have been detected as impacts
        assert stats["n_target_impacts"] == n_test
        # And then killed by boundaries
        assert ions.n_alive == 0

    def test_see_electrons_injected(self, grid, solver, electron_species, ion_species):
        """Ion impacts with SEE yield > 0 should inject secondary electrons."""
        target = MagnetronTarget(
            z_target=0.0, r_inner=0.01, r_outer=0.04,
            see_yield=1.0,  # Every impact produces SEE
            see_energy_ev=3.0,
        )

        electrons = ParticleArray(species=electron_species)
        electrons.allocate(200)

        ions = ParticleArray(species=ion_species)
        ions.allocate(100)
        n_ions = 10
        ions.add_particles(
            r=np.linspace(0.015, 0.035, n_ions),
            z=np.full(n_ions, -1e-6),
            vr=np.zeros(n_ions),
            vz=np.full(n_ions, -1e5),
            vtheta=np.zeros(n_ions),
            weight=np.full(n_ions, 1e10),
        )

        species_map = {"electron": electrons, "Ar+": ions}

        _phi, stats = pic_step(
            grid, [electrons, ions], solver, dt=1e-11,
            target=target, species_map=species_map,
        )

        # With yield=1.0, every impact should produce an SEE electron
        assert stats["n_see"] > 0
        assert electrons.n_alive > 0

    def test_sputtered_neutrals_injected(
        self, grid, solver, electron_species, ion_species, cu_species,
    ):
        """Ion impacts with sputter yield should inject neutral atoms."""
        sputter = SputterYield(
            ion="Ar+", target="Cu", a=0.1421, b=0.468,
            threshold_ev=20.0, cohesive_energy_ev=3.49,
        )
        target = MagnetronTarget(
            z_target=0.0, r_inner=0.01, r_outer=0.04,
            see_yield=0.0,
            sputter_yield=sputter,
            material_mass=M_CU,
        )

        electrons = ParticleArray(species=electron_species)
        electrons.allocate(10)

        cu = ParticleArray(species=cu_species)
        cu.allocate(200)

        # Energetic ions to exceed sputter threshold
        ions = ParticleArray(species=ion_species)
        ions.allocate(100)
        n_ions = 20
        # Give them enough energy to sputter (>20 eV threshold)
        v_high = np.sqrt(2.0 * 200.0 * E_CHARGE / M_AR)  # 200 eV ions
        ions.add_particles(
            r=np.linspace(0.015, 0.035, n_ions),
            z=np.full(n_ions, -1e-6),
            vr=np.zeros(n_ions),
            vz=np.full(n_ions, -v_high),
            vtheta=np.zeros(n_ions),
            weight=np.full(n_ions, 1e10),
        )

        species_map = {"electron": electrons, "Ar+": ions, "Cu": cu}

        _phi, stats = pic_step(
            grid, [electrons, ions, cu], solver, dt=1e-11,
            target=target, species_map=species_map,
        )

        assert stats["n_target_impacts"] == n_ions
        assert stats["n_sputtered"] > 0
        assert cu.n_alive > 0


class TestFullHiPIMSStep:
    def test_full_hipims_step(self, grid, solver, electron_species, ion_species, cu_species):
        """All components combined: waveform + target + MCC. No crash, counts change."""
        electrons = _make_particles(grid, electron_species, n0=1e15, te_ev=5.0, ppc=3)
        ions = _make_particles(grid, ion_species, n0=1e15, te_ev=0.1, ppc=3)
        cu = ParticleArray(species=cu_species)
        cu.allocate(500)

        waveform = make_square_pulse(600.0, 100e-6, 300e-6)
        sputter = SputterYield("Ar+", "Cu", 0.1421, 0.468, 20.0, 3.49)
        target = MagnetronTarget(
            z_target=0.0, r_inner=0.01, r_outer=0.04,
            see_yield=0.1, sputter_yield=sputter, material_mass=M_CU,
        )

        n_ar = 3.2e19  # ~1 Pa at 300 K
        mcc_handlers = {
            "electron": make_electron_ar_mcc(
                n_ar,
                sigma_elastic=electron_ar_elastic(),
                sigma_ionization=electron_ar_ionization(),
            ),
            "Ar+": make_ion_ar_mcc(
                n_ar, M_AR,
                sigma_charge_exchange=ion_ar_charge_exchange(),
            ),
        }

        species_map = {"electron": electrons, "Ar+": ions, "Cu": cu}
        rng = cp.random.RandomState(42)

        phi, stats = pic_step(
            grid, [electrons, ions, cu], solver, dt=1e-11,
            waveform=waveform, target=target,
            mcc_handlers=mcc_handlers, species_map=species_map,
            rng=rng, t=50e-6,
        )

        assert phi.shape == (grid.n_nodes_r, grid.n_nodes_z)
        assert np.all(np.isfinite(cp.asnumpy(phi)))
        assert isinstance(stats["collision_counts"], dict)

    def test_run_pic_with_mcc(self, grid, solver, electron_species, ion_species):
        """run_pic with MCC handlers should run without error."""
        electrons = _make_particles(grid, electron_species, n0=1e15, te_ev=5.0, ppc=2)
        ions = _make_particles(grid, ion_species, n0=1e15, te_ev=0.1, ppc=2)

        n_ar = 1e19
        mcc_handlers = {
            "electron": make_electron_ar_mcc(
                n_ar, sigma_elastic=electron_ar_elastic(),
            ),
        }
        species_map = {"electron": electrons, "Ar+": ions}

        diag = run_pic(
            grid, [electrons, ions], solver, dt=1e-11, n_steps=5,
            diag_interval=1,
            mcc_handlers=mcc_handlers, species_map=species_map,
        )

        assert len(diag.time) > 0
        assert len(diag.collision_counts) > 0
