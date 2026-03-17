"""Tests for magnetron geometry, SEE, and sputtering boundaries."""

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, M_AR
from plasma.data.sputtering import SputterYield
from plasma.pic.grid import CylindricalGrid
from plasma.pic.magnetron import (
    MagnetronGeometry,
    MagnetronTarget,
    apply_magnetron_boundaries,
    process_target_impacts,
)
from plasma.pic.particles import ParticleArray, Species


@pytest.fixture
def grid():
    return CylindricalGrid(nr=20, nz=40, r_max=0.06, z_max=0.1)


@pytest.fixture
def target():
    return MagnetronTarget(
        z_target=0.0,
        r_inner=0.015,
        r_outer=0.035,
        see_yield=0.1,
        see_energy_ev=3.0,
        sputter_yield=SputterYield(
            ion="Ar+", target="Cu",
            a=0.1421, b=0.468,
            threshold_ev=17.0,
            cohesive_energy_ev=3.49,
        ),
        surface_binding_ev=3.49,
        material_mass=63.546 * 1.66053906660e-27,
    )


@pytest.fixture
def ion_species():
    return Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)


@pytest.fixture
def cu_ion_species():
    return Species(name="Cu+", charge=E_CHARGE, mass=63.546 * 1.66053906660e-27, charge_state=1)


class TestMagnetronTarget:
    def test_on_target_check(self, target):
        """Points within erosion zone should be detected."""
        assert target.is_on_target(0.025)  # Between 0.015 and 0.035
        assert not target.is_on_target(0.005)  # Inside inner radius
        assert not target.is_on_target(0.045)  # Outside outer radius

    def test_on_target_array(self, target):
        r = np.array([0.005, 0.015, 0.025, 0.035, 0.045])
        mask = target.is_on_target_array(r)
        assert mask.tolist() == [False, True, True, True, False]


class TestMagnetronBoundaries:
    def test_ions_absorbed_at_target(self, grid, target, ion_species):
        """Ions at z <= 0 should be absorbed."""
        n = 100
        p = ParticleArray(species=ion_species)
        p.allocate(n)

        # Place ions just below target
        p.add_particles(
            r=np.full(n, 0.025),  # In erosion zone
            z=np.full(n, -0.001),  # Below target
            vr=np.zeros(n),
            vz=np.full(n, -1e4),  # Moving toward target
            vtheta=np.zeros(n),
            weight=np.ones(n),
        )

        n_absorbed, flags = apply_magnetron_boundaries(grid, p, target)
        assert n_absorbed == n
        assert p.n_alive == 0

    def test_target_flags_set(self, grid, target, ion_species):
        """Ions hitting erosion zone should be flagged."""
        n = 50
        p = ParticleArray(species=ion_species)
        p.allocate(n)

        # Half in erosion zone, half outside
        r_vals = np.concatenate([
            np.full(25, 0.025),   # In erosion zone
            np.full(25, 0.005),   # Inside inner radius
        ])
        p.add_particles(
            r=r_vals,
            z=np.full(n, -0.001),
            vr=np.zeros(n),
            vz=np.full(n, -1e4),
            vtheta=np.zeros(n),
            weight=np.ones(n),
        )

        _, flags = apply_magnetron_boundaries(grid, p, target)
        flags_cpu = cp.asnumpy(flags[:n])

        # 25 ions in erosion zone should have flag=1
        assert np.sum(flags_cpu) == 25

    def test_substrate_absorbs(self, grid, target, ion_species):
        """Ions at z >= z_max should be absorbed."""
        n = 10
        p = ParticleArray(species=ion_species)
        p.allocate(n)
        p.add_particles(
            r=np.full(n, 0.02),
            z=np.full(n, grid.z_max + 0.001),
            vr=np.zeros(n),
            vz=np.full(n, 1e4),
            vtheta=np.zeros(n),
            weight=np.ones(n),
        )

        n_absorbed, _ = apply_magnetron_boundaries(grid, p, target)
        assert n_absorbed == n


class TestSEE:
    def test_see_produces_electrons(self, grid, target, ion_species):
        """Ion impacts on target should produce secondary electrons."""
        n = 1000
        p = ParticleArray(species=ion_species)
        p.allocate(n)

        # Fast ions hitting target in erosion zone
        v_500ev = np.sqrt(2.0 * 500.0 * E_CHARGE / M_AR)
        p.add_particles(
            r=np.full(n, 0.025),
            z=np.full(n, -0.0001),  # Just past target
            vr=np.zeros(n),
            vz=np.full(n, -v_500ev),
            vtheta=np.zeros(n),
            weight=np.ones(n) * 1e8,
        )

        result = process_target_impacts(target, p, grid)

        assert result["n_impacts"] > 0

        # With yield=0.1, expect ~100 secondary electrons from 1000 impacts
        see = result["see_electrons"]
        assert see is not None
        n_see = len(see["r"])
        # Statistical: expect 100 ± ~30 (3 sigma)
        assert 30 < n_see < 200

        # SEE electrons should be just above target
        assert np.all(see["z"] > target.z_target)
        # Should have positive vz (away from target)
        assert np.all(see["vz"] > 0)


class TestSputtering:
    def test_species_specific_target_models_apply_to_cu_ions(self, grid, cu_ion_species):
        target = MagnetronTarget(
            z_target=0.0,
            r_inner=0.015,
            r_outer=0.035,
            see_yield=0.0,
            sputter_yield=None,
            species_see_yields={"Cu+": 1.0},
            species_sputter_yields={
                "Cu+": SputterYield(
                    ion="Cu+",
                    target="Cu",
                    a=0.0691,
                    b=0.556,
                    threshold_ev=20.0,
                    cohesive_energy_ev=3.49,
                )
            },
        )

        ions = ParticleArray(species=cu_ion_species)
        ions.allocate(100)
        v_200ev = np.sqrt(2.0 * 200.0 * E_CHARGE / cu_ion_species.mass)
        ions.add_particles(
            r=np.full(30, 0.025),
            z=np.full(30, -1.0e-4),
            vr=np.zeros(30),
            vz=np.full(30, -v_200ev),
            vtheta=np.zeros(30),
            weight=np.ones(30),
        )

        result = process_target_impacts(target, ions, grid)
        assert result["see_electrons"] is not None
        assert len(result["see_electrons"]["r"]) > 0
        assert result["sputtered_neutrals"] is not None
        assert len(result["sputtered_neutrals"]["r"]) > 0

    def test_sputtering_produces_neutrals(self, grid, target, ion_species):
        """Ion impacts should produce sputtered neutral atoms."""
        n = 500
        p = ParticleArray(species=ion_species)
        p.allocate(n)

        v_500ev = np.sqrt(2.0 * 500.0 * E_CHARGE / M_AR)
        p.add_particles(
            r=np.full(n, 0.025),
            z=np.full(n, -0.0001),
            vr=np.zeros(n),
            vz=np.full(n, -v_500ev),
            vtheta=np.zeros(n),
            weight=np.ones(n) * 1e8,
        )

        result = process_target_impacts(target, p, grid)
        sp = result["sputtered_neutrals"]

        assert sp is not None
        n_sputtered = len(sp["r"])
        # At 500 eV, Y(Ar+→Cu) ≈ 0.1421 * 500^0.468 ≈ 2.5
        # So 500 impacts × ~2.5 ≈ 1250 sputtered atoms (with statistical variation)
        assert n_sputtered > 100

        # Sputtered atoms should be above target
        assert np.all(sp["z"] > target.z_target)
        # Should have positive vz (away from target)
        assert np.all(sp["vz"] > 0)

    def test_below_threshold_no_sputtering(self, grid, ion_species):
        """Ions below sputter threshold should not produce neutrals."""
        target = MagnetronTarget(
            z_target=0.0,
            r_inner=0.015,
            r_outer=0.035,
            see_yield=0.0,  # Disable SEE for this test
            sputter_yield=SputterYield(
                ion="Ar+", target="Cu",
                a=0.1421, b=0.468,
                threshold_ev=17.0,
                cohesive_energy_ev=3.49,
            ),
        )

        n = 100
        p = ParticleArray(species=ion_species)
        p.allocate(n)

        # Slow ions: 5 eV (below 17 eV threshold)
        v_5ev = np.sqrt(2.0 * 5.0 * E_CHARGE / M_AR)
        p.add_particles(
            r=np.full(n, 0.025),
            z=np.full(n, -0.0001),
            vr=np.zeros(n),
            vz=np.full(n, -v_5ev),
            vtheta=np.zeros(n),
            weight=np.ones(n) * 1e8,
        )

        result = process_target_impacts(target, p, grid)
        assert result["sputtered_neutrals"] is None or len(result["sputtered_neutrals"]["r"]) == 0


class TestMagnetronGeometry:
    def test_make_target(self):
        """MagnetronGeometry should create a valid target."""
        geom = MagnetronGeometry(
            r_target=0.05,
            r_inner_erosion=0.015,
            r_outer_erosion=0.035,
        )
        target = geom.make_target(see_yield=0.1)
        assert target.r_inner == 0.015
        assert target.r_outer == 0.035
        assert target.see_yield == 0.1

    def test_make_bfield(self, grid):
        """Should compute a non-trivial magnetic field."""
        geom = MagnetronGeometry(
            r_target=0.05,
            r_inner_erosion=0.015,
            r_outer_erosion=0.035,
        )
        br, bz = geom.make_bfield(grid)

        assert br.shape == (grid.n_nodes_r, grid.n_nodes_z)
        assert bz.shape == (grid.n_nodes_r, grid.n_nodes_z)

        # Field should be non-zero
        assert np.max(np.abs(br)) > 0
        assert np.max(np.abs(bz)) > 0

        # Near the target (z~0), Br should be significant in the race-track region
        # (this is what traps electrons)
        br_at_target = br[:, 0]
        assert np.max(np.abs(br_at_target)) > 1e-4  # At least 0.1 mT
