"""Tests for HDF5 checkpoint save/load and diagnostic snapshots."""

import numpy as np
import pytest

pytest.importorskip("h5py")

from plasma.core.constants import E_CHARGE, M_AR, M_ELECTRON
from plasma.io.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint
from plasma.io.hdf5_diagnostics import load_diagnostics, save_diagnostics_snapshot
from plasma.pic.grid import CylindricalGrid
from plasma.pic.particles import ParticleArray, Species
from plasma.runtime.random import SimulationRNG, build_rng_from_state


@pytest.fixture
def grid():
    return CylindricalGrid(nr=5, nz=10, r_max=0.02, z_max=0.04)


@pytest.fixture
def electron_species():
    return Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)


@pytest.fixture
def ion_species():
    return Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)


class TestCheckpointRoundtrip:
    def test_save_load_roundtrip(self, tmp_path, grid, electron_species, ion_species):
        """Particle positions and phi should match after roundtrip."""
        import cupy as cp

        # Create particles
        electrons = ParticleArray(species=electron_species)
        electrons.allocate(50)
        n_e = 10
        r_vals = np.linspace(0.005, 0.015, n_e)
        z_vals = np.linspace(0.01, 0.03, n_e)
        electrons.add_particles(
            r=r_vals, z=z_vals,
            vr=np.ones(n_e) * 1e5, vz=np.ones(n_e) * -2e5,
            vtheta=np.zeros(n_e), weight=np.ones(n_e) * 1e10,
        )

        ions = ParticleArray(species=ion_species)
        ions.allocate(30)
        n_i = 5
        ions.add_particles(
            r=np.ones(n_i) * 0.01, z=np.ones(n_i) * 0.02,
            vr=np.zeros(n_i), vz=np.ones(n_i) * 1e3,
            vtheta=np.zeros(n_i), weight=np.ones(n_i) * 1e11,
        )

        phi = cp.ones((grid.n_nodes_r, grid.n_nodes_z)) * 42.0

        path = tmp_path / "test_ckpt.h5"
        save_checkpoint(path, step=100, time=1e-9, grid=grid,
                        species_dict={"electron": electrons, "Ar+": ions},
                        phi=phi)

        data = load_checkpoint(path)
        assert data["step"] == 100
        assert abs(data["time"] - 1e-9) < 1e-20
        np.testing.assert_allclose(data["phi"], 42.0)

        # Check particle data
        e_data = data["particles"]["electron"]
        np.testing.assert_allclose(e_data["r"], r_vals, atol=1e-15)
        np.testing.assert_allclose(e_data["z"], z_vals, atol=1e-15)

        i_data = data["particles"]["Ar+"]
        assert len(i_data["r"]) == n_i

    def test_species_metadata_preserved(self, tmp_path, grid, electron_species):
        """Species charge, mass, charge_state should survive checkpoint."""
        electrons = ParticleArray(species=electron_species)
        electrons.allocate(10)
        electrons.add_particles(
            r=np.array([0.01]), z=np.array([0.02]),
            vr=np.zeros(1), vz=np.zeros(1),
            vtheta=np.zeros(1), weight=np.array([1e10]),
        )

        path = tmp_path / "meta_test.h5"
        save_checkpoint(path, 0, 0.0, grid, {"electron": electrons})

        data = load_checkpoint(path)
        attrs = data["particles"]["electron"]["species_attrs"]
        assert abs(attrs["charge"] - (-E_CHARGE)) < 1e-25
        assert abs(attrs["mass"] - M_ELECTRON) < 1e-40
        assert attrs["charge_state"] == -1

    def test_empty_species_handled(self, tmp_path, grid, ion_species):
        """Zero alive particles should not crash."""
        ions = ParticleArray(species=ion_species)
        ions.allocate(10)

        path = tmp_path / "empty_test.h5"
        save_checkpoint(path, 0, 0.0, grid, {"Ar+": ions})

        data = load_checkpoint(path)
        assert len(data["particles"]["Ar+"]["r"]) == 0

    def test_grid_params_preserved(self, tmp_path, grid, ion_species):
        ions = ParticleArray(species=ion_species)
        ions.allocate(10)
        path = tmp_path / "grid_test.h5"
        save_checkpoint(path, 0, 0.0, grid, {"Ar+": ions})
        data = load_checkpoint(path)
        gp = data["grid_params"]
        assert gp["nr"] == grid.nr
        assert gp["nz"] == grid.nz
        assert abs(gp["r_max"] - grid.r_max) < 1e-15

    def test_background_state_and_metadata_preserved(self, tmp_path, grid, ion_species):
        ions = ParticleArray(species=ion_species)
        ions.allocate(10)
        path = tmp_path / "background_test.h5"
        save_checkpoint(
            path,
            12,
            4.0e-9,
            grid,
            {"Ar+": ions},
            background_state={"Ar_c": 2.0e20, "Cu": 4.0e17},
            metadata={"collision_package": "cu_ar_public_v1"},
        )

        data = load_checkpoint(path)
        assert data["background_state"]["Ar_c"] == pytest.approx(2.0e20)
        assert data["background_state"]["Cu"] == pytest.approx(4.0e17)
        assert data["metadata"]["collision_package"] == "cu_ar_public_v1"

    def test_rng_state_preserved(self, tmp_path, grid, ion_species):
        ions = ParticleArray(species=ion_species)
        ions.allocate(10)
        rng = SimulationRNG(123)
        _ = rng.rand(4)

        path = tmp_path / "rng_test.h5"
        save_checkpoint(
            path,
            3,
            2.0e-9,
            grid,
            {"Ar+": ions},
            rng_state=rng.state_dict(),
        )

        data = load_checkpoint(path)
        restored = build_rng_from_state(data["rng_state"])
        np.testing.assert_allclose(rng.rand(6), restored.rand(6))

    def test_list_checkpoints(self, tmp_path, grid, ion_species):
        ions = ParticleArray(species=ion_species)
        ions.allocate(10)
        for step in [100, 200, 50]:
            save_checkpoint(tmp_path / f"checkpoint_{step:06d}.h5", step, 0.0, grid, {"Ar+": ions})
        files = list_checkpoints(tmp_path)
        assert len(files) == 3
        # Should be sorted
        assert "000050" in files[0].name


class TestDiagnosticsSnapshot:
    def test_snapshot_appends(self, tmp_path):
        """Multiple snapshots should accumulate in the same file."""
        path = tmp_path / "diag.h5"
        phi = np.ones((6, 11)) * 10.0

        save_diagnostics_snapshot(path, step=0, time=0.0, phi=phi)
        save_diagnostics_snapshot(path, step=100, time=1e-9, phi=phi * 2)
        save_diagnostics_snapshot(path, step=200, time=2e-9, phi=phi * 3)

        data = load_diagnostics(path)
        assert len(data) == 3
        assert 0 in data
        assert 100 in data
        assert 200 in data
        np.testing.assert_allclose(data[0]["phi"], 10.0)
        np.testing.assert_allclose(data[200]["phi"], 30.0)

    def test_iedf_eedf_stored(self, tmp_path):
        path = tmp_path / "diag2.h5"
        energy = np.linspace(0, 100, 50)
        counts = np.random.rand(50)

        save_diagnostics_snapshot(
            path, step=0, time=0.0,
            iedf=(energy, counts),
            eedf=(energy, counts * 2),
        )

        data = load_diagnostics(path)
        np.testing.assert_allclose(data[0]["iedf_energy"], energy)
        np.testing.assert_allclose(data[0]["eedf_f"], counts * 2)

    def test_background_state_and_collision_counts_stored(self, tmp_path):
        path = tmp_path / "diag3.h5"
        save_diagnostics_snapshot(
            path,
            step=10,
            time=1e-6,
            background_state={"Ar_c": 2.1e20, "Cu": 3.0e17},
            collision_counts={"e_Ar_c_ionization": 4, "Ar+_Ar_c_cx": 7},
        )

        data = load_diagnostics(path)
        assert data[10]["background_state"]["Ar_c"] == pytest.approx(2.1e20)
        assert data[10]["collision_counts"]["Ar+_Ar_c_cx"] == 7
