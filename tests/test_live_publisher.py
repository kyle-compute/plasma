from __future__ import annotations

import numpy as np
import pytest

from plasma.diagnostics.collectors import SubstrateCollector
from plasma.live.contracts import LiveGeometry
from plasma.core.config import load_config
from plasma.global_model.irm import IRM
from plasma.live.builders import build_global_live_snapshot, build_pic_live_snapshot
from plasma.live.publisher import FileLiveSession, LiveCommandWriteError
from plasma.pic.grid import CylindricalGrid
from plasma.pic.particles import ParticleArray, Species


def test_file_live_session_round_trips_snapshot_and_commands(tmp_path):
    session = FileLiveSession(tmp_path / "live")
    cfg = load_config("config/hipims_cu_ar.yaml")
    result = IRM(cfg).run()

    snapshot = build_global_live_snapshot(cfg.name, result, end_index=8, state="running")
    session.publish(snapshot)

    loaded = session.load_snapshot()
    assert loaded is not None
    assert loaded.title == cfg.name
    assert loaded.series["current_a"].y

    command = session.send_command("pause")
    assert command.seq == 1
    assert session.poll_command() is not None
    assert session.poll_command() is None


def test_build_pic_live_snapshot_contains_fields_particles_and_histogram():
    grid = CylindricalGrid(8, 10, 0.02, 0.04)
    electron = ParticleArray(Species("electron", -1.0, 9.11e-31, -1))
    ion = ParticleArray(Species("Ar+", 1.0, 6.63e-26, 1))
    neutral = ParticleArray(Species("Cu", 0.0, 1.05e-25, 0))

    for array in (electron, ion, neutral):
        array.allocate(16)

    electron.add_particles(
        r=np.array([0.001, 0.002]),
        z=np.array([0.003, 0.004]),
        vr=np.array([1.0e5, 1.5e5]),
        vz=np.array([2.0e5, 1.0e5]),
        vtheta=np.array([0.0, 0.0]),
        weight=np.array([1.0, 1.0]),
    )
    ion.add_particles(
        r=np.array([0.0015, 0.0025]),
        z=np.array([0.0395, 0.0397]),
        vr=np.array([5.0e3, 6.0e3]),
        vz=np.array([7.0e3, 8.0e3]),
        vtheta=np.array([0.0, 0.0]),
        weight=np.array([1.0, 1.0]),
    )
    neutral.add_particles(
        r=np.array([0.0020]),
        z=np.array([0.0100]),
        vr=np.array([1.0e2]),
        vz=np.array([1.0e2]),
        vtheta=np.array([0.0]),
        weight=np.array([1.0]),
    )

    collector = SubstrateCollector(z_plane=0.04, dz_capture=1e-3)
    collector.record_absorbed(ion, t=1e-6)

    phi = np.zeros((grid.n_nodes_r, grid.n_nodes_z))
    history = {
        "time_s": [0.0, 1e-6],
        "target_voltage_v": [-600.0, -625.0],
        "electron_particles": [2.0, 2.0],
        "ar_ion_particles": [2.0, 2.0],
        "cu_neutral_particles": [1.0, 1.0],
        "substrate_hits_total": [0.0, float(collector.total_count)],
        "see_per_target_impact": [0.0, 0.5],
        "sputtered_per_target_impact": [0.0, 1.0],
        "source_activity_total_arb": [0.0, 3.0],
        "substrate_flux_total_arb": [0.0, 1.0],
        "substrate_mean_energy_ev": [0.0, collector.latest_mean_energy_ev()],
        "racetrack_peak_r_m": [0.0, 0.0015],
    }

    snapshot = build_pic_live_snapshot(
        "pic_live_test",
        step=20,
        time_s=1e-6,
        grid=grid,
        phi=phi,
        species_map={"electron": electron, "Ar+": ion, "Cu": neutral},
        history=history,
        br_grid=np.ones_like(phi) * 0.02,
        bz_grid=np.ones_like(phi) * 0.04,
        event_window={
            "e_Ar_excitation": {"r": np.array([0.001]), "z": np.array([0.005])},
            "secondary_electrons": {"r": np.array([0.0015]), "z": np.array([1e-6])},
        },
        geometry=LiveGeometry(r_max=0.02, z_max=0.04, z_target=0.0, z_substrate=0.04, r_inner=0.001, r_outer=0.01),
        substrate=collector,
        max_particles=8,
    )

    assert snapshot.model == "pic"
    assert "phi_v" in snapshot.fields
    assert "e_mag_v_m" in snapshot.fields
    assert "b_mag_t" in snapshot.fields
    assert "emissivity_arb" in snapshot.fields
    assert "see_source_arb" in snapshot.fields
    assert "sputter_source_arb" in snapshot.fields
    assert "ionization_source_arb" in snapshot.fields
    assert "substrate_flux_proxy_arb" in snapshot.fields
    assert "electron" in snapshot.particles
    assert "Ar+" in snapshot.particles
    assert "Cu" in snapshot.particles
    assert "substrate_iedf" in snapshot.histograms
    assert snapshot.metrics["max_e_field_v_m"] >= 0.0
    assert "see_yield_proxy" in snapshot.metrics
    assert "sputter_yield_proxy" in snapshot.metrics
    assert "pulse_phase_code" in snapshot.metrics
    assert "see_per_target_impact" in snapshot.series
    assert "substrate_mean_energy_ev" in snapshot.series
    assert snapshot.message is not None
    assert "Pulse phase:" in snapshot.message
    assert snapshot.geometry is not None


def test_file_live_session_reports_unwritable_live_dir(tmp_path, monkeypatch):
    session = FileLiveSession(tmp_path / "live")
    monkeypatch.setattr("plasma.live.publisher.os.access", lambda _path, _mode: False)

    message = session.command_write_error()

    assert message is not None
    assert "not writable" in message


def test_file_live_session_wraps_command_write_failures(tmp_path, monkeypatch):
    session = FileLiveSession(tmp_path / "live")

    def _raise_permission_error(*_args, **_kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr("plasma.live.publisher._atomic_write", _raise_permission_error)

    with pytest.raises(LiveCommandWriteError, match="docker compose run --user"):
        session.send_command("pause")
