"""Tests for run-mode truthfulness and runtime correctness fixes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plasma.core.config import load_config
from plasma.global_model.diagnostics import IRMDiagnostics
from plasma.global_model.irm import IRM
from plasma.global_model.rate_equations import STATE_INDICES
from plasma.io.checkpoint import load_checkpoint
from plasma.pic.grid import CylindricalGrid
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.runtime import _save_final_checkpoint
from plasma.runtime.cupy_compat import cp


def test_load_config_rejects_research_mode_with_exploratory_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "research_bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "base: /home/tax/Desktop/plasma/config/base.yaml",
                'name: "research_bad"',
                'mode: "research"',
                "case:",
                '  benchmark: "Bad research case"',
                "  inputs:",
                '    - name: "synthetic_waveform"',
                '      kind: "waveform"',
                '      provenance: "synthetic"',
            ],
        )
        + "\n",
    )

    with pytest.raises(ValueError, match="Research mode forbids"):
        load_config(config_path)


def test_irm_run_uses_configured_solver_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("config/hipims_cu_ar.yaml")
    assert cfg.numerics is not None
    cfg.numerics.solver = "Radau"
    cfg.numerics.rtol = 1e-5
    cfg.numerics.density_atol = 123.0
    cfg.numerics.current_atol = 7.0
    cfg.numerics.energy_atol = 456.0

    captured: dict[str, object] = {}

    def fake_solve_ivp(func, t_span, y0, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            t=np.array([t_span[0], t_span[1]], dtype=np.float64),
            y=np.column_stack([y0, y0]),
        )

    monkeypatch.setattr("scipy.integrate.solve_ivp", fake_solve_ivp)
    monkeypatch.setattr("plasma.global_model.irm.build_irm_diagnostics", lambda *args, **kwargs: IRMDiagnostics())

    IRM(cfg).run()

    assert captured["method"] == "Radau"
    assert captured["rtol"] == pytest.approx(1e-5)
    atol = captured["atol"]
    assert isinstance(atol, np.ndarray)
    assert float(atol[0]) == pytest.approx(123.0)
    assert float(atol[STATE_INDICES["current_circuit"]]) == pytest.approx(7.0)
    assert float(atol[-1]) == pytest.approx(456.0)


def test_save_final_checkpoint_persists_last_phi(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    grid = CylindricalGrid(nr=2, nz=3, r_max=0.01, z_max=0.02)
    electron = Species("electron", charge=-1.602176634e-19, mass=9.1093837015e-31, charge_state=-1)
    particles = ParticleArray(species=electron)
    particles.allocate(1)
    sim = {
        "grid": grid,
        "species_map": {"electron": particles},
        "n_steps": 5,
        "dt": 1e-9,
        "Br_grid": cp.zeros((grid.n_nodes_r, grid.n_nodes_z), dtype=cp.float64),
        "Bz_grid": cp.zeros((grid.n_nodes_r, grid.n_nodes_z), dtype=cp.float64),
    }
    phi = cp.full((grid.n_nodes_r, grid.n_nodes_z), 42.0, dtype=cp.float64)

    checkpoint_path = _save_final_checkpoint(tmp_path, sim, phi)

    assert checkpoint_path is not None
    data = load_checkpoint(checkpoint_path)
    np.testing.assert_allclose(data["phi"], 42.0)
