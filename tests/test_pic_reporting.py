"""Tests for typed PIC config and reporting artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plasma.data.waveforms import make_square_pulse
from plasma.pic.config import load_pic_config
from plasma.pic.plots import save_pic_quicklook
from plasma.pic.reporting import bundle_from_pic_run
from plasma.reporting import build_run_manifest, build_validation_report


@dataclass
class FakeCollector:
    total_count: int = 12

    def iedf(self, n_bins: int = 100, e_max_ev: float = 500.0):
        energy = np.linspace(0.0, 100.0, 8)
        counts = np.linspace(1.0, 8.0, 8)
        return energy, counts


@dataclass
class FakePICDiag:
    time: list[float] = field(default_factory=lambda: [0.0, 1e-6, 2e-6])
    n_particles: dict[str, list[int]] = field(
        default_factory=lambda: {"electron": [100, 120, 140], "Ar+": [80, 90, 95]},
    )
    field_energy: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.15])
    kinetic_energy: dict[str, list[float]] = field(
        default_factory=lambda: {"electron": [1.0, 1.1, 1.2], "Ar+": [0.5, 0.6, 0.7]},
    )
    total_energy: list[float] = field(default_factory=lambda: [1.6, 1.9, 2.0])
    collision_counts: list[dict[str, int]] = field(
        default_factory=lambda: [{"elastic": 4}, {"elastic": 5, "ionization": 1}, {"null": 8}],
    )
    n_see_total: list[int] = field(default_factory=lambda: [1, 2, 1])
    n_sputtered_total: list[int] = field(default_factory=lambda: [2, 3, 2])
    n_target_impacts: list[int] = field(default_factory=lambda: [4, 5, 6])


def test_pic_config_loads_case_metadata() -> None:
    cfg = load_pic_config("config/hipims_cu_ar_pic.yaml")

    assert cfg.output.dir.endswith("output/pic/hipims_cu_ar_pic")
    assert cfg.case is not None
    assert cfg.case.inputs[0].provenance == "synthetic"
    assert cfg.validation is not None
    assert cfg.validation.targets[0].metric == "peak_target_voltage_v"


def test_monitor_pic_config_loads_monitor_overrides() -> None:
    cfg = load_pic_config("config/hipims_cu_ar_pic_monitor.yaml")

    assert cfg.name == "hipims_cu_ar_pic_monitor"
    assert cfg.grid.nr == 60
    assert cfg.grid.nz == 100
    assert cfg.particles.ppc == 16
    assert cfg.magnetic_field.map_file is not None
    assert cfg.output.dir.endswith("output/pic/hipims_cu_ar_pic_monitor")


def test_pic_bundle_manifest_and_validation_are_exploratory(tmp_path) -> None:
    cfg = load_pic_config("config/hipims_cu_ar_pic.yaml")
    diag = FakePICDiag()
    waveform = make_square_pulse(600.0, 100e-6, 300e-6)
    bundle = bundle_from_pic_run(diag, waveform, collector=FakeCollector())
    report = build_validation_report(cfg, bundle)
    manifest = build_run_manifest(cfg, bundle, report)

    assert bundle.summary["peak_target_voltage_v"] == 600.0
    assert bundle.summary["total_target_impacts"] == 15.0
    assert "substrate_iedf" in bundle.distributions
    assert report.status == "exploratory"
    assert manifest.validation_status == "exploratory"

    plot_path = save_pic_quicklook(bundle, tmp_path / "pic_summary.png", title=cfg.name)
    assert plot_path.exists()
