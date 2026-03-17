"""Tests for config path resolution and case/result contracts."""

from __future__ import annotations

from plasma.core.config import load_config
from plasma.data.benchmark_packages import load_benchmark_package
from plasma.data.waveforms import load_waveform


def test_config_resolves_case_input_paths() -> None:
    cfg = load_config("config/hipims_cu_ar.yaml")

    assert cfg.material_package is not None
    assert cfg.material_package.endswith("data/material_packages/cu_ar_public_v1.yaml")
    assert cfg.surface_package is not None
    assert cfg.surface_package.endswith("data/surface_packages/cu_target_public_v1.yaml")
    assert cfg.pulse.waveform_file is not None
    assert cfg.pulse.waveform_file.endswith("data/waveforms/cu_ar_case1_literature_fit.csv")
    assert cfg.benchmark_package is not None
    assert cfg.benchmark_package.endswith("data/benchmarks/gudmundsson_cu_ar_case1.yaml")
    assert cfg.case is not None
    assert cfg.case.benchmark_package == "gudmundsson_cu_ar_case1"
    assert cfg.case.inputs[0].path is not None
    assert cfg.case.inputs[0].path.endswith("data/waveforms/cu_ar_case1_literature_fit.csv")
    assert cfg.validation is not None
    assert cfg.validation.suite_name == "gudmundsson_case1_global"


def test_benchmark_package_loader_merges_global_defaults() -> None:
    package = load_benchmark_package("data/benchmarks/gudmundsson_cu_ar_case1.yaml", model="global")

    assert package["reactions_file"].endswith("data/reactions/gudmundsson_cu_ar.yaml")
    assert package["pulse"]["waveform_file"].endswith("data/waveforms/cu_ar_case1_literature_fit.csv")
    assert package["case"]["benchmark_package"] == "gudmundsson_cu_ar_case1"
    assert package["validation"]["suite_name"] == "gudmundsson_case1_global"


def test_waveform_loader_preserves_provenance() -> None:
    waveform = load_waveform(
        "data/waveforms/cu_ar_case1_literature_fit.csv",
        provenance="literature-fit",
    )

    assert waveform.provenance == "literature-fit"
    assert waveform.current_a.max() > 0.0
