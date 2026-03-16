"""Tests for config path resolution and case/result contracts."""

from __future__ import annotations

from plasma.core.config import load_config
from plasma.data.waveforms import load_waveform


def test_config_resolves_case_input_paths() -> None:
    cfg = load_config("config/hipims_cu_ar.yaml")

    assert cfg.pulse.waveform_file is not None
    assert cfg.pulse.waveform_file.endswith("data/waveforms/cu_ar_case1_literature_fit.csv")
    assert cfg.case is not None
    assert cfg.case.inputs[0].path is not None
    assert cfg.case.inputs[0].path.endswith("data/waveforms/cu_ar_case1_literature_fit.csv")


def test_waveform_loader_preserves_provenance() -> None:
    waveform = load_waveform(
        "data/waveforms/cu_ar_case1_literature_fit.csv",
        provenance="literature-fit",
    )

    assert waveform.provenance == "literature-fit"
    assert waveform.current_a.max() > 0.0
