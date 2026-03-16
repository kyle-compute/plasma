"""Tests for 0D diagnostics bundles, manifests, and validation reports."""

from __future__ import annotations

from plasma.core.config import load_config
from plasma.diagnostics.bundles import bundle_from_irm_state
from plasma.global_model.irm import IRM
from plasma.reporting import build_run_manifest, build_validation_report


def test_irm_run_emits_transport_metrics() -> None:
    cfg = load_config("config/hipims_cu_ar.yaml")
    result = IRM(cfg).run()

    assert result.metric("alpha_t").shape == result.time.shape
    assert result.metric("xi_t").shape == result.time.shape
    assert result.metric("current_a").max() > 0.0
    assert result.metric("model_current_proxy_a").shape == result.time.shape
    assert result.metric("reference_current_a").shape == result.time.shape
    assert 0.0 <= result.metric("beta_t").max() <= 0.5


def test_bundle_manifest_and_validation_are_exploratory() -> None:
    cfg = load_config("config/hipims_cu_ar.yaml")
    result = IRM(cfg).run()
    bundle = bundle_from_irm_state(result)
    report = build_validation_report(cfg, bundle)
    manifest = build_run_manifest(cfg, bundle, report)

    assert "peak_current_a" in bundle.summary
    assert "current_proxy_rmse_a" in bundle.summary
    assert "current_proxy_peak_time_error_us" in bundle.summary
    assert report.status == "exploratory"
    assert manifest.validation_status == "exploratory"
    assert manifest.input_sources[0].provenance == "literature-fit"
