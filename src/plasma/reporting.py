"""Helpers to build manifests and validation reports for saved runs."""

from __future__ import annotations

from plasma.contracts.results import (
    MetricSummary,
    RunManifest,
    ValidationReport,
    ValidationResult,
)
from plasma.diagnostics.bundles import DiagnosticsBundle


def build_validation_report(config, bundle: DiagnosticsBundle) -> ValidationReport:
    """Compare configured validation targets against summary metrics."""

    targets = [] if config.validation is None else config.validation.targets
    if not targets:
        return ValidationReport(
            case_name=config.name,
            benchmark_package=None if config.case is None else config.case.benchmark_package,
            validation_suite=None if config.validation is None else config.validation.suite_name,
            status="not_validated",
        )

    results = []
    for target in targets:
        value = bundle.summary.get(target.metric)
        passed = value is not None
        if passed and target.lower is not None:
            passed = value >= target.lower
        if passed and target.upper is not None:
            passed = value <= target.upper
        results.append(
            ValidationResult(
                name=target.name,
                metric=target.metric,
                value=value,
                lower=target.lower,
                upper=target.upper,
                unit=target.unit,
                passed=passed,
                citation=target.citation,
                notes=None if value is not None else "Metric missing from diagnostics bundle.",
            )
        )

    exploratory = _has_exploratory_inputs(bundle)
    status = "validated" if all(result.passed for result in results) else "failed"
    if exploratory:
        status = "exploratory"
    return ValidationReport(
        case_name=config.name,
        benchmark_package=None if config.case is None else config.case.benchmark_package,
        validation_suite=None if config.validation is None else config.validation.suite_name,
        status=status,
        results=results,
    )


def build_run_manifest(config, bundle: DiagnosticsBundle, report: ValidationReport) -> RunManifest:
    """Create a typed run manifest from a config and diagnostics bundle."""

    metrics = {
        name: MetricSummary(
            value=value,
            provenance=_metric_provenance(bundle, name),
        )
        for name, value in bundle.summary.items()
    }
    inputs = [] if config.case is None else config.case.inputs

    status = report.status
    if status == "not_validated":
        status = _infer_exploratory_status(inputs, bundle)

    return RunManifest(
        case_name=config.name,
        model=config.model,
        run_mode=config.mode,
        config_path=config.config_path or "",
        benchmark_package=None if config.case is None else config.case.benchmark_package,
        validation_suite=None if config.validation is None else config.validation.suite_name,
        input_sources=inputs,
        metrics=metrics,
        validation_status=status,
    )


def _metric_provenance(bundle: DiagnosticsBundle, metric_name: str) -> str | None:
    metric_map = {
        "peak_current_a": "current_a",
        "peak_circuit_current_a": "current_a",
        "peak_model_current_proxy_a": "model_current_proxy_a",
        "peak_target_voltage_v": "target_voltage_v",
        "peak_alpha_t": "alpha_t",
        "final_xi_t": "xi_t",
        "peak_deposition_flux_m2s": "deposition_flux_m2s",
        "peak_electron_density": "electron_density",
        "current_rmse_a": "current_a",
        "current_peak_ratio": "current_a",
        "current_peak_time_error_us": "current_a",
        "current_proxy_rmse_a": "model_current_proxy_a",
        "current_proxy_peak_ratio": "model_current_proxy_a",
        "current_proxy_peak_time_error_us": "model_current_proxy_a",
        "peak_field_energy_j": "field_energy_j",
        "peak_total_energy_j": "total_energy_j",
        "total_target_impacts": "n_target_impacts_step",
        "total_sputtered_atoms": "n_sputtered_step",
        "total_secondary_electrons": "n_see_step",
        "total_collisions": "collisions_per_sample",
    }
    series_name = metric_map.get(metric_name)
    if series_name is None or series_name not in bundle.series:
        return None
    return bundle.series[series_name].provenance


def _infer_exploratory_status(inputs, bundle: DiagnosticsBundle) -> str:
    if any(source.provenance in {"surrogate", "heuristic", "synthetic"} for source in inputs):
        return "exploratory"
    if _has_exploratory_inputs(bundle):
        return "exploratory"
    return "not_validated"


def _has_exploratory_inputs(bundle: DiagnosticsBundle) -> bool:
    return any(
        series.provenance in {"surrogate", "heuristic", "synthetic"}
        for series in bundle.series.values()
        if series.provenance is not None
    )
