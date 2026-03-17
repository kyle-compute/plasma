"""Tests for ML dataset generation from stable run artifacts."""

from __future__ import annotations

from plasma.contracts.results import MetricSummary, RunManifest
from plasma.diagnostics.bundles import DiagnosticsBundle, DiagnosticSeriesBundle
from plasma.io.reports import save_json_model
from plasma.ml.dataset import build_scalar_dataset, load_scalar_dataset


def test_build_scalar_dataset_filters_exploratory_runs_and_roundtrips(tmp_path) -> None:
    validated_dir = tmp_path / "validated"
    exploratory_dir = tmp_path / "exploratory"
    validated_dir.mkdir()
    exploratory_dir.mkdir()

    bundle = DiagnosticsBundle(
        model="pic",
        time_s=[0.0, 1.0],
        series={
            "target_voltage_v": DiagnosticSeriesBundle(
                values=[580.0, 600.0],
                unit="V",
                provenance="literature-fit",
                description="Target voltage.",
            )
        },
        summary={"peak_target_voltage_v": 600.0, "total_collisions": 11.0},
    )
    validated = RunManifest(
        case_name="validated_case",
        model="pic",
        run_mode="benchmark",
        config_path="/tmp/validated.yaml",
        benchmark_package="bench",
        validation_suite="suite",
        metrics={"peak_target_voltage_v": MetricSummary(value=600.0, provenance="literature-fit")},
        validation_status="validated",
    )
    exploratory = RunManifest(
        case_name="exploratory_case",
        model="pic",
        run_mode="benchmark",
        config_path="/tmp/exploratory.yaml",
        benchmark_package="bench",
        validation_suite="suite",
        metrics={"peak_target_voltage_v": MetricSummary(value=600.0, provenance="literature-fit")},
        validation_status="exploratory",
    )
    save_json_model(validated_dir / "diagnostics_bundle.json", bundle)
    save_json_model(validated_dir / "run_manifest.json", validated)
    save_json_model(exploratory_dir / "diagnostics_bundle.json", bundle)
    save_json_model(exploratory_dir / "run_manifest.json", exploratory)

    dataset = build_scalar_dataset(
        [validated_dir, exploratory_dir],
        feature_specs=["series.target_voltage_v.max"],
        target_specs=["summary.total_collisions"],
        output_dir=tmp_path / "dataset",
    )

    assert dataset.features.shape == (1, 1)
    assert dataset.targets.shape == (1, 1)
    assert dataset.features[0, 0] == 600.0
    loaded = load_scalar_dataset(tmp_path / "dataset")
    assert loaded.manifest.examples[0].case_name == "validated_case"
    assert loaded.targets[0, 0] == 11.0
