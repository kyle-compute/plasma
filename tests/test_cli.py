"""Tests for the public workflow CLI."""

from __future__ import annotations

import json

from plasma.cli import main
from plasma.contracts.results import MetricSummary, RunManifest
from plasma.diagnostics.bundles import DiagnosticsBundle, DiagnosticSeriesBundle
from plasma.io.reports import save_json_model


def _write_run_artifacts(run_dir) -> None:
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
        summary={"peak_target_voltage_v": 600.0, "total_collisions": 12.0},
    )
    manifest = RunManifest(
        case_name="cli_case",
        model="pic",
        run_mode="benchmark",
        config_path="/tmp/cli_case.yaml",
        benchmark_package="bench",
        validation_suite="suite",
        metrics={"peak_target_voltage_v": MetricSummary(value=600.0, provenance="literature-fit")},
        validation_status="validated",
    )
    save_json_model(run_dir / "diagnostics_bundle.json", bundle)
    save_json_model(run_dir / "run_manifest.json", manifest)


def test_cli_builds_dataset_and_ranks_candidates(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    _write_run_artifacts(run_dir)

    dataset_dir = tmp_path / "dataset"
    exit_code = main(
        [
            "dataset-build",
            str(dataset_dir),
            str(run_dir),
            "--feature",
            "series.target_voltage_v.max",
            "--target",
            "summary.total_collisions",
        ]
    )
    assert exit_code == 0
    assert (dataset_dir / "training_dataset_manifest.json").exists()

    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"metal_delivery_efficiency": 0.7, "peak_current_a": 80.0},
                {"metal_delivery_efficiency": 0.8, "peak_current_a": 55.0},
            ]
        )
    )
    ranked_path = tmp_path / "ranked.json"
    exit_code = main(
        [
            "rank-candidates",
            str(predictions_path),
            "--objective",
            "metal_delivery_efficiency:maximize:1.0",
            "--constraint",
            "peak_current_a::60",
            "--output",
            str(ranked_path),
        ]
    )
    assert exit_code == 0
    ranked = json.loads(ranked_path.read_text())
    assert ranked[0]["index"] == 1


def test_cli_surrogate_train_and_predict(tmp_path) -> None:
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    _write_run_artifacts(run_dir)
    dataset_dir = tmp_path / "dataset"
    main(
        [
            "dataset-build",
            str(dataset_dir),
            str(run_dir),
            "--feature",
            "series.target_voltage_v.max",
            "--target",
            "summary.total_collisions",
        ]
    )

    model_dir = tmp_path / "model"
    exit_code = main(
        [
            "surrogate-train",
            str(dataset_dir),
            str(model_dir),
            "--epochs",
            "20",
            "--hidden",
            "8",
        ]
    )
    assert exit_code == 0
    assert (model_dir / "surrogate_model.pt").exists()

    features_path = tmp_path / "features.json"
    features_path.write_text(json.dumps([[600.0]]))
    predictions_path = tmp_path / "predictions_out.json"
    exit_code = main(
        [
            "surrogate-predict",
            str(model_dir),
            str(features_path),
            "--output",
            str(predictions_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(predictions_path.read_text())
    assert payload["feature_names"] == ["series.target_voltage_v.max"]
    assert payload["target_names"] == ["summary.total_collisions"]
    assert len(payload["predictions"]) == 1
