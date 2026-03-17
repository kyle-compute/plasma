"""Stable ML dataset builder over run manifests and diagnostics bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from plasma.contracts.results import RunManifest
from plasma.diagnostics.bundles import DiagnosticsBundle
from plasma.io.reports import save_json_model


class ScalarDatasetExample(BaseModel):
    """One scalar-training example derived from stable run artifacts."""

    run_dir: str
    case_name: str
    benchmark_package: str | None = None
    validation_status: Literal["validated", "exploratory", "failed", "not_validated"]
    features: dict[str, float] = Field(default_factory=dict)
    targets: dict[str, float] = Field(default_factory=dict)


class TrainingDatasetManifest(BaseModel):
    """Machine-readable manifest for one saved scalar dataset."""

    name: str
    tier: Literal["scalar_run_summaries"] = "scalar_run_summaries"
    feature_specs: list[str]
    target_specs: list[str]
    examples: list[ScalarDatasetExample] = Field(default_factory=list)


@dataclass
class ScalarDataset:
    """In-memory scalar dataset and its manifest."""

    manifest: TrainingDatasetManifest
    features: np.ndarray
    targets: np.ndarray


def build_scalar_dataset(
    run_dirs: list[str | Path],
    *,
    feature_specs: list[str],
    target_specs: list[str],
    dataset_name: str = "scalar_dataset",
    include_exploratory: bool = False,
    output_dir: str | Path | None = None,
) -> ScalarDataset:
    """Build a scalar dataset from saved run directories."""

    examples: list[ScalarDatasetExample] = []
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []

    for run_dir in [Path(value).resolve() for value in run_dirs]:
        manifest = _load_manifest(run_dir / "run_manifest.json")
        bundle = _load_bundle(run_dir / "diagnostics_bundle.json")
        if not include_exploratory and manifest.validation_status == "exploratory":
            continue
        features = {spec: _resolve_metric_spec(spec, manifest, bundle) for spec in feature_specs}
        targets = {spec: _resolve_metric_spec(spec, manifest, bundle) for spec in target_specs}
        examples.append(
            ScalarDatasetExample(
                run_dir=str(run_dir),
                case_name=manifest.case_name,
                benchmark_package=manifest.benchmark_package,
                validation_status=manifest.validation_status,
                features=features,
                targets=targets,
            )
        )
        feature_rows.append([features[spec] for spec in feature_specs])
        target_rows.append([targets[spec] for spec in target_specs])

    dataset = ScalarDataset(
        manifest=TrainingDatasetManifest(
            name=dataset_name,
            feature_specs=feature_specs,
            target_specs=target_specs,
            examples=examples,
        ),
        features=np.asarray(feature_rows, dtype=np.float64),
        targets=np.asarray(target_rows, dtype=np.float64),
    )
    if output_dir is not None:
        _save_dataset(Path(output_dir), dataset)
    return dataset


def load_scalar_dataset(directory: str | Path) -> ScalarDataset:
    """Load a saved scalar dataset from disk."""

    target_dir = Path(directory).resolve()
    manifest = TrainingDatasetManifest.model_validate_json((target_dir / "training_dataset_manifest.json").read_text())
    payload = np.load(target_dir / "scalar_dataset.npz")
    return ScalarDataset(
        manifest=manifest,
        features=np.asarray(payload["features"], dtype=np.float64),
        targets=np.asarray(payload["targets"], dtype=np.float64),
    )


def _save_dataset(directory: Path, dataset: ScalarDataset) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_json_model(directory / "training_dataset_manifest.json", dataset.manifest)
    np.savez_compressed(
        directory / "scalar_dataset.npz",
        features=dataset.features,
        targets=dataset.targets,
        feature_specs=np.asarray(dataset.manifest.feature_specs),
        target_specs=np.asarray(dataset.manifest.target_specs),
    )


def _load_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text())


def _load_bundle(path: Path) -> DiagnosticsBundle:
    return DiagnosticsBundle.model_validate_json(path.read_text())


def _resolve_metric_spec(spec: str, manifest: RunManifest, bundle: DiagnosticsBundle) -> float:
    """Resolve one scalar spec against a diagnostics bundle and run manifest."""

    if spec.startswith("summary."):
        key = spec.split(".", 1)[1]
        return float(bundle.summary[key])
    if spec.startswith("series."):
        _, series_name, reducer = spec.split(".", 2)
        return _reduce_series(bundle.series[series_name].values, reducer)
    if spec in bundle.summary:
        return float(bundle.summary[spec])
    if spec in bundle.series:
        return _reduce_series(bundle.series[spec].values, "last")
    if spec in manifest.metrics:
        return float(manifest.metrics[spec].value)
    raise KeyError(f"Metric spec '{spec}' not present in run artifacts.")


def _reduce_series(values: list[float], reducer: str) -> float:
    data = np.asarray(values, dtype=np.float64)
    if reducer == "last":
        return float(data[-1])
    if reducer == "max":
        return float(np.max(data))
    if reducer == "min":
        return float(np.min(data))
    if reducer == "mean":
        return float(np.mean(data))
    raise ValueError(f"Unsupported scalar reducer '{reducer}'")
