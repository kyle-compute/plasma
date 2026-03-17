"""Load named benchmark packages and merge them into runtime configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml


def load_benchmark_package(path: str | Path, *, model: Literal["global", "pic"]) -> dict[str, Any]:
    """Load a benchmark package and return config fragments for one model."""

    package_path = Path(path).resolve()
    with open(package_path) as handle:
        raw = yaml.safe_load(handle)

    package = raw.get("package", {})
    merged: dict[str, Any] = {}
    model_block = raw.get(model, {})
    if model_block:
        merged = _deep_merge(merged, model_block)

    case = {
        "benchmark": package.get("benchmark"),
        "material_system": package.get("material_system"),
        "benchmark_package": package.get("name"),
        "benchmark_version": package.get("version"),
        "inputs": raw.get("inputs", []),
        "notes": package.get("notes"),
    }
    merged["case"] = _deep_merge(case, merged.get("case", {}))
    merged.setdefault("validation", raw.get("validation", {}))
    _resolve_paths(merged, package_path.parent)
    return merged


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> None:
    if raw.get("reactions_file"):
        raw["reactions_file"] = str(_resolve_path(base_dir, raw["reactions_file"]))
    if raw.get("collision_package"):
        raw["collision_package"] = str(_resolve_path(base_dir, raw["collision_package"]))

    pulse = raw.get("pulse", {})
    if pulse.get("waveform_file"):
        pulse["waveform_file"] = str(_resolve_path(base_dir, pulse["waveform_file"]))

    magnetic_field = raw.get("magnetic_field", {})
    if magnetic_field.get("map_file"):
        magnetic_field["map_file"] = str(_resolve_path(base_dir, magnetic_field["map_file"]))

    cross_sections = raw.get("cross_sections", {})
    for key in ("elastic_file", "excitation_file", "ionization_file", "manifest_file"):
        if cross_sections.get(key):
            cross_sections[key] = str(_resolve_path(base_dir, cross_sections[key]))

    case = raw.get("case", {})
    for item in case.get("inputs", []):
        if item.get("path"):
            item["path"] = str(_resolve_path(base_dir, item["path"]))


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
