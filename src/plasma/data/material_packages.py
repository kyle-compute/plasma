"""Typed material-package loader for shared Cu/Ar and Ti/Ar defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MaterialPackageMetadata(BaseModel):
    """Identity and provenance for one material package."""

    name: str
    version: str
    material_system: str
    target_material: str
    provenance: str = "model-derived"
    notes: str | None = None


class MaterialPackage(BaseModel):
    """Config fragments shared across 0D and PIC runs."""

    model_config = ConfigDict(populate_by_name=True)

    package: MaterialPackageMetadata
    shared: dict[str, Any] = Field(default_factory=dict)
    global_config: dict[str, Any] = Field(default_factory=dict, alias="global")
    pic: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None

    def fragment(self, *, model: Literal["global", "pic"]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged = _deep_merge(merged, self.shared)
        merged = _deep_merge(merged, self.global_config if model == "global" else self.pic)
        merged.setdefault("target", {})
        merged["target"].setdefault("material", self.package.target_material)
        return merged


def load_material_package(path: str | Path, *, model: Literal["global", "pic"]) -> dict[str, Any]:
    """Load one material package and return the config fragment for one model."""

    package_path = Path(path).resolve()
    with open(package_path) as handle:
        raw = yaml.safe_load(handle)

    package = MaterialPackage(**raw)
    package.source_path = str(package_path)
    fragment = package.fragment(model=model)
    fragment["material_package"] = str(package_path)
    _resolve_paths(fragment, package_path.parent)
    return fragment


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> None:
    for key in ("reactions_file", "collision_package", "surface_package", "benchmark_package"):
        if raw.get(key):
            raw[key] = str(_resolve_path(base_dir, raw[key]))


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
