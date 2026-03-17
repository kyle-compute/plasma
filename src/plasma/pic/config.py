"""Typed configuration for PIC HiPIMS runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from plasma.contracts.cases import CaseMetadata, RunMode, ValidationConfig
from plasma.core.config import OutputConfig
from plasma.core.run_mode import validate_run_mode
from plasma.data.benchmark_packages import load_benchmark_package
from plasma.data.material_packages import load_material_package
from plasma.data.surface_packages import load_surface_package_fragment


class PICGeometryConfig(BaseModel):
    r_target: float
    r_inner: float
    r_outer: float
    z_target: float = 0.0
    z_substrate: float = 0.1
    r_max: float


class PICGridConfig(BaseModel):
    nr: int
    nz: int
    permittivity_factor: float = 1.0


class PICGasConfig(BaseModel):
    species: str = "Ar"
    pressure_pa: float
    temperature_k: float = 300.0


class PICTargetConfig(BaseModel):
    material: str = "Cu"
    cohesive_energy_ev: float
    sputter_yield_a: float
    sputter_yield_b: float
    self_sputter_yield_a: float | None = None
    self_sputter_yield_b: float | None = None
    sputter_threshold_ev: float = 20.0
    secondary_electron_yield: float = 0.1
    metal_ion_secondary_electron_yield: float | None = None
    see_energy_ev: float = 3.0


class PICPulseConfig(BaseModel):
    voltage_v: float
    t_pulse_us: float
    t_total_us: float
    rise_time_us: float = 0.1
    waveform_file: str | None = None


class PICParticlesConfig(BaseModel):
    ppc: int = 50
    n0_electron: float = 1e15
    n0_ion: float = 1e15
    n0_metal_ion: float = 0.0
    n0_sputtered_neutral: float = 0.0
    te_ev: float = 3.0
    ti_ev: float = 0.1


class PICMagneticFieldConfig(BaseModel):
    map_file: str | None = None
    inner_magnet_r: float = 0.012
    outer_magnet_r: float = 0.038
    magnet_z: float = -0.005
    current_inner: float = 1000.0
    current_outer: float = -600.0


class PICTimeConfig(BaseModel):
    dt: float
    n_steps: int
    diag_interval: int = 100
    compact_interval: int = 500
    checkpoint_interval: int = 0


class PICCrossSectionConfig(BaseModel):
    source: Literal["synthetic", "normalized_files"] = "synthetic"
    elastic_file: str | None = None
    excitation_file: str | None = None
    ionization_file: str | None = None
    manifest_file: str | None = None


class PICBackgroundModelConfig(BaseModel):
    densities_m3: dict[str, float] = Field(default_factory=dict)


class PICConfig(BaseModel):
    name: str
    model: Literal["pic"]
    mode: RunMode = Field("benchmark")
    geometry: PICGeometryConfig
    grid: PICGridConfig
    gas: PICGasConfig
    target: PICTargetConfig
    pulse: PICPulseConfig
    particles: PICParticlesConfig
    magnetic_field: PICMagneticFieldConfig = Field(default_factory=PICMagneticFieldConfig)
    cross_sections: PICCrossSectionConfig = Field(default_factory=PICCrossSectionConfig)
    material_package: str | None = None
    surface_package: str | None = None
    collision_package: str | None = None
    background_model: PICBackgroundModelConfig = Field(default_factory=PICBackgroundModelConfig)
    time: PICTimeConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    benchmark_package: str | None = None
    case: CaseMetadata | None = None
    validation: ValidationConfig | None = None
    config_path: str | None = None


def load_pic_config(path: str | Path) -> PICConfig:
    config_path = Path(path).resolve()
    raw = _load_raw_config(config_path)
    raw = _merge_material_package(raw, config_path.parent)
    raw = _merge_benchmark_package(raw, config_path.parent)
    raw = _merge_surface_package(raw, config_path.parent)
    _resolve_relative_paths(raw, config_path.parent)
    raw["config_path"] = str(config_path)
    if raw.get("material_package"):
        raw["material_package"] = str(_resolve_path(config_path.parent, raw["material_package"]))
    if raw.get("surface_package"):
        raw["surface_package"] = str(_resolve_path(config_path.parent, raw["surface_package"]))
    if raw.get("benchmark_package"):
        raw["benchmark_package"] = str(_resolve_path(config_path.parent, raw["benchmark_package"]))
    if raw.get("collision_package"):
        raw["collision_package"] = str(_resolve_path(config_path.parent, raw["collision_package"]))
    config = PICConfig(**raw)
    validate_run_mode(config.mode, config.case)
    return config


def _load_raw_config(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if "base" not in raw:
        return raw

    base_path = (path.parent / raw.pop("base")).resolve()
    base_raw = _load_raw_config(base_path)
    return _deep_merge(base_raw, raw)


def _resolve_relative_paths(raw: dict[str, Any], base_dir: Path) -> None:
    if raw.get("material_package"):
        raw["material_package"] = str(_resolve_path(base_dir, raw["material_package"]))
    if raw.get("surface_package"):
        raw["surface_package"] = str(_resolve_path(base_dir, raw["surface_package"]))
    if raw.get("benchmark_package"):
        raw["benchmark_package"] = str(_resolve_path(base_dir, raw["benchmark_package"]))
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


def _merge_benchmark_package(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if not raw.get("benchmark_package"):
        return raw
    package_path = _resolve_path(base_dir, raw["benchmark_package"])
    package_raw = load_benchmark_package(package_path, model="pic")
    merged = _deep_merge(package_raw, raw)
    merged["benchmark_package"] = str(package_path)
    return merged


def _merge_material_package(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if not raw.get("material_package"):
        return raw
    package_path = _resolve_path(base_dir, raw["material_package"])
    package_raw = load_material_package(package_path, model="pic")
    merged = _deep_merge(package_raw, raw)
    merged["material_package"] = str(package_path)
    return merged


def _merge_surface_package(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if not raw.get("surface_package"):
        return raw
    package_path = _resolve_path(base_dir, raw["surface_package"])
    package_raw = load_surface_package_fragment(package_path)
    merged = _deep_merge(package_raw, raw)
    merged["surface_package"] = str(package_path)
    return merged
