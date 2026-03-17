"""Pydantic configuration schema for plasma simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from plasma.contracts.cases import CaseMetadata, RunMode, ValidationConfig
from plasma.core.run_mode import validate_run_mode
from plasma.data.benchmark_packages import load_benchmark_package
from plasma.data.material_packages import load_material_package
from plasma.data.surface_packages import load_surface_package_fragment


class GeometryConfig(BaseModel):
    """Ionization region / magnetron geometry."""

    r_target: float = Field(description="Target radius [m]")
    r_inner: float = Field(description="Inner race-track radius [m]")
    r_outer: float = Field(description="Outer race-track radius [m]")
    z_ir: float = Field(description="Ionization region height [m]")
    area_target: float | None = Field(None, description="Target erosion area [m]")
    volume_ir: float | None = Field(None, description="Ionization region volume [m^3]")


class GasConfig(BaseModel):
    """Background gas parameters."""

    species: str = "Ar"
    pressure_pa: float = Field(description="Gas pressure [Pa]")
    temperature_k: float = Field(300.0, description="Gas temperature [K]")


class TargetConfig(BaseModel):
    """Sputtering target material."""

    material: str = "Cu"
    cohesive_energy_ev: float = Field(description="Surface binding energy [eV]")
    sputter_yield_a: float = Field(description="Yamamura fit parameter a")
    sputter_yield_b: float = Field(description="Yamamura fit parameter b")
    self_sputter_yield_a: float = Field(description="Self-sputter Yamamura a")
    self_sputter_yield_b: float = Field(description="Self-sputter Yamamura b")
    secondary_electron_yield: float = Field(0.1, description="Ion SEE coefficient")


class PulseConfig(BaseModel):
    """HiPIMS pulse parameters."""

    voltage_v: float = Field(description="Discharge voltage [V]")
    t_pulse_us: float = Field(description="Pulse-on time [us]")
    t_afterglow_us: float = Field(description="Afterglow time [us]")
    frequency_hz: float | None = Field(None, description="Repetition frequency [Hz]")
    waveform_file: str | None = Field(None, description="Path to V_D(t), I_D(t) CSV")


class NumericsConfig(BaseModel):
    """Numerical parameters for the 0D global model."""

    t_start: float = Field(0.0, description="Simulation start time [s]")
    t_end: float = Field(description="Simulation end time [s]")
    dt_max: float = Field(1e-8, description="Maximum timestep for ODE solver [s]")
    rtol: float = Field(1e-6, description="Relative tolerance")
    atol: float = Field(1e-10, description="Legacy scalar absolute tolerance")
    density_atol: float = Field(1e8, description="Absolute tolerance for density states")
    current_atol: float = Field(1e-3, description="Absolute tolerance for circuit current state [A]")
    energy_atol: float = Field(1e12, description="Absolute tolerance for energy state")
    solver: Literal["BDF", "Radau", "RK45", "LSODA"] = Field("BDF")


class CircuitConfig(BaseModel):
    """External-circuit parameters for the 0D discharge model."""

    mode: Literal["waveform_current", "rl"] = Field(
        "rl",
        description="Use the literature waveform current directly or solve an RL discharge circuit.",
    )
    series_resistance_ohm: float = Field(6.0, description="Series resistance external to the plasma [ohm]")
    series_inductance_h: float = Field(15e-6, description="Series inductance of the pulser/cable loop [H]")
    plasma_resistance_floor_ohm: float = Field(0.5, description="Lower clamp for plasma resistance [ohm]")
    plasma_resistance_ceiling_ohm: float = Field(500.0, description="Upper clamp for plasma resistance [ohm]")
    electron_neutral_collision_hz: float = Field(
        4e7,
        description="Effective electron-neutral momentum transfer frequency [Hz]",
    )
    electron_conductivity_scale: float = Field(
        1.0,
        description="Empirical conductivity multiplier for benchmark tuning.",
    )


class PICNumericsConfig(BaseModel):
    """Numerical parameters for 2D PIC-MCC."""

    nr: int = Field(description="Number of radial cells")
    nz: int = Field(description="Number of axial cells")
    dt: float = Field(description="Timestep [s]")
    n_steps: int = Field(description="Total number of timesteps")
    particles_per_cell: int = Field(50, description="Initial macro-particles per cell")
    permittivity_factor: float = Field(1.0, description="Artificial permittivity scaling")
    checkpoint_interval: int = Field(1000, description="Steps between HDF5 checkpoints")


class OutputConfig(BaseModel):
    """Persisted output locations for a run."""

    dir: str = Field("output", description="Directory for plots, manifests, and diagnostics")


class SimulationConfig(BaseModel):
    """Top-level simulation configuration."""

    name: str = Field(description="Simulation case name")
    model: Literal["global", "pic"] = Field(description="Which model to run")
    mode: RunMode = Field("benchmark", description="Run intent: smoke, benchmark, or research")
    geometry: GeometryConfig
    gas: GasConfig
    target: TargetConfig
    pulse: PulseConfig
    circuit: CircuitConfig = Field(default_factory=CircuitConfig)
    numerics: NumericsConfig | None = None
    pic_numerics: PICNumericsConfig | None = None
    material_package: str | None = Field(None, description="Path to a material-system package YAML")
    surface_package: str | None = Field(None, description="Path to a surface-interaction package YAML")
    reactions_file: str = Field(description="Path to reactions YAML")
    cross_sections_dir: str | None = Field(None, description="Path to cross-section TSVs")
    benchmark_package: str | None = Field(None, description="Path to a benchmark package YAML")
    case: CaseMetadata | None = None
    validation: ValidationConfig | None = None
    output: OutputConfig | None = None
    config_path: str | None = Field(None, description="Resolved path to the source config")


def load_config(path: str | Path) -> SimulationConfig:
    """Load, merge, resolve, and validate a simulation config."""

    config_path = Path(path).resolve()
    raw = _load_raw_config(config_path)
    raw = _merge_material_package(raw, config_path.parent, model="global")
    raw = _merge_benchmark_package(raw, config_path.parent, model="global")
    raw = _merge_surface_package(raw, config_path.parent)
    _resolve_relative_paths(raw, config_path.parent)
    raw["config_path"] = str(config_path)
    if raw.get("material_package"):
        raw["material_package"] = str(_resolve_path(config_path.parent, raw["material_package"]))
    if raw.get("surface_package"):
        raw["surface_package"] = str(_resolve_path(config_path.parent, raw["surface_package"]))
    if raw.get("benchmark_package"):
        raw["benchmark_package"] = str(_resolve_path(config_path.parent, raw["benchmark_package"]))
    config = SimulationConfig(**raw)
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
    for key in ("reactions_file", "cross_sections_dir", "benchmark_package", "material_package", "surface_package"):
        if raw.get(key):
            raw[key] = str(_resolve_path(base_dir, raw[key]))

    pulse = raw.get("pulse", {})
    if pulse.get("waveform_file"):
        pulse["waveform_file"] = str(_resolve_path(base_dir, pulse["waveform_file"]))

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


def _merge_benchmark_package(raw: dict[str, Any], base_dir: Path, *, model: Literal["global", "pic"]) -> dict[str, Any]:
    if not raw.get("benchmark_package"):
        return raw
    package_path = _resolve_path(base_dir, raw["benchmark_package"])
    package_raw = load_benchmark_package(package_path, model=model)
    merged = _deep_merge(package_raw, raw)
    merged["benchmark_package"] = str(package_path)
    return merged


def _merge_material_package(raw: dict[str, Any], base_dir: Path, *, model: Literal["global", "pic"]) -> dict[str, Any]:
    if not raw.get("material_package"):
        return raw
    package_path = _resolve_path(base_dir, raw["material_package"])
    package_raw = load_material_package(package_path, model=model)
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
