"""Typed desktop-launcher configuration loaded from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class DesktopProfileConfig(BaseModel):
    """One selectable base PIC profile in the desktop launcher."""

    key: str
    title: str
    base_config: str
    notes: str


class DesktopDurationConfig(BaseModel):
    """One selectable runtime duration in the desktop launcher."""

    key: str
    title: str
    n_steps: int = Field(gt=0)
    est_wallclock: str


class DesktopGenerationConfig(BaseModel):
    """Rules for generated run configs and output directories."""

    runtime_dir: str = "output/.desktop"
    output_root: str = "output/pic"
    live_root: str = "output/live"
    run_name_template: str = "desktop_{profile}_{steps}"
    diag_divisor: int = Field(100, gt=0)
    diag_min: int = Field(20, gt=0)
    compact_divisor: int = Field(40, gt=0)
    compact_min: int = Field(100, gt=0)
    checkpoint_interval: int = Field(0, ge=0)


class DesktopLauncherConfig(BaseModel):
    """Top-level YAML schema for the local desktop launcher."""

    window_title: str = "Plasma Desktop Launcher"
    default_profile: str
    default_duration: str
    generation: DesktopGenerationConfig = Field(default_factory=DesktopGenerationConfig)
    profiles: list[DesktopProfileConfig]
    durations: list[DesktopDurationConfig]

    @model_validator(mode="after")
    def validate_defaults(self) -> DesktopLauncherConfig:
        profile_keys = {item.key for item in self.profiles}
        duration_keys = {item.key for item in self.durations}
        if self.default_profile not in profile_keys:
            raise ValueError(f"default_profile '{self.default_profile}' is not defined in profiles")
        if self.default_duration not in duration_keys:
            raise ValueError(f"default_duration '{self.default_duration}' is not defined in durations")
        return self

    def profile_map(self) -> dict[str, DesktopProfileConfig]:
        """Return profiles keyed by launcher key."""

        return {item.key: item for item in self.profiles}

    def duration_map(self) -> dict[str, DesktopDurationConfig]:
        """Return durations keyed by launcher key."""

        return {item.key: item for item in self.durations}


def default_desktop_config_path(root: Path) -> Path:
    """Return the default launcher config path or an env override."""

    override = os.environ.get("PLASMA_DESKTOP_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return (root / "config" / "desktop" / "launcher.yaml").resolve()


def load_desktop_config(path: str | Path) -> DesktopLauncherConfig:
    """Load a desktop launcher config with recursive base support."""

    config_path = Path(path).resolve()
    raw = _load_raw_config(config_path)
    _resolve_relative_paths(raw, config_path.parent)
    return DesktopLauncherConfig(**raw)


def _load_raw_config(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if "base" not in raw:
        return raw
    base_path = (path.parent / raw.pop("base")).resolve()
    return _deep_merge(_load_raw_config(base_path), raw)


def _resolve_relative_paths(raw: dict[str, Any], base_dir: Path) -> None:
    for profile in raw.get("profiles", []):
        if profile.get("base_config"):
            profile["base_config"] = str(_resolve_path(base_dir, profile["base_config"]))


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
