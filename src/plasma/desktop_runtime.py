"""Runtime helpers for the YAML-driven desktop launcher."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from plasma.desktop_config import DesktopDurationConfig, DesktopLauncherConfig, DesktopProfileConfig


def project_root() -> Path:
    """Return the repo root for this workspace."""

    return Path(__file__).resolve().parents[2]


def default_python_executable(root: Path) -> str:
    """Prefer the local project venv when it exists."""

    venv_python = root / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable).resolve())


def find_profile(config: DesktopLauncherConfig, key: str) -> DesktopProfileConfig:
    """Resolve one configured launcher profile by key."""

    return config.profile_map()[key]


def find_duration(config: DesktopLauncherConfig, key: str) -> DesktopDurationConfig:
    """Resolve one configured launcher duration by key."""

    return config.duration_map()[key]


def build_run_name(config: DesktopLauncherConfig, profile_key: str, n_steps: int) -> str:
    """Render the generated run name from launcher config."""

    return config.generation.run_name_template.format(profile=profile_key, steps=n_steps)


def write_launcher_config(
    root: Path,
    launcher_config: DesktopLauncherConfig,
    profile: DesktopProfileConfig,
    *,
    n_steps: int,
    run_name: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write one generated YAML config for a launcher profile."""

    generation = launcher_config.generation
    runtime_dir = (root / generation.runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or build_run_name(launcher_config, profile.key, n_steps)
    output_dir = (root / generation.output_root / name).resolve()
    live_dir = (root / generation.live_root / name).resolve()
    config_path = runtime_dir / f"{name}.yaml"
    payload = {
        "base": str(Path(profile.base_config).resolve()),
        "name": name,
        "time": {
            "n_steps": int(n_steps),
            "diag_interval": max(generation.diag_min, int(n_steps) // generation.diag_divisor),
            "compact_interval": max(generation.compact_min, int(n_steps) // generation.compact_divisor),
            "checkpoint_interval": generation.checkpoint_interval,
        },
        "output": {"dir": str(output_dir)},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path, output_dir, live_dir


def build_run_command(root: Path, python_exe: str, config_path: Path, live_dir: Path) -> list[str]:
    """Build the PIC run command for one generated config."""

    return [python_exe, str(root / "scripts" / "run_pic.py"), str(config_path), "--live-dir", str(live_dir)]


def build_viewer_command(root: Path, python_exe: str, live_dir: Path) -> list[str]:
    """Build the live-viewer command for one run."""

    return [python_exe, str(root / "scripts" / "live_viewer.py"), str(live_dir)]
