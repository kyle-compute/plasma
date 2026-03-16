"""Helpers for writable runtime cache/config directories."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_runtime_environment(project_root: str | Path) -> dict[str, str]:
    """Ensure runtime cache directories point to writable locations."""
    root = Path(project_root)
    runtime_root = Path(os.environ.get("PLASMA_RUNTIME_HOME", root / "output" / ".runtime"))
    home = _ensure_env_dir("HOME", runtime_root)
    config_home = _ensure_env_dir("XDG_CONFIG_HOME", home / ".config")
    cache_home = _ensure_env_dir("XDG_CACHE_HOME", home / ".cache")
    mpl_config = _ensure_env_dir("MPLCONFIGDIR", config_home / "matplotlib")
    cupy_cache = _ensure_env_dir("CUPY_CACHE_DIR", cache_home / "cupy")
    numba_cache = _ensure_env_dir("NUMBA_CACHE_DIR", cache_home / "numba")
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_CACHE_HOME": str(cache_home),
        "MPLCONFIGDIR": str(mpl_config),
        "CUPY_CACHE_DIR": str(cupy_cache),
        "NUMBA_CACHE_DIR": str(numba_cache),
    }


def _ensure_env_dir(name: str, fallback: Path) -> Path:
    current = os.environ.get(name)
    if current:
        current_path = Path(current).expanduser()
        try:
            _prepare_writable_dir(current_path)
            os.environ[name] = str(current_path)
            return current_path
        except OSError:
            pass
    fallback_path = Path(fallback).expanduser()
    _prepare_writable_dir(fallback_path)
    os.environ[name] = str(fallback_path)
    return fallback_path


def _prepare_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK | os.X_OK):
        raise PermissionError(f"{path} is not writable")
