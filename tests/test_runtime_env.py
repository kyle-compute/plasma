from __future__ import annotations

from pathlib import Path

from plasma.core.runtime_env import ensure_runtime_environment


def test_ensure_runtime_environment_falls_back_when_home_is_unwritable(tmp_path, monkeypatch):
    fallback_root = tmp_path / "runtime-home"
    monkeypatch.setenv("HOME", "/")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("PLASMA_RUNTIME_HOME", str(fallback_root))

    env = ensure_runtime_environment(tmp_path)

    assert env["HOME"] == str(fallback_root)
    assert Path(env["MPLCONFIGDIR"]).is_dir()
    assert Path(env["CUPY_CACHE_DIR"]).is_dir()
    assert Path(env["NUMBA_CACHE_DIR"]).is_dir()


def test_ensure_runtime_environment_preserves_writable_env_dirs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = home / ".config"
    cache = home / ".cache"
    mpl = config / "matplotlib"
    cupy = cache / "cupy"
    numba = cache / "numba"
    for path in (home, config, cache, mpl, cupy, numba):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl))
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cupy))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(numba))

    env = ensure_runtime_environment(tmp_path)

    assert env["HOME"] == str(home)
    assert env["MPLCONFIGDIR"] == str(mpl)
    assert env["CUPY_CACHE_DIR"] == str(cupy)
    assert env["NUMBA_CACHE_DIR"] == str(numba)
