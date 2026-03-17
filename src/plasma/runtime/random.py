"""Deterministic RNG helpers shared across CPU and CUDA fallback paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma.runtime.cupy_compat import cp


def _normalise_size(size: int | tuple[int, ...]) -> tuple[int, ...]:
    return (size,) if isinstance(size, int) else size


@dataclass
class SimulationRNG:
    """Repo-owned RNG wrapper with checkpointable NumPy state."""

    seed: int = 0

    def __post_init__(self) -> None:
        self._state = np.random.RandomState(self.seed)

    def rand(self, *shape: int, dtype: Any = None) -> cp.ndarray:
        values = self._state.rand(*shape)
        if dtype is not None:
            values = values.astype(dtype)
        return cp.asarray(values)

    def uniform_cpu(self, size: int | tuple[int, ...]) -> np.ndarray:
        return np.asarray(self._state.random_sample(_normalise_size(size)), dtype=np.float64)

    def state_dict(self) -> dict[str, Any]:
        algo, keys, pos, has_gauss, cached = self._state.get_state()
        return {
            "type": "numpy_random_state",
            "algorithm": str(algo),
            "keys": np.asarray(keys, dtype=np.uint32).tolist(),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }

    def set_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("type") != "numpy_random_state":
            raise ValueError(f"Unsupported RNG state type '{state.get('type')}'.")
        self._state.set_state(
            (
                str(state["algorithm"]),
                np.asarray(state["keys"], dtype=np.uint32),
                int(state["pos"]),
                int(state["has_gauss"]),
                float(state["cached_gaussian"]),
            )
        )


def uniform_cpu(rng: Any, size: int | tuple[int, ...]) -> np.ndarray:
    """Draw deterministic uniform samples on CPU from any supported RNG."""

    shape = _normalise_size(size)
    if rng is None:
        return np.asarray(np.random.random_sample(shape), dtype=np.float64)
    if hasattr(rng, "uniform_cpu"):
        return np.asarray(rng.uniform_cpu(shape), dtype=np.float64)
    values = rng.rand(*shape, dtype=cp.float64)
    return np.asarray(cp.asnumpy(values), dtype=np.float64)


def uniform_gpu(rng: Any, size: int | tuple[int, ...]) -> cp.ndarray:
    """Draw deterministic uniform samples as a GPU array.

    Uses the same underlying RNG state as uniform_cpu but returns
    a CuPy array, avoiding an extra CPU->GPU copy when the caller
    only needs values on the device.
    """

    shape = _normalise_size(size)
    if rng is None:
        return cp.asarray(np.random.random_sample(shape), dtype=cp.float64)
    return rng.rand(*shape, dtype=cp.float64)


def export_rng_state(rng: Any) -> dict[str, Any] | None:
    """Serialize RNG state when the generator supports it."""

    if rng is None:
        return None
    if hasattr(rng, "state_dict"):
        return rng.state_dict()
    if hasattr(rng, "get_state"):
        algo, keys, pos, has_gauss, cached = rng.get_state()
        return {
            "type": "numpy_random_state",
            "algorithm": str(algo),
            "keys": np.asarray(keys, dtype=np.uint32).tolist(),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }
    return None


def build_rng_from_state(state: dict[str, Any] | None, *, seed: int = 0) -> SimulationRNG:
    """Restore a SimulationRNG from serialized state."""

    rng = SimulationRNG(seed)
    if state is not None:
        rng.set_state_dict(state)
    return rng
