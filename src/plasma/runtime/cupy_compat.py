"""CuPy compatibility helpers with a NumPy/scipy fallback."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse.linalg import spsolve as scipy_spsolve


def _build_cupy_stub() -> types.ModuleType:
    cupy = types.ModuleType("cupy")
    cupy.ndarray = np.ndarray
    cupy.float64 = np.float64
    cupy.int32 = np.int32
    cupy.asarray = np.asarray
    cupy.asnumpy = np.asarray
    cupy.array = np.array
    cupy.zeros = np.zeros
    cupy.ones = np.ones
    cupy.empty = np.empty
    cupy.full = np.full
    cupy.zeros_like = np.zeros_like
    cupy.sum = np.sum
    cupy.maximum = np.maximum
    cupy.where = np.where
    cupy.sqrt = np.sqrt

    class _RandomState(np.random.RandomState):
        def rand(self, *args: Any, dtype: Any = None) -> np.ndarray:  # type: ignore[override]
            values = super().rand(*args)
            return values.astype(dtype) if dtype is not None else values

    cupy.random = types.SimpleNamespace(RandomState=_RandomState)
    cupy.__getattr__ = lambda name: getattr(np, name)
    return cupy


def _load_cupy() -> tuple[types.ModuleType, bool]:
    try:
        import cupy as real_cupy  # type: ignore
    except Exception:
        return _build_cupy_stub(), False

    try:
        if real_cupy.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("No CUDA devices detected")
        _ = real_cupy.zeros(1, dtype=real_cupy.float64)
    except Exception:
        return _build_cupy_stub(), False

    return real_cupy, True


cp, CUPY_AVAILABLE = _load_cupy()

if CUPY_AVAILABLE:
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    from cupyx.scipy.sparse.linalg import cg as gpu_cg
    from cupyx.scipy.sparse.linalg import spsolve as gpu_spsolve

    try:
        from cupyx.scipy.sparse.linalg import LinearOperator as GpuLinearOperator
    except ImportError:
        from scipy.sparse.linalg import LinearOperator as GpuLinearOperator
else:
    gpu_csr_matrix = scipy_csr_matrix
    gpu_spsolve = scipy_spsolve
    from scipy.sparse.linalg import cg as gpu_cg
    from scipy.sparse.linalg import LinearOperator as GpuLinearOperator

