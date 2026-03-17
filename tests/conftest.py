"""Test bootstrap for local src imports and CPU-only CuPy fallback."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_cupy_stub() -> None:
    cupy = types.ModuleType("cupy")
    cupy.ndarray = np.ndarray
    cupy.float64 = np.float64
    cupy.int32 = np.int32
    cupy.asarray = np.asarray
    cupy.asnumpy = np.asarray
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
        def rand(self, *args, dtype=None):  # type: ignore[override]
            values = super().rand(*args)
            return values.astype(dtype) if dtype is not None else values

    cupy.random = types.SimpleNamespace(RandomState=_RandomState)
    cupy.__getattr__ = lambda name: getattr(np, name)
    sys.modules["cupy"] = cupy

    cupyx = types.ModuleType("cupyx")
    cupyx_scipy = types.ModuleType("cupyx.scipy")
    cupyx_sparse = types.ModuleType("cupyx.scipy.sparse")
    cupyx_sparse.csr_matrix = csr_matrix
    cupyx_sparse_linalg = types.ModuleType("cupyx.scipy.sparse.linalg")
    cupyx_sparse_linalg.spsolve = spsolve

    sys.modules["cupyx"] = cupyx
    sys.modules["cupyx.scipy"] = cupyx_scipy
    sys.modules["cupyx.scipy.sparse"] = cupyx_sparse
    sys.modules["cupyx.scipy.sparse.linalg"] = cupyx_sparse_linalg


def _install_numba_stub() -> None:
    numba = types.ModuleType("numba")
    cuda = types.ModuleType("numba.cuda")
    launch_state = {"index": 0}

    class _Atomic:
        @staticmethod
        def add(arr, idx, value):
            arr[idx] += value

    class _Kernel:
        def __init__(self, func):
            self.func = func

        def __getitem__(self, config):
            blocks, threads = config

            def launcher(*args, **kwargs):
                total_threads = blocks * threads
                for idx in range(total_threads):
                    launch_state["index"] = idx
                    self.func(*args, **kwargs)

            return launcher

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    def _jit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return _Kernel(args[0])

        def decorator(func):
            return _Kernel(func)

        return decorator

    def _grid(_dim):
        return launch_state["index"]

    cuda.jit = _jit
    cuda.grid = _grid
    cuda.atomic = _Atomic()
    numba.cuda = cuda
    sys.modules["numba"] = numba
    sys.modules["numba.cuda"] = cuda


def _cupy_is_usable() -> bool:
    try:
        import cupy
    except ModuleNotFoundError:
        return False
    try:
        if cupy.cuda.runtime.getDeviceCount() <= 0:
            return False
        _ = cupy.zeros(1, dtype=cupy.float64)
    except Exception:
        return False
    return True


def _numba_cuda_is_usable() -> bool:
    try:
        from numba import cuda
    except ModuleNotFoundError:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


if not _cupy_is_usable():
    _install_cupy_stub()

if not _numba_cuda_is_usable():
    _install_numba_stub()
