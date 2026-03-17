"""Numba CUDA compatibility helpers with a CPU execution stub."""

from __future__ import annotations

import types
from typing import Any

from plasma.runtime.cupy_compat import CUPY_AVAILABLE


def _build_cuda_stub() -> types.ModuleType:
    cuda = types.ModuleType("numba.cuda")
    launch_state = {"index": 0}

    class _Atomic:
        @staticmethod
        def add(arr: Any, idx: Any, value: Any) -> None:
            arr[idx] += value

    class _Kernel:
        def __init__(self, func: Any):
            self.func = func

        def __getitem__(self, config: tuple[int, int]) -> Any:
            blocks, threads = config

            def launcher(*args: Any, **kwargs: Any) -> None:
                total_threads = blocks * threads
                for idx in range(total_threads):
                    launch_state["index"] = idx
                    self.func(*args, **kwargs)

            return launcher

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.func(*args, **kwargs)

    def _jit(*args: Any, **kwargs: Any) -> Any:
        if args and callable(args[0]) and not kwargs:
            return _Kernel(args[0])

        def decorator(func: Any) -> _Kernel:
            return _Kernel(func)

        return decorator

    def _grid(_dim: int) -> int:
        return int(launch_state["index"])

    cuda.jit = _jit
    cuda.grid = _grid
    cuda.atomic = _Atomic()
    return cuda


def _load_cuda() -> tuple[types.ModuleType, bool]:
    if not CUPY_AVAILABLE:
        return _build_cuda_stub(), False

    try:
        from numba import cuda as real_cuda  # type: ignore
    except Exception:
        return _build_cuda_stub(), False

    try:
        if not real_cuda.is_available():
            raise RuntimeError("CUDA runtime unavailable")
    except Exception:
        return _build_cuda_stub(), False

    return real_cuda, True


cuda, CUDA_AVAILABLE = _load_cuda()

