"""HDF5 diagnostic snapshots: time-series of fields and distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _require_h5py() -> Any:
    import h5py

    return h5py


def save_diagnostics_snapshot(
    path: str | Path,
    step: int,
    time: float,
    phi: NDArray | None = None,
    n_e: NDArray | None = None,
    n_i: NDArray | None = None,
    te_ev: NDArray | None = None,
    iedf: tuple[NDArray, NDArray] | None = None,
    eedf: tuple[NDArray, NDArray] | None = None,
    background_state: dict[str, float] | None = None,
    collision_counts: dict[str, int] | None = None,
) -> None:
    """Append a diagnostics snapshot to an HDF5 time-series file.

    Each snapshot is stored under /snapshots/{step:06d}/.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h5py_module = _require_h5py()
    with h5py_module.File(path, "a") as f:
        key = f"snapshots/{step:06d}"
        grp = f.create_group(key)
        grp.attrs["step"] = step
        grp.attrs["time"] = time

        if phi is not None:
            grp.create_dataset("phi", data=phi)
        if n_e is not None:
            grp.create_dataset("n_e", data=n_e)
        if n_i is not None:
            grp.create_dataset("n_i", data=n_i)
        if te_ev is not None:
            grp.create_dataset("te_ev", data=te_ev)
        if iedf is not None:
            grp.create_dataset("iedf_energy", data=iedf[0])
            grp.create_dataset("iedf_counts", data=iedf[1])
        if eedf is not None:
            grp.create_dataset("eedf_energy", data=eedf[0])
            grp.create_dataset("eedf_f", data=eedf[1])
        if background_state is not None:
            bg_grp = grp.create_group("background_state")
            for name, density in background_state.items():
                bg_grp.attrs[name] = float(density)
        if collision_counts is not None:
            counts_grp = grp.create_group("collision_counts")
            for name, count in collision_counts.items():
                counts_grp.attrs[name] = int(count)


def load_diagnostics(path: str | Path) -> dict[int, dict]:
    """Load all diagnostic snapshots from HDF5 file.

    Returns:
        Dict keyed by step number, each value a dict of arrays.
    """
    path = Path(path)
    result: dict[int, dict] = {}

    h5py_module = _require_h5py()
    with h5py_module.File(path, "r") as f:
        if "snapshots" not in f:
            return result
        for step_key in sorted(f["snapshots"]):
            grp = f["snapshots"][step_key]
            step = int(grp.attrs["step"])
            snap: dict = {"time": float(grp.attrs["time"])}
            for dset_name in grp:
                if dset_name == "background_state":
                    snap[dset_name] = {
                        str(name): float(value)
                        for name, value in grp[dset_name].attrs.items()
                    }
                    continue
                if dset_name == "collision_counts":
                    snap[dset_name] = {
                        str(name): int(value)
                        for name, value in grp[dset_name].attrs.items()
                    }
                    continue
                snap[dset_name] = np.array(grp[dset_name])
            result[step] = snap

    return result
