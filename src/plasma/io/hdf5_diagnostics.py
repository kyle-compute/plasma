"""HDF5 diagnostic snapshots: time-series of fields and distributions."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


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
) -> None:
    """Append a diagnostics snapshot to an HDF5 time-series file.

    Each snapshot is stored under /snapshots/{step:06d}/.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as f:
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


def load_diagnostics(path: str | Path) -> dict[int, dict]:
    """Load all diagnostic snapshots from HDF5 file.

    Returns:
        Dict keyed by step number, each value a dict of arrays.
    """
    path = Path(path)
    result: dict[int, dict] = {}

    with h5py.File(path, "r") as f:
        if "snapshots" not in f:
            return result
        for step_key in sorted(f["snapshots"]):
            grp = f["snapshots"][step_key]
            step = int(grp.attrs["step"])
            snap: dict = {"time": float(grp.attrs["time"])}
            for dset_name in grp:
                snap[dset_name] = np.array(grp[dset_name])
            result[step] = snap

    return result
