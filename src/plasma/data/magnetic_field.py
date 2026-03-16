"""Magnetic field map loaders for PIC benchmark cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_magnetic_field_map(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Br and Bz arrays from a YAML or NPZ field map."""

    source = Path(path)
    if source.suffix == ".npz":
        data = np.load(source)
        return np.asarray(data["Br"], dtype=np.float64), np.asarray(data["Bz"], dtype=np.float64)
    if source.suffix in {".yaml", ".yml"}:
        with open(source) as handle:
            data = yaml.safe_load(handle)
        return np.asarray(data["Br"], dtype=np.float64), np.asarray(data["Bz"], dtype=np.float64)
    raise ValueError(f"Unsupported magnetic field map format: {source.suffix}")


def validate_field_map_shape(
    br_grid: np.ndarray,
    bz_grid: np.ndarray,
    *,
    n_r: int,
    n_z: int,
) -> None:
    """Ensure a field map matches the expected node layout."""

    expected = (n_r, n_z)
    if br_grid.shape != expected or bz_grid.shape != expected:
        raise ValueError(
            f"Expected field map shape {expected}, got Br={br_grid.shape}, Bz={bz_grid.shape}",
        )
