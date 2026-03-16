"""HDF5 checkpointing: save and restore full PIC simulation state."""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import h5py
import numpy as np


def save_checkpoint(
    path: str | Path,
    step: int,
    time: float,
    grid,
    species_dict: dict,
    phi: cp.ndarray | None = None,
    br_grid: cp.ndarray | np.ndarray | None = None,
    bz_grid: cp.ndarray | np.ndarray | None = None,
    metadata: dict | None = None,
) -> None:
    """Save simulation state to HDF5 checkpoint.

    Args:
        path: Output file path.
        step: Current timestep number.
        time: Current simulation time [s].
        grid: CylindricalGrid instance.
        species_dict: Dict of {name: ParticleArray}.
        phi: Potential on grid nodes.
        br_grid, bz_grid: Magnetic field arrays.
        metadata: Optional extra metadata dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Metadata
        meta = f.create_group("meta")
        meta.attrs["step"] = step
        meta.attrs["time"] = time
        meta.attrs["grid_nr"] = grid.nr
        meta.attrs["grid_nz"] = grid.nz
        meta.attrs["grid_r_max"] = grid.r_max
        meta.attrs["grid_z_max"] = grid.z_max
        if metadata:
            for k, v in metadata.items():
                meta.attrs[k] = v

        # Fields
        fields = f.create_group("fields")
        if phi is not None:
            phi_np = cp.asnumpy(phi) if isinstance(phi, cp.ndarray) else phi
            fields.create_dataset("phi", data=phi_np)
        if br_grid is not None:
            br = cp.asnumpy(br_grid) if isinstance(br_grid, cp.ndarray) else br_grid
            fields.create_dataset("Br", data=br)
        if bz_grid is not None:
            bz = cp.asnumpy(bz_grid) if isinstance(bz_grid, cp.ndarray) else bz_grid
            fields.create_dataset("Bz", data=bz)

        # Particles — compact before saving
        particles_grp = f.create_group("particles")
        for name, particles in species_dict.items():
            particles.compact()
            data = particles.to_numpy()
            sp_grp = particles_grp.create_group(name)
            for key, arr in data.items():
                sp_grp.create_dataset(key, data=arr)

            # Species metadata as attributes
            sp_grp.attrs["charge"] = particles.species.charge
            sp_grp.attrs["mass"] = particles.species.mass
            sp_grp.attrs["charge_state"] = particles.species.charge_state
            sp_grp.attrs["n_alive"] = particles.n_alive


def load_checkpoint(path: str | Path) -> dict:
    """Load simulation state from HDF5 checkpoint.

    Returns:
        Dict with keys:
            step, time, grid_params (dict),
            phi, Br, Bz (numpy arrays or None),
            particles (dict of {name: {r, z, vr, vz, vtheta, weight, species_attrs}})
    """
    path = Path(path)
    result: dict = {}

    with h5py.File(path, "r") as f:
        meta = f["meta"]
        result["step"] = int(meta.attrs["step"])
        result["time"] = float(meta.attrs["time"])
        result["grid_params"] = {
            "nr": int(meta.attrs["grid_nr"]),
            "nz": int(meta.attrs["grid_nz"]),
            "r_max": float(meta.attrs["grid_r_max"]),
            "z_max": float(meta.attrs["grid_z_max"]),
        }

        # Fields
        fields = f["fields"]
        result["phi"] = np.array(fields["phi"]) if "phi" in fields else None
        result["Br"] = np.array(fields["Br"]) if "Br" in fields else None
        result["Bz"] = np.array(fields["Bz"]) if "Bz" in fields else None

        # Particles
        particles = {}
        for name in f["particles"]:
            sp_grp = f["particles"][name]
            sp_data = {}
            for key in ("r", "z", "vr", "vz", "vtheta", "weight"):
                sp_data[key] = np.array(sp_grp[key]) if key in sp_grp else np.empty(0)
            sp_data["species_attrs"] = {
                "charge": float(sp_grp.attrs["charge"]),
                "mass": float(sp_grp.attrs["mass"]),
                "charge_state": int(sp_grp.attrs["charge_state"]),
            }
            particles[name] = sp_data
        result["particles"] = particles

    return result


def list_checkpoints(directory: str | Path) -> list[Path]:
    """List checkpoint files sorted by step number."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = sorted(directory.glob("checkpoint_*.h5"))
    return files
