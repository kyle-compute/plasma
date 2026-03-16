"""IO: HDF5 checkpointing and diagnostic snapshots."""

from plasma.io.reports import save_json_model

try:
    from plasma.io.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint
except ModuleNotFoundError:  # pragma: no cover - depends on optional GPU stack
    list_checkpoints = None
    load_checkpoint = None
    save_checkpoint = None

try:
    from plasma.io.hdf5_diagnostics import load_diagnostics, save_diagnostics_snapshot
except ModuleNotFoundError:  # pragma: no cover - depends on optional HDF5 stack
    load_diagnostics = None
    save_diagnostics_snapshot = None

__all__ = [
    "list_checkpoints",
    "load_checkpoint",
    "load_diagnostics",
    "save_checkpoint",
    "save_diagnostics_snapshot",
    "save_json_model",
]
