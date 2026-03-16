"""Convenience wrapper for running the 0D IRM."""

from __future__ import annotations

from pathlib import Path

from plasma.core.config import load_config
from plasma.global_model.irm import IRM, IRMState


def run_global_model(config_path: str | Path) -> IRMState:
    """Load config, build IRM, and run simulation.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        IRMState with time-series of all species densities and T_e.
    """
    cfg = load_config(config_path)
    irm = IRM(cfg)
    return irm.run()
