#!/usr/bin/env python3
"""Generate a surrogate magnetron field map from a PIC config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plasma.pic.config import load_pic_config
from plasma.pic.grid import CylindricalGrid
from plasma.pic.magnetic import magnetron_field, save_field


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a surrogate PIC field map")
    parser.add_argument("config", help="PIC YAML config")
    parser.add_argument("output", help="Output .npz field-map path")
    args = parser.parse_args()

    cfg = load_pic_config(args.config)
    grid = CylindricalGrid(cfg.grid.nr, cfg.grid.nz, cfg.geometry.r_max, cfg.geometry.z_substrate)
    br_grid, bz_grid = magnetron_field(
        grid,
        inner_loop_r=cfg.magnetic_field.inner_magnet_r,
        outer_loop_r=cfg.magnetic_field.outer_magnet_r,
        loop_z=cfg.magnetic_field.magnet_z,
        current_inner=cfg.magnetic_field.current_inner,
        current_outer=cfg.magnetic_field.current_outer,
    )
    save_field(args.output, br_grid, bz_grid, r_edges=grid.r_edges, z_edges=grid.z_edges)
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
