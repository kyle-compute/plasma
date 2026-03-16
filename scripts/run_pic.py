#!/usr/bin/env python3
"""Run HiPIMS PIC-MCC simulation from YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plasma.core.runtime_env import ensure_runtime_environment

ensure_runtime_environment(ROOT)

from plasma.pic.runtime import run_pic_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HiPIMS PIC-MCC simulation")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--live-dir", help="Directory used to publish live-view snapshots")
    parser.add_argument(
        "--live-max-particles",
        type=int,
        default=1200,
        help="Maximum particles per species to stream to the live viewer",
    )
    args = parser.parse_args()
    return run_pic_case(
        args.config,
        output_dir=args.output_dir,
        live_dir=args.live_dir,
        live_max_particles=args.live_max_particles,
    )


if __name__ == "__main__":
    raise SystemExit(main())
