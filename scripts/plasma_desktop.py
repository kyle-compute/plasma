#!/usr/bin/env python3
"""Entrypoint wrapper for the local desktop launcher."""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plasma.desktop_launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
