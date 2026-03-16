#!/usr/bin/env python3
"""Launch the preferred live plasma viewer with a dark fallback."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    try:
        from plasma.viewer.app import main as qt_main
    except Exception as exc:
        from plasma.viewer.mpl_legacy import main as legacy_main

        print(
            "Qt viewer unavailable; falling back to dark Matplotlib viewer.\n"
            "Install the optional viz extras for the full 2D + 3D desktop UI.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return legacy_main(sys.argv[1:])
    return qt_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
