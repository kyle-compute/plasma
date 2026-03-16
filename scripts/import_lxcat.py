#!/usr/bin/env python3
"""Normalize a downloaded LXCat archive into explicit cross-section tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plasma.data.lxcat_import import normalize_biagi_argon_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a LXCat archive")
    parser.add_argument("archive", help="Path to the downloaded LXCat .zip or .txt")
    parser.add_argument("output_dir", help="Directory for normalized .tsv files and manifest")
    args = parser.parse_args()
    manifest = normalize_biagi_argon_archive(args.archive, args.output_dir)
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
