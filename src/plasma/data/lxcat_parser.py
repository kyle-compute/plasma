"""Parse cross-section data files from LXCat (TSV format).

LXCat files are two-column: energy [eV] and cross-section [m^2].
Lines starting with '#' or containing non-numeric data are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def parse_lxcat_tsv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse a two-column TSV file of (energy_eV, cross_section_m2).

    Returns:
        energy: 1D array of energies [eV], sorted ascending.
        sigma: 1D array of cross-sections [m^2], same length.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cross-section file not found: {path}")

    energies = []
    sigmas = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                e = float(parts[0])
                s = float(parts[1])
                energies.append(e)
                sigmas.append(s)
            except ValueError:
                continue

    if not energies:
        raise ValueError(f"No valid data found in {path}")

    energy = np.array(energies, dtype=np.float64)
    sigma = np.array(sigmas, dtype=np.float64)

    # Ensure sorted by energy
    order = np.argsort(energy)
    return energy[order], sigma[order]


def parse_lxcat_txt(path: str | Path) -> dict[str, tuple[NDArray, NDArray]]:
    """Parse a multi-section LXCat .txt download file."""
    path = Path(path)
    with open(path) as f:
        return parse_lxcat_text(f.read())


def parse_lxcat_text(text: str) -> dict[str, tuple[NDArray, NDArray]]:
    """Parse LXCat text content into named cross-section sections."""

    sections: dict[str, tuple[NDArray, NDArray]] = {}
    current_name = None
    energies: list[float] = []
    sigmas: list[float] = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Section headers in LXCat format
        if line.startswith("PROCESS:"):
            if current_name and energies:
                e = np.array(energies, dtype=np.float64)
                s = np.array(sigmas, dtype=np.float64)
                order = np.argsort(e)
                sections[current_name] = (e[order], s[order])
            current_name = line.split(":", 1)[1].strip()
            energies = []
            sigmas = []
            in_data = False
        elif line == "-----------------------------":
            in_data = True
        elif in_data and line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[0]))
                    sigmas.append(float(parts[1]))
                except ValueError:
                    in_data = False

    # Last section
    if current_name and energies:
        e = np.array(energies, dtype=np.float64)
        s = np.array(sigmas, dtype=np.float64)
        order = np.argsort(e)
        sections[current_name] = (e[order], s[order])

    return sections
