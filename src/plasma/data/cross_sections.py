"""Cross-section table with log-log interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plasma.data.lxcat_parser import parse_lxcat_tsv


class CrossSectionTable:
    """Tabulated cross-section sigma(E) with log-log interpolation.

    Cross-sections in plasma physics span many orders of magnitude, so
    log-log interpolation gives much better accuracy than linear interpolation
    for the same number of data points.
    """

    def __init__(self, energy_ev: NDArray, sigma_m2: NDArray, name: str = ""):
        if len(energy_ev) != len(sigma_m2):
            raise ValueError("Energy and sigma arrays must have same length")
        if len(energy_ev) < 2:
            raise ValueError("Need at least 2 data points")

        # Filter out zero/negative values for log interpolation
        mask = (energy_ev > 0) & (sigma_m2 > 0)
        self.energy_ev = energy_ev[mask].copy()
        self.sigma_m2 = sigma_m2[mask].copy()
        self.name = name

        # Pre-compute log values for interpolation
        self._log_energy = np.log(self.energy_ev)
        self._log_sigma = np.log(self.sigma_m2)

        self.e_min = self.energy_ev[0]
        self.e_max = self.energy_ev[-1]

    @classmethod
    def from_file(cls, path: str | Path, name: str = "") -> CrossSectionTable:
        """Load from a two-column TSV file."""
        energy, sigma = parse_lxcat_tsv(path)
        if not name:
            name = Path(path).stem
        return cls(energy, sigma, name)

    def __call__(self, energy_ev: float | NDArray) -> NDArray:
        """Evaluate cross-section at given energy(ies) [m^2].

        Returns 0 for energies below the data range (below threshold).
        Extrapolates with last slope for energies above range.
        """
        energy_ev = np.atleast_1d(np.asarray(energy_ev, dtype=np.float64))
        result = np.zeros_like(energy_ev)

        # Only interpolate where energy is within data range
        in_range = energy_ev >= self.e_min
        if not np.any(in_range):
            return result

        e_clip = np.clip(energy_ev[in_range], self.e_min, self.e_max)
        log_e = np.log(e_clip)
        log_sigma = np.interp(log_e, self._log_energy, self._log_sigma)
        result[in_range] = np.exp(log_sigma)

        return result

    @property
    def max_sigma(self) -> float:
        """Maximum cross-section value [m^2]."""
        return float(np.max(self.sigma_m2))

    def __repr__(self) -> str:
        return (
            f"CrossSectionTable('{self.name}', "
            f"{len(self.energy_ev)} pts, "
            f"E=[{self.e_min:.2f}, {self.e_max:.2f}] eV)"
        )
