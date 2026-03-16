"""Yamamura sputtering yield model: Y(E) = a * E^b."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray


@dataclass
class SputterYield:
    """Sputter yield Y(E) = a * E^b for E > threshold, else 0."""

    ion: str
    target: str
    a: float
    b: float
    threshold_ev: float
    cohesive_energy_ev: float

    def __call__(self, energy_ev: float | NDArray) -> NDArray:
        """Compute sputter yield at given ion energy(ies)."""
        e = np.atleast_1d(np.asarray(energy_ev, dtype=np.float64))
        result = np.zeros_like(e)
        above = e > self.threshold_ev
        result[above] = self.a * e[above] ** self.b
        return result

    def sputtered_energy_peak(self) -> float:
        """Peak of Thompson energy distribution ≈ E_b / 2."""
        return self.cohesive_energy_ev / 2.0


def load_sputter_yields(path: str | Path) -> dict[str, SputterYield]:
    """Load all sputter yield fits from YAML.

    Returns dict keyed by '{ion}_{target}', e.g. 'Ar_Cu'.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    yields = {}
    for key, ydata in data["yields"].items():
        yields[key] = SputterYield(
            ion=ydata["ion"],
            target=ydata["target"],
            a=ydata["a"],
            b=ydata["b"],
            threshold_ev=ydata["threshold_ev"],
            cohesive_energy_ev=ydata["cohesive_energy_ev"],
        )
    return yields
