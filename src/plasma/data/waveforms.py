"""Load and interpolate discharge waveforms V_D(t), I_D(t)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plasma.contracts.cases import Provenance


@dataclass
class DischargeWaveform:
    """Time-dependent discharge voltage and current.

    Provides linear interpolation of measured V_D(t) and I_D(t).
    """

    time_s: NDArray[np.float64]
    voltage_v: NDArray[np.float64]
    current_a: NDArray[np.float64]
    provenance: Provenance = "synthetic"

    def V(self, t: float | NDArray) -> NDArray:  # noqa: N802
        """Interpolate discharge voltage [V] at time t [s]."""
        return np.interp(t, self.time_s, self.voltage_v)

    def I(self, t: float | NDArray) -> NDArray:  # noqa: N802, E743
        """Interpolate discharge current [A] at time t [s]."""
        return np.interp(t, self.time_s, self.current_a)

    def power(self, t: float | NDArray) -> NDArray:
        """Instantaneous power P = V * I [W]."""
        return self.V(t) * self.I(t)

    @property
    def t_max(self) -> float:
        return float(self.time_s[-1])


def load_waveform(
    path: str | Path,
    provenance: Provenance = "measured",
) -> DischargeWaveform:
    """Load waveform from CSV with columns: time_s, voltage_v, current_a."""
    path = Path(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.shape[1] < 3:
        raise ValueError(f"Waveform CSV must have >= 3 columns, got {data.shape[1]}")
    return DischargeWaveform(
        time_s=data[:, 0],
        voltage_v=data[:, 1],
        current_a=data[:, 2],
        provenance=provenance,
    )


def make_square_pulse(
    voltage_v: float,
    t_pulse_s: float,
    t_total_s: float,
    rise_time_s: float = 1e-7,
    n_points: int = 1000,
) -> DischargeWaveform:
    """Generate an idealized square pulse waveform.

    Useful when measured waveforms aren't available. Creates a trapezoidal
    voltage pulse with finite rise/fall time and zero voltage during afterglow.
    Current is set to zero (the 0D model computes it self-consistently).
    """
    t = np.linspace(0, t_total_s, n_points)
    v = np.zeros_like(t)

    # Rise
    rise_mask = (t >= 0) & (t < rise_time_s)
    v[rise_mask] = voltage_v * t[rise_mask] / rise_time_s

    # Flat top
    flat_mask = (t >= rise_time_s) & (t < t_pulse_s - rise_time_s)
    v[flat_mask] = voltage_v

    # Fall
    fall_mask = (t >= t_pulse_s - rise_time_s) & (t < t_pulse_s)
    v[fall_mask] = voltage_v * (t_pulse_s - t[fall_mask]) / rise_time_s

    return DischargeWaveform(
        time_s=t,
        voltage_v=v,
        current_a=np.zeros_like(t),
        provenance="synthetic",
    )
