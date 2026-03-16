"""Cheap derived diagnostics for the live HiPIMS monitor."""

from __future__ import annotations

import numpy as np

PULSE_PHASES = ("early", "ramp", "peak", "decay")


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio for live proxy metrics."""

    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def peak_radius_m(r_axis: np.ndarray, field: np.ndarray) -> float:
    """Return the radial location of the peak of a node-centered field."""

    if field.size == 0 or r_axis.size == 0:
        return 0.0
    radial_profile = np.max(np.asarray(field, dtype=np.float64), axis=1)
    if not np.any(radial_profile > 0.0):
        return 0.0
    return float(r_axis[int(np.argmax(radial_profile))])


def pulse_phase_label(voltage_history: list[float], emissivity_history: list[float]) -> tuple[str, float]:
    """Infer a coarse pulse phase from the recent live traces."""

    if not voltage_history:
        return PULSE_PHASES[0], 0.0

    voltage = np.abs(np.asarray(voltage_history, dtype=np.float64))
    latest_voltage = float(voltage[-1])
    peak_voltage = float(np.max(voltage)) if voltage.size else 0.0
    emissivity = np.asarray(emissivity_history, dtype=np.float64) if emissivity_history else np.zeros(0, dtype=np.float64)
    latest_glow = float(emissivity[-1]) if emissivity.size else 0.0
    peak_glow = float(np.max(emissivity)) if emissivity.size else 0.0

    if peak_voltage <= 0.0 or latest_voltage < 0.35 * peak_voltage:
        return PULSE_PHASES[0], 0.0
    if peak_glow <= 0.0 or latest_glow < 0.55 * peak_glow:
        return PULSE_PHASES[1], 1.0
    if emissivity.size >= 2 and latest_glow >= float(emissivity[-2]):
        return PULSE_PHASES[2], 2.0
    return PULSE_PHASES[3], 3.0


def monitor_message(phase: str, peak_radius: float, capture_ratio: float) -> str:
    """Build the short viewer subtitle for the live monitor."""

    return f"Pulse phase: {phase} | racetrack r={peak_radius * 1e3:.1f} mm | capture={capture_ratio:.2f}"
