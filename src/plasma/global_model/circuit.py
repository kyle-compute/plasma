"""External-circuit closures for the 0D Cu/Ar discharge model."""

from __future__ import annotations

from plasma.core.constants import E_CHARGE, M_ELECTRON


def plasma_resistance_ohm(
    *,
    n_e: float,
    gas_density: float,
    area_m2: float,
    length_m: float,
    te_ev: float,
    circuit,
) -> float:
    """Estimate plasma resistance from a Drude-like conductivity model."""

    density_scale = max(gas_density / 1.0e20, 0.05)
    collision_frequency = circuit.electron_neutral_collision_hz * density_scale
    conductivity = (
        circuit.electron_conductivity_scale
        * (E_CHARGE**2)
        * max(n_e, 1.0)
        / (M_ELECTRON * max(collision_frequency, 1.0))
    )
    conductivity *= max(te_ev / 3.0, 0.2)
    resistance = length_m / max(conductivity * max(area_m2, 1e-12), 1e-12)
    return min(max(resistance, circuit.plasma_resistance_floor_ohm), circuit.plasma_resistance_ceiling_ohm)


def circuit_current_rhs(
    *,
    current_a: float,
    source_voltage_v: float,
    plasma_resistance_ohm: float,
    circuit,
) -> float:
    """Current derivative for a series RL circuit feeding the plasma."""

    if circuit.mode != "rl":
        return 0.0
    total_resistance = plasma_resistance_ohm + circuit.series_resistance_ohm
    inductance = max(circuit.series_inductance_h, 1e-12)
    return (source_voltage_v - total_resistance * current_a) / inductance


def discharge_current_from_circuit(
    *,
    source_voltage_v: float,
    current_a: float,
    plasma_resistance_ohm: float,
    circuit,
) -> float:
    """Return the model discharge current magnitude."""

    if circuit.mode == "waveform_current":
        return current_a
    if circuit.series_inductance_h <= 0.0:
        return source_voltage_v / max(plasma_resistance_ohm + circuit.series_resistance_ohm, 1e-12)
    return max(current_a, 0.0)


def target_voltage_from_circuit(*, current_a: float, plasma_resistance_ohm: float) -> float:
    """Voltage dropped across the plasma/sheath."""

    return current_a * plasma_resistance_ohm
