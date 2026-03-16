"""Unit conversion helpers for plasma simulation."""

from plasma.core.constants import E_CHARGE, K_BOLTZMANN


def ev_to_joule(energy_ev: float) -> float:
    """Convert electron-volts to joules."""
    return energy_ev * E_CHARGE


def joule_to_ev(energy_j: float) -> float:
    """Convert joules to electron-volts."""
    return energy_j / E_CHARGE


def ev_to_kelvin(temp_ev: float) -> float:
    """Convert temperature in eV to Kelvin."""
    return temp_ev * E_CHARGE / K_BOLTZMANN


def kelvin_to_ev(temp_k: float) -> float:
    """Convert temperature in Kelvin to eV."""
    return temp_k * K_BOLTZMANN / E_CHARGE


def pa_to_torr(pressure_pa: float) -> float:
    """Convert Pascal to Torr."""
    return pressure_pa / 133.322


def torr_to_pa(pressure_torr: float) -> float:
    """Convert Torr to Pascal."""
    return pressure_torr * 133.322


def pa_to_mtorr(pressure_pa: float) -> float:
    """Convert Pascal to milliTorr."""
    return pressure_pa / 0.133322


def mtorr_to_pa(pressure_mtorr: float) -> float:
    """Convert milliTorr to Pascal."""
    return pressure_mtorr * 0.133322


def number_density_from_pressure(pressure_pa: float, temp_k: float) -> float:
    """Ideal gas number density n = p / (k_B * T) [m^-3]."""
    return pressure_pa / (K_BOLTZMANN * temp_k)
