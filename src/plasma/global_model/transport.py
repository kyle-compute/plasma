"""Transport fluxes at the ionization region boundary.

The IR (ionization region) exchanges particles with:
  - The target (sputtered neutrals in, ions back-attracted out)
  - The diffusion region / bulk plasma (ambipolar diffusion losses)
  - The gas reservoir (Ar refill from chamber walls)

Key parameters from Brenning 2021:
  - alpha_t: ionization probability of sputtered target species
  - beta_t: back-attraction probability (ionized target species returning to target)
"""

from __future__ import annotations

import numpy as np

from plasma.core.constants import E_CHARGE, K_BOLTZMANN, M_AR, M_ELECTRON, PI


def bohm_velocity(te_ev: float, mass_kg: float) -> float:
    """Bohm velocity u_B = sqrt(e * T_e / m_i) [m/s]."""
    return np.sqrt(E_CHARGE * te_ev / mass_kg)


def thermal_velocity(temp_ev: float, mass_kg: float) -> float:
    """Mean thermal speed <v> = sqrt(8 * e * T / (pi * m)) [m/s]."""
    return np.sqrt(8.0 * E_CHARGE * temp_ev / (PI * mass_kg))


def electron_thermal_velocity(te_ev: float) -> float:
    """Electron thermal speed [m/s]."""
    return thermal_velocity(te_ev, M_ELECTRON)


def ion_loss_rate(
    n_ion: float,
    te_ev: float,
    mass_kg: float,
    area_loss: float,
    volume: float,
) -> float:
    """Ion loss rate from IR due to ambipolar diffusion [m^-3 s^-1].

    Flux = n_i * u_B * A_loss / V_IR
    """
    u_b = bohm_velocity(te_ev, mass_kg)
    return n_ion * u_b * area_loss / volume


def neutral_refill_rate(
    n_gas_0: float,
    n_gas: float,
    temp_k: float,
    area_wall: float,
    volume: float,
    mass_kg: float = M_AR,
) -> float:
    """Neutral gas refill rate from chamber walls [m^-3 s^-1].

    Gas flows back into IR from the chamber at thermal velocity,
    driven by the density difference (n_0 - n_gas).
    """
    v_th = np.sqrt(8.0 * K_BOLTZMANN * temp_k / (PI * mass_kg))
    return max(0.0, (n_gas_0 - n_gas)) * v_th * area_wall / (4.0 * volume)


def sputter_flux(
    n_ion: float,
    te_ev: float,
    mass_ion_kg: float,
    voltage_v: float,
    yield_func,
    area_target: float,
    volume: float,
) -> float:
    """Sputtered neutral influx to IR [m^-3 s^-1].

    Sputtered flux = ion flux to target * Y(E_ion).
    Ion energy ≈ e * V_discharge (simplified sheath model).
    """
    u_b = bohm_velocity(te_ev, mass_ion_kg)
    ion_flux = n_ion * u_b  # [m^-2 s^-1]
    energy = abs(voltage_v)  # Ion energy in eV (≈ discharge voltage)
    y = float(yield_func(energy).item())
    return ion_flux * y * area_target / volume


def back_attraction_rate(
    n_metal_ion: float,
    beta_t: float,
    te_ev: float,
    mass_kg: float,
    area_target: float,
    volume: float,
) -> float:
    """Rate of metal ions pulled back to target [m^-3 s^-1]."""
    u_b = bohm_velocity(te_ev, mass_kg)
    return n_metal_ion * beta_t * u_b * area_target / volume


def compute_beta_t(voltage_v: float, te_ev: float, b_field_t: float = 0.05) -> float:
    """Estimate back-attraction probability beta_t.

    From Brenning 2021: beta_t depends on the electric field structure
    in the magnetic presheath. Higher V_D and stronger B give higher beta_t.
    Simplified model: beta_t ≈ 0.5 * (1 - exp(-V_D / (10 * T_e))).
    """
    ratio = abs(voltage_v) / max(10.0 * te_ev, 1.0)
    return 0.5 * (1.0 - np.exp(-ratio))
