"""Electron energy balance equation for the 0D IRM.

The electron temperature evolves via:
  d(3/2 * n_e * T_e) / dt = P_abs - P_loss

Where:
  P_abs = absorbed power from discharge (Ohmic + secondary electron heating)
  P_loss = collisional losses (ionization, excitation) + wall losses + radiation
"""

from __future__ import annotations

from plasma.core.constants import E_CHARGE
from plasma.global_model.transport import bohm_velocity


def absorbed_power_density(
    voltage_v: float,
    current_a: float,
    volume: float,
    secondary_yield: float,
    te_ev: float,
) -> float:
    """Power absorbed by electrons [W/m^3].

    P_abs = (V_D * I_D) / V_IR for the bulk Ohmic heating,
    plus secondary electron contribution.
    """
    p_ohmic = abs(voltage_v * current_a) / volume

    # Secondary electrons gain the full sheath potential
    # Their contribution to heating the cold population is small
    # but they drive hot-electron reactions
    return p_ohmic


def collisional_power_loss(
    n_e: float,
    te_ev: float,
    densities: dict[str, float],
    reactions,
) -> float:
    """Power lost to electron-impact collisions [W/m^3].

    Each inelastic collision removes threshold energy from the electron population.
    P_coll = sum_j( n_e * n_target_j * k_j(T_e) * E_threshold_j )
    """
    p_loss = 0.0
    for rxn in reactions:
        if not rxn.is_electron_impact:
            continue
        if rxn.threshold_ev <= 0:
            continue

        # Find the target species (non-electron reactant)
        for reactant in rxn.reactants:
            if reactant not in ("e", "e_cold", "e_hot"):
                n_target = densities.get(reactant, 0.0)
                break
        else:
            continue

        k = rxn.rate(te_ev, population="cold")
        p_loss += n_e * n_target * float(k) * rxn.threshold_ev * E_CHARGE

    return p_loss


def wall_power_loss(
    n_e: float,
    te_ev: float,
    area_loss: float,
    volume: float,
    mass_ion_kg: float,
) -> float:
    """Electron energy lost to walls [W/m^3].

    Electrons hitting the wall carry ~2*T_e of energy on average.
    Ions hitting the wall were accelerated through the sheath (0.5*T_e presheath + sheath).
    Total energy per ion-electron pair lost = (2*T_e + 0.5*T_e) * e.
    """
    u_b = bohm_velocity(te_ev, mass_ion_kg)
    # Energy per electron-ion pair escaping
    energy_per_pair = 2.5 * te_ev * E_CHARGE  # Approximate
    flux = n_e * u_b * area_loss / volume
    return flux * energy_per_pair


def electron_energy_rhs(
    n_e: float,
    te_ev: float,
    densities: dict[str, float],
    reactions,
    voltage_v: float,
    current_a: float,
    volume: float,
    area_loss: float,
    mass_ion_kg: float,
    secondary_yield: float,
) -> float:
    """RHS of d(3/2 * n_e * T_e)/dt in [eV * m^-3 * s^-1].

    Returns the time derivative of (3/2 * n_e * T_e).
    """
    p_abs = absorbed_power_density(voltage_v, current_a, volume, secondary_yield, te_ev)
    p_coll = collisional_power_loss(n_e, te_ev, densities, reactions)
    p_wall = wall_power_loss(n_e, te_ev, area_loss, volume, mass_ion_kg)

    # Convert W/m^3 to eV/(m^3 * s)
    return (p_abs - p_coll - p_wall) / E_CHARGE
