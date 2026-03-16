"""Rate equations (dy/dt) for all species densities in the 0D IRM.

State vector y = [n_e, n_Ar, n_Ar_m, n_Ar_r, n_Ar+, n_Ar2+,
                  n_Cu, n_Cu_m1, n_Cu_m2, n_Cu_ex, n_Cu+, n_Cu2+,
                  3/2 * n_e * T_e]

Each species density evolves as:
  dn_s/dt = sum(creation rates) - sum(destruction rates) - transport_losses + transport_gains
"""

from __future__ import annotations

from numpy.typing import NDArray

from plasma.data.reactions import ReactionSet

# Index mapping for the state vector
STATE_INDICES = {
    "e_cold": 0,
    "Ar_c": 1,
    "Ar_m": 2,
    "Ar_r": 3,
    "Ar+": 4,
    "Ar2+": 5,
    "Cu": 6,
    "Cu_m1": 7,
    "Cu_m2": 8,
    "Cu_ex": 9,
    "Cu+": 10,
    "Cu2+": 11,
    "energy": 12,  # 3/2 * n_e * T_e [eV * m^-3]
}

N_STATES = len(STATE_INDICES)


def state_to_densities(y: NDArray) -> dict[str, float]:
    """Convert state vector to species density dict."""
    return {name: float(y[idx]) for name, idx in STATE_INDICES.items() if name != "energy"}


def compute_reaction_rates(
    densities: dict[str, float],
    reactions: ReactionSet,
    te_ev: float,
) -> dict[str, float]:
    """Compute volumetric reaction rate [m^-3 s^-1] for each reaction.

    For two-body: R = k(T_e) * n_1 * n_2
    For decay: R = k * n_1
    """
    rates = {}
    n_e = densities.get("e_cold", 0.0)

    for rxn in reactions:
        if rxn.is_decay:
            # Unimolecular: R = k * n_species
            n_1 = densities.get(rxn.reactants[0], 0.0)
            rates[rxn.id] = rxn.rate_constant * n_1
        elif rxn.is_electron_impact:
            # Electron impact: R = k(T_e) * n_e * n_target
            for reactant in rxn.reactants:
                if reactant not in ("e", "e_cold", "e_hot"):
                    n_target = densities.get(reactant, 0.0)
                    break
            else:
                n_target = 0.0
            k = float(rxn.rate(te_ev, population="cold"))
            rates[rxn.id] = k * n_e * n_target
        else:
            # Heavy particle (charge exchange, Penning): R = k * n_1 * n_2
            n_1 = densities.get(rxn.reactants[0], 0.0)
            n_2 = densities.get(rxn.reactants[1], 0.0)
            k = float(rxn.rate(te_ev))
            rates[rxn.id] = k * n_1 * n_2

    return rates


def species_rhs(
    densities: dict[str, float],
    reaction_rates: dict[str, float],
    reactions: ReactionSet,
) -> dict[str, float]:
    """Compute dn_s/dt from chemistry only (no transport).

    For each reaction, reactant densities decrease and product densities increase
    by the volumetric reaction rate.
    """
    # Map from species symbol to the symbols tracked in state vector
    # "e" in reactions maps to "e_cold" in state
    symbol_map = {"e": "e_cold"}

    ddt: dict[str, float] = {name: 0.0 for name in STATE_INDICES if name != "energy"}

    for rxn in reactions:
        r = reaction_rates.get(rxn.id, 0.0)
        if r == 0.0:
            continue

        # Subtract from reactants
        for reactant in rxn.reactants:
            sym = symbol_map.get(reactant, reactant)
            if sym in ddt:
                ddt[sym] -= r

        # Add to products
        for product in rxn.products:
            sym = symbol_map.get(product, product)
            if sym in ddt:
                ddt[sym] += r

    return ddt
