"""Chemistry assembly helpers for the Cu/Ar 0D global model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from plasma.data.reactions import ReactionSet
from plasma.global_model.state import (
    DEFAULT_STATE_LAYOUT,
    N_STATES,
    STATE_INDICES,
    StateLayout,
    state_to_densities,
)

ELECTRON_TOKENS = {"e", "e_cold", "e_hot"}


@dataclass(frozen=True)
class ReactionPopulationRates:
    """Per-reaction rate split by electron population."""

    cold: float = 0.0
    hot: float = 0.0

    @property
    def total(self) -> float:
        return self.cold + self.hot


def compute_population_reaction_rates(
    densities: dict[str, float],
    reactions: ReactionSet,
    te_cold_ev: float,
    te_hot_ev: float,
) -> dict[str, ReactionPopulationRates]:
    """Compute volumetric rates for each reaction and electron population."""

    rates: dict[str, ReactionPopulationRates] = {}
    n_cold = densities.get("e_cold", 0.0)
    n_hot = densities.get("e_hot", 0.0)

    for reaction in reactions:
        if reaction.is_decay:
            density = densities.get(reaction.reactants[0], 0.0)
            rates[reaction.id] = ReactionPopulationRates(cold=reaction.rate_constant * density)
            continue

        if reaction.is_electron_impact:
            target = _electron_impact_target_density(densities, reaction.reactants)
            rates[reaction.id] = ReactionPopulationRates(
                cold=float(reaction.rate(te_cold_ev, population="cold")) * n_cold * target,
                hot=float(reaction.rate(te_hot_ev, population="hot")) * n_hot * target,
            )
            continue

        n_1 = densities.get(reaction.reactants[0], 0.0)
        n_2 = densities.get(reaction.reactants[1], 0.0)
        rates[reaction.id] = ReactionPopulationRates(cold=float(reaction.rate(te_cold_ev)) * n_1 * n_2)

    return rates


def compute_reaction_rates(
    densities: dict[str, float],
    reactions: ReactionSet,
    te_ev: float,
    te_hot_ev: float | None = None,
) -> dict[str, float]:
    """Compute total volumetric rate [m^-3 s^-1] for each reaction."""

    hot_temperature = te_ev if te_hot_ev is None else te_hot_ev
    rates = compute_population_reaction_rates(densities, reactions, te_ev, hot_temperature)
    return {reaction_id: rate.total for reaction_id, rate in rates.items()}


def species_rhs(
    densities: dict[str, float],
    reaction_rates: Mapping[str, float | ReactionPopulationRates],
    reactions: ReactionSet,
    *,
    state_layout: StateLayout = DEFAULT_STATE_LAYOUT,
) -> dict[str, float]:
    """Compute chemistry-only dn/dt for all tracked species."""

    del densities
    ddt = {
        name: 0.0
        for name in state_layout.indices
        if not name.startswith("energy_") and not name.startswith("current_")
    }

    for reaction in reactions:
        rate = reaction_rates.get(reaction.id, 0.0)
        if isinstance(rate, ReactionPopulationRates):
            _apply_reaction(ddt, reaction.reactants, reaction.products, rate.cold, "e_cold")
            _apply_reaction(ddt, reaction.reactants, reaction.products, rate.hot, "e_hot")
            continue
        _apply_reaction(ddt, reaction.reactants, reaction.products, float(rate), "e_cold")

    return ddt


def _apply_reaction(
    ddt: dict[str, float],
    reactants: list[str],
    products: list[str],
    rate: float,
    driver_electron: str,
) -> None:
    if rate == 0.0:
        return
    for reactant in reactants:
        symbol = _map_symbol(reactant, driver_electron, is_product=False)
        if symbol in ddt:
            ddt[symbol] -= rate
    for product in products:
        symbol = _map_symbol(product, driver_electron, is_product=True)
        if symbol in ddt:
            ddt[symbol] += rate


def _map_symbol(symbol: str, driver_electron: str, *, is_product: bool) -> str:
    if symbol == "e":
        return "e_cold" if is_product else driver_electron
    return symbol


def _electron_impact_target_density(densities: dict[str, float], reactants: list[str]) -> float:
    for reactant in reactants:
        if reactant not in ELECTRON_TOKENS:
            return densities.get(reactant, 0.0)
    return 0.0


__all__ = [
    "N_STATES",
    "STATE_INDICES",
    "ReactionPopulationRates",
    "compute_population_reaction_rates",
    "compute_reaction_rates",
    "species_rhs",
    "state_to_densities",
]
