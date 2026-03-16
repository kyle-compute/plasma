"""Reaction set: load from YAML, compute rate coefficients k(T_e)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray


@dataclass
class RateCoeffFit:
    """Arrhenius-type rate coefficient: k = a * T_e^b * exp(-c / T_e)."""

    a: float
    b: float
    c: float

    def __call__(self, te_ev: float | NDArray) -> float | NDArray:
        te = np.asarray(te_ev, dtype=np.float64)
        # Avoid division by zero and overflow
        safe_te = np.maximum(te, 1e-3)
        return self.a * safe_te**self.b * np.exp(-self.c / safe_te)


@dataclass
class Reaction:
    """A single plasma-chemical reaction with rate coefficient."""

    id: str
    name: str
    reactants: list[str]
    products: list[str]
    reaction_type: str
    threshold_ev: float
    rate_cold: RateCoeffFit | None = None
    rate_hot: RateCoeffFit | None = None
    rate_constant: float | None = None  # For decay/heavy-particle reactions [1/s or m^3/s]

    def rate(self, te_ev: float | NDArray, population: str = "cold") -> float | NDArray:
        """Compute rate coefficient at electron temperature T_e [eV].

        For decay reactions, returns the constant rate (independent of T_e).
        For heavy-particle reactions (charge_exchange, penning), returns constant rate.
        """
        if self.rate_constant is not None:
            return self.rate_constant

        fit = self.rate_cold if population == "cold" else self.rate_hot
        if fit is None or (fit.a == 0.0):
            return 0.0
        return fit(te_ev)

    @property
    def is_electron_impact(self) -> bool:
        return any(r in ("e", "e_cold", "e_hot") for r in self.reactants)

    @property
    def is_decay(self) -> bool:
        return self.reaction_type == "decay"

    @property
    def is_ionization(self) -> bool:
        return self.reaction_type == "ionization"


@dataclass
class ReactionSet:
    """Collection of all reactions for a discharge chemistry."""

    reactions: dict[str, Reaction] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Reaction:
        return self.reactions[key]

    def __iter__(self):
        return iter(self.reactions.values())

    def __len__(self) -> int:
        return len(self.reactions)

    @property
    def ids(self) -> list[str]:
        return list(self.reactions.keys())

    def by_type(self, reaction_type: str) -> list[Reaction]:
        return [r for r in self if r.reaction_type == reaction_type]

    def involving(self, species_symbol: str) -> list[Reaction]:
        """All reactions where species appears as reactant or product."""
        return [
            r for r in self
            if species_symbol in r.reactants or species_symbol in r.products
        ]

    def electron_impact(self) -> list[Reaction]:
        return [r for r in self if r.is_electron_impact]

    def heavy_particle(self) -> list[Reaction]:
        return [r for r in self if not r.is_electron_impact and not r.is_decay]


def load_reactions(path: str | Path) -> ReactionSet:
    """Load reaction set from YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    reactions = {}
    for rid, rdata in data["reactions"].items():
        rate_cold = None
        rate_hot = None
        rate_const = None

        if "rate_cold" in rdata:
            rc = rdata["rate_cold"]
            rate_cold = RateCoeffFit(a=rc["a"], b=rc["b"], c=rc["c"])
        if "rate_hot" in rdata:
            rh = rdata["rate_hot"]
            rate_hot = RateCoeffFit(a=rh["a"], b=rh["b"], c=rh["c"])
        if "rate_constant" in rdata:
            rate_const = float(rdata["rate_constant"])

        reactions[rid] = Reaction(
            id=rid,
            name=rdata["name"],
            reactants=rdata["reactants"],
            products=rdata["products"],
            reaction_type=rdata["type"],
            threshold_ev=rdata["threshold_ev"],
            rate_cold=rate_cold,
            rate_hot=rate_hot,
            rate_constant=rate_const,
        )

    return ReactionSet(reactions=reactions)
