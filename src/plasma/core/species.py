"""Species definitions for plasma simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from plasma.core.constants import AMU, E_CHARGE, M_ELECTRON


@dataclass(frozen=True)
class Species:
    """A plasma species (electrons, ions, neutrals, or excited states).

    Attributes:
        name: Human-readable identifier (e.g. "Ar+", "Cu_m1").
        symbol: Short symbol for equations/plots.
        mass: Particle mass [kg].
        charge: Charge in units of elementary charge (0 for neutrals, +1 for singly ionized).
        energy_level: Internal energy above ground state [eV]. 0.0 for ground state.
        is_metastable: Whether this is a metastable state (long-lived excited).
    """

    name: str
    symbol: str
    mass: float
    charge: int = 0
    energy_level: float = 0.0
    is_metastable: bool = False

    @property
    def charge_si(self) -> float:
        """Charge in Coulombs."""
        return self.charge * E_CHARGE

    @property
    def mass_amu(self) -> float:
        """Mass in atomic mass units."""
        return self.mass / AMU

    @property
    def is_electron(self) -> bool:
        return self.mass < 1e-28  # lighter than any ion


@dataclass
class SpeciesSet:
    """Collection of all species in a discharge model."""

    species: list[Species] = field(default_factory=list)

    def __getitem__(self, symbol: str) -> Species:
        for s in self.species:
            if s.symbol == symbol:
                return s
        raise KeyError(f"Species '{symbol}' not found")

    def __contains__(self, symbol: str) -> bool:
        return any(s.symbol == symbol for s in self.species)

    def __len__(self) -> int:
        return len(self.species)

    def __iter__(self):
        return iter(self.species)

    @property
    def symbols(self) -> list[str]:
        return [s.symbol for s in self.species]


# ── Pre-built species sets ──────────────────────────────────────────

def _build_electron_species() -> list[Species]:
    return [
        Species("Cold electrons", "e_cold", M_ELECTRON, charge=-1),
        Species("Hot electrons", "e_hot", M_ELECTRON, charge=-1),
    ]


def _build_argon_species() -> list[Species]:
    m = 39.948 * AMU
    return [
        Species("Argon (cold)", "Ar_c", m),
        Species("Argon (hot)", "Ar_h", m),
        Species("Argon (warm)", "Ar_w", m),
        Species("Argon metastable", "Ar_m", m, energy_level=11.55, is_metastable=True),
        Species("Argon resonant", "Ar_r", m, energy_level=11.72),
        Species("Argon 4p", "Ar_4p", m, energy_level=13.0),
        Species("Ar+", "Ar+", m, charge=+1),
        Species("Ar2+", "Ar2+", m, charge=+2),
    ]


def _build_copper_species() -> list[Species]:
    m = 63.546 * AMU
    return [
        Species("Copper ground", "Cu", m),
        Species("Copper metastable 1", "Cu_m1", m, energy_level=1.389, is_metastable=True),
        Species("Copper metastable 2", "Cu_m2", m, energy_level=1.642, is_metastable=True),
        Species("Copper excited", "Cu_ex", m, energy_level=3.786),
        Species("Cu+", "Cu+", m, charge=+1),
        Species("Cu2+", "Cu2+", m, charge=+2),
    ]


def _build_titanium_species() -> list[Species]:
    m = 47.867 * AMU
    return [
        Species("Titanium ground", "Ti", m),
        Species("Titanium metastable 1", "Ti_m1", m, energy_level=0.813, is_metastable=True),
        Species("Titanium metastable 2", "Ti_m2", m, energy_level=0.900, is_metastable=True),
        Species("Titanium excited", "Ti_ex", m, energy_level=2.0),
        Species("Ti+", "Ti+", m, charge=+1),
        Species("Ti2+", "Ti2+", m, charge=+2),
    ]


def cu_ar_species() -> SpeciesSet:
    """Full species set for Cu/Ar HiPIMS discharge (Gudmundsson 2022)."""
    return SpeciesSet(
        _build_electron_species() + _build_argon_species() + _build_copper_species()
    )


def ti_ar_species() -> SpeciesSet:
    """Full species set for Ti/Ar HiPIMS discharge."""
    return SpeciesSet(
        _build_electron_species() + _build_argon_species() + _build_titanium_species()
    )
