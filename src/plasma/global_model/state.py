"""State-layout helpers for material-aware HiPIMS global models."""

from __future__ import annotations

from dataclasses import dataclass

from plasma.core.constants import material_mass_kg
from plasma.core.species import SpeciesSet, cu_ar_species, ti_ar_species

CONTROL_KEYS = ("current_circuit",)
ENERGY_KEYS = ("energy_cold", "energy_hot")

@dataclass(frozen=True)
class StateLayout:
    """Material-aware state layout for the 0D global model."""

    material: str
    species_set: SpeciesSet
    species_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    indices: dict[str, int]
    species_charges: dict[str, int]
    species_masses: dict[str, float]
    electron_species: tuple[str, ...]
    argon_neutral_species: tuple[str, ...]
    argon_ion_species: tuple[str, ...]
    metal_neutral_species: tuple[str, ...]
    metal_ion_species: tuple[str, ...]

    @property
    def ion_species(self) -> tuple[str, ...]:
        return self.argon_ion_species + self.metal_ion_species

    @property
    def primary_argon_ion(self) -> str:
        return next(symbol for symbol in self.argon_ion_species if self.species_charges[symbol] == 1)

    @property
    def primary_metal_ion(self) -> str:
        return next(symbol for symbol in self.metal_ion_species if self.species_charges[symbol] == 1)

    @property
    def metal_ground_species(self) -> str:
        return next(
            symbol
            for symbol in self.metal_neutral_species
            if symbol == self.material
        )

    @property
    def metal_mass_kg(self) -> float:
        return material_mass_kg(self.material)

    @property
    def n_states(self) -> int:
        return len(self.state_keys)


def build_state_layout(material: str) -> StateLayout:
    """Build a state layout for one supported Ar/metal material system."""

    if material == "Cu":
        species_set = cu_ar_species()
    elif material == "Ti":
        species_set = ti_ar_species()
    else:
        raise ValueError(f"Unsupported 0D material system '{material}'.")

    species_keys = tuple(species.symbol for species in species_set)
    species_charges = {species.symbol: species.charge for species in species_set}
    species_masses = {species.symbol: species.mass for species in species_set}
    electron_species = tuple(species.symbol for species in species_set if species.is_electron)
    argon_neutral_species = tuple(
        species.symbol for species in species_set if species.symbol.startswith("Ar") and species.charge == 0
    )
    argon_ion_species = tuple(
        species.symbol for species in species_set if species.symbol.startswith("Ar") and species.charge > 0
    )
    metal_neutral_species = tuple(
        species.symbol
        for species in species_set
        if not species.symbol.startswith("Ar") and not species.is_electron and species.charge == 0
    )
    metal_ion_species = tuple(
        species.symbol
        for species in species_set
        if not species.symbol.startswith("Ar") and not species.is_electron and species.charge > 0
    )
    state_keys = species_keys + CONTROL_KEYS + ENERGY_KEYS
    return StateLayout(
        material=material,
        species_set=species_set,
        species_keys=species_keys,
        state_keys=state_keys,
        indices={name: idx for idx, name in enumerate(state_keys)},
        species_charges=species_charges,
        species_masses=species_masses,
        electron_species=electron_species,
        argon_neutral_species=argon_neutral_species,
        argon_ion_species=argon_ion_species,
        metal_neutral_species=metal_neutral_species,
        metal_ion_species=metal_ion_species,
    )


DEFAULT_STATE_LAYOUT = build_state_layout("Cu")
SPECIES_SET: SpeciesSet = DEFAULT_STATE_LAYOUT.species_set
SPECIES_KEYS = DEFAULT_STATE_LAYOUT.species_keys
STATE_KEYS = DEFAULT_STATE_LAYOUT.state_keys
STATE_INDICES = DEFAULT_STATE_LAYOUT.indices
N_STATES = DEFAULT_STATE_LAYOUT.n_states
SPECIES_CHARGES = DEFAULT_STATE_LAYOUT.species_charges
SPECIES_MASSES = DEFAULT_STATE_LAYOUT.species_masses
ELECTRON_SPECIES = DEFAULT_STATE_LAYOUT.electron_species
ARGON_NEUTRAL_SPECIES = DEFAULT_STATE_LAYOUT.argon_neutral_species
ARGON_ION_SPECIES = DEFAULT_STATE_LAYOUT.argon_ion_species
METAL_NEUTRAL_SPECIES = DEFAULT_STATE_LAYOUT.metal_neutral_species
METAL_ION_SPECIES = DEFAULT_STATE_LAYOUT.metal_ion_species
ION_SPECIES = DEFAULT_STATE_LAYOUT.ion_species


def state_to_densities(y: object, *, layout: StateLayout = DEFAULT_STATE_LAYOUT) -> dict[str, float]:
    """Convert a state vector to a species-density mapping."""

    auxiliary_keys = set(CONTROL_KEYS + ENERGY_KEYS)
    return {name: float(y[idx]) for name, idx in layout.indices.items() if name not in auxiliary_keys}


def electron_density(densities: dict[str, float], *, layout: StateLayout = DEFAULT_STATE_LAYOUT) -> float:
    """Total electron density [m^-3]."""

    return sum(max(densities.get(symbol, 0.0), 0.0) for symbol in layout.electron_species)


def ion_charge_density(densities: dict[str, float], *, layout: StateLayout = DEFAULT_STATE_LAYOUT) -> float:
    """Total positive charge density in units of particles per m^3."""

    return sum(
        max(layout.species_charges[symbol], 0) * max(densities.get(symbol, 0.0), 0.0)
        for symbol in layout.ion_species
    )


def electron_temperature_ev(y: object, population: str, *, layout: StateLayout = DEFAULT_STATE_LAYOUT) -> float:
    """Electron temperature for one population [eV]."""

    density_key = f"e_{population}"
    energy_key = f"energy_{population}"
    density = max(float(y[layout.indices[density_key]]), 1.0)
    energy_density = max(float(y[layout.indices[energy_key]]), 0.0)
    return max((2.0 / 3.0) * energy_density / density, 0.1)
