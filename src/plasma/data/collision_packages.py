"""Versioned Cu/Ar collision-package contracts and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from plasma.contracts.cases import Provenance
from plasma.data.cross_sections import CrossSectionTable
from plasma.data.synthetic_cross_sections import (
    electron_ar_elastic,
    electron_ar_excitation,
    electron_ar_ionization,
    electron_cu_excitation,
    electron_cu_ionization,
    ion_ar_charge_exchange,
    ion_cu_charge_exchange,
)

SpeciesRole = Literal["kinetic", "background"]
CollisionProcessKind = Literal["elastic", "excitation", "ionization", "charge_exchange", "penning", "rate_only"]
CollisionExecution = Literal["mcc", "background_rate", "disabled"]


class CollisionPackageMetadata(BaseModel):
    """Top-level identity and provenance for one collision package."""

    name: str
    version: str
    material_system: str
    notes: str | None = None


class SpeciesDefinition(BaseModel):
    """One species entry for a collision package."""

    name: str
    role: SpeciesRole
    charge_state: int
    mass_kg: float
    initial_density_m3: float = 0.0
    notes: str | None = None


class CollisionChannelDefinition(BaseModel):
    """One collision or background-rate channel."""

    name: str
    projectile: str
    background: str
    process: CollisionProcessKind
    execution: CollisionExecution = "mcc"
    threshold_ev: float = 0.0
    product_species_name: str | None = None
    product_ion_name: str | None = None
    cross_section_file: str | None = None
    builtin_cross_section: str | None = None
    provenance: Provenance
    citation: str | None = None
    notes: str | None = None

    @property
    def uses_cross_section(self) -> bool:
        return self.execution == "mcc" and (self.cross_section_file is not None or self.builtin_cross_section is not None)


class CollisionPackage(BaseModel):
    """Machine-readable collision package for PIC runtime construction."""

    package: CollisionPackageMetadata
    species: list[SpeciesDefinition] = Field(default_factory=list)
    channels: list[CollisionChannelDefinition] = Field(default_factory=list)
    source_path: str | None = None

    def species_by_name(self) -> dict[str, SpeciesDefinition]:
        return {entry.name: entry for entry in self.species}

    def kinetic_species(self) -> list[SpeciesDefinition]:
        return [entry for entry in self.species if entry.role == "kinetic"]

    def background_species(self) -> list[SpeciesDefinition]:
        return [entry for entry in self.species if entry.role == "background"]

    def channels_for_projectile(
        self,
        projectile: str,
        *,
        execution: CollisionExecution | None = None,
    ) -> list[CollisionChannelDefinition]:
        return [
            channel
            for channel in self.channels
            if channel.projectile == projectile and (execution is None or channel.execution == execution)
        ]

    def channel_provenance(self) -> dict[str, Provenance]:
        return {channel.name: channel.provenance for channel in self.channels}


def load_collision_package(path: str | Path) -> CollisionPackage:
    """Load a collision package and resolve any relative cross-section paths."""

    package_path = Path(path).resolve()
    with open(package_path) as handle:
        raw = yaml.safe_load(handle)

    package = CollisionPackage(**raw)
    package.source_path = str(package_path)
    for channel in package.channels:
        if channel.cross_section_file:
            channel.cross_section_file = str(_resolve_path(package_path.parent, channel.cross_section_file))
    return package


def load_channel_cross_section(channel: CollisionChannelDefinition) -> CrossSectionTable | None:
    """Resolve one channel to a tabulated cross section, if it has one."""

    if channel.cross_section_file:
        return CrossSectionTable.from_file(channel.cross_section_file, name=channel.name)
    if channel.builtin_cross_section:
        try:
            factory = _BUILTIN_CROSS_SECTIONS[channel.builtin_cross_section]
        except KeyError as exc:
            raise ValueError(f"Unknown builtin cross section '{channel.builtin_cross_section}'") from exc
        return factory()
    return None


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


_BUILTIN_CROSS_SECTIONS = {
    "electron_ar_elastic": electron_ar_elastic,
    "electron_ar_excitation": electron_ar_excitation,
    "electron_ar_ionization": electron_ar_ionization,
    "electron_cu_excitation": electron_cu_excitation,
    "electron_cu_ionization": electron_cu_ionization,
    "ion_ar_charge_exchange": ion_ar_charge_exchange,
    "ion_cu_charge_exchange": ion_cu_charge_exchange,
}
