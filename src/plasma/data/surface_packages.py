"""Typed surface-package loader for sputter and SEE defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SurfacePackageMetadata(BaseModel):
    """Identity and provenance for one surface-interaction package."""

    name: str
    version: str
    target_material: str
    material_system: str
    provenance: str = "model-derived"
    notes: str | None = None


class SurfaceTargetDefinition(BaseModel):
    """Target-level defaults shared across interacting ions."""

    cohesive_energy_ev: float
    sputter_threshold_ev: float
    see_energy_ev: float = 3.0


class SurfaceInteractionDefinition(BaseModel):
    """Per-ion sputter and SEE parameters."""

    ion: str
    sputter_yield_a: float
    sputter_yield_b: float
    threshold_ev: float | None = None
    secondary_electron_yield: float
    citation: str | None = None


class SurfacePackage(BaseModel):
    """Package of target interactions keyed by impacting ion species."""

    package: SurfacePackageMetadata
    target: SurfaceTargetDefinition
    interactions: list[SurfaceInteractionDefinition] = Field(default_factory=list)
    source_path: str | None = None

    def interaction_for(self, ion: str) -> SurfaceInteractionDefinition | None:
        for interaction in self.interactions:
            if interaction.ion == ion:
                return interaction
        return None

    def to_target_fragment(self) -> dict[str, Any]:
        target: dict[str, Any] = {
            "material": self.package.target_material,
            "cohesive_energy_ev": self.target.cohesive_energy_ev,
            "sputter_threshold_ev": self.target.sputter_threshold_ev,
            "see_energy_ev": self.target.see_energy_ev,
        }
        gas = self.interaction_for("Ar+")
        metal = self.interaction_for(f"{self.package.target_material}+")
        if gas is not None:
            target["sputter_yield_a"] = gas.sputter_yield_a
            target["sputter_yield_b"] = gas.sputter_yield_b
            target["secondary_electron_yield"] = gas.secondary_electron_yield
            target["sputter_threshold_ev"] = gas.threshold_ev or self.target.sputter_threshold_ev
        if metal is not None:
            target["self_sputter_yield_a"] = metal.sputter_yield_a
            target["self_sputter_yield_b"] = metal.sputter_yield_b
            target["metal_ion_secondary_electron_yield"] = metal.secondary_electron_yield
        return {"target": target}


def load_surface_package(path: str | Path) -> SurfacePackage:
    """Load one surface package from YAML."""

    package_path = Path(path).resolve()
    with open(package_path) as handle:
        raw = yaml.safe_load(handle)

    package = SurfacePackage(**raw)
    package.source_path = str(package_path)
    return package


def load_surface_package_fragment(path: str | Path) -> dict[str, Any]:
    """Load one surface package and convert it into config defaults."""

    package = load_surface_package(path)
    fragment = package.to_target_fragment()
    fragment["surface_package"] = package.source_path
    return fragment
