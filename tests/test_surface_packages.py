"""Tests for surface and material package loaders."""

from __future__ import annotations

import pytest

from plasma.data.material_packages import load_material_package
from plasma.data.surface_packages import load_surface_package


def test_load_surface_package_exposes_cu_interactions() -> None:
    package = load_surface_package("data/surface_packages/cu_target_public_v1.yaml")

    assert package.package.target_material == "Cu"
    assert package.target.cohesive_energy_ev == pytest.approx(3.49)
    assert package.interaction_for("Ar+") is not None
    assert package.interaction_for("Cu+") is not None
    assert package.interaction_for("Cu+").secondary_electron_yield == pytest.approx(0.08)


def test_material_package_merges_ti_surface_defaults_into_pic_fragment() -> None:
    fragment = load_material_package("data/material_packages/ti_ar_exploratory_v1.yaml", model="pic")

    assert fragment["target"]["material"] == "Ti"
    assert fragment["surface_package"].endswith("data/surface_packages/ti_target_exploratory_v1.yaml")
