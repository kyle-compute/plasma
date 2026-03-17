"""Tests for material-aware 0D and PIC plumbing."""

from __future__ import annotations

import pytest

from plasma.core.config import load_config
from plasma.core.constants import M_TI
from plasma.data.reactions import ChemistrySpecies, ReactionSet
from plasma.global_model.irm import IRM
from plasma.global_model.state import build_state_layout, ion_charge_density, state_to_densities
from plasma.pic.config import load_pic_config
from plasma.pic.runtime import build_simulation


def _empty_reaction_set(material: str) -> ReactionSet:
    layout = build_state_layout(material)
    species = tuple(
        ChemistrySpecies(
            symbol=entry.symbol,
            family="electron" if entry.is_electron else ("argon" if entry.symbol.startswith("Ar") else "metal"),
            charge=entry.charge,
            population=None,
            role=None,
            energy_level_ev=entry.energy_level,
            metastable=entry.is_metastable,
        )
        for entry in layout.species_set
    )
    return ReactionSet(reactions={}, package_name=f"{material.lower()}_empty", source="test", species=species)


def test_build_state_layout_for_ti() -> None:
    layout = build_state_layout("Ti")

    assert layout.material == "Ti"
    assert layout.metal_ground_species == "Ti"
    assert layout.primary_metal_ion == "Ti+"
    assert "Ti2+" in layout.metal_ion_species


def test_ti_irm_initial_state_uses_ti_species() -> None:
    cfg = load_config("config/hipims_ti_ar.yaml")
    irm = IRM(cfg, reactions=_empty_reaction_set("Ti"))
    y0 = irm.initial_state()

    assert y0[irm.state_layout.indices["Ti+"]] > 0.0
    assert "Cu+" not in irm.state_layout.indices
    densities = state_to_densities(y0, layout=irm.state_layout)
    assert densities["Ti+"] > 0.0
    assert y0[irm.state_layout.indices["e_cold"]] + y0[irm.state_layout.indices["e_hot"]] == pytest.approx(
        ion_charge_density(densities, layout=irm.state_layout),
        rel=1e-6,
    )


def test_ti_pic_build_uses_ti_mass(tmp_path) -> None:
    config_path = tmp_path / "pic_ti_small.yaml"
    config_path.write_text(
        "\n".join(
            [
                'name: "pic_ti_small"',
                'model: "pic"',
                'material_package: "data/material_packages/ti_ar_exploratory_v1.yaml"',
                "geometry:",
                "  r_target: 0.05",
                "  r_inner: 0.015",
                "  r_outer: 0.04",
                "  z_target: 0.0",
                "  z_substrate: 0.02",
                "  r_max: 0.03",
                "grid:",
                "  nr: 2",
                "  nz: 3",
                "gas:",
                '  species: "Ar"',
                "  pressure_pa: 1.0",
                "  temperature_k: 300.0",
                "target:",
                '  material: "Ti"',
                "pulse:",
                "  voltage_v: 625.0",
                "  t_pulse_us: 40.0",
                "  t_total_us: 300.0",
                "particles:",
                "  ppc: 2",
                "  n0_electron: 1.0e+14",
                "  n0_ion: 1.0e+14",
                "  te_ev: 3.0",
                "  ti_ev: 0.1",
                "time:",
                "  dt: 5.0e-12",
                "  n_steps: 10",
            ],
        )
        + "\n",
    )

    cfg = load_pic_config(config_path)
    sim = build_simulation(cfg)

    assert cfg.surface_package is not None
    assert cfg.target.cohesive_energy_ev == pytest.approx(4.89)
    assert cfg.target.sputter_yield_a == pytest.approx(0.0813)
    assert sim["target"].material_mass == pytest.approx(M_TI)
    assert sim["species_map"]["Ti"].species.mass == pytest.approx(M_TI)
