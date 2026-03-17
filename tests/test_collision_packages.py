"""Tests for versioned Cu/Ar collision packages."""

from __future__ import annotations

from plasma.data.collision_packages import load_channel_cross_section, load_collision_package
from plasma.pic.config import load_pic_config
from plasma.pic.runtime import _update_background_state, build_simulation


def test_public_collision_package_resolves_paths_and_metadata() -> None:
    package = load_collision_package("data/collision_packages/cu_ar_public_v1.yaml")

    assert package.package.name == "cu_ar_public_v1"
    assert any(spec.name == "Cu+" and spec.role == "kinetic" for spec in package.species)
    elastic = next(channel for channel in package.channels if channel.name == "e_Ar_c_elastic")
    assert elastic.cross_section_file is not None
    assert elastic.cross_section_file.endswith("data/cross_sections/lxcat_biagi_e_ar/electron_ar_elastic.tsv")


def test_smoke_collision_package_loads_builtin_cu_channels() -> None:
    package = load_collision_package("data/collision_packages/cu_ar_smoke_v1.yaml")

    ionization = next(channel for channel in package.channels if channel.name == "e_Cu_ground_ionization")
    sigma = load_channel_cross_section(ionization)

    assert sigma is not None
    assert sigma.name == "e_Cu_ionization_synthetic"
    assert sigma.max_sigma > 0.0


def test_background_state_updates_from_active_argon_channels() -> None:
    cfg = load_pic_config("config/hipims_cu_ar_pic_public.yaml")
    sim = build_simulation(cfg)

    initial_ar_c = sim["background_state"]["Ar_c"]
    initial_ar_m = sim["background_state"]["Ar_m"]
    _update_background_state(
        sim["background_state"],
        sim["collision_package"],
        {"e_Ar_c_excitation": 1.0e10},
        sim["grid"],
        sim["mcc_handlers"],
    )

    assert sim["background_state"]["Ar_c"] < initial_ar_c
    assert sim["background_state"]["Ar_m"] > initial_ar_m
    electron_handler = sim["mcc_handlers"]["electron"]
    densities = {
        child.background_species_name: child.background_density
        for child in getattr(electron_handler, "handlers", [electron_handler])
    }
    assert densities["Ar_c"] == sim["background_state"]["Ar_c"]
    assert densities["Cu"] == sim["background_state"]["Cu"]
