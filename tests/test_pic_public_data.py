"""Tests for the public-data PIC config and runtime wiring."""

from __future__ import annotations

from plasma.pic.config import load_pic_config
from plasma.pic.runtime import build_simulation


def test_public_pic_config_resolves_normalized_files() -> None:
    cfg = load_pic_config("config/hipims_cu_ar_pic_public.yaml")

    assert cfg.cross_sections.source == "normalized_files"
    assert cfg.collision_package is not None
    assert cfg.collision_package.endswith("data/collision_packages/cu_ar_public_v1.yaml")
    assert cfg.cross_sections.elastic_file is not None
    assert cfg.cross_sections.elastic_file.endswith("electron_ar_elastic.tsv")
    assert cfg.case is not None
    assert cfg.case.inputs[1].provenance == "public-download"
    assert cfg.case.inputs[2].kind == "collision_package"
    assert cfg.case.inputs[2].provenance == "synthetic"
    assert cfg.case.inputs[3].provenance == "surrogate"


def test_build_simulation_uses_public_cross_sections(tmp_path) -> None:
    config_path = tmp_path / "pic_public_small.yaml"
    config_path.write_text(
        "\n".join(
            [
                'name: "pic_public_small"',
                'model: "pic"',
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
                "  permittivity_factor: 10.0",
                "gas:",
                '  species: "Ar"',
                "  pressure_pa: 1.0",
                "  temperature_k: 300.0",
                "target:",
                '  material: "Cu"',
                "  cohesive_energy_ev: 3.49",
                "  sputter_yield_a: 0.1421",
                "  sputter_yield_b: 0.468",
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
                "cross_sections:",
                '  source: "normalized_files"',
                '  elastic_file: "data/cross_sections/lxcat_biagi_e_ar/electron_ar_elastic.tsv"',
                '  excitation_file: "data/cross_sections/lxcat_biagi_e_ar/electron_ar_excitation_total.tsv"',
                '  ionization_file: "data/cross_sections/lxcat_biagi_e_ar/electron_ar_ionization.tsv"',
                'collision_package: "data/collision_packages/cu_ar_public_v1.yaml"',
                "time:",
                "  dt: 5.0e-12",
                "  n_steps: 10",
            ],
        )
        + "\n",
    )

    cfg = load_pic_config(config_path)
    sim = build_simulation(cfg)

    electron_handler = sim["mcc_handlers"]["electron"]
    electron_names = {process.name for process in electron_handler.processes}
    ion_names = {process.name for process in sim["mcc_handlers"]["Ar+"].processes}
    assert electron_names == {
        "e_Ar_c_elastic",
        "e_Ar_c_excitation",
        "e_Ar_c_ionization",
        "e_Cu_ground_excitation",
        "e_Cu_ground_ionization",
    }
    assert ion_names == {"Ar+_Ar_c_cx", "Ar+_Cu_cx"}
