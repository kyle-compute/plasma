"""Tests for magnetic field map ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from plasma.data.magnetic_field import load_magnetic_field_map, validate_field_map_shape
from plasma.pic.config import load_pic_config


def test_load_yaml_magnetic_field_map() -> None:
    br_grid, bz_grid = load_magnetic_field_map("data/field_maps/sample_magnetron_map.yaml")

    assert br_grid.shape == (2, 3)
    assert bz_grid.shape == (2, 3)
    assert float(br_grid[1, 2]) == pytest.approx(0.05)
    assert float(bz_grid[0, 1]) == pytest.approx(0.11)


def test_validate_field_map_shape() -> None:
    br_grid, bz_grid = load_magnetic_field_map("data/field_maps/sample_magnetron_map.yaml")
    validate_field_map_shape(br_grid, bz_grid, n_r=2, n_z=3)

    with pytest.raises(ValueError):
        validate_field_map_shape(br_grid, bz_grid, n_r=3, n_z=3)


def test_pic_config_resolves_field_map_paths(tmp_path) -> None:
    fixture = Path("data/field_maps/sample_magnetron_map.yaml").resolve()
    config_path = tmp_path / "pic_with_map.yaml"
    config_path.write_text(
        "\n".join(
            [
                'name: "pic_with_map"',
                'model: "pic"',
                "geometry:",
                "  r_target: 0.05",
                "  r_inner: 0.015",
                "  r_outer: 0.04",
                "  z_target: 0.0",
                "  z_substrate: 0.1",
                "  r_max: 0.06",
                "grid:",
                "  nr: 1",
                "  nz: 2",
                "gas:",
                '  species: "Ar"',
                "  pressure_pa: 1.0",
                "target:",
                '  material: "Cu"',
                "  cohesive_energy_ev: 3.49",
                "  sputter_yield_a: 0.1421",
                "  sputter_yield_b: 0.468",
                "pulse:",
                "  voltage_v: 600.0",
                "  t_pulse_us: 100.0",
                "  t_total_us: 300.0",
                "particles:",
                "  ppc: 10",
                "time:",
                "  dt: 1.0e-11",
                "  n_steps: 10",
                "magnetic_field:",
                f'  map_file: "{fixture}"',
            ],
        )
        + "\n",
    )

    cfg = load_pic_config(config_path)
    assert cfg.magnetic_field.map_file == str(fixture)
