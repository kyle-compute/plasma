from __future__ import annotations

import numpy as np

from plasma.live.contracts import LiveSeries
from plasma.viewer.presentation import (
    grouped_series,
    ordered_field_names,
    preferred_field_name,
    preset_field_name,
    tone_map_field,
)


def test_field_order_and_preferred_defaults_prioritize_emissivity():
    names = ["phi_v", "e_z_v_m", "emissivity_arb", "target_activity_arb", "sputter_source_arb"]
    assert ordered_field_names(names) == ["target_activity_arb", "sputter_source_arb", "emissivity_arb", "e_z_v_m", "phi_v"]
    assert preferred_field_name(names) == "target_activity_arb"
    assert preset_field_name(names, "Fields") == "e_z_v_m"
    assert preset_field_name(names, "HiPIMS Monitor") == "target_activity_arb"


def test_grouped_series_preserves_domain_specific_diagnostic_panels():
    series = {
        "target_voltage_v": LiveSeries(x=[0.0], y=[1.0]),
        "electron_particles": LiveSeries(x=[0.0], y=[10.0]),
        "ionization_collisions": LiveSeries(x=[0.0], y=[2.0]),
        "custom_trace": LiveSeries(x=[0.0], y=[3.0]),
    }

    groups = grouped_series(series)

    assert [title for title, _ in groups] == ["Pulse", "Transport", "Additional"]
    assert [name for name, _ in groups[0][1]] == ["target_voltage_v"]
    assert [name for name, _ in groups[1][1]] == ["electron_particles", "ionization_collisions"]
    assert [name for name, _ in groups[2][1]] == ["custom_trace"]


def test_tone_map_field_uses_stable_normalized_ranges():
    diverging = tone_map_field("e_z_v_m", np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float64))
    assert diverging.gradient_name == "diverging"
    assert np.isclose(diverging.image[0, 1], 0.5, atol=1.0e-6)

    emissive = tone_map_field("emissivity_arb", np.asarray([[0.0, 1.0, 4.0]], dtype=np.float64))
    assert emissive.gradient_name == "glow"
    assert float(emissive.image.min()) >= 0.0
    assert float(emissive.image.max()) <= 1.0
