from __future__ import annotations

import numpy as np

from plasma.live.hipims_monitor import monitor_message, peak_radius_m, pulse_phase_label, safe_ratio


def test_pulse_phase_label_tracks_ramp_and_decay() -> None:
    phase, code = pulse_phase_label([-100.0, -400.0], [0.4, 0.2])
    assert phase == "ramp"
    assert code == 1.0

    phase, code = pulse_phase_label([-600.0, -600.0, -600.0], [0.4, 0.9, 0.7])
    assert phase == "decay"
    assert code == 3.0


def test_monitor_message_and_peak_radius_are_stable() -> None:
    field = np.zeros((4, 3), dtype=np.float64)
    field[2, 1] = 5.0
    radius = peak_radius_m(np.linspace(0.0, 0.03, 4), field)

    assert np.isclose(radius, 0.02)
    assert safe_ratio(3.0, 0.0) == 0.0
    assert "Pulse phase: peak" in monitor_message("peak", radius, 0.25)
