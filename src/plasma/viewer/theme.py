"""Color and theme constants for the live plasma viewers."""

from __future__ import annotations

SERIES_COLORS = {
    "target_voltage_v": "#8fd3ff",
    "field_max_v_m": "#ff7b72",
    "emissivity_total_arb": "#ffd166",
    "electron_mean_energy_ev": "#8ef0ff",
    "ar_ion_mean_energy_ev": "#f6bd60",
    "target_impacts_window": "#ff8c42",
    "see_window": "#9bf6ff",
    "sputtered_window": "#f4a261",
    "see_per_target_impact": "#b8f2e6",
    "sputtered_per_target_impact": "#ffe066",
    "source_activity_total_arb": "#f28482",
    "racetrack_peak_r_m": "#cdb4db",
    "substrate_mean_energy_ev": "#7ad7ff",
    "substrate_flux_total_arb": "#72efdd",
    "electron_particles": "#57c7ff",
    "ar_ion_particles": "#ffb347",
    "cu_neutral_particles": "#d98f43",
    "substrate_hits_total": "#d4f5ff",
    "collisions_per_sample": "#b9c0ff",
    "excitation_collisions": "#7ae582",
    "ionization_collisions": "#ffe66d",
    "charge_exchange_collisions": "#c77dff",
}

SPECIES_COLORS = {
    "electron": "#57c7ff",
    "Ar+": "#ffb347",
    "Cu": "#d98f43",
}

FIELD_GRADIENTS = {
    "glow": [
        (0.0, (4, 5, 10, 255)),
        (0.12, (26, 9, 40, 255)),
        (0.32, (90, 18, 110, 255)),
        (0.58, (37, 120, 210, 255)),
        (0.82, (255, 158, 69, 255)),
        (1.0, (255, 248, 210, 255)),
    ],
    "surface": [
        (0.0, (4, 5, 10, 255)),
        (0.2, (56, 14, 36, 255)),
        (0.5, (160, 42, 76, 255)),
        (0.78, (255, 122, 69, 255)),
        (1.0, (255, 235, 195, 255)),
    ],
    "density": [
        (0.0, (4, 7, 14, 255)),
        (0.2, (16, 42, 78, 255)),
        (0.5, (32, 112, 174, 255)),
        (0.78, (110, 214, 255, 255)),
        (1.0, (245, 253, 255, 255)),
    ],
    "field": [
        (0.0, (4, 7, 14, 255)),
        (0.18, (14, 34, 56, 255)),
        (0.45, (28, 122, 170, 255)),
        (0.75, (120, 226, 215, 255)),
        (1.0, (245, 253, 255, 255)),
    ],
    "diverging": [
        (0.0, (20, 48, 120, 255)),
        (0.2, (47, 128, 237, 255)),
        (0.5, (12, 16, 24, 255)),
        (0.8, (255, 139, 61, 255)),
        (1.0, (255, 245, 210, 255)),
    ],
}

BACKGROUND = "#06080d"
SURFACE = "#10151d"
SURFACE_ALT = "#151c27"
TEXT = "#edf2f7"
MUTED = "#8190a7"
BORDER = "#263140"
BADGE = "#0f1824"

QT_STYLE_SHEET = f"""
QMainWindow {{
    background: {BACKGROUND};
}}
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
}}
QFrame#panel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QComboBox, QSpinBox, QPushButton, QLabel {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px;
}}
QCheckBox {{
    color: {TEXT};
    spacing: 6px;
}}
QLabel#title {{
    background: transparent;
    border: none;
    font-size: 15px;
    font-weight: 700;
    padding: 0;
}}
QLabel#subtitle {{
    background: transparent;
    border: none;
    color: {MUTED};
    padding: 0;
}}
QLabel#badge {{
    background: {BADGE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    font-family: monospace;
}}
QLabel#metrics {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    padding: 10px;
    font-family: monospace;
}}
"""
