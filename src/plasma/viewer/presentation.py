"""Pure presentation helpers for the live plasma viewer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma.live.contracts import LiveSeries

FIELD_ORDER = (
    "target_activity_arb",
    "sputter_source_arb",
    "see_source_arb",
    "ionization_source_arb",
    "substrate_flux_proxy_arb",
    "emissivity_arb",
    "electron_density_m3",
    "ion_density_m3",
    "cu_density_m3",
    "e_mag_v_m",
    "e_z_v_m",
    "e_r_v_m",
    "b_mag_t",
    "b_z_t",
    "b_r_t",
    "rho_c_m3",
    "phi_v",
)

FIELD_PRESETS: dict[str, tuple[str, ...]] = {
    "HiPIMS Monitor": ("target_activity_arb", "sputter_source_arb", "substrate_flux_proxy_arb", "emissivity_arb"),
    "Source": ("sputter_source_arb", "see_source_arb", "ionization_source_arb", "target_activity_arb"),
    "Fields": ("e_mag_v_m", "e_z_v_m", "e_r_v_m", "b_mag_t", "b_z_t", "b_r_t", "phi_v"),
    "Densities": ("electron_density_m3", "ion_density_m3", "cu_density_m3", "rho_c_m3"),
    "Surface Activity": ("target_activity_arb", "sputter_source_arb", "see_source_arb", "phi_v"),
}

SERIES_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Pulse",
        ("target_voltage_v", "field_max_v_m", "emissivity_total_arb", "electron_mean_energy_ev", "ar_ion_mean_energy_ev"),
    ),
    (
        "Source",
        (
            "target_impacts_window",
            "see_window",
            "sputtered_window",
            "see_per_target_impact",
            "sputtered_per_target_impact",
            "source_activity_total_arb",
            "racetrack_peak_r_m",
        ),
    ),
    (
        "Transport",
        (
            "substrate_hits_total",
            "substrate_mean_energy_ev",
            "substrate_flux_total_arb",
            "electron_particles",
            "ar_ion_particles",
            "cu_neutral_particles",
            "collisions_per_sample",
            "excitation_collisions",
            "ionization_collisions",
            "charge_exchange_collisions",
        ),
    ),
)


@dataclass(frozen=True)
class FieldRenderStyle:
    """Normalized image data and the gradient family to use."""

    image: np.ndarray
    gradient_name: str


def ordered_field_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return fields in a stable, intent-driven order."""

    available = set(names)
    ordered = [name for name in FIELD_ORDER if name in available]
    ordered.extend(sorted(name for name in names if name not in FIELD_ORDER))
    return ordered


def preferred_field_name(names: list[str] | tuple[str, ...], current: str | None = None) -> str | None:
    """Keep the current field when valid, else choose the best default."""

    if current and current in names:
        return current
    for name in FIELD_ORDER:
        if name in names:
            return name
    return names[0] if names else None


def preset_field_name(names: list[str] | tuple[str, ...], preset: str, current: str | None = None) -> str | None:
    """Choose the first field available in the selected preset."""

    for name in FIELD_PRESETS.get(preset, ()):
        if name in names:
            return name
    return preferred_field_name(names, current=current)


def grouped_series(series: dict[str, LiveSeries]) -> list[tuple[str, list[tuple[str, LiveSeries]]]]:
    """Group related live series into diagnostic panels."""

    groups: list[tuple[str, list[tuple[str, LiveSeries]]]] = []
    seen: set[str] = set()
    for title, names in SERIES_GROUPS:
        entries = [(name, series[name]) for name in names if name in series]
        if entries:
            seen.update(name for name, _ in entries)
            groups.append((title, entries))
    leftovers = [(name, payload) for name, payload in series.items() if name not in seen]
    if leftovers:
        groups.append(("Additional", leftovers))
    return groups


def format_metric(value: float) -> str:
    """Format metrics for the compact HUD."""

    magnitude = abs(value)
    if magnitude >= 1.0e4 or (0.0 < magnitude < 1.0e-2):
        return f"{value:.2e}"
    return f"{value:.2f}"


def tone_map_field(name: str, values: np.ndarray) -> FieldRenderStyle:
    """Convert raw field values into a stable normalized image."""

    data = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if data.size == 0:
        return FieldRenderStyle(np.zeros_like(data), "glow")
    if name in {"phi_v", "rho_c_m3", "e_r_v_m", "e_z_v_m", "b_r_t", "b_z_t"}:
        return FieldRenderStyle(_diverging_map(data), "diverging")
    if name in {"target_activity_arb", "sputter_source_arb"}:
        return FieldRenderStyle(_positive_map(data, percentile=99.0, gamma=0.58), "surface")
    if name.endswith("_density_m3"):
        return FieldRenderStyle(_positive_map(data, percentile=98.5, gamma=0.82), "density")
    if name in {"emissivity_arb", "see_source_arb", "ionization_source_arb"}:
        return FieldRenderStyle(_positive_map(data, percentile=99.2, gamma=0.62), "glow")
    return FieldRenderStyle(_positive_map(np.abs(data), percentile=98.0, gamma=0.78), "field")


def _positive_map(values: np.ndarray, *, percentile: float, gamma: float) -> np.ndarray:
    positive = np.clip(values, 0.0, None)
    if not np.any(positive > 0.0):
        return np.zeros_like(positive)
    high = float(np.nanpercentile(positive[positive > 0.0], percentile))
    if high <= 0.0:
        return np.zeros_like(positive)
    normalized = np.clip(positive / high, 0.0, 1.0)
    return np.power(normalized, gamma)


def _diverging_map(values: np.ndarray) -> np.ndarray:
    peak = float(np.nanpercentile(np.abs(values), 97.5))
    if peak <= 0.0:
        return np.full_like(values, 0.5)
    normalized = np.clip(values / peak, -1.0, 1.0)
    signed = np.sign(normalized) * np.power(np.abs(normalized), 0.9)
    return 0.5 * (signed + 1.0)
