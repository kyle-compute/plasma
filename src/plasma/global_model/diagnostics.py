"""Post-run diagnostics for the 0D ionization-region model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plasma.contracts.cases import Provenance
from plasma.core.constants import E_CHARGE
from plasma.data.reactions import ReactionSet
from plasma.global_model.rate_equations import compute_reaction_rates, state_to_densities
from plasma.global_model.state import (
    DEFAULT_STATE_LAYOUT,
    StateLayout,
    electron_density,
    electron_temperature_ev,
)
from plasma.global_model.transport import compute_beta_t, sputter_flux, thermal_velocity


@dataclass
class MetricSeries:
    """One named transport or deposition metric."""

    values: NDArray[np.float64]
    unit: str
    provenance: Provenance
    description: str


@dataclass
class IRMDiagnostics:
    """Structured diagnostics attached to an IRM result."""

    series: dict[str, MetricSeries] = field(default_factory=dict)

    def values(self, name: str) -> NDArray[np.float64]:
        return self.series[name].values

    def summary(self) -> dict[str, float]:
        summary = {
            "peak_electron_density": float(self.series["electron_density"].values.max()),
            "peak_current_a": float(self.series["current_a"].values.max()),
            "peak_circuit_current_a": float(self.series["current_a"].values.max()),
            "peak_model_current_proxy_a": float(self.series["model_current_proxy_a"].values.max()),
            "peak_target_voltage_v": float(self.series["target_voltage_v"].values.max()),
            "peak_alpha_t": float(self.series["alpha_t"].values.max()),
            "final_xi_t": float(self.series["xi_t"].values[-1]),
            "peak_deposition_flux_m2s": float(self.series["deposition_flux_m2s"].values.max()),
        }
        if "reference_current_a" in self.series:
            reference = self.series["reference_current_a"].values
            current = self.series["current_a"].values
            proxy = self.series["model_current_proxy_a"].values
            summary["current_rmse_a"] = float(np.sqrt(np.mean((current - reference) ** 2)))
            summary["current_peak_ratio"] = float(np.max(current) / max(np.max(reference), 1e-30))
            summary["current_peak_time_error_us"] = float(
                abs(
                    self.series["time_reference_s"].values[np.argmax(current)]
                    - self.series["time_reference_s"].values[np.argmax(reference)]
                )
                * 1e6,
            )
            summary["current_proxy_rmse_a"] = float(np.sqrt(np.mean((proxy - reference) ** 2)))
            summary["current_proxy_peak_ratio"] = float(np.max(proxy) / max(np.max(reference), 1e-30))
            summary["current_proxy_peak_time_error_us"] = float(
                abs(
                    self.series["time_reference_s"].values[np.argmax(proxy)]
                    - self.series["time_reference_s"].values[np.argmax(reference)]
                )
                * 1e6,
            )
        return summary


def build_irm_diagnostics(
    time_s: NDArray[np.float64],
    states: NDArray[np.float64],
    *,
    reactions: ReactionSet,
    waveform,
    geom,
    yield_gas,
    yield_self,
    gas_ion_mass_kg: float,
    metal_mass_kg: float,
    drive_current_func,
    model_current_func,
    target_voltage_func,
    plasma_resistance_func,
    current_provenance: Provenance,
    state_layout: StateLayout = DEFAULT_STATE_LAYOUT,
) -> IRMDiagnostics:
    """Compute transport and calibration diagnostics from the solved 0D state."""

    source_voltage = np.asarray([abs(float(waveform.V(t))) for t in time_s], dtype=np.float64)
    target_voltage = np.asarray(
        [float(target_voltage_func(t, row)) for t, row in zip(time_s, states, strict=False)],
        dtype=np.float64,
    )
    plasma_resistance = np.asarray([float(plasma_resistance_func(row)) for row in states], dtype=np.float64)
    current = np.asarray(
        [float(drive_current_func(t, row)) for t, row in zip(time_s, states, strict=False)],
        dtype=np.float64,
    )
    model_current = np.asarray(
        [float(model_current_func(t, row)) for t, row in zip(time_s, states, strict=False)],
        dtype=np.float64,
    )
    reference_current = np.asarray([abs(float(waveform.I(t))) for t in time_s], dtype=np.float64)
    electron_density_series = np.asarray(
        [electron_density(state_to_densities(np.maximum(row, 0.0), layout=state_layout), layout=state_layout) for row in states],
        dtype=np.float64,
    )
    electron_density_cold = np.asarray(states[:, state_layout.indices["e_cold"]], dtype=np.float64)
    electron_density_hot = np.asarray(states[:, state_layout.indices["e_hot"]], dtype=np.float64)
    te_cold_series = np.asarray(
        [electron_temperature_ev(np.maximum(row, 0.0), "cold", layout=state_layout) for row in states],
        dtype=np.float64,
    )
    te_hot_series = np.asarray(
        [electron_temperature_ev(np.maximum(row, 0.0), "hot", layout=state_layout) for row in states],
        dtype=np.float64,
    )

    alpha_t = np.zeros_like(time_s, dtype=np.float64)
    beta_t = np.zeros_like(time_s, dtype=np.float64)
    xi_t = np.zeros_like(time_s, dtype=np.float64)
    epsilon_ti_ev = np.zeros_like(time_s, dtype=np.float64)
    deposition_flux = np.zeros_like(time_s, dtype=np.float64)
    sputtered_influx = np.zeros_like(time_s, dtype=np.float64)
    ion_escape_rate = np.zeros_like(time_s, dtype=np.float64)
    ion_back_attraction_rate = np.zeros_like(time_s, dtype=np.float64)
    neutral_escape_rate = np.zeros_like(time_s, dtype=np.float64)

    neutral_vth = thermal_velocity(0.5, metal_mass_kg)

    for idx, (row, voltage_v, current_a) in enumerate(
        zip(states, target_voltage, current, strict=False)
    ):
        densities = state_to_densities(np.maximum(row, 0.0), layout=state_layout)
        te_cold_ev = te_cold_series[idx]
        te_hot_ev = te_hot_series[idx]
        rates = compute_reaction_rates(densities, reactions, te_cold_ev, te_hot_ev)
        metal_ion_prod = _metal_ionization_rate(reactions, rates, state_layout=state_layout)
        sputtered = _sputtered_influx(
            densities,
            te_cold_ev,
            voltage_v,
            yield_gas,
            yield_self,
            geom.area_target,
            geom.volume,
            gas_ion_mass_kg,
            metal_mass_kg,
            state_layout=state_layout,
        )
        flux_terms = _resolved_flux_terms(
            densities,
            te_cold_ev,
            voltage_v,
            geom,
            metal_mass_kg,
            state_layout=state_layout,
        )
        neutral_metal = sum(max(densities.get(name, 0.0), 0.0) for name in state_layout.metal_neutral_species)

        sputtered_influx[idx] = sputtered
        ion_escape_rate[idx] = flux_terms["metal_ion_escape"]
        ion_back_attraction_rate[idx] = flux_terms["metal_ion_back_attraction"]
        neutral_escape_rate[idx] = flux_terms["metal_neutral_escape"]
        alpha_t[idx] = np.clip(metal_ion_prod / max(sputtered, 1e-30), 0.0, 1.0)
        beta_t[idx] = ion_back_attraction_rate[idx] / max(ion_back_attraction_rate[idx] + ion_escape_rate[idx], 1e-30)
        xi_t[idx] = ion_escape_rate[idx] / max(ion_escape_rate[idx] + neutral_escape_rate[idx], 1e-30)
        epsilon_ti_ev[idx] = abs(voltage_v * current_a) / (E_CHARGE * max(ion_escape_rate[idx] * geom.volume, 1e-30))
        deposition_flux[idx] = 0.25 * neutral_metal * neutral_vth + ion_escape_rate[idx] * geom.z_ir

    diagnostics = IRMDiagnostics(
        series={
            "electron_density": MetricSeries(electron_density_series, "m^-3", "model-derived", "Total electron density in the ionization region."),
            "electron_density_cold": MetricSeries(electron_density_cold, "m^-3", "model-derived", "Cold-electron density in the ionization region."),
            "electron_density_hot": MetricSeries(electron_density_hot, "m^-3", "model-derived", "Hot-electron density in the ionization region."),
            "electron_temperature_ev": MetricSeries(te_cold_series, "eV", "model-derived", "Cold-electron temperature used for bulk rate coefficients."),
            "electron_temperature_hot_ev": MetricSeries(te_hot_series, "eV", "model-derived", "Effective hot-electron temperature used for secondary-electron chemistry."),
            "source_voltage_v": MetricSeries(source_voltage, "V", _waveform_provenance(waveform), "Applied source voltage waveform magnitude."),
            "target_voltage_v": MetricSeries(target_voltage, "V", "model-derived", "Voltage dropped across the plasma/sheath from the circuit model."),
            "plasma_resistance_ohm": MetricSeries(plasma_resistance, "ohm", "model-derived", "Effective plasma resistance used by the circuit model."),
            "current_a": MetricSeries(current, "A", current_provenance, "Circuit-coupled discharge current."),
            "model_current_proxy_a": MetricSeries(model_current, "A", "heuristic", "Legacy density-based current proxy for comparison."),
            "alpha_t": MetricSeries(alpha_t, "1", "model-derived", "Ionization probability of sputtered target material from resolved production and sputter source terms."),
            "beta_t": MetricSeries(beta_t, "1", "heuristic", "Back-attraction probability from resolved metal-ion loss channels."),
            "xi_t": MetricSeries(xi_t, "1", "model-derived", "Resolved escaping metal-ion fraction relative to total metal outflow."),
            "epsilon_ti_ev": MetricSeries(epsilon_ti_ev, "eV/ion", "model-derived", "Absorbed electrical energy per escaping metal ion."),
            "deposition_flux_m2s": MetricSeries(deposition_flux, "m^-2 s^-1", "model-derived", "Resolved metal outflow proxy combining neutral escape and escaping metal-ion flux."),
            "sputtered_source_m3s": MetricSeries(sputtered_influx, "m^-3 s^-1", "model-derived", "Target-metal sputter source term entering the ionization region."),
            "metal_ion_escape_m3s": MetricSeries(ion_escape_rate, "m^-3 s^-1", "model-derived", "Metal-ion escape rate from the ionization region."),
            "metal_ion_back_attraction_m3s": MetricSeries(ion_back_attraction_rate, "m^-3 s^-1", "heuristic", "Metal-ion back-attraction rate to the target."),
            "time_reference_s": MetricSeries(np.asarray(time_s, dtype=np.float64), "s", _waveform_provenance(waveform), "Reference time axis for calibration metrics."),
        }
    )
    if np.any(reference_current > 0.0):
        diagnostics.series["reference_current_a"] = MetricSeries(
            reference_current,
            "A",
            waveform.provenance,
            "Reference discharge current from the literature-fit waveform.",
        )
    return diagnostics


def _metal_ionization_rate(
    reactions: ReactionSet,
    reaction_rates: dict[str, float],
    *,
    state_layout: StateLayout,
) -> float:
    total = 0.0
    for reaction in reactions:
        if not any(product in state_layout.metal_ion_species for product in reaction.products):
            continue
        if not any(reactant in state_layout.metal_neutral_species for reactant in reaction.reactants):
            continue
        total += reaction_rates.get(reaction.id, 0.0)
    return total


def _sputtered_influx(
    densities: dict[str, float],
    te_ev: float,
    voltage_v: float,
    yield_gas,
    yield_self,
    area_target: float,
    volume: float,
    gas_ion_mass_kg: float,
    metal_mass_kg: float,
    *,
    state_layout: StateLayout,
) -> float:
    sput_ar = sputter_flux(
        densities.get(state_layout.primary_argon_ion, 0.0),
        te_ev,
        gas_ion_mass_kg,
        voltage_v,
        yield_gas,
        area_target,
        volume,
    )
    sput_self = sputter_flux(
        densities.get(state_layout.primary_metal_ion, 0.0),
        te_ev,
        metal_mass_kg,
        voltage_v,
        yield_self,
        area_target,
        volume,
    )
    return sput_ar + sput_self


def _resolved_flux_terms(
    densities: dict[str, float],
    te_ev: float,
    voltage_v: float,
    geom,
    metal_mass_kg: float,
    *,
    state_layout: StateLayout,
) -> dict[str, float]:
    beta_guess = compute_beta_t(voltage_v, te_ev)
    metal_ion_density = sum(max(densities.get(name, 0.0), 0.0) for name in state_layout.metal_ion_species)
    ion_speed = np.sqrt(E_CHARGE * te_ev / max(metal_mass_kg, 1e-30))
    escape = metal_ion_density * ion_speed * geom.area_loss / geom.volume
    back_attraction = metal_ion_density * beta_guess * ion_speed * geom.area_target / geom.volume
    neutral_density = sum(max(densities.get(name, 0.0), 0.0) for name in state_layout.metal_neutral_species)
    neutral_escape = neutral_density * thermal_velocity(0.5, metal_mass_kg) * geom.area_loss / (4.0 * geom.volume)
    return {
        "metal_ion_escape": escape,
        "metal_ion_back_attraction": back_attraction,
        "metal_neutral_escape": neutral_escape,
    }


def _waveform_provenance(waveform) -> Provenance:
    return waveform.provenance
