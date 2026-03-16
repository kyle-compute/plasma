"""Post-run diagnostics for the 0D ionization-region model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plasma.contracts.cases import Provenance
from plasma.core.constants import E_CHARGE
from plasma.data.reactions import ReactionSet
from plasma.global_model.rate_equations import STATE_INDICES, compute_reaction_rates, state_to_densities
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
            "peak_model_current_proxy_a": float(self.series["model_current_proxy_a"].values.max()),
            "peak_alpha_t": float(self.series["alpha_t"].values.max()),
            "final_xi_t": float(self.series["xi_t"].values[-1]),
            "peak_deposition_flux_m2s": float(self.series["deposition_flux_m2s"].values.max()),
        }
        if "reference_current_a" in self.series:
            reference = self.series["reference_current_a"].values
            model = self.series["model_current_proxy_a"].values
            summary["current_proxy_rmse_a"] = float(np.sqrt(np.mean((model - reference) ** 2)))
            summary["current_proxy_peak_ratio"] = float(
                np.max(model) / max(np.max(reference), 1e-30),
            )
            summary["current_proxy_peak_time_error_us"] = float(
                abs(
                    self.series["time_reference_s"].values[np.argmax(model)]
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
) -> IRMDiagnostics:
    """Compute transport and usability diagnostics from the solved 0D state."""

    voltage = np.asarray([float(waveform.V(t)) for t in time_s], dtype=np.float64)
    current = np.asarray([float(drive_current_func(t, row)) for t, row in zip(time_s, states)], dtype=np.float64)
    model_current = np.asarray(
        [float(model_current_func(t, row)) for t, row in zip(time_s, states)],
        dtype=np.float64,
    )
    reference_current = np.asarray([float(waveform.I(t)) for t in time_s], dtype=np.float64)
    electron_density = np.asarray(states[:, STATE_INDICES["e_cold"]], dtype=np.float64)

    alpha_t = np.zeros_like(time_s, dtype=np.float64)
    beta_t = np.zeros_like(time_s, dtype=np.float64)
    xi_t = np.zeros_like(time_s, dtype=np.float64)
    epsilon_ti_ev = np.zeros_like(time_s, dtype=np.float64)
    deposition_flux = np.zeros_like(time_s, dtype=np.float64)

    metal_neutral_keys = [name for name in STATE_INDICES if name.startswith("Cu") and not name.endswith("+")]
    metal_ion_keys = [name for name in STATE_INDICES if name.startswith("Cu") and name.endswith("+")]

    neutral_vth = thermal_velocity(0.5, metal_mass_kg)

    for idx, (t, row, v_d, i_d) in enumerate(zip(time_s, states, voltage, current)):
        densities = state_to_densities(np.maximum(row, 0.0))
        n_e = max(densities.get("e_cold", 1.0), 1.0)
        te_ev = max((2.0 / 3.0) * row[STATE_INDICES["energy"]] / n_e, 0.1)
        rxn_rates = compute_reaction_rates(densities, reactions, te_ev)

        metal_ion_prod = _metal_ionization_rate(reactions, rxn_rates)
        sputtered = _sputtered_influx(
            densities,
            te_ev,
            v_d,
            yield_gas,
            yield_self,
            geom.area_target,
            geom.volume,
            gas_ion_mass_kg,
            metal_mass_kg,
        )
        total_metal = sum(max(densities.get(name, 0.0), 0.0) for name in metal_neutral_keys + metal_ion_keys)
        ionized_metal = sum(max(densities.get(name, 0.0), 0.0) for name in metal_ion_keys)
        neutral_metal = sum(max(densities.get(name, 0.0), 0.0) for name in metal_neutral_keys)

        alpha_t[idx] = np.clip(metal_ion_prod / max(sputtered, 1e-30), 0.0, 1.0)
        beta_t[idx] = compute_beta_t(v_d, te_ev)
        xi_t[idx] = ionized_metal / max(total_metal, 1e-30)
        epsilon_ti_ev[idx] = abs(v_d * i_d) / (E_CHARGE * max(metal_ion_prod * geom.volume, 1e-30))
        deposition_flux[idx] = 0.25 * neutral_metal * neutral_vth

    diagnostics = IRMDiagnostics(
        series={
            "electron_density": MetricSeries(
                values=electron_density,
                unit="m^-3",
                provenance="heuristic",
                description="Electron density in the ionization region.",
            ),
            "voltage_v": MetricSeries(
                values=voltage,
                unit="V",
                provenance=_waveform_provenance(waveform),
                description="Applied discharge voltage waveform.",
            ),
            "current_a": MetricSeries(
                values=current,
                unit="A",
                provenance=_current_provenance(waveform),
                description="Measured or model-derived discharge current.",
            ),
            "model_current_proxy_a": MetricSeries(
                values=model_current,
                unit="A",
                provenance="heuristic",
                description="Density-based model current proxy for comparison against literature current.",
            ),
            "alpha_t": MetricSeries(
                values=alpha_t,
                unit="1",
                provenance="heuristic",
                description="Ionization probability of sputtered target material.",
            ),
            "beta_t": MetricSeries(
                values=beta_t,
                unit="1",
                provenance="heuristic",
                description="Back-attraction probability of target ions.",
            ),
            "xi_t": MetricSeries(
                values=xi_t,
                unit="1",
                provenance="heuristic",
                description="Ionized metal fraction in the ionization region.",
            ),
            "epsilon_ti_ev": MetricSeries(
                values=epsilon_ti_ev,
                unit="eV/ion",
                provenance="heuristic",
                description="Absorbed energy per produced metal ion.",
            ),
            "deposition_flux_m2s": MetricSeries(
                values=deposition_flux,
                unit="m^-2 s^-1",
                provenance="heuristic",
                description="Ballistic neutral metal flux proxy leaving the ionization region.",
            ),
            "time_reference_s": MetricSeries(
                values=np.asarray(time_s, dtype=np.float64),
                unit="s",
                provenance=_waveform_provenance(waveform),
                description="Reference time axis for calibration metrics.",
            ),
        }
    )
    if np.any(np.abs(reference_current) > 0.0):
        diagnostics.series["reference_current_a"] = MetricSeries(
            values=reference_current,
            unit="A",
            provenance=waveform.provenance,
            description="Reference discharge current from the literature-fit waveform.",
        )
    return diagnostics


def _metal_ionization_rate(reactions: ReactionSet, reaction_rates: dict[str, float]) -> float:
    total = 0.0
    for rxn in reactions:
        if not any(product.startswith("Cu") and product.endswith("+") for product in rxn.products):
            continue
        if not any(reactant.startswith("Cu") and not reactant.endswith("+") for reactant in rxn.reactants):
            continue
        total += reaction_rates.get(rxn.id, 0.0)
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
) -> float:
    sput_ar = sputter_flux(
        densities.get("Ar+", 0.0),
        te_ev,
        gas_ion_mass_kg,
        voltage_v,
        yield_gas,
        area_target,
        volume,
    )
    sput_self = sputter_flux(
        densities.get("Cu+", 0.0),
        te_ev,
        metal_mass_kg,
        voltage_v,
        yield_self,
        area_target,
        volume,
    )
    return sput_ar + sput_self


def _waveform_provenance(waveform) -> Provenance:
    return waveform.provenance


def _current_provenance(waveform) -> Provenance:
    return waveform.provenance if np.any(np.abs(waveform.current_a) > 0.0) else "heuristic"
