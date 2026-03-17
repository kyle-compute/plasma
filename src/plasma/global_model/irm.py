"""Ionization Region Model (IRM) for material-aware Ar/metal HiPIMS discharge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plasma.core.config import SimulationConfig
from plasma.core.constants import E_CHARGE, K_BOLTZMANN, M_AR, PI, material_mass_kg
from plasma.data.reactions import ReactionSet, load_reactions
from plasma.data.sputtering import SputterYield
from plasma.data.waveforms import DischargeWaveform, load_waveform, make_square_pulse
from plasma.global_model.circuit import (
    circuit_current_rhs,
    discharge_current_from_circuit,
    plasma_resistance_ohm,
    target_voltage_from_circuit,
)
from plasma.global_model.diagnostics import IRMDiagnostics, build_irm_diagnostics
from plasma.global_model.rate_equations import (
    compute_population_reaction_rates,
    species_rhs,
)
from plasma.global_model.state import (
    build_state_layout,
    electron_density,
    electron_temperature_ev,
    ion_charge_density,
    state_to_densities,
)
from plasma.global_model.transport import (
    back_attraction_rate,
    bohm_velocity,
    compute_beta_t,
    electron_thermal_velocity,
    neutral_refill_rate,
    sputter_flux,
)


@dataclass
class IRMGeometry:
    """Pre-computed geometric quantities for the ionization region."""

    r_target: float
    r_inner: float
    r_outer: float
    z_ir: float
    area_target: float
    volume: float
    area_loss: float
    area_wall: float

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> IRMGeometry:
        geometry = cfg.geometry
        area_target = geometry.area_target or PI * (geometry.r_outer**2 - geometry.r_inner**2)
        volume = geometry.volume_ir or area_target * geometry.z_ir
        area_loss = area_target + 2.0 * PI * geometry.r_outer * geometry.z_ir
        return cls(
            r_target=geometry.r_target,
            r_inner=geometry.r_inner,
            r_outer=geometry.r_outer,
            z_ir=geometry.z_ir,
            area_target=area_target,
            volume=volume,
            area_loss=area_loss,
            area_wall=area_loss,
        )


@dataclass
class IRMState:
    """Container for IRM simulation results."""

    time: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    states: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    diagnostics: IRMDiagnostics = field(default_factory=IRMDiagnostics)
    state_layout: object | None = None

    def density(self, species: str) -> NDArray[np.float64]:
        layout = self.state_layout or build_state_layout("Cu")
        return self.states[:, layout.indices[species]]

    @property
    def n_e(self) -> NDArray[np.float64]:
        return self.density("e_cold") + self.density("e_hot")

    @property
    def n_e_cold(self) -> NDArray[np.float64]:
        return self.density("e_cold")

    @property
    def n_e_hot(self) -> NDArray[np.float64]:
        return self.density("e_hot")

    @property
    def current_a(self) -> NDArray[np.float64]:
        layout = self.state_layout or build_state_layout("Cu")
        return self.states[:, layout.indices["current_circuit"]]

    @property
    def te_ev(self) -> NDArray[np.float64]:
        layout = self.state_layout or build_state_layout("Cu")
        density = np.maximum(self.n_e_cold, 1.0)
        energy = self.states[:, layout.indices["energy_cold"]]
        return (2.0 / 3.0) * energy / density

    @property
    def te_hot_ev(self) -> NDArray[np.float64]:
        layout = self.state_layout or build_state_layout("Cu")
        density = np.maximum(self.n_e_hot, 1.0)
        energy = self.states[:, layout.indices["energy_hot"]]
        return (2.0 / 3.0) * energy / density

    def metric(self, name: str) -> NDArray[np.float64]:
        return self.diagnostics.values(name)


class IRM:
    """Volume-averaged Ar/metal discharge model with a circuit-coupled current state."""

    def __init__(
        self,
        config: SimulationConfig,
        reactions: ReactionSet | None = None,
        sputter_yield_gas: SputterYield | None = None,
        sputter_yield_self: SputterYield | None = None,
        waveform: DischargeWaveform | None = None,
    ) -> None:
        self.config = config
        self.geom = IRMGeometry.from_config(config)
        self.state_layout = build_state_layout(config.target.material)
        self.reactions = reactions if reactions is not None else load_reactions(config.reactions_file)
        if self.reactions.species_order and self.reactions.species_order != self.state_layout.species_keys:
            raise ValueError("Reaction package species order does not match the configured IRM state layout")

        self.circuit = config.circuit
        target = config.target
        self.yield_gas = sputter_yield_gas or SputterYield(
            ion="Ar+",
            target=target.material,
            a=target.sputter_yield_a,
            b=target.sputter_yield_b,
            threshold_ev=17.0,
            cohesive_energy_ev=target.cohesive_energy_ev,
        )
        self.yield_self = sputter_yield_self or SputterYield(
            ion=f"{target.material}+",
            target=target.material,
            a=target.self_sputter_yield_a,
            b=target.self_sputter_yield_b,
            threshold_ev=15.0,
            cohesive_energy_ev=target.cohesive_energy_ev,
        )
        self.waveform = waveform or self._load_waveform()
        self.n_gas_0 = config.gas.pressure_pa / (K_BOLTZMANN * config.gas.temperature_k)
        self.mass_ion = M_AR
        self.metal_mass = material_mass_kg(target.material)
        self.secondary_electron_yield = target.secondary_electron_yield
        self.reference_peak_current_a = max(50.0, float(np.max(np.abs(self.waveform.current_a))))

    def _load_waveform(self) -> DischargeWaveform:
        if self.config.pulse.waveform_file:
            return load_waveform(
                self.config.pulse.waveform_file,
                provenance=self._waveform_provenance_from_config(),
            )
        pulse = self.config.pulse
        return make_square_pulse(
            voltage_v=pulse.voltage_v,
            t_pulse_s=pulse.t_pulse_us * 1e-6,
            t_total_s=(pulse.t_pulse_us + pulse.t_afterglow_us) * 1e-6,
        )

    def _waveform_provenance_from_config(self) -> str:
        if self.config.case is None:
            return "measured"
        for input_source in self.config.case.inputs:
            if input_source.kind == "waveform":
                return input_source.provenance
        return "measured"

    def initial_state(self) -> NDArray[np.float64]:
        """Create a seeded Ar/metal state with dual electron populations and circuit current."""

        idx = self.state_layout.indices
        y0 = np.zeros(self.state_layout.n_states)
        y0[idx["Ar_c"]] = self.n_gas_0
        y0[idx[self.state_layout.primary_argon_ion]] = 1.0e15
        y0[idx[self.state_layout.primary_metal_ion]] = 1.0e13
        total_charge = ion_charge_density(state_to_densities(y0, layout=self.state_layout), layout=self.state_layout)
        y0[idx["e_hot"]] = total_charge * 5.0e-3
        y0[idx["e_cold"]] = max(total_charge - y0[idx["e_hot"]], 1.0e12)
        y0[idx["current_circuit"]] = self._reference_current_a(0.0) if self.circuit.mode == "waveform_current" else 0.0
        y0[idx["energy_cold"]] = 1.5 * y0[idx["e_cold"]] * 3.0
        y0[idx["energy_hot"]] = 1.5 * y0[idx["e_hot"]] * 250.0
        return y0

    def _effective_confinement_time(self, te_ev: float, mass_kg: float) -> float:
        return self.geom.volume / (bohm_velocity(te_ev, mass_kg) * 0.3 * self.geom.area_loss + 1e-30)

    def _hot_electron_confinement_time(self, te_ev: float) -> float:
        return self.geom.volume / (electron_thermal_velocity(max(te_ev, 1.0)) * 0.15 * self.geom.area_loss + 1e-30)

    def _hot_electron_thermalization_time(self, gas_density: float) -> float:
        return 5.0e-7 * max(self.n_gas_0 / max(gas_density, 1.0), 0.2)

    def _secondary_electron_source_rate(self, charge_loss_rate: float, pulse_on: bool) -> float:
        return self.secondary_electron_yield * charge_loss_rate if pulse_on else 0.0

    def _secondary_electron_energy_ev(self, target_voltage_v: float) -> float:
        return max(75.0, 0.35 * abs(target_voltage_v))

    def _source_voltage_v(self, t: float) -> float:
        return abs(float(self.waveform.V(t)))

    def _reference_current_a(self, t: float) -> float:
        return abs(float(self.waveform.I(t)))

    def _model_current_proxy(self, voltage_v: float, n_e: float, te_ev: float) -> float:
        if abs(voltage_v) < 10.0:
            return 0.0
        proxy = n_e * E_CHARGE * bohm_velocity(te_ev, M_AR) * self.geom.area_target
        return min(proxy, self.reference_peak_current_a)

    def _plasma_resistance(self, densities: dict[str, float], te_ev: float) -> float:
        gas_density = densities.get("Ar_c", 0.0) + densities.get("Ar_h", 0.0) + densities.get("Ar_w", 0.0)
        return plasma_resistance_ohm(
            n_e=max(electron_density(densities, layout=self.state_layout), 1.0),
            gas_density=max(gas_density, 1.0),
            area_m2=self.geom.area_target,
            length_m=self.geom.z_ir,
            te_ev=te_ev,
            circuit=self.circuit,
        )

    def _current_from_row(self, t: float, row: NDArray[np.float64]) -> float:
        if self.circuit.mode == "waveform_current":
            return self._reference_current_a(t)
        idx = self.state_layout.indices
        densities = state_to_densities(np.maximum(row, 0.0), layout=self.state_layout)
        resistance = self._plasma_resistance(
            densities,
            min(max(electron_temperature_ev(row, "cold", layout=self.state_layout), 0.1), 20.0),
        )
        return discharge_current_from_circuit(
            source_voltage_v=self._source_voltage_v(t),
            current_a=max(float(row[idx["current_circuit"]]), 0.0),
            plasma_resistance_ohm=resistance,
            circuit=self.circuit,
        )

    def _target_voltage_from_row(self, t: float, row: NDArray[np.float64]) -> float:
        current = self._current_from_row(t, row)
        resistance = self._plasma_resistance(
            state_to_densities(np.maximum(row, 0.0), layout=self.state_layout),
            min(max(electron_temperature_ev(row, "cold", layout=self.state_layout), 0.1), 20.0),
        )
        return min(self._source_voltage_v(t), target_voltage_from_circuit(current_a=current, plasma_resistance_ohm=resistance))

    def _plasma_resistance_from_row(self, row: NDArray[np.float64]) -> float:
        return self._plasma_resistance(
            state_to_densities(np.maximum(row, 0.0), layout=self.state_layout),
            min(max(electron_temperature_ev(row, "cold", layout=self.state_layout), 0.1), 20.0),
        )

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Right-hand side of the configured Ar/metal IRM state."""

        idx = self.state_layout.indices
        y = np.maximum(y, 0.0)
        densities = state_to_densities(y, layout=self.state_layout)
        n_cold = max(densities.get("e_cold", 0.0), 0.0)
        n_hot = max(densities.get("e_hot", 0.0), 0.0)
        te_cold_ev = min(max(electron_temperature_ev(y, "cold", layout=self.state_layout), 0.1), 20.0)
        te_hot_ev = self._hot_temperature_ev(y)
        source_voltage_v = self._source_voltage_v(t)
        pulse_on = source_voltage_v > 10.0
        plasma_resistance = self._plasma_resistance(densities, te_cold_ev)
        current_a = self._current_from_row(t, y)
        target_voltage_v = min(
            source_voltage_v,
            target_voltage_from_circuit(current_a=current_a, plasma_resistance_ohm=plasma_resistance),
        )

        reaction_rates = compute_population_reaction_rates(densities, self.reactions, te_cold_ev, te_hot_ev)
        chemistry_rhs = species_rhs(densities, reaction_rates, self.reactions, state_layout=self.state_layout)
        dydt = np.zeros(self.state_layout.n_states, dtype=np.float64)
        for name, rate in chemistry_rhs.items():
            dydt[idx[name]] += rate

        ion_losses = {
            symbol: self._ion_loss_rate(
                densities.get(symbol, 0.0),
                te_cold_ev,
                M_AR if symbol.startswith("Ar") else self.metal_mass,
            )
            for symbol in self.state_layout.ion_species
        }
        total_ion_charge_loss = 0.0
        for symbol, loss in ion_losses.items():
            dydt[idx[symbol]] -= loss
            total_ion_charge_loss += (2.0 if symbol.endswith("2+") else 1.0) * loss

        self._apply_electron_charge_loss(dydt, n_cold, n_hot, total_ion_charge_loss)

        gas_density = densities.get("Ar_c", 0.0) + densities.get("Ar_h", 0.0) + densities.get("Ar_w", 0.0)
        dydt[idx["Ar_c"]] += neutral_refill_rate(
            self.n_gas_0,
            gas_density,
            self.config.gas.temperature_k,
            self.geom.area_wall,
            self.geom.volume,
        )

        argon_return = sum(ion_losses[symbol] for symbol in self.state_layout.argon_ion_species)
        dydt[idx["Ar_h"]] += 0.7 * argon_return
        dydt[idx["Ar_w"]] += 0.3 * argon_return

        if pulse_on:
            dydt[idx[self.state_layout.metal_ground_species]] += self._sputter_sources(densities, te_cold_ev, target_voltage_v)
            beta_t = compute_beta_t(target_voltage_v, te_cold_ev)
            for symbol in self.state_layout.metal_ion_species:
                dydt[idx[symbol]] -= back_attraction_rate(
                    densities.get(symbol, 0.0),
                    beta_t,
                    te_cold_ev,
                    self.metal_mass,
                    self.geom.area_target,
                    self.geom.volume,
                )

        self._apply_neutral_transport(dydt, densities)
        self._apply_argon_relaxation(dydt, densities)

        hot_source = self._secondary_electron_source_rate(total_ion_charge_loss, pulse_on)
        thermalization = n_hot / self._hot_electron_thermalization_time(gas_density)
        hot_wall_loss = n_hot / self._hot_electron_confinement_time(te_hot_ev)
        dydt[idx["e_hot"]] += hot_source - thermalization - hot_wall_loss
        dydt[idx["e_cold"]] += thermalization

        dydt[idx["current_circuit"]] = circuit_current_rhs(
            current_a=max(float(y[idx["current_circuit"]]), 0.0),
            source_voltage_v=source_voltage_v,
            plasma_resistance_ohm=plasma_resistance,
            circuit=self.circuit,
        )
        if self.circuit.mode == "waveform_current":
            dydt[idx["current_circuit"]] = 0.0

        power_terms = self._electron_power_terms(
            densities=densities,
            reaction_rates=reaction_rates,
            te_cold_ev=te_cold_ev,
            te_hot_ev=te_hot_ev,
            target_voltage_v=target_voltage_v,
            current_a=current_a,
            hot_source=hot_source,
            thermalization=thermalization,
        )
        dydt[idx["energy_cold"]] = self._clip_energy_rhs(
            density=max(densities.get("e_cold", 0.0), 0.0),
            energy_density=float(y[idx["energy_cold"]]),
            rhs=power_terms["cold"] / E_CHARGE,
        )
        dydt[idx["energy_hot"]] = self._clip_energy_rhs(
            density=max(densities.get("e_hot", 0.0), 0.0),
            energy_density=float(y[idx["energy_hot"]]),
            rhs=power_terms["hot"] / E_CHARGE,
        )
        return dydt

    def run(self, y0: NDArray[np.float64] | None = None) -> IRMState:
        """Integrate the global-model ODE system."""

        from scipy.integrate import solve_ivp

        initial_state = self.initial_state() if y0 is None else y0
        numerics = self.config.numerics
        atol = np.full(self.state_layout.n_states, numerics.density_atol, dtype=np.float64)
        atol[self.state_layout.indices["current_circuit"]] = numerics.current_atol
        atol[self.state_layout.indices["energy_cold"]] = numerics.energy_atol
        atol[self.state_layout.indices["energy_hot"]] = numerics.energy_atol
        if numerics.atol > 0.0:
            atol = np.maximum(atol, numerics.atol)

        solution = solve_ivp(
            self.rhs,
            (numerics.t_start, numerics.t_end),
            initial_state,
            method=numerics.solver,
            rtol=numerics.rtol,
            atol=atol,
            max_step=numerics.dt_max,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")

        diagnostics = build_irm_diagnostics(
            solution.t,
            solution.y.T,
            reactions=self.reactions,
            waveform=self.waveform,
            geom=self.geom,
            yield_gas=self.yield_gas,
            yield_self=self.yield_self,
            gas_ion_mass_kg=self.mass_ion,
            metal_mass_kg=self.metal_mass,
            drive_current_func=self._current_from_state,
            model_current_func=self._model_current_from_state,
            target_voltage_func=self._target_voltage_from_row,
            plasma_resistance_func=self._plasma_resistance_from_row,
            current_provenance="model-derived" if self.circuit.mode == "rl" else self.waveform.provenance,
            state_layout=self.state_layout,
        )
        return IRMState(time=solution.t, states=solution.y.T, diagnostics=diagnostics, state_layout=self.state_layout)

    def _current_from_state(self, t: float, row: NDArray[np.float64]) -> float:
        return self._current_from_row(t, row)

    def _model_current_from_state(self, t: float, row: NDArray[np.float64]) -> float:
        densities = state_to_densities(np.maximum(row, 0.0), layout=self.state_layout)
        return self._model_current_proxy(
            self._source_voltage_v(t),
            max(electron_density(densities, layout=self.state_layout), 1.0),
            min(max(electron_temperature_ev(row, "cold", layout=self.state_layout), 0.1), 20.0),
        )

    def _hot_temperature_ev(self, y: NDArray[np.float64]) -> float:
        idx = self.state_layout.indices
        density = max(float(y[idx["e_hot"]]), 0.0)
        if density < 1.0 and float(y[idx["energy_hot"]]) <= 0.0:
            return 250.0
        return min(max(electron_temperature_ev(y, "hot", layout=self.state_layout), 5.0), 1000.0)

    def _ion_loss_rate(self, density: float, te_ev: float, mass_kg: float) -> float:
        if density < 1.0:
            return 0.0
        return density / self._effective_confinement_time(te_ev, mass_kg)

    def _apply_electron_charge_loss(
        self,
        dydt: NDArray[np.float64],
        n_cold: float,
        n_hot: float,
        total_charge_loss: float,
    ) -> None:
        idx = self.state_layout.indices
        total_electrons = max(n_cold + n_hot, 1.0)
        dydt[idx["e_cold"]] -= total_charge_loss * (n_cold / total_electrons)
        dydt[idx["e_hot"]] -= total_charge_loss * (n_hot / total_electrons)

    def _sputter_sources(self, densities: dict[str, float], te_ev: float, voltage_v: float) -> float:
        sput_ar = sputter_flux(
            densities.get(self.state_layout.primary_argon_ion, 0.0),
            te_ev,
            M_AR,
            voltage_v,
            self.yield_gas,
            self.geom.area_target,
            self.geom.volume,
        )
        sput_self = sputter_flux(
            densities.get(self.state_layout.primary_metal_ion, 0.0),
            te_ev,
            self.metal_mass,
            voltage_v,
            self.yield_self,
            self.geom.area_target,
            self.geom.volume,
        )
        return sput_ar + sput_self

    def _apply_neutral_transport(self, dydt: NDArray[np.float64], densities: dict[str, float]) -> None:
        idx = self.state_layout.indices
        v_th_metal = np.sqrt(8.0 * E_CHARGE * 0.5 / (PI * self.metal_mass))
        for symbol in self.state_layout.metal_neutral_species:
            loss = densities.get(symbol, 0.0) * v_th_metal * self.geom.area_loss / (4.0 * self.geom.volume)
            dydt[idx[symbol]] -= loss

    def _apply_argon_relaxation(self, dydt: NDArray[np.float64], densities: dict[str, float]) -> None:
        idx = self.state_layout.indices
        hot_to_warm = densities.get("Ar_h", 0.0) / 3.0e-6
        warm_to_cold = densities.get("Ar_w", 0.0) / 1.5e-5
        metastable_loss = densities.get("Ar_m", 0.0) * 1.0e4
        resonant_loss = densities.get("Ar_r", 0.0) * 1.2e4
        four_p_loss = densities.get("Ar_4p", 0.0) * 5.0e5
        dydt[idx["Ar_h"]] -= hot_to_warm
        dydt[idx["Ar_w"]] += hot_to_warm - warm_to_cold
        dydt[idx["Ar_c"]] += warm_to_cold + metastable_loss + resonant_loss + four_p_loss
        dydt[idx["Ar_m"]] -= metastable_loss
        dydt[idx["Ar_r"]] -= resonant_loss
        dydt[idx["Ar_4p"]] -= four_p_loss

    def _electron_power_terms(
        self,
        *,
        densities: dict[str, float],
        reaction_rates: dict[str, object],
        te_cold_ev: float,
        te_hot_ev: float,
        target_voltage_v: float,
        current_a: float,
        hot_source: float,
        thermalization: float,
    ) -> dict[str, float]:
        p_abs_cold = abs(target_voltage_v * current_a) / self.geom.volume
        p_abs_hot = 1.5 * hot_source * self._secondary_electron_energy_ev(target_voltage_v) * E_CHARGE
        p_coll_cold = 0.0
        p_coll_hot = 0.0
        for reaction in self.reactions:
            if not reaction.is_electron_impact or reaction.threshold_ev <= 0.0:
                continue
            rate = reaction_rates[reaction.id]
            p_coll_cold += rate.cold * reaction.threshold_ev * E_CHARGE
            p_coll_hot += rate.hot * reaction.threshold_ev * E_CHARGE

        n_cold = max(densities.get("e_cold", 0.0), 0.0)
        n_hot = max(densities.get("e_hot", 0.0), 0.0)
        p_wall_cold = n_cold * 2.5 * te_cold_ev * E_CHARGE / self._effective_confinement_time(te_cold_ev, M_AR)
        p_wall_hot = n_hot * 2.5 * te_hot_ev * E_CHARGE / self._hot_electron_confinement_time(te_hot_ev)
        p_thermalize = 1.5 * thermalization * te_hot_ev * E_CHARGE
        return {
            "cold": p_abs_cold + p_thermalize - p_coll_cold - p_wall_cold,
            "hot": p_abs_hot - p_thermalize - p_coll_hot - p_wall_hot,
        }

    def _clip_energy_rhs(self, density: float, energy_density: float, rhs: float) -> float:
        min_energy = 1.5 * max(density, 1.0) * 0.1
        if energy_density < min_energy * 2.0 and rhs < 0.0:
            return rhs * max(0.0, (energy_density - min_energy) / min_energy)
        return rhs
