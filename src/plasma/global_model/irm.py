"""Ionization Region Model (IRM) — 0D global model for HiPIMS.

This is the main class implementing the volume-averaged ODE system
from Gudmundsson et al. (2022). It tracks species densities and electron
temperature in the ionization region (IR) of a magnetron sputtering discharge.

The state vector contains:
  - Electron density (cold population)
  - Neutral Ar density (cold background)
  - Ar metastable, resonant densities
  - Ar+, Ar2+ densities
  - Metal neutral (ground, metastable, excited) densities
  - Metal+ and Metal2+ densities
  - Electron energy density (3/2 * n_e * T_e)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plasma.core.config import SimulationConfig
from plasma.core.constants import E_CHARGE, K_BOLTZMANN, M_AR, PI
from plasma.data.reactions import ReactionSet, load_reactions
from plasma.data.sputtering import SputterYield
from plasma.data.waveforms import DischargeWaveform, load_waveform, make_square_pulse
from plasma.global_model.diagnostics import IRMDiagnostics, build_irm_diagnostics
from plasma.global_model.rate_equations import (
    N_STATES,
    STATE_INDICES,
    compute_reaction_rates,
    species_rhs,
    state_to_densities,
)
from plasma.global_model.transport import (
    back_attraction_rate,
    bohm_velocity,
    compute_beta_t,
    neutral_refill_rate,
    sputter_flux,
)


@dataclass
class IRMGeometry:
    """Pre-computed geometric quantities for the ionization region."""

    r_target: float        # [m]
    r_inner: float         # [m]
    r_outer: float         # [m]
    z_ir: float            # [m]
    area_target: float     # Erosion area [m^2]
    volume: float          # IR volume [m^3]
    area_loss: float       # Total loss area (sides + top of IR) [m^2]
    area_wall: float       # Area for gas refill [m^2]

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> IRMGeometry:
        g = cfg.geometry
        area_target = g.area_target or PI * (g.r_outer**2 - g.r_inner**2)
        volume = g.volume_ir or area_target * g.z_ir
        # Loss area: annular top + cylindrical sides of IR
        area_loss = area_target + 2.0 * PI * g.r_outer * g.z_ir
        area_wall = area_loss  # Gas refills through same surfaces
        return cls(
            r_target=g.r_target,
            r_inner=g.r_inner,
            r_outer=g.r_outer,
            z_ir=g.z_ir,
            area_target=area_target,
            volume=volume,
            area_loss=area_loss,
            area_wall=area_wall,
        )


@dataclass
class IRMState:
    """Container for IRM simulation results."""

    time: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    states: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    diagnostics: IRMDiagnostics = field(default_factory=IRMDiagnostics)

    def density(self, species: str) -> NDArray:
        """Get density time-series for a species [m^-3]."""
        idx = STATE_INDICES[species]
        return self.states[:, idx]

    @property
    def n_e(self) -> NDArray:
        return self.density("e_cold")

    @property
    def te_ev(self) -> NDArray:
        """Electron temperature [eV] from energy density."""
        n_e = np.maximum(self.n_e, 1.0)  # Avoid divide by zero
        energy_density = self.states[:, STATE_INDICES["energy"]]
        return (2.0 / 3.0) * energy_density / n_e

    def metric(self, name: str) -> NDArray[np.float64]:
        """Get a named diagnostic metric."""
        return self.diagnostics.values(name)


class IRM:
    """Ionization Region Model for HiPIMS discharge.

    Usage:
        cfg = load_config("config/hipims_cu_ar.yaml")
        irm = IRM(cfg)
        result = irm.run()
        plt.plot(result.time * 1e6, result.n_e)
    """

    def __init__(
        self,
        config: SimulationConfig,
        reactions: ReactionSet | None = None,
        sputter_yield_gas: SputterYield | None = None,
        sputter_yield_self: SputterYield | None = None,
        waveform: DischargeWaveform | None = None,
    ):
        self.config = config
        self.geom = IRMGeometry.from_config(config)

        # Load reactions if not provided
        self.reactions = reactions or load_reactions(config.reactions_file)

        # Sputter yields
        t = config.target
        self.yield_gas = sputter_yield_gas or SputterYield(
            ion="Ar+", target=t.material,
            a=t.sputter_yield_a, b=t.sputter_yield_b,
            threshold_ev=17.0, cohesive_energy_ev=t.cohesive_energy_ev,
        )
        self.yield_self = sputter_yield_self or SputterYield(
            ion=f"{t.material}+", target=t.material,
            a=t.self_sputter_yield_a, b=t.self_sputter_yield_b,
            threshold_ev=15.0, cohesive_energy_ev=t.cohesive_energy_ev,
        )

        # Waveform
        if waveform:
            self.waveform = waveform
        elif config.pulse.waveform_file:
            self.waveform = load_waveform(
                config.pulse.waveform_file,
                provenance=self._waveform_provenance_from_config(),
            )
        else:
            pulse = config.pulse
            self.waveform = make_square_pulse(
                voltage_v=pulse.voltage_v,
                t_pulse_s=pulse.t_pulse_us * 1e-6,
                t_total_s=(pulse.t_pulse_us + pulse.t_afterglow_us) * 1e-6,
            )

        # Initial neutral gas density from ideal gas law
        self.n_gas_0 = config.gas.pressure_pa / (K_BOLTZMANN * config.gas.temperature_k)

        # Ion mass for transport calculations
        self.mass_ion = M_AR

        # Cu mass constant
        self.m_cu = 63.546 * 1.66053906660e-27

        # Prescribed peak discharge current [A] — from measurement or estimate
        # Gudmundsson 2022 Case I: peak current ~ 40-60 A
        self.peak_current_a = max(50.0, float(np.max(np.abs(self.waveform.current_a))))

    def _waveform_provenance_from_config(self) -> str:
        if self.config.case is None:
            return "measured"
        for input_source in self.config.case.inputs:
            if input_source.kind == "waveform":
                return input_source.provenance
        return "measured"

    def initial_state(self) -> NDArray:
        """Create initial state vector with small seed densities."""
        y0 = np.zeros(N_STATES)

        # Background Ar at gas pressure
        y0[STATE_INDICES["Ar_c"]] = self.n_gas_0

        # Seed plasma density (small, represents pre-ionization)
        n_seed = 1e15  # m^-3
        y0[STATE_INDICES["Ar+"]] = n_seed
        y0[STATE_INDICES["Cu+"]] = n_seed * 0.01

        # Quasineutrality: n_e = sum(Z_i * n_i)
        y0[STATE_INDICES["e_cold"]] = (
            y0[STATE_INDICES["Ar+"]]
            + 2.0 * y0[STATE_INDICES["Ar2+"]]
            + y0[STATE_INDICES["Cu+"]]
            + 2.0 * y0[STATE_INDICES["Cu2+"]]
        )

        # Initial electron temperature ~ 3 eV
        te0 = 3.0  # eV
        y0[STATE_INDICES["energy"]] = 1.5 * y0[STATE_INDICES["e_cold"]] * te0

        return y0

    def _effective_confinement_time(self, te_ev: float, mass_kg: float) -> float:
        """Effective ion confinement time in the IR [s].

        tau_eff = V / (u_B * A_eff), where A_eff is a reduced loss area
        accounting for magnetic confinement. In HiPIMS the magnetic field
        traps electrons, which in turn confines ions via ambipolarity.
        Typical tau ~ 10-100 us.
        """
        u_b = bohm_velocity(te_ev, mass_kg)
        # Magnetic confinement factor: reduces effective loss area
        # In HiPIMS, B-field reduces cross-field electron transport by ~10-100x
        # Higher factor = more losses = lower peak density
        magnetic_conf_factor = 0.3  # A_eff = factor * A_geometric
        a_eff = magnetic_conf_factor * self.geom.area_loss
        tau = self.geom.volume / (u_b * a_eff + 1e-30)
        return tau

    def _discharge_current(self, t: float, v_d: float, n_e: float, te_ev: float) -> float:
        """Discharge current [A] — prescribed ramp tied to plasma density.

        In a real HiPIMS discharge, current is determined by the external circuit
        and the plasma impedance. Here we use a hybrid model: the current
        grows with plasma density (more charge carriers → more current)
        but is capped at the peak value from measurement.
        """
        measured_current = abs(float(self.waveform.I(t)))
        if measured_current > 0.0:
            return measured_current
        return self._model_current_proxy(v_d, n_e, te_ev)

    def _model_current_proxy(self, v_d: float, n_e: float, te_ev: float) -> float:
        """Internal density-based current proxy [A]."""

        if abs(v_d) < 10.0:
            return 0.0
        # Current ∝ n_e * u_B * A_target * e, capped at measured peak
        u_b = bohm_velocity(te_ev, M_AR)
        i_plasma = n_e * E_CHARGE * u_b * self.geom.area_target
        return min(i_plasma, self.peak_current_a)

    def rhs(self, t: float, y: NDArray) -> NDArray:
        """Right-hand side of the ODE system dy/dt = f(t, y)."""
        # Enforce non-negative densities
        y = np.maximum(y, 0.0)

        # Unpack state
        n_e = max(y[STATE_INDICES["e_cold"]], 1.0)
        energy_density = max(y[STATE_INDICES["energy"]], 1.0)
        te_ev = float((2.0 / 3.0) * energy_density / n_e)
        te_ev = max(0.1, min(te_ev, 100.0))

        densities = state_to_densities(y)
        v_d = float(self.waveform.V(t))
        is_pulse_on = abs(v_d) > 10.0
        current_a = self._discharge_current(t, v_d, n_e, te_ev)

        # ── Chemistry ──
        rxn_rates = compute_reaction_rates(densities, self.reactions, te_ev)
        chem_ddt = species_rhs(densities, rxn_rates, self.reactions)

        dydt = np.zeros(N_STATES)
        for name, rate in chem_ddt.items():
            if name in STATE_INDICES:
                dydt[STATE_INDICES[name]] += rate

        # ── Ion transport losses (confinement time model) ──
        ion_species = [
            ("Ar+", M_AR, 1),
            ("Ar2+", M_AR, 2),
            ("Cu+", self.m_cu, 1),
            ("Cu2+", self.m_cu, 2),
        ]

        total_ion_loss_charge = 0.0
        for ion_sym, mass, charge_z in ion_species:
            n_ion = densities.get(ion_sym, 0.0)
            if n_ion < 1.0:
                continue
            tau = self._effective_confinement_time(te_ev, mass)
            loss = n_ion / tau
            dydt[STATE_INDICES[ion_sym]] -= loss
            total_ion_loss_charge += charge_z * loss

        # Electron loss = total ion charge loss (quasineutrality)
        dydt[STATE_INDICES["e_cold"]] -= total_ion_loss_charge

        # ── Neutral Ar refill ──
        n_ar = densities.get("Ar_c", 0.0)
        refill = neutral_refill_rate(
            self.n_gas_0, n_ar, self.config.gas.temperature_k,
            self.geom.area_wall, self.geom.volume,
        )
        dydt[STATE_INDICES["Ar_c"]] += refill

        # ── Sputtering (pulse on only) ──
        if is_pulse_on:
            n_ar_ion = densities.get("Ar+", 0.0)
            sput_ar = sputter_flux(
                n_ar_ion, te_ev, M_AR, v_d,
                self.yield_gas, self.geom.area_target, self.geom.volume,
            )
            dydt[STATE_INDICES["Cu"]] += sput_ar

            n_cu_ion = densities.get("Cu+", 0.0)
            sput_self = sputter_flux(
                n_cu_ion, te_ev, self.m_cu, v_d,
                self.yield_self, self.geom.area_target, self.geom.volume,
            )
            dydt[STATE_INDICES["Cu"]] += sput_self

            beta = compute_beta_t(v_d, te_ev)
            ba_rate = back_attraction_rate(
                n_cu_ion, beta, te_ev, self.m_cu,
                self.geom.area_target, self.geom.volume,
            )
            dydt[STATE_INDICES["Cu+"]] -= ba_rate

        # ── Neutral metal escape ──
        v_th_cu = np.sqrt(8.0 * E_CHARGE * 0.5 / (PI * self.m_cu))
        for sym in ("Cu", "Cu_m1", "Cu_m2", "Cu_ex"):
            n_m = densities.get(sym, 0.0)
            loss = n_m * v_th_cu * self.geom.area_loss / (4.0 * self.geom.volume)
            dydt[STATE_INDICES[sym]] -= loss

        # ── Ar metastable/resonant diffusion losses ──
        for sym in ("Ar_m", "Ar_r"):
            n_s = densities.get(sym, 0.0)
            dydt[STATE_INDICES[sym]] -= n_s * 1e4

        # ── Electron energy ──
        # Absorbed power = V_D * I_D / V_IR
        p_abs = abs(v_d * current_a) / self.geom.volume

        # Collisional losses
        p_coll = 0.0
        for rxn in self.reactions:
            if not rxn.is_electron_impact or rxn.threshold_ev <= 0:
                continue
            for reactant in rxn.reactants:
                if reactant not in ("e", "e_cold", "e_hot"):
                    n_target = densities.get(reactant, 0.0)
                    break
            else:
                continue
            k = float(rxn.rate(te_ev, population="cold"))
            p_coll += n_e * n_target * k * rxn.threshold_ev * E_CHARGE

        # Wall losses (using effective confinement)
        tau_e = self._effective_confinement_time(te_ev, M_AR)
        p_wall = n_e * 2.5 * te_ev * E_CHARGE / tau_e

        energy_rhs = (p_abs - p_coll - p_wall) / E_CHARGE

        # Prevent energy density from going negative: if energy is near floor
        # and derivative is negative, clamp the cooling rate
        min_energy = 1.5 * n_e * 0.1  # Floor at T_e = 0.1 eV
        if energy_density < min_energy * 2.0 and energy_rhs < 0:
            energy_rhs *= max(0.0, (energy_density - min_energy) / min_energy)

        dydt[STATE_INDICES["energy"]] = energy_rhs

        return dydt

    def run(self, y0: NDArray | None = None) -> IRMState:
        """Integrate the ODE system and return results."""
        from scipy.integrate import solve_ivp

        if y0 is None:
            y0 = self.initial_state()

        num = self.config.numerics
        t_span = (num.t_start, num.t_end)

        # Per-variable absolute tolerances: energy and density differ by
        # many orders of magnitude, so uniform atol causes problems.
        # State order: [n_e, n_Ar, n_Ar_m, n_Ar_r, n_Ar+, n_Ar2+,
        #               n_Cu, n_Cu_m1, n_Cu_m2, n_Cu_ex, n_Cu+, n_Cu2+, energy]
        atol = np.ones(N_STATES) * 1e8  # density atol: 1e8 m^-3
        atol[STATE_INDICES["energy"]] = 1e12  # energy atol: 1e12 eV*m^-3

        sol = solve_ivp(
            self.rhs,
            t_span,
            y0,
            method="LSODA",  # Auto stiff/non-stiff switching
            rtol=1e-4,
            atol=atol,
            max_step=num.dt_max,
            dense_output=True,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        states = sol.y.T
        diagnostics = build_irm_diagnostics(
            sol.t,
            states,
            reactions=self.reactions,
            waveform=self.waveform,
            geom=self.geom,
            yield_gas=self.yield_gas,
            yield_self=self.yield_self,
            gas_ion_mass_kg=self.mass_ion,
            metal_mass_kg=self.m_cu,
            drive_current_func=self._current_from_state,
            model_current_func=self._model_current_from_state,
        )
        return IRMState(time=sol.t, states=states, diagnostics=diagnostics)

    def _current_from_state(self, t: float, row: NDArray) -> float:
        densities = state_to_densities(np.maximum(row, 0.0))
        n_e = max(densities.get("e_cold", 1.0), 1.0)
        energy_density = max(row[STATE_INDICES["energy"]], 1.0)
        te_ev = max((2.0 / 3.0) * energy_density / n_e, 0.1)
        v_d = float(self.waveform.V(t))
        return self._discharge_current(t, v_d, n_e, te_ev)

    def _model_current_from_state(self, t: float, row: NDArray) -> float:
        densities = state_to_densities(np.maximum(row, 0.0))
        n_e = max(densities.get("e_cold", 1.0), 1.0)
        energy_density = max(row[STATE_INDICES["energy"]], 1.0)
        te_ev = max((2.0 / 3.0) * energy_density / n_e, 0.1)
        v_d = float(self.waveform.V(t))
        return self._model_current_proxy(v_d, n_e, te_ev)
