"""Helpers that build live-view snapshots from simulation state."""

from __future__ import annotations

import time

import numpy as np

from plasma.core.constants import E_CHARGE
from plasma.global_model.irm import IRMState
from plasma.live.contracts import (
    LiveGeometry,
    LiveHistogram,
    LiveParticleCloud,
    LiveSeries,
    LiveSnapshot,
)
from plasma.live.hipims_monitor import monitor_message, peak_radius_m, pulse_phase_label, safe_ratio
from plasma.live.pic_fields import (
    charge_density_view,
    electric_field_components,
    emissivity_proxy,
    event_map,
    field_bundle,
    magnetic_field_magnitude,
    normalize_field,
    number_density_view,
    substrate_flux_proxy,
)
from plasma.live.pic_window import EventWindow, event_counts
from plasma.runtime.cupy_compat import cp

_GLOBAL_SERIES_META = {
    "current_a": ("s", "A", "Discharge current"),
    "reference_current_a": ("s", "A", "Reference current"),
    "model_current_proxy_a": ("s", "A", "Model current proxy"),
    "voltage_v": ("s", "V", "Discharge voltage"),
    "electron_density": ("s", "m^-3", "Electron density"),
    "alpha_t": ("s", "1", "alpha_t"),
    "beta_t": ("s", "1", "beta_t"),
    "xi_t": ("s", "1", "xi_t"),
    "deposition_flux_m2s": ("s", "m^-2 s^-1", "Deposition flux"),
}

_PIC_SERIES_META = {
    "target_voltage_v": ("s", "V", "Target voltage"),
    "electron_particles": ("s", "count", "Electron particles"),
    "ar_ion_particles": ("s", "count", "Ar+ particles"),
    "metal_neutral_particles": ("s", "count", "Target-metal neutrals"),
    "cu_neutral_particles": ("s", "count", "Cu particles"),
    "substrate_hits_total": ("s", "count", "Substrate ion samples"),
    "electron_mean_energy_ev": ("s", "eV", "Electron mean energy"),
    "ar_ion_mean_energy_ev": ("s", "eV", "Ar+ mean energy"),
    "field_max_v_m": ("s", "V/m", "Peak electric field"),
    "emissivity_total_arb": ("s", "a.u.", "Emissivity proxy"),
    "collisions_per_sample": ("s", "count", "Collisions"),
    "excitation_collisions": ("s", "count", "Excitation collisions"),
    "ionization_collisions": ("s", "count", "Ionization collisions"),
    "charge_exchange_collisions": ("s", "count", "Charge-exchange collisions"),
    "target_impacts_window": ("s", "count", "Target impacts"),
    "see_window": ("s", "count", "SEE electrons"),
    "sputtered_window": ("s", "count", "Sputtered neutrals"),
    "see_per_target_impact": ("s", "count", "SEE per impact"),
    "sputtered_per_target_impact": ("s", "count", "Sputtered per impact"),
    "source_activity_total_arb": ("s", "a.u.", "Source activity"),
    "substrate_flux_total_arb": ("s", "a.u.", "Substrate flux proxy"),
    "substrate_mean_energy_ev": ("s", "eV", "Substrate mean ion energy"),
    "racetrack_peak_r_m": ("s", "m", "Racetrack peak radius"),
}


def _metal_species_name(species_map: dict) -> str | None:
    for name, particles in species_map.items():
        if name == "electron":
            continue
        if particles.species.charge_state != 0:
            continue
        if name.startswith("Ar"):
            continue
        return name
    return None


def build_global_live_snapshot(
    title: str,
    result: IRMState,
    *,
    end_index: int,
    state: str,
    message: str | None = None,
) -> LiveSnapshot:
    """Build a live snapshot from a partial 0D result slice."""

    t = np.asarray(result.time[:end_index], dtype=np.float64)
    series: dict[str, LiveSeries] = {}
    for name, (x_unit, y_unit, label) in _GLOBAL_SERIES_META.items():
        if name not in result.diagnostics.series:
            continue
        values = np.asarray(result.metric(name)[:end_index], dtype=np.float64)
        series[name] = LiveSeries(
            x=t.tolist(),
            y=values.tolist(),
            x_unit=x_unit,
            y_unit=y_unit,
            label=label,
        )

    return LiveSnapshot(
        model="global",
        state=state,
        title=title,
        step=end_index,
        time_s=float(t[-1]) if len(t) else 0.0,
        updated_at_s=time.time(),
        message=message,
        series=series,
    )


def build_pic_live_snapshot(
    title: str,
    *,
    step: int,
    time_s: float,
    grid,
    phi,
    species_map: dict,
    history: dict[str, list[float]],
    br_grid=None,
    bz_grid=None,
    event_window: EventWindow | None = None,
    geometry: LiveGeometry | None = None,
    substrate=None,
    max_particles: int = 1200,
    state: str = "running",
    message: str | None = None,
    precomputed_electron_density: np.ndarray | None = None,
    precomputed_ion_density: np.ndarray | None = None,
) -> LiveSnapshot:
    """Build a live snapshot from the current PIC state."""

    phi_np = np.asarray(cp.asnumpy(phi) if isinstance(phi, cp.ndarray) else phi, dtype=np.float64)
    fields = {
        "phi_v": field_bundle(phi_np, x=grid.z_edges, y=grid.r_edges, unit="V", label="Potential"),
    }

    electrons = species_map.get("electron")
    electron_density = precomputed_electron_density
    if electron_density is None and electrons is not None:
        electron_density = number_density_view(grid, electrons)
    if electron_density is not None:
        fields["electron_density_m3"] = field_bundle(
            electron_density, x=grid.z_edges, y=grid.r_edges, unit="m^-3", label="Electron density"
        )

    ions = species_map.get("Ar+")
    ion_density = precomputed_ion_density
    if ion_density is None and ions is not None:
        ion_density = number_density_view(grid, ions)
    if ion_density is not None:
        fields["ion_density_m3"] = field_bundle(
            ion_density, x=grid.z_edges, y=grid.r_edges, unit="m^-3", label="Ar+ density"
        )

    metal_species_name = _metal_species_name(species_map)
    neutrals = species_map.get(metal_species_name) if metal_species_name is not None else None
    if neutrals is not None and metal_species_name is not None:
        metal_density = number_density_view(grid, neutrals)
        fields["metal_density_m3"] = field_bundle(
            metal_density,
            x=grid.z_edges,
            y=grid.r_edges,
            unit="m^-3",
            label=f"{metal_species_name} density",
        )
        fields[f"{metal_species_name.lower()}_density_m3"] = field_bundle(
            metal_density,
            x=grid.z_edges,
            y=grid.r_edges,
            unit="m^-3",
            label=f"{metal_species_name} density",
        )

    fields["rho_c_m3"] = field_bundle(
        charge_density_view(grid, list(species_map.values())),
        x=grid.z_edges,
        y=grid.r_edges,
        unit="C/m^3",
        label="Charge density",
    )

    er, ez, e_mag = electric_field_components(grid, phi_np)
    fields["e_mag_v_m"] = field_bundle(e_mag, x=grid.z_edges, y=grid.r_edges, unit="V/m", label="|E|")
    fields["e_r_v_m"] = field_bundle(er, x=grid.z_edges, y=grid.r_edges, unit="V/m", label="E_r")
    fields["e_z_v_m"] = field_bundle(ez, x=grid.z_edges, y=grid.r_edges, unit="V/m", label="E_z")

    if br_grid is not None and bz_grid is not None:
        br, bz, b_mag = magnetic_field_magnitude(br_grid, bz_grid)
        fields["b_mag_t"] = field_bundle(b_mag, x=grid.z_edges, y=grid.r_edges, unit="T", label="|B|")
        fields["b_r_t"] = field_bundle(br, x=grid.z_edges, y=grid.r_edges, unit="T", label="B_r")
        fields["b_z_t"] = field_bundle(bz, x=grid.z_edges, y=grid.r_edges, unit="T", label="B_z")

    excitation_map = event_map(grid, event_window, ("e_Ar_excitation",))
    ionization_map = event_map(grid, event_window, ("e_Ar_ionization",))
    see_map = event_map(grid, event_window, ("secondary_electrons",))
    sputter_map = event_map(grid, event_window, ("sputtered_neutrals",))
    target_map = event_map(grid, event_window, ("target_impacts", "secondary_electrons", "sputtered_neutrals"))
    substrate_map = substrate_flux_proxy(grid, substrate)
    emissivity = emissivity_proxy(electron_density, e_mag, excitation_map, ionization_map, see_map)
    fields["emissivity_arb"] = field_bundle(
        emissivity, x=grid.z_edges, y=grid.r_edges, unit="a.u.", label="Emissivity"
    )
    fields["target_activity_arb"] = field_bundle(
        normalize_field(target_map), x=grid.z_edges, y=grid.r_edges, unit="a.u.", label="Target activity"
    )
    fields["see_source_arb"] = field_bundle(
        normalize_field(see_map), x=grid.z_edges, y=grid.r_edges, unit="a.u.", label="SEE source"
    )
    fields["sputter_source_arb"] = field_bundle(
        normalize_field(sputter_map), x=grid.z_edges, y=grid.r_edges, unit="a.u.", label="Sputter source"
    )
    fields["ionization_source_arb"] = field_bundle(
        normalize_field(ionization_map), x=grid.z_edges, y=grid.r_edges, unit="a.u.", label="Ionization source"
    )
    fields["substrate_flux_proxy_arb"] = field_bundle(
        substrate_map, x=grid.z_edges, y=grid.r_edges, unit="a.u.", label="Substrate flux proxy"
    )

    series: dict[str, LiveSeries] = {}
    time_axis = history.get("time_s", [])
    for name, (x_unit, y_unit, label) in _PIC_SERIES_META.items():
        if name not in history:
            continue
        series[name] = LiveSeries(
            x=[float(value) for value in time_axis],
            y=[float(value) for value in history[name]],
            x_unit=x_unit,
            y_unit=y_unit,
            label=label,
        )

    particles: dict[str, LiveParticleCloud] = {}
    particle_species = ["electron", "Ar+"]
    if metal_species_name is not None:
        particle_species.append(metal_species_name)
    for species_name in particle_species:
        particle_array = species_map.get(species_name)
        if particle_array is None:
            continue
        cloud = _sample_particles(particle_array, max_particles=max_particles)
        if cloud is not None:
            particles[species_name] = cloud

    histograms: dict[str, LiveHistogram] = {}
    if substrate is not None and getattr(substrate, "total_count", 0) > 0:
        energy_ev, counts = substrate.iedf(n_bins=64, e_max_ev=300.0)
        histograms["substrate_iedf"] = LiveHistogram(
            axis=[float(value) for value in energy_ev],
            values=[float(value) for value in counts],
            axis_unit="eV",
            value_unit="count",
            label="Substrate IEDF",
        )

    counts = event_counts(event_window)
    phase_label, phase_code = pulse_phase_label(history.get("target_voltage_v", []), history.get("emissivity_total_arb", []))
    peak_racetrack_radius = peak_radius_m(np.asarray(grid.r_edges, dtype=np.float64), target_map)
    source_activity_total = float(np.sum(target_map + ionization_map + sputter_map))
    substrate_flux_total = float(np.sum(substrate_map))
    target_impacts = float(counts.get("target_impacts", 0))
    see_events = float(counts.get("secondary_electrons", 0))
    sputtered_events = float(counts.get("sputtered_neutrals", 0))
    substrate_mean_energy = float(substrate.mean_energy_ev()) if substrate is not None else 0.0
    substrate_capture = safe_ratio(float(substrate.latest_count()) if substrate is not None else 0.0, sputtered_events)
    metrics = {
        "max_e_field_v_m": float(np.max(e_mag)) if e_mag.size else 0.0,
        "mean_electron_energy_ev": float(electrons.mean_energy_ev()) if electrons is not None else 0.0,
        "mean_ion_energy_ev": float(ions.mean_energy_ev()) if ions is not None else 0.0,
        "emissivity_total_arb": float(np.sum(emissivity)),
        "peak_target_activity_arb": float(np.max(normalize_field(target_map))) if target_map.size else 0.0,
        "racetrack_peak_r_m": peak_racetrack_radius,
        "see_yield_proxy": safe_ratio(see_events, target_impacts),
        "sputter_yield_proxy": safe_ratio(sputtered_events, target_impacts),
        "substrate_capture_proxy": substrate_capture,
        "source_activity_total_arb": source_activity_total,
        "substrate_flux_total_arb": substrate_flux_total,
        "substrate_mean_energy_ev": substrate_mean_energy,
        "pulse_phase_code": phase_code,
    }
    metrics.update({name: float(count) for name, count in counts.items()})

    resolved_message = message or monitor_message(phase_label, peak_racetrack_radius, substrate_capture)

    return LiveSnapshot(
        model="pic",
        state=state,
        title=title,
        step=step,
        time_s=float(time_s),
        updated_at_s=time.time(),
        message=resolved_message,
        series=series,
        fields=fields,
        particles=particles,
        histograms=histograms,
        metrics=metrics,
        geometry=geometry,
    )


def _sample_particles(particles, *, max_particles: int) -> LiveParticleCloud | None:
    n = particles.count
    if n == 0:
        return None

    stride = max(n // max_particles, 1)
    sample_idx = cp.arange(0, n, stride, dtype=cp.int32)[:max_particles]
    if sample_idx.size == 0:
        return None
    alive_mask = particles.alive[sample_idx] == 1
    sample_idx = sample_idx[alive_mask]
    if sample_idx.size == 0:
        return None

    r = cp.asnumpy(particles.r[sample_idx])
    z = cp.asnumpy(particles.z[sample_idx])
    vr = cp.asnumpy(particles.vr[sample_idx])
    vz = cp.asnumpy(particles.vz[sample_idx])
    vt = cp.asnumpy(particles.vtheta[sample_idx])
    speed = np.sqrt(vr**2 + vz**2 + vt**2)
    energy_ev = 0.5 * particles.species.mass * (vr**2 + vz**2 + vt**2) / E_CHARGE

    return LiveParticleCloud(
        r=[float(value) for value in r],
        z=[float(value) for value in z],
        energy_ev=[float(value) for value in energy_ev],
        speed_m_s=[float(value) for value in speed],
        label=particles.species.name,
        species=particles.species.name,
    )
