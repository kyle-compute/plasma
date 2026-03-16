"""Runtime helpers for PIC case execution."""

from __future__ import annotations

import time
from pathlib import Path
from time import sleep

import cupy as cp

from plasma.core.constants import E_CHARGE, K_BOLTZMANN, M_AR, M_CU, M_ELECTRON
from plasma.data.cross_sections import CrossSectionTable
from plasma.data.magnetic_field import load_magnetic_field_map, validate_field_map_shape
from plasma.data.sputtering import SputterYield
from plasma.data.synthetic_cross_sections import (
    electron_ar_elastic,
    electron_ar_excitation,
    electron_ar_ionization,
    ion_ar_charge_exchange,
)
from plasma.data.waveforms import load_waveform, make_square_pulse
from plasma.diagnostics.collectors import SubstrateCollector
from plasma.io.reports import save_json_model
from plasma.live.contracts import LiveGeometry
from plasma.live.builders import build_pic_live_snapshot
from plasma.live.pic_window import clear_event_clouds, event_counts, merge_event_clouds
from plasma.live.publisher import FileLiveSession
from plasma.pic.config import PICConfig, load_pic_config
from plasma.pic.grid import CylindricalGrid
from plasma.pic.loop import run_pic
from plasma.pic.magnetic import magnetron_field
from plasma.pic.magnetron import MagnetronTarget
from plasma.pic.mcc import make_electron_ar_mcc, make_ion_ar_mcc
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.plots import save_pic_quicklook
from plasma.pic.poisson import PoissonSolverCylindrical
from plasma.pic.reporting import bundle_from_pic_run
from plasma.pic.weighting import initialize_particles_uniform
from plasma.reporting import build_run_manifest, build_validation_report


def run_pic_case(
    config_path: str,
    output_dir: str | None = None,
    *,
    live_dir: str | None = None,
    live_max_particles: int = 1200,
) -> int:
    """Run a PIC case and persist quick-look artifacts."""

    cfg = load_pic_config(config_path)
    sim = build_simulation(cfg)
    target_dir = Path(output_dir or cfg.output.dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    live_session = FileLiveSession(live_dir) if live_dir else None

    substrate = SubstrateCollector(z_plane=cfg.geometry.z_substrate, dz_capture=cfg.geometry.z_substrate * 0.02)
    species_map = sim["species_map"]
    live_history: dict[str, list[float]] = {
        "time_s": [],
        "target_voltage_v": [],
        "electron_particles": [],
        "ar_ion_particles": [],
        "cu_neutral_particles": [],
        "substrate_hits_total": [],
        "electron_mean_energy_ev": [],
        "ar_ion_mean_energy_ev": [],
        "field_max_v_m": [],
        "emissivity_total_arb": [],
        "collisions_per_sample": [],
        "excitation_collisions": [],
        "ionization_collisions": [],
        "charge_exchange_collisions": [],
        "target_impacts_window": [],
        "see_window": [],
        "sputtered_window": [],
        "see_per_target_impact": [],
        "sputtered_per_target_impact": [],
        "source_activity_total_arb": [],
        "substrate_flux_total_arb": [],
        "substrate_mean_energy_ev": [],
        "racetrack_peak_r_m": [],
    }
    live_event_window: dict[str, dict] = {}
    control_state = {"paused": False, "step_budget": 0}
    live_geometry = LiveGeometry(
        r_max=float(cfg.geometry.r_max),
        z_max=float(cfg.geometry.z_substrate),
        z_target=float(cfg.geometry.z_target),
        z_substrate=float(cfg.geometry.z_substrate),
        r_inner=float(cfg.geometry.r_inner),
        r_outer=float(cfg.geometry.r_outer),
    )

    def record_substrate(step: int, t: float, _phi, _species_list, stats: dict) -> None:
        merge_event_clouds(live_event_window, stats.get("event_clouds"))
        if step % cfg.time.diag_interval != 0:
            return
        ions = species_map.get("Ar+")
        if ions is not None:
            substrate.record_absorbed(ions, t=t)
        window_counts = event_counts(live_event_window)
        provisional_snapshot = build_pic_live_snapshot(
            cfg.name,
            step=step,
            time_s=t,
            grid=sim["grid"],
            phi=_phi,
            species_map=species_map,
            history=live_history,
            br_grid=sim["Br_grid"],
            bz_grid=sim["Bz_grid"],
            event_window=live_event_window,
            geometry=live_geometry,
            substrate=substrate,
            max_particles=live_max_particles,
        )
        live_history["time_s"].append(float(t))
        live_history["target_voltage_v"].append(float(sim["waveform"].V(t)))
        live_history["electron_particles"].append(float(species_map["electron"].n_alive))
        live_history["ar_ion_particles"].append(float(species_map["Ar+"].n_alive))
        live_history["cu_neutral_particles"].append(float(species_map["Cu"].n_alive))
        live_history["substrate_hits_total"].append(float(substrate.total_count))
        live_history["electron_mean_energy_ev"].append(float(species_map["electron"].mean_energy_ev()))
        live_history["ar_ion_mean_energy_ev"].append(float(species_map["Ar+"].mean_energy_ev()))
        live_history["field_max_v_m"].append(float(provisional_snapshot.metrics.get("max_e_field_v_m", 0.0)))
        live_history["emissivity_total_arb"].append(float(provisional_snapshot.metrics.get("emissivity_total_arb", 0.0)))
        live_history["collisions_per_sample"].append(float(sum(stats.get("collision_counts", {}).values())))
        live_history["excitation_collisions"].append(float(window_counts.get("e_Ar_excitation", 0)))
        live_history["ionization_collisions"].append(float(window_counts.get("e_Ar_ionization", 0)))
        live_history["charge_exchange_collisions"].append(float(window_counts.get("ion_Ar_cx", 0)))
        live_history["target_impacts_window"].append(float(window_counts.get("target_impacts", 0)))
        live_history["see_window"].append(float(window_counts.get("secondary_electrons", 0)))
        live_history["sputtered_window"].append(float(window_counts.get("sputtered_neutrals", 0)))
        target_impacts = float(window_counts.get("target_impacts", 0))
        see_events = float(window_counts.get("secondary_electrons", 0))
        sputtered_events = float(window_counts.get("sputtered_neutrals", 0))
        live_history["see_per_target_impact"].append(see_events / target_impacts if target_impacts > 0.0 else 0.0)
        live_history["sputtered_per_target_impact"].append(
            sputtered_events / target_impacts if target_impacts > 0.0 else 0.0
        )
        live_history["source_activity_total_arb"].append(
            float(provisional_snapshot.metrics.get("source_activity_total_arb", 0.0))
        )
        live_history["substrate_flux_total_arb"].append(
            float(provisional_snapshot.metrics.get("substrate_flux_total_arb", 0.0))
        )
        live_history["substrate_mean_energy_ev"].append(float(substrate.latest_mean_energy_ev()))
        live_history["racetrack_peak_r_m"].append(float(provisional_snapshot.metrics.get("racetrack_peak_r_m", 0.0)))
        if live_session is not None:
            snapshot = build_pic_live_snapshot(
                cfg.name,
                step=step,
                time_s=t,
                grid=sim["grid"],
                phi=_phi,
                species_map=species_map,
                history=live_history,
                br_grid=sim["Br_grid"],
                bz_grid=sim["Bz_grid"],
                event_window=live_event_window,
                geometry=live_geometry,
                substrate=substrate,
                max_particles=live_max_particles,
            )
            live_session.publish(snapshot)
            _handle_live_commands(live_session, control_state)
        clear_event_clouds(live_event_window)

    print(f"HiPIMS PIC-MCC Simulation: {cfg.name}")
    print(f"Grid: {sim['grid'].nr} x {sim['grid'].nz}")
    print(f"dt = {sim['dt']:.2e} s, n_steps = {sim['n_steps']}")
    print(f"Output: {target_dir}")

    rng = cp.random.RandomState(42)
    started = time.time()
    diag = run_pic(
        sim["grid"], sim["species_list"], sim["solver"], sim["dt"], sim["n_steps"],
        Br_grid=sim["Br_grid"], Bz_grid=sim["Bz_grid"],
        diag_interval=sim["diag_interval"], compact_interval=sim["compact_interval"],
        waveform=sim["waveform"], target=sim["target"], mcc_handlers=sim["mcc_handlers"],
        species_map=species_map, rng=rng, callback=record_substrate,
        checkpoint_interval=sim["checkpoint_interval"], checkpoint_path=str(target_dir),
    )
    elapsed = time.time() - started

    collision_provenance = "public-download" if cfg.cross_sections.source == "normalized_files" else "synthetic"
    bundle = bundle_from_pic_run(
        diag,
        sim["waveform"],
        collector=substrate,
        collision_provenance=collision_provenance,
    )
    report = build_validation_report(cfg, bundle)
    manifest = build_run_manifest(cfg, bundle, report)

    plot_path = save_pic_quicklook(bundle, target_dir / "summary.png", title=cfg.name)
    bundle_path = target_dir / "diagnostics_bundle.json"
    report_path = target_dir / "validation_report.json"
    manifest_path = target_dir / "run_manifest.json"
    save_json_model(bundle_path, bundle)
    save_json_model(report_path, report)
    manifest.outputs = {
        "diagnostics_bundle": str(bundle_path),
        "validation_report": str(report_path),
        "summary_plot": str(plot_path),
    }
    checkpoint_path = _save_final_checkpoint(target_dir, sim)
    if checkpoint_path is not None:
        manifest.outputs["final_checkpoint"] = str(checkpoint_path)
    save_json_model(manifest_path, manifest)
    if live_session is not None and diag.last_phi is not None:
        live_session.publish(
            build_pic_live_snapshot(
                cfg.name,
                step=cfg.time.n_steps,
                time_s=cfg.time.n_steps * cfg.time.dt,
                grid=sim["grid"],
                phi=diag.last_phi,
                species_map=species_map,
                history=live_history,
                br_grid=sim["Br_grid"],
                bz_grid=sim["Bz_grid"],
                event_window=live_event_window,
                geometry=live_geometry,
                substrate=substrate,
                max_particles=live_max_particles,
                state="completed",
                message=f"Simulation complete in {elapsed:.1f}s",
            )
        )

    print(f"\nSimulation complete in {elapsed:.1f}s")
    print(f"Peak target voltage = {bundle.summary['peak_target_voltage_v']:.1f} V")
    print(f"Total target impacts = {bundle.summary['total_target_impacts']:.0f}")
    print(f"Validation status = {manifest.validation_status}")
    return 0


def build_simulation(cfg: PICConfig) -> dict:
    """Build PIC components from typed config."""

    grid = CylindricalGrid(cfg.grid.nr, cfg.grid.nz, cfg.geometry.r_max, cfg.geometry.z_substrate)
    solver = PoissonSolverCylindrical(grid, permittivity_factor=cfg.grid.permittivity_factor)
    electrons, ions, cu = _initialize_species(grid, cfg)
    br_grid, bz_grid = _build_magnetic_field(grid, cfg)

    return {
        "grid": grid,
        "solver": solver,
        "species_list": [electrons, ions, cu],
        "species_map": {"electron": electrons, "Ar+": ions, "Cu": cu},
        "Br_grid": cp.asarray(br_grid),
        "Bz_grid": cp.asarray(bz_grid),
        "waveform": _build_waveform(cfg),
        "target": _build_target(cfg),
        "mcc_handlers": _build_mcc_handlers(cfg),
        "dt": cfg.time.dt,
        "n_steps": cfg.time.n_steps,
        "diag_interval": cfg.time.diag_interval,
        "compact_interval": cfg.time.compact_interval,
        "checkpoint_interval": cfg.time.checkpoint_interval,
    }


def _initialize_species(grid, cfg: PICConfig) -> tuple[ParticleArray, ParticleArray, ParticleArray]:
    electron_sp = Species("electron", -E_CHARGE, M_ELECTRON, -1)
    ion_sp = Species("Ar+", E_CHARGE, M_AR, 1)
    cu_sp = Species("Cu", 0.0, M_CU, 0)
    e_data = initialize_particles_uniform(grid, electron_sp, cfg.particles.n0_electron, cfg.particles.te_ev, cfg.particles.ppc)
    i_data = initialize_particles_uniform(grid, ion_sp, cfg.particles.n0_ion, cfg.particles.ti_ev, cfg.particles.ppc)
    electrons = ParticleArray(species=electron_sp); electrons.allocate(len(e_data["r"]) * 4); electrons.add_particles(**e_data)
    ions = ParticleArray(species=ion_sp); ions.allocate(len(i_data["r"]) * 4); ions.add_particles(**i_data)
    cu = ParticleArray(species=cu_sp); cu.allocate(len(e_data["r"]))
    return electrons, ions, cu


def _build_waveform(cfg: PICConfig):
    if cfg.pulse.waveform_file:
        provenance = "measured"
        if cfg.case is not None:
            for source in cfg.case.inputs:
                if source.kind == "waveform":
                    provenance = source.provenance
                    break
        return load_waveform(cfg.pulse.waveform_file, provenance=provenance)
    return make_square_pulse(cfg.pulse.voltage_v, cfg.pulse.t_pulse_us * 1e-6, cfg.pulse.t_total_us * 1e-6, rise_time_s=cfg.pulse.rise_time_us * 1e-6)


def _build_magnetic_field(grid, cfg: PICConfig) -> tuple:
    if cfg.magnetic_field.map_file:
        br_grid, bz_grid = load_magnetic_field_map(cfg.magnetic_field.map_file)
        validate_field_map_shape(br_grid, bz_grid, n_r=grid.n_nodes_r, n_z=grid.n_nodes_z)
        return br_grid, bz_grid
    return magnetron_field(
        grid,
        inner_loop_r=cfg.magnetic_field.inner_magnet_r,
        outer_loop_r=cfg.magnetic_field.outer_magnet_r,
        loop_z=cfg.magnetic_field.magnet_z,
        current_inner=cfg.magnetic_field.current_inner,
        current_outer=cfg.magnetic_field.current_outer,
    )


def _build_target(cfg: PICConfig) -> MagnetronTarget:
    sputter = SputterYield("Ar+", "Cu", cfg.target.sputter_yield_a, cfg.target.sputter_yield_b, cfg.target.sputter_threshold_ev, cfg.target.cohesive_energy_ev)
    return MagnetronTarget(
        z_target=cfg.geometry.z_target,
        r_inner=cfg.geometry.r_inner,
        r_outer=cfg.geometry.r_outer,
        see_yield=cfg.target.secondary_electron_yield,
        see_energy_ev=cfg.target.see_energy_ev,
        sputter_yield=sputter,
        surface_binding_ev=cfg.target.cohesive_energy_ev,
        material_mass=M_CU,
    )


def _build_mcc_handlers(cfg: PICConfig) -> dict:
    n_ar = cfg.gas.pressure_pa / (K_BOLTZMANN * cfg.gas.temperature_k)
    electron_sigma = _build_electron_cross_sections(cfg)
    return {
        "electron": make_electron_ar_mcc(
            n_ar,
            sigma_elastic=electron_sigma["elastic"],
            sigma_excitation=electron_sigma["excitation"],
            sigma_ionization=electron_sigma["ionization"],
        ),
        "Ar+": make_ion_ar_mcc(n_ar, M_AR, sigma_charge_exchange=ion_ar_charge_exchange()),
    }


def _build_electron_cross_sections(cfg: PICConfig) -> dict[str, CrossSectionTable]:
    if cfg.cross_sections.source == "normalized_files":
        return {
            "elastic": CrossSectionTable.from_file(cfg.cross_sections.elastic_file, name="lxcat_elastic"),
            "excitation": CrossSectionTable.from_file(cfg.cross_sections.excitation_file, name="lxcat_excitation"),
            "ionization": CrossSectionTable.from_file(cfg.cross_sections.ionization_file, name="lxcat_ionization"),
        }
    return {
        "elastic": electron_ar_elastic(),
        "excitation": electron_ar_excitation(),
        "ionization": electron_ar_ionization(),
    }


def _save_final_checkpoint(output_dir: Path, sim: dict) -> Path | None:
    try:
        from plasma.io.checkpoint import save_checkpoint
    except ModuleNotFoundError:
        return None
    phi = cp.zeros((sim["grid"].n_nodes_r, sim["grid"].n_nodes_z), dtype=cp.float64)
    checkpoint_path = output_dir / "final_checkpoint.h5"
    save_checkpoint(checkpoint_path, sim["n_steps"], sim["n_steps"] * sim["dt"], sim["grid"], sim["species_map"], phi, sim["Br_grid"], sim["Bz_grid"])
    return checkpoint_path


def _handle_live_commands(session: FileLiveSession, state: dict[str, int | bool]) -> None:
    """Process viewer commands at publish boundaries."""

    if state["step_budget"] > 0:
        state["step_budget"] -= 1
        if state["step_budget"] <= 0:
            state["paused"] = True

    command = session.poll_command()
    if command is not None:
        if command.command == "pause":
            state["paused"] = True
            state["step_budget"] = 0
        elif command.command == "resume":
            state["paused"] = False
            state["step_budget"] = 0
        elif command.command == "single_step":
            state["paused"] = False
            state["step_budget"] = 1

    while state["paused"]:
        sleep(0.05)
        command = session.poll_command()
        if command is None:
            continue
        if command.command == "resume":
            state["paused"] = False
            state["step_budget"] = 0
        elif command.command == "single_step":
            state["paused"] = False
            state["step_budget"] = 1
