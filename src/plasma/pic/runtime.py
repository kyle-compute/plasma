"""Runtime helpers for PIC case execution."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from time import sleep

import numpy as np

from plasma.core.constants import E_CHARGE, K_BOLTZMANN, M_AR, M_ELECTRON, material_mass_kg
from plasma.data.collision_packages import (
    CollisionPackage,
    load_channel_cross_section,
    load_collision_package,
)
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
from plasma.live.builders import build_pic_live_snapshot
from plasma.live.contracts import LiveGeometry
from plasma.live.pic_fields import number_density_view
from plasma.live.pic_window import clear_event_clouds, event_counts, merge_event_clouds
from plasma.live.publisher import FileLiveSession
from plasma.pic.config import PICConfig, load_pic_config
from plasma.pic.grid import CylindricalGrid
from plasma.pic.loop import run_pic
from plasma.pic.magnetic import magnetron_field
from plasma.pic.magnetron import MagnetronTarget
from plasma.pic.mcc import (
    CollisionProcess,
    CollisionType,
    CompositeMCCHandler,
    MCCHandler,
    make_electron_ar_mcc,
    make_ion_ar_mcc,
)
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.plots import save_pic_quicklook
from plasma.pic.poisson import PoissonSolverCylindrical
from plasma.pic.reporting import bundle_from_pic_run
from plasma.pic.weighting import initialize_particles_uniform
from plasma.reporting import build_run_manifest, build_validation_report
from plasma.runtime.cupy_compat import cp
from plasma.runtime.random import SimulationRNG, export_rng_state


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
    diagnostics_h5_path = target_dir / "diagnostics_snapshots.h5"
    metal_species_name = cfg.target.material

    substrate = SubstrateCollector(z_plane=cfg.geometry.z_substrate, dz_capture=cfg.geometry.z_substrate * 0.02)
    hdf5_writer = _AsyncHDF5Writer()
    species_map = sim["species_map"]
    live_history: dict[str, list[float]] = {
        "time_s": [],
        "target_voltage_v": [],
        "electron_particles": [],
        "ar_ion_particles": [],
        "metal_neutral_particles": [],
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
    if metal_species_name == "Cu":
        live_history["cu_neutral_particles"] = []
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
        _update_background_state(
            sim.get("background_state"),
            sim.get("collision_package"),
            stats.get("collision_weight_sums", {}),
            sim["grid"],
            sim.get("mcc_handlers"),
        )
        merge_event_clouds(live_event_window, stats.get("event_clouds"))
        if step % cfg.time.diag_interval != 0:
            return
        for species_name, incident in stats.get("substrate_incident", {}).items():
            particles = species_map.get(species_name)
            if particles is None:
                continue
            substrate.record_particle_data(
                r=incident["r"],
                vr=incident["vr"],
                vz=incident["vz"],
                vtheta=incident["vtheta"],
                weight=incident["weight"],
                mass_kg=particles.species.mass,
                t=t,
                species_name=species_name,
            )
        window_counts = event_counts(live_event_window)
        # Build the snapshot ONCE (previously built twice per diag step)
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
        live_history["time_s"].append(float(t))
        live_history["target_voltage_v"].append(float(sim["waveform"].V(t)))
        live_history["electron_particles"].append(float(species_map["electron"].n_alive))
        live_history["ar_ion_particles"].append(float(species_map["Ar+"].n_alive))
        metal_particles = species_map.get(metal_species_name)
        live_history["metal_neutral_particles"].append(0.0 if metal_particles is None else float(metal_particles.n_alive))
        if metal_species_name == "Cu":
            live_history["cu_neutral_particles"].append(0.0 if metal_particles is None else float(metal_particles.n_alive))
        live_history["substrate_hits_total"].append(float(substrate.total_count))
        live_history["electron_mean_energy_ev"].append(float(species_map["electron"].mean_energy_ev()))
        live_history["ar_ion_mean_energy_ev"].append(float(species_map["Ar+"].mean_energy_ev()))
        live_history["field_max_v_m"].append(float(snapshot.metrics.get("max_e_field_v_m", 0.0)))
        live_history["emissivity_total_arb"].append(float(snapshot.metrics.get("emissivity_total_arb", 0.0)))
        collision_counts = stats.get("collision_counts", {})
        live_history["collisions_per_sample"].append(float(sum(collision_counts.values())))
        live_history["excitation_collisions"].append(float(_sum_collision_family(collision_counts, "excitation")))
        live_history["ionization_collisions"].append(float(_sum_collision_family(collision_counts, "ionization")))
        live_history["charge_exchange_collisions"].append(float(_sum_collision_family(collision_counts, "cx")))
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
            float(snapshot.metrics.get("source_activity_total_arb", 0.0))
        )
        live_history["substrate_flux_total_arb"].append(
            float(snapshot.metrics.get("substrate_flux_total_arb", 0.0))
        )
        live_history["substrate_mean_energy_ev"].append(float(substrate.latest_mean_energy_ev()))
        live_history["racetrack_peak_r_m"].append(float(snapshot.metrics.get("racetrack_peak_r_m", 0.0)))
        if live_session is not None:
            live_session.publish(snapshot)
            _handle_live_commands(live_session, control_state)
        _save_hdf5_snapshot(
            diagnostics_h5_path,
            step,
            t,
            sim["grid"],
            _phi,
            species_map,
            sim.get("background_state"),
            collision_counts,
            async_writer=hdf5_writer,
        )
        clear_event_clouds(live_event_window)

    print(f"HiPIMS PIC-MCC Simulation: {cfg.name}")
    print(f"Grid: {sim['grid'].nr} x {sim['grid'].nz}")
    print(f"dt = {sim['dt']:.2e} s, n_steps = {sim['n_steps']}")
    print(f"Output: {target_dir}")

    rng = SimulationRNG(42)
    started = time.time()
    checkpoint_metadata = {}
    if sim.get("collision_package") is not None:
        checkpoint_metadata["collision_package"] = sim["collision_package"].package.name
        checkpoint_metadata["collision_package_version"] = sim["collision_package"].package.version
    diag = run_pic(
        sim["grid"], sim["species_list"], sim["solver"], sim["dt"], sim["n_steps"],
        Br_grid=sim["Br_grid"], Bz_grid=sim["Bz_grid"],
        diag_interval=sim["diag_interval"], compact_interval=sim["compact_interval"],
        waveform=sim["waveform"], target=sim["target"], mcc_handlers=sim["mcc_handlers"],
        species_map=species_map, rng=rng, callback=record_substrate,
        checkpoint_interval=sim["checkpoint_interval"], checkpoint_path=str(target_dir),
        checkpoint_background_state=sim.get("background_state"),
        checkpoint_metadata=checkpoint_metadata,
    )
    elapsed = time.time() - started
    hdf5_writer.shutdown()

    collision_provenance = (
        "model-derived"
        if sim.get("collision_package") is not None
        else ("public-download" if cfg.cross_sections.source == "normalized_files" else "synthetic")
    )
    bundle = bundle_from_pic_run(
        diag,
        sim["waveform"],
        collector=substrate,
        collision_provenance=collision_provenance,
        collision_channel_provenance=sim.get("collision_channel_provenance"),
        background_state=sim.get("background_state"),
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
    if diagnostics_h5_path.exists():
        manifest.outputs["diagnostics_hdf5"] = str(diagnostics_h5_path)
    checkpoint_path = _save_final_checkpoint(target_dir, sim, diag.last_phi, rng=rng)
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
    collision_package = load_collision_package(cfg.collision_package) if cfg.collision_package else None
    background_state = _build_background_state(cfg, collision_package)
    species_map = _initialize_species(grid, cfg, collision_package)
    br_grid, bz_grid = _build_magnetic_field(grid, cfg)
    mcc_handlers = _build_mcc_handlers(cfg, species_map, collision_package, background_state)

    return {
        "grid": grid,
        "solver": solver,
        "species_list": list(species_map.values()),
        "species_map": species_map,
        "Br_grid": cp.asarray(br_grid),
        "Bz_grid": cp.asarray(bz_grid),
        "waveform": _build_waveform(cfg),
        "target": _build_target(cfg),
        "mcc_handlers": mcc_handlers,
        "collision_package": collision_package,
        "collision_channel_provenance": None if collision_package is None else collision_package.channel_provenance(),
        "background_state": background_state,
        "dt": cfg.time.dt,
        "n_steps": cfg.time.n_steps,
        "diag_interval": cfg.time.diag_interval,
        "compact_interval": cfg.time.compact_interval,
        "checkpoint_interval": cfg.time.checkpoint_interval,
    }


def _initialize_species(
    grid,
    cfg: PICConfig,
    collision_package: CollisionPackage | None = None,
) -> dict[str, ParticleArray]:
    if collision_package is None:
        electron_sp = Species("electron", -E_CHARGE, M_ELECTRON, -1)
        ion_sp = Species("Ar+", E_CHARGE, M_AR, 1)
        metal_mass = material_mass_kg(cfg.target.material)
        metal_sp = Species(cfg.target.material, 0.0, metal_mass, 0)
        return {
            "electron": _seed_species(grid, electron_sp, cfg.particles.n0_electron, cfg.particles.te_ev, cfg.particles.ppc),
            "Ar+": _seed_species(grid, ion_sp, cfg.particles.n0_ion, cfg.particles.ti_ev, cfg.particles.ppc),
            cfg.target.material: _seed_species(grid, metal_sp, 0.0, cfg.particles.ti_ev, cfg.particles.ppc),
        }

    species_map: dict[str, ParticleArray] = {}
    for definition in collision_package.kinetic_species():
        species = Species(
            definition.name,
            definition.charge_state * E_CHARGE,
            definition.mass_kg,
            definition.charge_state,
        )
        density = _initial_density_for_species(definition.name, cfg)
        temperature_ev = cfg.particles.te_ev if definition.name == "electron" else cfg.particles.ti_ev
        species_map[definition.name] = _seed_species(grid, species, density, temperature_ev, cfg.particles.ppc)

    # Maintain the neutral target species expected by the live/reporting path.
    if cfg.target.material not in species_map:
        metal_mass = material_mass_kg(cfg.target.material)
        species_map[cfg.target.material] = _seed_species(
            grid,
            Species(cfg.target.material, 0.0, metal_mass, 0),
            cfg.particles.n0_sputtered_neutral,
            cfg.particles.ti_ev,
            cfg.particles.ppc,
        )
    return species_map


def _seed_species(grid, species: Species, density: float, temperature_ev: float, ppc: int) -> ParticleArray:
    data = initialize_particles_uniform(grid, species, density, temperature_ev, ppc)
    capacity = max(256, int(len(data["r"]) * 4))
    particles = ParticleArray(species=species)
    particles.allocate(capacity)
    if len(data["r"]) > 0:
        particles.add_particles(**data)
    return particles


def _initial_density_for_species(name: str, cfg: PICConfig) -> float:
    if name == "electron":
        return cfg.particles.n0_electron
    if name == "Ar+":
        return cfg.particles.n0_ion
    if name == cfg.target.material:
        return cfg.particles.n0_sputtered_neutral
    if name.endswith("+"):
        return cfg.particles.n0_metal_ion
    return 0.0


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
    sputter = SputterYield(
        "Ar+",
        cfg.target.material,
        cfg.target.sputter_yield_a,
        cfg.target.sputter_yield_b,
        cfg.target.sputter_threshold_ev,
        cfg.target.cohesive_energy_ev,
    )
    species_sputter_yields = {"Ar+": sputter}
    if cfg.target.self_sputter_yield_a is not None and cfg.target.self_sputter_yield_b is not None:
        species_sputter_yields[f"{cfg.target.material}+"] = SputterYield(
            f"{cfg.target.material}+",
            cfg.target.material,
            cfg.target.self_sputter_yield_a,
            cfg.target.self_sputter_yield_b,
            cfg.target.sputter_threshold_ev,
            cfg.target.cohesive_energy_ev,
        )
    species_see_yields = {
        "Ar+": cfg.target.secondary_electron_yield,
        f"{cfg.target.material}+": (
            cfg.target.metal_ion_secondary_electron_yield
            if cfg.target.metal_ion_secondary_electron_yield is not None
            else cfg.target.secondary_electron_yield
        ),
    }
    return MagnetronTarget(
        z_target=cfg.geometry.z_target,
        r_inner=cfg.geometry.r_inner,
        r_outer=cfg.geometry.r_outer,
        see_yield=cfg.target.secondary_electron_yield,
        see_energy_ev=cfg.target.see_energy_ev,
        sputter_yield=sputter,
        surface_binding_ev=cfg.target.cohesive_energy_ev,
        material_mass=material_mass_kg(cfg.target.material),
        species_see_yields=species_see_yields,
        species_sputter_yields=species_sputter_yields,
    )


def _build_mcc_handlers(
    cfg: PICConfig,
    species_map: dict[str, ParticleArray],
    collision_package: CollisionPackage | None,
    background_state: dict[str, float],
) -> dict:
    if collision_package is not None:
        return _build_mcc_handlers_from_package(species_map, collision_package, background_state)

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


def _build_background_state(cfg: PICConfig, collision_package: CollisionPackage | None) -> dict[str, float]:
    n_ar = cfg.gas.pressure_pa / (K_BOLTZMANN * cfg.gas.temperature_k)
    state: dict[str, float] = {}
    if collision_package is not None:
        for species in collision_package.species:
            state[species.name] = float(species.initial_density_m3)
    state.update({name: float(value) for name, value in cfg.background_model.densities_m3.items()})
    state.setdefault("Ar_c", n_ar)
    if state["Ar_c"] <= 0.0:
        state["Ar_c"] = n_ar
    return state


def _build_mcc_handlers_from_package(
    species_map: dict[str, ParticleArray],
    collision_package: CollisionPackage,
    background_state: dict[str, float],
) -> dict[str, MCCHandler]:
    species_defs = collision_package.species_by_name()
    grouped: dict[str, dict[str, list[CollisionProcess]]] = defaultdict(lambda: defaultdict(list))
    handler_meta: dict[tuple[str, str], tuple[float, float, float]] = {}

    for channel in collision_package.channels:
        if channel.execution != "mcc" or channel.projectile not in species_map or not channel.uses_cross_section:
            continue
        background_def = species_defs.get(channel.background)
        projectile_particles = species_map[channel.projectile]
        if background_def is None:
            continue
        sigma = load_channel_cross_section(channel)
        if sigma is None:
            continue
        grouped[channel.projectile][channel.background].append(
            CollisionProcess(
                name=channel.name,
                collision_type=_collision_type(channel.process),
                cross_section=sigma,
                threshold_ev=channel.threshold_ev,
                product_species_name=channel.product_species_name,
                product_ion_name=channel.product_ion_name,
            )
        )
        handler_meta[(channel.projectile, channel.background)] = (
            projectile_particles.species.mass,
            background_def.mass_kg,
            float(background_state.get(channel.background, 0.0)),
        )

    result: dict[str, MCCHandler] = {}
    for projectile_name, background_map in grouped.items():
        handlers: list[MCCHandler] = []
        for background_name, processes in background_map.items():
            projectile_mass, background_mass, density = handler_meta[(projectile_name, background_name)]
            handler = MCCHandler(
                projectile_mass=projectile_mass,
                background_mass=background_mass,
                background_density=density,
            )
            handler.background_species_name = background_name
            for process in processes:
                handler.add_process(process)
            handlers.append(handler)
        if not handlers:
            continue
        result[projectile_name] = handlers[0] if len(handlers) == 1 else CompositeMCCHandler(handlers=handlers)
    return result


def _collision_type(name: str) -> CollisionType:
    mapping = {
        "elastic": CollisionType.ELASTIC,
        "excitation": CollisionType.EXCITATION,
        "ionization": CollisionType.IONIZATION,
        "charge_exchange": CollisionType.CHARGE_EXCHANGE,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported MCC collision type '{name}'") from exc


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


def _save_final_checkpoint(output_dir: Path, sim: dict, phi: cp.ndarray | None, *, rng=None) -> Path | None:
    try:
        from plasma.io.checkpoint import save_checkpoint
    except ModuleNotFoundError:
        return None
    if phi is None:
        phi = cp.zeros((sim["grid"].n_nodes_r, sim["grid"].n_nodes_z), dtype=cp.float64)
    checkpoint_path = output_dir / "final_checkpoint.h5"
    metadata = {}
    if sim.get("collision_package") is not None:
        metadata["collision_package"] = sim["collision_package"].package.name
        metadata["collision_package_version"] = sim["collision_package"].package.version
    save_checkpoint(
        checkpoint_path,
        sim["n_steps"],
        sim["n_steps"] * sim["dt"],
        sim["grid"],
        sim["species_map"],
        phi,
        sim["Br_grid"],
        sim["Bz_grid"],
        background_state=sim.get("background_state"),
        metadata=metadata,
        rng_state=export_rng_state(rng),
    )
    return checkpoint_path


def _sum_collision_family(counts: dict[str, int], token: str) -> int:
    return sum(value for name, value in counts.items() if token in name)


def _update_background_state(
    background_state: dict[str, float] | None,
    collision_package: CollisionPackage | None,
    collision_weight_sums: dict[str, float],
    grid,
    mcc_handlers: dict[str, MCCHandler] | None,
) -> None:
    if background_state is None or collision_package is None or not collision_weight_sums:
        return

    channel_map = {channel.name: channel for channel in collision_package.channels}
    domain_volume = float(np.pi * grid.r_max**2 * grid.z_max)
    if domain_volume <= 0.0:
        return

    for channel_name, weight_sum in collision_weight_sums.items():
        channel = channel_map.get(channel_name)
        if channel is None:
            continue
        delta_n = float(weight_sum) / domain_volume
        if delta_n <= 0.0:
            continue
        if channel.process in {"excitation", "ionization"} and channel.background in background_state:
            background_state[channel.background] = max(background_state[channel.background] - delta_n, 0.0)
        if (
            channel.process == "excitation"
            and channel.product_species_name
            and channel.product_species_name != "electron"
            and channel.product_species_name in background_state
        ):
            background_state[channel.product_species_name] = background_state.get(channel.product_species_name, 0.0) + delta_n

    if mcc_handlers is not None:
        for handler in mcc_handlers.values():
            if hasattr(handler, "update_background_state"):
                handler.update_background_state(background_state)
                continue
            background_name = getattr(handler, "background_species_name", None)
            if background_name is not None and background_name in background_state:
                handler.update_background_density(background_state[background_name])


class _AsyncHDF5Writer:
    """Background thread for non-blocking HDF5 snapshot writes."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                _do_save_hdf5_snapshot(**item)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def submit(self, **kwargs) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        try:
            self._queue.put_nowait(kwargs)
        except queue.Full:
            pass  # drop snapshot rather than block the simulation

    def shutdown(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=10.0)


def _do_save_hdf5_snapshot(
    path,
    step: int,
    time_s: float,
    phi_np,
    n_e,
    n_i,
    background_state,
    collision_counts,
) -> None:
    try:
        from plasma.io.hdf5_diagnostics import save_diagnostics_snapshot
    except ModuleNotFoundError:
        return

    save_diagnostics_snapshot(
        path,
        step=step,
        time=time_s,
        phi=phi_np,
        n_e=n_e,
        n_i=n_i,
        background_state=background_state,
        collision_counts=collision_counts,
    )


def _save_hdf5_snapshot(
    path: Path,
    step: int,
    time_s: float,
    grid,
    phi,
    species_map: dict[str, ParticleArray],
    background_state: dict[str, float] | None,
    collision_counts: dict[str, int],
    *,
    async_writer: _AsyncHDF5Writer | None = None,
) -> None:
    phi_np = cp.asnumpy(phi) if isinstance(phi, cp.ndarray) else phi
    electrons = species_map.get("electron")
    electron_density = number_density_view(grid, electrons) if electrons is not None else None
    ion_density = None
    for particles in species_map.values():
        if particles.species.charge_state <= 0:
            continue
        density = number_density_view(grid, particles)
        ion_density = density if ion_density is None else (ion_density + density)

    payload = dict(
        path=path,
        step=step,
        time_s=time_s,
        phi_np=phi_np,
        n_e=electron_density,
        n_i=ion_density,
        background_state=background_state,
        collision_counts=collision_counts,
    )

    if async_writer is not None:
        async_writer.submit(**payload)
    else:
        _do_save_hdf5_snapshot(**payload)


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
