"""Main PIC simulation loop with optional MCC collisions and magnetron surface physics.

Implements the electrostatic PIC-MCC cycle from Taccogna 2023 Fig. 1:

    1. Deposit charge onto grid (CIC)
    2. Solve Poisson equation for potential
    3. Compute electric field E = -grad(phi)
    4. Gather fields to particle positions (CIC)
    5. Push particles (Boris or electrostatic leapfrog)
    6. [Optional] Process target surface impacts (SEE, sputtering)
    7. Apply boundary conditions (absorb/reflect)
    8. [Optional] Inject surface products (SEE electrons, sputtered atoms)
    9. [Optional] Perform MCC collisions and inject products
    10. Repeat

The loop maintains leapfrog synchronization:
    - Positions at integer steps: r^n
    - Velocities at half-steps: v^{n-1/2}
    - Fields at integer steps: E^n

This module provides both a single-step function (for testing) and a
full time-loop driver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import signature

import numpy as np

from plasma.pic.boundaries import apply_boundaries
from plasma.pic.deposit import deposit_charge
from plasma.pic.gather import gather_electric_field, gather_magnetic_field
from plasma.pic.grid import CylindricalGrid
from plasma.pic.particles import ParticleArray
from plasma.pic.poisson import PoissonSolverCylindrical
from plasma.pic.pusher import boris_push, electrostatic_push
from plasma.runtime.cupy_compat import cp
from plasma.runtime.random import export_rng_state


@dataclass
class PICDiagnostics:
    """Time-series diagnostics collected during PIC loop."""

    time: list[float] = field(default_factory=list)
    n_particles: dict[str, list[int]] = field(default_factory=dict)
    field_energy: list[float] = field(default_factory=list)
    kinetic_energy: dict[str, list[float]] = field(default_factory=dict)
    total_energy: list[float] = field(default_factory=list)

    # Extended diagnostics for MCC + magnetron
    collision_counts: list[dict[str, int]] = field(default_factory=list)
    n_see_total: list[int] = field(default_factory=list)
    n_sputtered_total: list[int] = field(default_factory=list)
    n_target_impacts: list[int] = field(default_factory=list)
    last_phi: cp.ndarray | None = None

    def record(
        self,
        t: float,
        species_list: list[ParticleArray],
        phi: cp.ndarray,
        grid: CylindricalGrid,
    ) -> None:
        """Record diagnostics at current timestep."""
        self.time.append(t)

        # Particle counts
        for sp in species_list:
            name = sp.species.name
            if name not in self.n_particles:
                self.n_particles[name] = []
                self.kinetic_energy[name] = []
            self.n_particles[name].append(sp.n_alive)
            self.kinetic_energy[name].append(sp.kinetic_energy())

        # Field energy: (1/2) * epsilon_0 * sum(|E|^2 * dV)
        from plasma.core.constants import EPSILON_0

        Er = cp.zeros_like(phi)
        Ez = cp.zeros_like(phi)
        if phi.shape[0] > 2:
            Er[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * grid.dr)
        if phi.shape[1] > 2:
            Ez[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * grid.dz)

        node_vol = grid.node_volumes_gpu()
        e_field = 0.5 * EPSILON_0 * float(cp.sum((Er**2 + Ez**2) * node_vol).item())
        self.field_energy.append(e_field)

        # Total energy
        ke_total = sum(sp.kinetic_energy() for sp in species_list)
        self.total_energy.append(ke_total + e_field)


def _inject_particles(target_array: ParticleArray, data: dict) -> int:
    """Inject particle data dict into a ParticleArray. Returns count added."""
    if data is None:
        return 0
    return target_array.add_particles(
        r=data["r"], z=data["z"],
        vr=data["vr"], vz=data["vz"], vtheta=data["vtheta"],
        weight=data["weight"],
    )


def _merge_event_clouds(target: dict[str, dict[str, np.ndarray]], name: str, cloud: dict | None) -> None:
    """Append one event cloud into the per-step stats payload."""

    if cloud is None:
        return

    r = np.asarray(cloud.get("r", np.empty(0)), dtype=np.float64)
    z = np.asarray(cloud.get("z", np.empty(0)), dtype=np.float64)
    if r.size == 0 or z.size == 0:
        return
    if name not in target:
        target[name] = {"r": r.copy(), "z": z.copy()}
        return
    target[name]["r"] = np.concatenate((target[name]["r"], r))
    target[name]["z"] = np.concatenate((target[name]["z"], z))


def _capture_boundary_incident_particles(particles: ParticleArray, *, z_plane: float) -> dict[str, np.ndarray] | None:
    """Capture alive particles that have crossed the substrate boundary.

    GPU pre-filter: hit detection runs on device so only the compact
    incident subset is transferred to CPU.
    """

    n = particles.count
    if n == 0 or particles.species.charge_state <= 0:
        return None

    # GPU pre-filter — avoids transferring all z/alive arrays
    mask_gpu = (particles.alive[:n] == 1) & (particles.z[:n] >= z_plane)
    hit_idx = cp.where(mask_gpu)[0]
    if len(hit_idx) == 0:
        return None

    # Transfer only the compact incident subset
    return {
        "r": cp.asnumpy(particles.r[:n][hit_idx]),
        "z": cp.asnumpy(particles.z[:n][hit_idx]),
        "vr": cp.asnumpy(particles.vr[:n][hit_idx]),
        "vz": cp.asnumpy(particles.vz[:n][hit_idx]),
        "vtheta": cp.asnumpy(particles.vtheta[:n][hit_idx]),
        "weight": cp.asnumpy(particles.weight[:n][hit_idx]),
    }


def _invoke_callback(callback: Callable, step: int, t: float, phi: cp.ndarray, species_list: list[ParticleArray], stats: dict) -> None:
    """Call the callback with backwards-compatible argument count."""

    n_params = len(signature(callback).parameters)
    if n_params >= 5:
        callback(step, t, phi, species_list, stats)
    else:
        callback(step, t, phi, species_list)


def pic_step(
    grid: CylindricalGrid,
    species_list: list[ParticleArray],
    solver: PoissonSolverCylindrical,
    dt: float,
    Br_grid: cp.ndarray | None = None,
    Bz_grid: cp.ndarray | None = None,
    phi_wall_r: float = 0.0,
    phi_wall_z0: float = 0.0,
    phi_wall_zL: float = 0.0,
    *,
    t: float = 0.0,
    waveform=None,
    target=None,
    mcc_handlers: dict | None = None,
    species_map: dict[str, ParticleArray] | None = None,
    rng=None,
) -> tuple[cp.ndarray, dict]:
    """Execute one PIC timestep with optional MCC and magnetron physics.

    Args:
        grid: Simulation grid.
        species_list: List of ParticleArray (one per species).
        solver: Pre-built Poisson solver.
        dt: Timestep [s].
        Br_grid, Bz_grid: Static magnetic field on nodes [T]. None for ES-only.
        phi_wall_*: Wall potentials for Poisson BCs [V].
        t: Current simulation time [s] (for waveform evaluation).
        waveform: DischargeWaveform instance driving the target potential.
        target: MagnetronTarget instance for surface physics.
        mcc_handlers: Dict of {species_name: MCCHandler} for collisions.
        species_map: Dict of {species_name: ParticleArray} for product injection.
        rng: CuPy RandomState for MCC candidate selection.

    Returns:
        (phi, step_stats): Potential on grid nodes and step statistics dict.
    """
    has_B = Br_grid is not None and Bz_grid is not None
    stats: dict = {}
    event_clouds: dict[str, dict[str, np.ndarray]] = {}

    # 1. Deposit charge
    rho = deposit_charge(grid, species_list)

    # 2. Solve Poisson — waveform drives target potential (cathode at z=0)
    wall_z0 = phi_wall_z0
    if waveform is not None:
        wall_z0 = -abs(float(waveform.V(t)))
    phi = solver.solve(rho, phi_wall_r, wall_z0, phi_wall_zL)

    # 3. Compute E-field on grid
    Er_grid_arr, Ez_grid_arr = solver.electric_field(phi)

    # 4. Gather fields & push each species
    for particles in species_list:
        if particles.count == 0:
            continue
        Er_p, Ez_p = gather_electric_field(grid, Er_grid_arr, Ez_grid_arr, particles)
        if has_B:
            Br_p, Bz_p = gather_magnetic_field(grid, Br_grid, Bz_grid, particles)
            boris_push(particles, Er_p, Ez_p, Br_p, Bz_p, dt)
        else:
            electrostatic_push(particles, Er_p, Ez_p, dt)

    # 5. Surface physics BEFORE boundaries (detect hits while particles alive)
    n_see = 0
    n_sputtered = 0
    n_impacts = 0
    substrate_incident: dict[str, dict[str, np.ndarray]] = {}
    if target is not None:
        from plasma.pic.magnetron import process_target_impacts

        # Process ions (charge_state > 0) hitting the target
        for particles in species_list:
            if particles.species.charge_state <= 0:
                continue
            result = process_target_impacts(target, particles, grid, rng=rng)
            n_impacts += result["n_impacts"]
            _merge_event_clouds(event_clouds, "target_impacts", result.get("impact_positions"))

            # Inject SEE electrons
            if result["see_electrons"] is not None and species_map is not None:
                electron_arr = species_map.get("electron")
                if electron_arr is not None:
                    n_see += _inject_particles(electron_arr, result["see_electrons"])
                _merge_event_clouds(event_clouds, "secondary_electrons", result["see_electrons"])

            # Inject sputtered neutrals
            if result["sputtered_neutrals"] is not None and species_map is not None:
                # Look for a neutral target-material species (e.g. "Cu", "Ti")
                for name, arr in species_map.items():
                    if arr.species.charge_state == 0 and name != "electron":
                        n_sputtered += _inject_particles(arr, result["sputtered_neutrals"])
                        break
                _merge_event_clouds(event_clouds, "sputtered_neutrals", result["sputtered_neutrals"])

    stats["n_see"] = n_see
    stats["n_sputtered"] = n_sputtered
    stats["n_target_impacts"] = n_impacts

    for particles in species_list:
        incident = _capture_boundary_incident_particles(particles, z_plane=grid.z_max)
        if incident is not None:
            substrate_incident[particles.species.name] = incident
    stats["substrate_incident"] = substrate_incident

    # 6. Apply boundaries
    if target is not None:
        from plasma.pic.magnetron import apply_magnetron_boundaries
        for particles in species_list:
            apply_magnetron_boundaries(grid, particles, target)
    else:
        for particles in species_list:
            apply_boundaries(grid, particles)

    # 7. MCC collisions
    collision_counts: dict[str, int] = {}
    collision_weight_sums: dict[str, float] = {}
    if mcc_handlers is not None and species_map is not None:
        if rng is None:
            rng = cp.random.RandomState()

        for sp_name, handler in mcc_handlers.items():
            particles = species_map.get(sp_name)
            if particles is None or particles.count == 0:
                continue

            counts = handler.perform_collisions(particles, dt, rng=rng)
            for k, v in counts.items():
                collision_counts[k] = collision_counts.get(k, 0) + v
            if hasattr(handler, "get_collision_weight_sums"):
                for k, v in handler.get_collision_weight_sums().items():
                    collision_weight_sums[k] = collision_weight_sums.get(k, 0.0) + float(v)
            for name, cloud in handler.get_event_positions().items():
                _merge_event_clouds(event_clouds, name, cloud)

            # Inject ionization products
            new_electrons = handler.get_new_electrons()
            if new_electrons is not None:
                electron_arr = species_map.get("electron")
                if electron_arr is not None:
                    _inject_particles(electron_arr, new_electrons)

            if hasattr(handler, "get_new_ions_by_species"):
                for ion_name, ion_data in handler.get_new_ions_by_species().items():
                    if ion_name in species_map:
                        _inject_particles(species_map[ion_name], ion_data)
            else:
                new_ions = handler.get_new_ions()
                if new_ions is not None:
                    # Backwards-compatible single-ion fallback.
                    for proc in handler.processes:
                        if proc.product_ion_name and proc.product_ion_name in species_map:
                            _inject_particles(species_map[proc.product_ion_name], new_ions)
                            break

    stats["collision_counts"] = collision_counts
    stats["collision_weight_sums"] = collision_weight_sums
    stats["event_clouds"] = event_clouds

    return phi, stats


def run_pic(
    grid: CylindricalGrid,
    species_list: list[ParticleArray],
    solver: PoissonSolverCylindrical,
    dt: float,
    n_steps: int,
    Br_grid: cp.ndarray | None = None,
    Bz_grid: cp.ndarray | None = None,
    phi_wall_r: float = 0.0,
    phi_wall_z0: float = 0.0,
    phi_wall_zL: float = 0.0,
    diag_interval: int = 10,
    compact_interval: int = 100,
    callback: Callable | None = None,
    *,
    waveform=None,
    target=None,
    mcc_handlers: dict | None = None,
    species_map: dict[str, ParticleArray] | None = None,
    rng=None,
    checkpoint_interval: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_background_state: dict[str, float] | None = None,
    checkpoint_metadata: dict | None = None,
) -> PICDiagnostics:
    """Run the full PIC time loop.

    Args:
        grid: Simulation grid.
        species_list: List of ParticleArray.
        solver: Poisson solver.
        dt: Timestep [s].
        n_steps: Total number of timesteps.
        Br_grid, Bz_grid: Static magnetic field [T].
        phi_wall_*: Wall potentials [V].
        diag_interval: Record diagnostics every N steps.
        compact_interval: Compact dead particles every N steps.
        callback: Optional function(step, t, phi, species_list) called each step.
        waveform: DischargeWaveform for time-dependent target voltage.
        target: MagnetronTarget for surface physics.
        mcc_handlers: Dict of {species_name: MCCHandler}.
        species_map: Dict of {species_name: ParticleArray} for product injection.
        rng: CuPy RandomState for reproducibility.
        checkpoint_interval: Save checkpoint every N steps (0 = disabled).
        checkpoint_path: Directory for checkpoint files.
        checkpoint_background_state: Optional background reservoir state to persist with checkpoints.
        checkpoint_metadata: Optional metadata persisted into checkpoints.

    Returns:
        PICDiagnostics with time-series data.
    """
    diag = PICDiagnostics()
    phi = cp.zeros((grid.n_nodes_r, grid.n_nodes_z), dtype=cp.float64)

    for step in range(n_steps):
        t = step * dt

        phi, stats = pic_step(
            grid, species_list, solver, dt,
            Br_grid, Bz_grid,
            phi_wall_r, phi_wall_z0, phi_wall_zL,
            t=t, waveform=waveform, target=target,
            mcc_handlers=mcc_handlers, species_map=species_map, rng=rng,
        )

        # Record diagnostics
        if step % diag_interval == 0:
            diag.record(t, species_list, phi, grid)
            diag.collision_counts.append(stats.get("collision_counts", {}))
            diag.n_see_total.append(stats.get("n_see", 0))
            diag.n_sputtered_total.append(stats.get("n_sputtered", 0))
            diag.n_target_impacts.append(stats.get("n_target_impacts", 0))

        # Periodic compaction
        if step % compact_interval == 0 and step > 0:
            for sp in species_list:
                sp.compact()

        if callback is not None:
            _invoke_callback(callback, step, t, phi, species_list, stats)

        # Checkpointing
        should_ckpt = (
            checkpoint_interval > 0 and checkpoint_path
            and step > 0 and step % checkpoint_interval == 0
        )
        if should_ckpt:
            try:
                from plasma.io.checkpoint import save_checkpoint
                save_checkpoint(
                    f"{checkpoint_path}/checkpoint_{step:06d}.h5",
                    step, t, grid,
                    {sp.species.name: sp for sp in species_list},
                    phi, Br_grid, Bz_grid,
                    background_state=checkpoint_background_state,
                    metadata=checkpoint_metadata,
                    rng_state=export_rng_state(rng),
                )
            except ImportError:
                pass

    diag.last_phi = phi
    return diag
