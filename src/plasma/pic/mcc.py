"""Monte Carlo Collisions (MCC) for PIC projectiles against background species.

Null-collision scheme loosely following Vahedi & Surendra (1995) and
Taccogna 2023 Eq. 5.  All collision physics — candidate selection,
accept/reject, scattering, and product generation — runs on GPU via
CuPy arrays.  Only aggregated counters and compact event-position
arrays for live visualisation are transferred to CPU.

Algorithm:
    1. Compute null-collision frequency from a *conservative sampled
       upper bound*: nu_null = n_bg * max_sample(sum_i sigma_i(E) * v(E)),
       where the max is taken over 2000 log-spaced energies in [0.01, 10000] eV.
       This is not an exact supremum — it can under-estimate if the true
       peak lies between sample points, but is sufficient for typical
       monotonic or single-peaked cross-sections.
    2. Collision probability: P_null = 1 - exp(-nu_null * dt)
    3. For each particle, draw random u ~ U(0,1) on GPU:
       - If u > P_null: no collision (skip)
       - If u <= P_null: accept/reject per-process
         using sigma_i(E) * v_rel / max_sample(sigma * v_rel).

Collision types supported:
    - Elastic scattering (isotropic, energy loss ~ 2 m_e/M per collision)
    - Excitation (inelastic, threshold energy removed, isotropic scatter)
    - Ionization (equal energy split between scattered and ejected electron)
    - Charge exchange (ion velocity replaced by thermal neutral velocity)

References:
    - Vahedi, V. & Surendra, M. (1995). Comp. Phys. Comm. 87, 179-198.
    - Nanbu, K. (2000). IEEE Trans. Plasma Sci. 28, 971-990.
    - Taccogna et al. (2023), J. Appl. Phys. 134, 150901.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from plasma.core.constants import E_CHARGE, M_ELECTRON
from plasma.data.cross_sections import CrossSectionTable, eval_cross_section_gpu
from plasma.runtime.cupy_compat import cp
from plasma.runtime.random import SimulationRNG, uniform_cpu, uniform_gpu


class CollisionType(Enum):
    """Types of binary collisions in a plasma."""

    ELASTIC = auto()
    EXCITATION = auto()
    IONIZATION = auto()
    CHARGE_EXCHANGE = auto()


@dataclass
class CollisionProcess:
    """Definition of a single collision process.

    Attributes:
        name: Human-readable label (e.g., "e + Ar -> Ar+ + 2e").
        collision_type: Type of collision.
        cross_section: Energy-dependent cross-section table sigma(E) [m^2].
        threshold_ev: Threshold energy for inelastic processes [eV].
        product_species_name: Name of species created (for ionization).
        product_ion_name: Name of ion created (for ionization).
    """

    name: str
    collision_type: CollisionType
    cross_section: CrossSectionTable
    threshold_ev: float = 0.0
    product_species_name: str | None = None
    product_ion_name: str | None = None


def _concat_particle_dicts(buffers: list[dict[str, cp.ndarray]]) -> dict[str, cp.ndarray]:
    """Concatenate a list of particle dicts (same keys) into one.

    Handles both CuPy and NumPy arrays.
    """
    keys = buffers[0].keys()
    first = buffers[0][next(iter(keys))]
    if isinstance(first, np.ndarray):
        return {k: np.concatenate([b[k] for b in buffers]) for k in keys}
    return {k: cp.concatenate([b[k] for b in buffers]) for k in keys}


@dataclass
class MCCHandler:
    """Null-collision MCC handler for one projectile vs. one background species.

    All collision physics runs on GPU.  Only aggregated counters and
    compact event-position subsets are transferred to CPU.

    Attributes:
        projectile_mass: Mass of projectile species [kg].
        background_mass: Mass of background species [kg].
        background_density: Number density of background [m^-3].
        processes: List of collision processes.
    """

    projectile_mass: float
    background_mass: float
    background_density: float
    processes: list[CollisionProcess] = field(default_factory=list)

    # Cached null-collision quantities
    _max_sigma_vrel: float = 0.0
    _nu_null: float = 0.0

    def add_process(self, process: CollisionProcess) -> None:
        self.processes.append(process)
        self._update_null_frequency()

    def _update_null_frequency(self) -> None:
        """Recompute the null-collision frequency upper bound.

        nu_null = n_bg * max_sample(sum_i(sigma_i(E) * v_rel(E)))

        The maximum is estimated by sampling 2000 log-spaced energies in
        [0.01, 10000] eV.
        """
        if not self.processes:
            self._max_sigma_vrel = 0.0
            self._nu_null = 0.0
            return

        # Sample energies (in eV) on log scale
        e_sample = np.logspace(-2, 4, 2000)

        # Reduced mass for relative velocity calculation
        m_r = (self.projectile_mass * self.background_mass /
               (self.projectile_mass + self.background_mass))

        if self.projectile_mass < 1e-28:
            v_rel = np.sqrt(2.0 * e_sample * E_CHARGE / self.projectile_mass)
        else:
            v_rel = np.sqrt(2.0 * e_sample * E_CHARGE / m_r)

        # Sum sigma*v over all processes at each energy
        total_sigma_v = np.zeros_like(e_sample)
        for proc in self.processes:
            sigma = proc.cross_section(e_sample)
            total_sigma_v += sigma * v_rel

        self._max_sigma_vrel = float(np.max(total_sigma_v))
        self._nu_null = self.background_density * self._max_sigma_vrel

    def collision_probability(self, dt: float) -> float:
        """Null-collision probability for timestep dt."""
        if self._nu_null <= 0:
            return 0.0
        return 1.0 - np.exp(-self._nu_null * dt)

    def perform_collisions(
        self,
        particles,
        dt: float,
        rng=None,
    ) -> dict[str, int]:
        """Perform MCC collisions on all particles of this species.

        All collision physics runs on GPU.  Velocities are never
        transferred to CPU; only final counters and compact event
        positions leave the device.

        Args:
            particles: ParticleArray of projectile species.
            dt: Timestep [s].
            rng: SimulationRNG (for reproducibility).

        Returns:
            Dict of collision counts by process name.
        """
        if rng is None:
            rng = SimulationRNG()

        # Clear ionization product buffers from any prior call
        self._new_electron_buffers: list[dict[str, cp.ndarray]] = []
        self._new_ion_buffers_by_species: dict[str, list[dict[str, cp.ndarray]]] = {}
        self._event_buffers: dict[str, dict[str, np.ndarray]] = {}
        self._weight_sums: dict[str, float] = {}

        n = particles.count
        if n == 0 or not self.processes:
            return {}

        p_null = self.collision_probability(dt)
        if p_null <= 0:
            return {}

        # Step 1: Determine which particles undergo a potential collision (GPU)
        rand_select = rng.rand(n, dtype=cp.float64)
        alive_mask = particles.alive[:n] == 1
        collide_mask = (rand_select < p_null) & alive_mask

        collide_indices = cp.where(collide_mask)[0]
        n_collide = len(collide_indices)

        if n_collide == 0:
            return {}

        # Step 2: Extract candidate velocities on GPU (no CPU transfer!)
        vr = particles.vr[:n][collide_indices]
        vz = particles.vz[:n][collide_indices]
        vt = particles.vtheta[:n][collide_indices]
        v2 = vr**2 + vz**2 + vt**2
        v_mag = cp.sqrt(v2)

        # Kinetic energy in eV (GPU)
        energy_ev = 0.5 * self.projectile_mass * v2 / E_CHARGE

        # Step 3: Accept/reject for each collision type (GPU)
        counts: dict[str, int] = {}
        processed = cp.zeros(n_collide, dtype=cp.bool_)

        rand_type = uniform_gpu(rng, n_collide)
        cumulative_prob = cp.zeros(n_collide, dtype=cp.float64)

        for proc in self.processes:
            # GPU cross-section evaluation
            log_e_gpu, log_s_gpu = proc.cross_section.gpu_log_data()
            sigma = eval_cross_section_gpu(
                energy_ev, log_e_gpu, log_s_gpu,
                proc.cross_section.e_min, proc.cross_section.e_max,
            )
            sigma_vrel = sigma * v_mag

            if self._max_sigma_vrel > 0:
                p_type = sigma_vrel / self._max_sigma_vrel
            else:
                p_type = cp.zeros_like(sigma_vrel)
            cumulative_prob = cumulative_prob + p_type

            this_collision = (~processed) & (rand_type < cumulative_prob)
            collision_idx = cp.where(this_collision)[0]

            if len(collision_idx) == 0:
                counts[proc.name] = 0
                continue

            # Get original particle indices (GPU)
            orig_idx = collide_indices[collision_idx]

            # Pull compact event positions to CPU for live visualisation
            self._event_buffers[proc.name] = {
                "r": cp.asnumpy(particles.r[orig_idx]),
                "z": cp.asnumpy(particles.z[orig_idx]),
            }
            self._weight_sums[proc.name] = float(
                cp.sum(particles.weight[orig_idx]).item()
            )

            # Apply collision on GPU
            n_events = self._apply_collision(
                proc, particles, orig_idx,
                energy_ev[collision_idx], v_mag[collision_idx],
                vr[collision_idx], vz[collision_idx], vt[collision_idx],
                rng=rng,
            )
            counts[proc.name] = n_events
            processed[collision_idx] = True

        counts["null"] = int(cp.sum(~processed).item())
        return counts

    def _apply_collision(
        self,
        process: CollisionProcess,
        particles,
        indices: cp.ndarray,
        energy_ev: cp.ndarray,
        v_mag: cp.ndarray,
        vr: cp.ndarray,
        vz: cp.ndarray,
        vt: cp.ndarray,
        *,
        rng,
    ) -> int:
        """Apply a specific collision to selected particles (all GPU)."""
        n = len(indices)
        if n == 0:
            return 0

        if process.collision_type == CollisionType.ELASTIC:
            self._elastic_scatter(particles, indices, energy_ev, v_mag, rng=rng)
        elif process.collision_type == CollisionType.EXCITATION:
            self._excitation(particles, indices, energy_ev, v_mag, process.threshold_ev, rng=rng)
        elif process.collision_type == CollisionType.IONIZATION:
            self._ionization(
                particles, indices, energy_ev, v_mag,
                process.threshold_ev,
                product_ion_name=process.product_ion_name,
                rng=rng,
            )
        elif process.collision_type == CollisionType.CHARGE_EXCHANGE:
            self._charge_exchange(particles, indices, rng=rng)

        return n

    def _elastic_scatter(
        self,
        particles,
        indices: cp.ndarray,
        energy_ev: cp.ndarray,
        v_mag: cp.ndarray,
        *,
        rng,
    ) -> None:
        """Isotropic elastic scattering (GPU)."""
        n = len(indices)
        cos_theta = 2.0 * uniform_gpu(rng, n) - 1.0
        sin_theta = cp.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * cp.pi * uniform_gpu(rng, n)

        mass_ratio = self.projectile_mass / self.background_mass
        energy_loss_frac = 2.0 * mass_ratio
        new_speed = v_mag * cp.sqrt(1.0 - energy_loss_frac)

        particles.vr[indices] = new_speed * sin_theta * cp.cos(phi)
        particles.vz[indices] = new_speed * cos_theta
        particles.vtheta[indices] = new_speed * sin_theta * cp.sin(phi)

    def _excitation(
        self,
        particles,
        indices: cp.ndarray,
        energy_ev: cp.ndarray,
        v_mag: cp.ndarray,
        threshold_ev: float,
        *,
        rng,
    ) -> None:
        """Inelastic excitation (GPU)."""
        n = len(indices)
        new_energy_ev = cp.maximum(energy_ev - threshold_ev, 0.01)
        new_speed = cp.sqrt(2.0 * new_energy_ev * E_CHARGE / self.projectile_mass)

        cos_theta = 2.0 * uniform_gpu(rng, n) - 1.0
        sin_theta = cp.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * cp.pi * uniform_gpu(rng, n)

        particles.vr[indices] = new_speed * sin_theta * cp.cos(phi)
        particles.vz[indices] = new_speed * cos_theta
        particles.vtheta[indices] = new_speed * sin_theta * cp.sin(phi)

    def _ionization(
        self,
        particles,
        indices: cp.ndarray,
        energy_ev: cp.ndarray,
        v_mag: cp.ndarray,
        threshold_ev: float,
        *,
        product_ion_name: str | None,
        rng,
    ) -> None:
        """Electron-impact ionization (GPU).

        Scattered incident electron and new ejected electron + ion
        are all computed on GPU.
        """
        n = len(indices)
        remaining_ev = cp.maximum(energy_ev - threshold_ev, 0.01)
        share_ev = remaining_ev * 0.5

        # Scattered incident electron (GPU)
        new_speed = cp.sqrt(2.0 * share_ev * E_CHARGE / self.projectile_mass)
        cos_theta = 2.0 * uniform_gpu(rng, n) - 1.0
        sin_theta = cp.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * cp.pi * uniform_gpu(rng, n)

        particles.vr[indices] = new_speed * sin_theta * cp.cos(phi)
        particles.vz[indices] = new_speed * cos_theta
        particles.vtheta[indices] = new_speed * sin_theta * cp.sin(phi)

        # Ejected electron — born at collision site (GPU)
        ejected_speed = cp.sqrt(2.0 * share_ev * E_CHARGE / self.projectile_mass)
        cos_theta2 = 2.0 * uniform_gpu(rng, n) - 1.0
        sin_theta2 = cp.sqrt(1.0 - cos_theta2**2)
        phi2 = 2.0 * cp.pi * uniform_gpu(rng, n)

        # Positions and weights stay on GPU
        r_new = particles.r[indices]
        z_new = particles.z[indices]
        w_new = particles.weight[indices]

        electron_data = {
            "r": r_new.copy(),
            "z": z_new.copy(),
            "vr": ejected_speed * sin_theta2 * cp.cos(phi2),
            "vz": ejected_speed * cos_theta2,
            "vtheta": ejected_speed * sin_theta2 * cp.sin(phi2),
            "weight": w_new.copy(),
        }
        if not hasattr(self, "_new_electron_buffers"):
            self._new_electron_buffers = []
        self._new_electron_buffers.append(electron_data)

        # New ions born cold at collision site (GPU)
        ion_data = {
            "r": r_new.copy(),
            "z": z_new.copy(),
            "vr": cp.zeros(n, dtype=cp.float64),
            "vz": cp.zeros(n, dtype=cp.float64),
            "vtheta": cp.zeros(n, dtype=cp.float64),
            "weight": w_new.copy(),
        }
        if not hasattr(self, "_new_ion_buffers_by_species"):
            self._new_ion_buffers_by_species = {}
        ion_name = product_ion_name or "ion"
        self._new_ion_buffers_by_species.setdefault(ion_name, []).append(ion_data)

    def _charge_exchange(
        self,
        particles,
        indices: cp.ndarray,
        *,
        rng,
    ) -> None:
        """Charge exchange: fast ion → slow ion (GPU)."""
        n = len(indices)
        v_th = np.sqrt(0.026 * E_CHARGE / self.background_mass)

        cos_theta = 2.0 * uniform_gpu(rng, n) - 1.0
        sin_theta = cp.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * cp.pi * uniform_gpu(rng, n)

        particles.vr[indices] = v_th * sin_theta * cp.cos(phi)
        particles.vz[indices] = v_th * cos_theta
        particles.vtheta[indices] = v_th * sin_theta * cp.sin(phi)

    def get_new_electrons(self) -> dict[str, cp.ndarray] | None:
        """Retrieve newly created electrons from ionization events.

        Returns GPU (CuPy) arrays — ParticleArray.add_particles handles
        these directly via cp.asarray (no-op for device arrays).
        """
        buffers = getattr(self, "_new_electron_buffers", None)
        if not buffers:
            self._new_electron_buffers = []
            return None
        result = _concat_particle_dicts(buffers)
        self._new_electron_buffers = []
        return result

    def get_event_positions(self) -> dict[str, dict[str, np.ndarray]]:
        """Retrieve collision-site positions from the last collision pass."""

        buffers = getattr(self, "_event_buffers", {})
        self._event_buffers = {}
        return {
            name: {"r": cloud["r"].copy(), "z": cloud["z"].copy()}
            for name, cloud in buffers.items()
        }

    def get_new_ions(self) -> dict[str, cp.ndarray] | None:
        """Retrieve newly created ions from ionization events."""
        ion_map = self.get_new_ions_by_species()
        if not ion_map:
            return None
        result = _concat_particle_dicts(list(ion_map.values()))
        return result

    def get_new_ions_by_species(self) -> dict[str, dict[str, cp.ndarray]]:
        """Retrieve newly created ions, keyed by product species name."""

        buffers_by_species = getattr(self, "_new_ion_buffers_by_species", None)
        if not buffers_by_species:
            self._new_ion_buffers_by_species = {}
            return {}
        result = {
            name: _concat_particle_dicts(buffers)
            for name, buffers in buffers_by_species.items()
            if buffers
        }
        self._new_ion_buffers_by_species = {}
        return result

    def get_collision_weight_sums(self) -> dict[str, float]:
        """Retrieve weighted physical collision totals from the last pass."""

        weights = getattr(self, "_weight_sums", {})
        self._weight_sums = {}
        return {name: float(value) for name, value in weights.items()}

    def update_background_density(self, n_bg: float) -> None:
        """Update background neutral density (e.g., for gas rarefaction)."""
        self.background_density = n_bg
        self._update_null_frequency()


def _merge_event_cloud(
    target: dict[str, dict[str, np.ndarray]],
    name: str,
    cloud: dict[str, np.ndarray],
) -> None:
    if name not in target:
        target[name] = {
            "r": np.asarray(cloud["r"], dtype=np.float64).copy(),
            "z": np.asarray(cloud["z"], dtype=np.float64).copy(),
        }
        return
    target[name]["r"] = np.concatenate((target[name]["r"], np.asarray(cloud["r"], dtype=np.float64)))
    target[name]["z"] = np.concatenate((target[name]["z"], np.asarray(cloud["z"], dtype=np.float64)))


@dataclass
class CompositeMCCHandler:
    """Aggregate multiple background-specific MCC handlers for one projectile.

    GPU-accelerated: candidate selection, velocity extraction, energy
    computation, accept/reject, and scattering all run on GPU.
    """

    handlers: list[MCCHandler] = field(default_factory=list)
    _new_electron_buffers: list[dict[str, cp.ndarray]] = field(default_factory=list)
    _new_ion_buffers_by_species: dict[str, list[dict[str, cp.ndarray]]] = field(default_factory=dict)
    _event_buffers: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    _weight_sums: dict[str, float] = field(default_factory=dict)

    @property
    def processes(self) -> list[CollisionProcess]:
        result: list[CollisionProcess] = []
        for handler in self.handlers:
            result.extend(handler.processes)
        return result

    def perform_collisions(self, particles, dt: float, rng=None) -> dict[str, int]:
        self._new_electron_buffers = []
        self._new_ion_buffers_by_species = {}
        self._event_buffers = {}
        self._weight_sums = {}
        counts: dict[str, int] = {}
        if rng is None:
            rng = SimulationRNG()
        for handler in self.handlers:
            handler._new_electron_buffers = []
            handler._new_ion_buffers_by_species = {}
            handler._event_buffers = {}
            handler._weight_sums = {}

        n = particles.count
        total_upper = self._combined_upper_rate()
        if n == 0 or total_upper <= 0.0:
            return counts

        p_null = 1.0 - np.exp(-total_upper * dt)
        if p_null <= 0.0:
            return counts

        # Candidate selection (GPU)
        rand_select = rng.rand(n, dtype=cp.float64)
        alive_mask = particles.alive[:n] == 1
        collide_mask = (rand_select < p_null) & alive_mask
        collide_indices = cp.where(collide_mask)[0]
        n_collide = len(collide_indices)
        if n_collide == 0:
            return counts

        # Extract candidate velocities on GPU (no CPU transfer!)
        vr = particles.vr[:n][collide_indices]
        vz = particles.vz[:n][collide_indices]
        vt = particles.vtheta[:n][collide_indices]
        v2 = vr**2 + vz**2 + vt**2
        v_mag = cp.sqrt(v2)
        energy_ev = 0.5 * self.handlers[0].projectile_mass * v2 / E_CHARGE

        rand_type = uniform_gpu(rng, n_collide)
        processed = cp.zeros(n_collide, dtype=cp.bool_)
        cumulative_prob = cp.zeros(n_collide, dtype=cp.float64)

        for handler in self.handlers:
            for proc in handler.processes:
                # GPU cross-section evaluation
                log_e_gpu, log_s_gpu = proc.cross_section.gpu_log_data()
                sigma = eval_cross_section_gpu(
                    energy_ev, log_e_gpu, log_s_gpu,
                    proc.cross_section.e_min, proc.cross_section.e_max,
                )
                rate = handler.background_density * sigma * v_mag
                cumulative_prob = cumulative_prob + cp.where(
                    total_upper > 0.0, rate / total_upper, 0.0,
                )
                this_collision = (~processed) & (rand_type < cumulative_prob)
                collision_idx = cp.where(this_collision)[0]
                if len(collision_idx) == 0:
                    counts.setdefault(proc.name, 0)
                    continue

                orig_idx = collide_indices[collision_idx]

                # Compact event positions to CPU for live viz
                _merge_event_cloud(
                    self._event_buffers,
                    proc.name,
                    {
                        "r": cp.asnumpy(particles.r[orig_idx]),
                        "z": cp.asnumpy(particles.z[orig_idx]),
                    },
                )
                self._weight_sums[proc.name] = self._weight_sums.get(proc.name, 0.0) + float(
                    cp.sum(particles.weight[orig_idx]).item()
                )
                counts[proc.name] = counts.get(proc.name, 0) + handler._apply_collision(
                    proc, particles, orig_idx,
                    energy_ev[collision_idx], v_mag[collision_idx],
                    vr[collision_idx], vz[collision_idx], vt[collision_idx],
                    rng=rng,
                )
                processed[collision_idx] = True

        counts["null"] = int(cp.sum(~processed).item())
        for handler in self.handlers:
            child_electrons = handler.get_new_electrons()
            if child_electrons is not None:
                self._new_electron_buffers.append(child_electrons)
            for species_name, ion_data in handler.get_new_ions_by_species().items():
                self._new_ion_buffers_by_species.setdefault(species_name, []).append(ion_data)
        return counts

    def get_new_electrons(self) -> dict[str, cp.ndarray] | None:
        if not self._new_electron_buffers:
            return None
        result = _concat_particle_dicts(self._new_electron_buffers)
        self._new_electron_buffers = []
        return result

    def get_new_ions(self) -> dict[str, cp.ndarray] | None:
        ions_by_species = self.get_new_ions_by_species()
        if not ions_by_species:
            return None
        return _concat_particle_dicts(list(ions_by_species.values()))

    def get_new_ions_by_species(self) -> dict[str, dict[str, cp.ndarray]]:
        if not self._new_ion_buffers_by_species:
            return {}
        result = {
            name: _concat_particle_dicts(buffers)
            for name, buffers in self._new_ion_buffers_by_species.items()
            if buffers
        }
        self._new_ion_buffers_by_species = {}
        return result

    def get_event_positions(self) -> dict[str, dict[str, np.ndarray]]:
        buffers = self._event_buffers
        self._event_buffers = {}
        return buffers

    def get_collision_weight_sums(self) -> dict[str, float]:
        weights = self._weight_sums
        self._weight_sums = {}
        return {name: float(value) for name, value in weights.items()}

    def update_background_state(self, background_state: dict[str, float]) -> None:
        for handler in self.handlers:
            background_name = getattr(handler, "background_species_name", None)
            if background_name is None:
                continue
            if background_name in background_state:
                handler.update_background_density(background_state[background_name])

    def _combined_upper_rate(self) -> float:
        return float(sum(max(handler._nu_null, 0.0) for handler in self.handlers))


def make_electron_ar_mcc(
    n_ar: float,
    sigma_elastic: CrossSectionTable | None = None,
    sigma_excitation: CrossSectionTable | None = None,
    sigma_ionization: CrossSectionTable | None = None,
) -> MCCHandler:
    """Create MCC handler for electron-Ar collisions."""
    from plasma.core.constants import M_AR

    handler = MCCHandler(
        projectile_mass=M_ELECTRON,
        background_mass=M_AR,
        background_density=n_ar,
    )

    if sigma_elastic is not None:
        handler.add_process(CollisionProcess(
            name="e_Ar_elastic",
            collision_type=CollisionType.ELASTIC,
            cross_section=sigma_elastic,
        ))

    if sigma_excitation is not None:
        handler.add_process(CollisionProcess(
            name="e_Ar_excitation",
            collision_type=CollisionType.EXCITATION,
            cross_section=sigma_excitation,
            threshold_ev=11.55,
        ))

    if sigma_ionization is not None:
        handler.add_process(CollisionProcess(
            name="e_Ar_ionization",
            collision_type=CollisionType.IONIZATION,
            cross_section=sigma_ionization,
            threshold_ev=15.76,
            product_species_name="electron",
            product_ion_name="Ar+",
        ))

    return handler


def make_ion_ar_mcc(
    n_ar: float,
    ion_mass: float,
    sigma_charge_exchange: CrossSectionTable | None = None,
    sigma_elastic: CrossSectionTable | None = None,
) -> MCCHandler:
    """Create MCC handler for ion-Ar collisions (charge exchange + elastic)."""
    from plasma.core.constants import M_AR

    handler = MCCHandler(
        projectile_mass=ion_mass,
        background_mass=M_AR,
        background_density=n_ar,
    )

    if sigma_charge_exchange is not None:
        handler.add_process(CollisionProcess(
            name="ion_Ar_cx",
            collision_type=CollisionType.CHARGE_EXCHANGE,
            cross_section=sigma_charge_exchange,
        ))

    if sigma_elastic is not None:
        handler.add_process(CollisionProcess(
            name="ion_Ar_elastic",
            collision_type=CollisionType.ELASTIC,
            cross_section=sigma_elastic,
        ))

    return handler
