"""Monte Carlo Collisions (MCC) — hybrid CPU/GPU prototype.

Null-collision scheme loosely following Vahedi & Surendra (1995) and
Taccogna 2023 Eq. 5.  Candidate selection (step 1) runs on GPU via
CuPy; collision physics (accept/reject, scattering angles, energy
partition) runs on CPU with NumPy.  This is adequate for prototype
validation but is not a fully-GPU, reproducible-RNG implementation.

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
       - If u <= P_null: transfer to CPU for accept/reject per-process
         using sigma_i(E) * v_rel / max_sample(sigma * v_rel).

Collision types supported:
    - Elastic scattering (isotropic, energy loss ~ 2 m_e/M per collision)
    - Excitation (inelastic, threshold energy removed, isotropic scatter)
    - Ionization (equal energy split between scattered and ejected electron)
    - Charge exchange (ion velocity replaced by thermal neutral velocity)

Limitations:
    - CPU-side collision physics uses unseeded np.random.default_rng()
      inside each method; the CuPy RNG passed to perform_collisions()
      controls only candidate selection, not scattering angles.
    - Ionization products are accumulated in lists per perform_collisions()
      call, but are overwritten across successive calls — caller must
      retrieve via get_new_electrons()/get_new_ions() before the next call.

References:
    - Vahedi, V. & Surendra, M. (1995). Comp. Phys. Comm. 87, 179-198.
    - Nanbu, K. (2000). IEEE Trans. Plasma Sci. 28, 971-990.
    - Taccogna et al. (2023), J. Appl. Phys. 134, 150901.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cupy as cp
import numpy as np

from plasma.core.constants import E_CHARGE, M_ELECTRON
from plasma.data.cross_sections import CrossSectionTable


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


def _concat_particle_dicts(buffers: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate a list of particle dicts (same keys) into one."""
    keys = buffers[0].keys()
    return {k: np.concatenate([b[k] for b in buffers]) for k in keys}


@dataclass
class MCCHandler:
    """Null-collision MCC handler for one projectile vs. one background species.

    Hybrid CPU/GPU prototype: candidate selection on GPU (CuPy RNG),
    accept/reject and scattering physics on CPU (NumPy RNG).  The CPU-side
    RNG is *not* seeded by the caller — reproducibility is limited to
    which particles are selected as collision candidates.

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
        [0.01, 10000] eV.  This is a conservative sampled bound, not an
        exact supremum — adequate for cross-sections without narrow
        resonance features between sample points.
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

        # v_rel from kinetic energy: E = 0.5 * m_r * v^2 → v = sqrt(2E/m_r)
        # But for electron-neutral: m_r ≈ m_e, and E is the electron energy
        # For ion-neutral: use reduced mass
        if self.projectile_mass < 1e-28:
            # Electron projectile: v = sqrt(2*E*e / m_e)
            v_rel = np.sqrt(2.0 * e_sample * E_CHARGE / self.projectile_mass)
        else:
            # Ion projectile: v = sqrt(2*E*e / m_reduced)
            v_rel = np.sqrt(2.0 * e_sample * E_CHARGE / m_r)

        # Sum sigma*v over all processes at each energy
        total_sigma_v = np.zeros_like(e_sample)
        for proc in self.processes:
            sigma = proc.cross_section(e_sample)
            total_sigma_v += sigma * v_rel

        self._max_sigma_vrel = float(np.max(total_sigma_v))
        self._nu_null = self.background_density * self._max_sigma_vrel

    def collision_probability(self, dt: float) -> float:
        """Null-collision probability for timestep dt.

        P = 1 - exp(-nu_null * dt)
        """
        if self._nu_null <= 0:
            return 0.0
        return 1.0 - np.exp(-self._nu_null * dt)

    def perform_collisions(
        self,
        particles,
        dt: float,
        rng: cp.random.RandomState | None = None,
    ) -> dict[str, int]:
        """Perform MCC collisions on all particles of this species.

        Args:
            particles: ParticleArray of projectile species.
            dt: Timestep [s].
            rng: CuPy random state (for reproducibility).

        Returns:
            Dict of collision counts by process name.
        """
        if rng is None:
            rng = cp.random.RandomState()

        # Clear ionization product buffers from any prior call
        self._new_electron_buffers = []
        self._new_ion_buffers = []
        self._event_buffers = {}

        n = particles.count
        if n == 0 or not self.processes:
            return {}

        p_null = self.collision_probability(dt)
        if p_null <= 0:
            return {}

        # Step 1: Determine which particles undergo a potential collision
        rand_select = rng.rand(n, dtype=cp.float64)
        alive_mask = particles.alive[:n] == 1
        collide_mask = (rand_select < p_null) & alive_mask

        # Get indices of particles that potentially collide
        collide_indices = cp.where(collide_mask)[0]
        n_collide = len(collide_indices)

        if n_collide == 0:
            return {}

        # Step 2: Get particle velocities and compute energies
        vr = cp.asnumpy(particles.vr[:n][collide_indices])
        vz = cp.asnumpy(particles.vz[:n][collide_indices])
        vt = cp.asnumpy(particles.vtheta[:n][collide_indices])
        v2 = vr**2 + vz**2 + vt**2
        v_mag = np.sqrt(v2)

        # Kinetic energy in eV
        energy_ev = 0.5 * self.projectile_mass * v2 / E_CHARGE

        # Step 3: Accept/reject for each collision type
        # For each potential collision, compute sigma_i * v_rel / max(sigma * v_rel)
        counts: dict[str, int] = {}
        processed = np.zeros(n_collide, dtype=bool)

        # Random number for accept/reject (CPU-side, unseeded)
        cpu_rng = np.random.default_rng()
        rand_type = cpu_rng.random(n_collide)

        cumulative_prob = np.zeros(n_collide)

        for proc in self.processes:
            sigma = proc.cross_section(energy_ev)
            sigma_vrel = sigma * v_mag

            # Probability of this collision type (relative to null)
            p_type = np.where(
                self._max_sigma_vrel > 0,
                sigma_vrel / self._max_sigma_vrel,
                0.0,
            )

            cumulative_prob += p_type

            # Particles that get this collision type
            this_collision = (~processed) & (rand_type < cumulative_prob)
            collision_idx = np.where(this_collision)[0]

            if len(collision_idx) == 0:
                counts[proc.name] = 0
                continue

            # Get original particle indices
            orig_idx = cp.asnumpy(collide_indices[collision_idx])
            self._event_buffers[proc.name] = {
                "r": cp.asnumpy(particles.r[cp.asarray(orig_idx)]),
                "z": cp.asnumpy(particles.z[cp.asarray(orig_idx)]),
            }

            # Apply collision
            n_events = self._apply_collision(
                proc, particles, orig_idx,
                energy_ev[collision_idx], v_mag[collision_idx],
                vr[collision_idx], vz[collision_idx], vt[collision_idx],
            )
            counts[proc.name] = n_events
            processed[collision_idx] = True

        # Remaining particles had null collisions (no effect)
        counts["null"] = int(np.sum(~processed))

        return counts

    def _apply_collision(
        self,
        process: CollisionProcess,
        particles,
        indices: np.ndarray,
        energy_ev: np.ndarray,
        v_mag: np.ndarray,
        vr: np.ndarray,
        vz: np.ndarray,
        vt: np.ndarray,
    ) -> int:
        """Apply a specific collision to selected particles.

        Returns number of collisions applied.
        """
        n = len(indices)
        if n == 0:
            return 0

        if process.collision_type == CollisionType.ELASTIC:
            self._elastic_scatter(particles, indices, energy_ev, v_mag)
        elif process.collision_type == CollisionType.EXCITATION:
            self._excitation(particles, indices, energy_ev, v_mag, process.threshold_ev)
        elif process.collision_type == CollisionType.IONIZATION:
            self._ionization(particles, indices, energy_ev, v_mag, process.threshold_ev)
        elif process.collision_type == CollisionType.CHARGE_EXCHANGE:
            self._charge_exchange(particles, indices)

        return n

    def _elastic_scatter(
        self,
        particles,
        indices: np.ndarray,
        energy_ev: np.ndarray,
        v_mag: np.ndarray,
    ) -> None:
        """Isotropic elastic scattering.

        For electron-neutral: electron is deflected, loses energy fraction
        ~ 2*m_e/M (very small). We randomize direction, keeping |v| constant.

        For ion-neutral: isotropic scattering in CM frame.
        """
        n = len(indices)
        rng = np.random.default_rng()

        # Isotropic scattering: random direction, same speed
        cos_theta = 2.0 * rng.random(n) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random(n)

        # Energy loss for electrons: delta_E/E = 2*m_e/M per collision
        mass_ratio = self.projectile_mass / self.background_mass
        energy_loss_frac = 2.0 * mass_ratio
        new_speed = v_mag * np.sqrt(1.0 - energy_loss_frac)

        new_vr = new_speed * sin_theta * np.cos(phi)
        new_vz = new_speed * cos_theta
        new_vt = new_speed * sin_theta * np.sin(phi)

        idx_gpu = cp.asarray(indices)
        particles.vr[idx_gpu] = cp.asarray(new_vr)
        particles.vz[idx_gpu] = cp.asarray(new_vz)
        particles.vtheta[idx_gpu] = cp.asarray(new_vt)

    def _excitation(
        self,
        particles,
        indices: np.ndarray,
        energy_ev: np.ndarray,
        v_mag: np.ndarray,
        threshold_ev: float,
    ) -> None:
        """Inelastic excitation: electron loses threshold energy.

        Post-collision speed: v' = sqrt(v^2 - 2*E_threshold*e/m)
        Direction is randomized (isotropic).
        """
        n = len(indices)
        rng = np.random.default_rng()

        # New kinetic energy after excitation
        new_energy_ev = np.maximum(energy_ev - threshold_ev, 0.01)
        new_speed = np.sqrt(2.0 * new_energy_ev * E_CHARGE / self.projectile_mass)

        # Isotropic post-collision direction
        cos_theta = 2.0 * rng.random(n) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random(n)

        idx_gpu = cp.asarray(indices)
        particles.vr[idx_gpu] = cp.asarray(new_speed * sin_theta * np.cos(phi))
        particles.vz[idx_gpu] = cp.asarray(new_speed * cos_theta)
        particles.vtheta[idx_gpu] = cp.asarray(new_speed * sin_theta * np.sin(phi))

    def _ionization(
        self,
        particles,
        indices: np.ndarray,
        energy_ev: np.ndarray,
        v_mag: np.ndarray,
        threshold_ev: float,
    ) -> None:
        """Electron-impact ionization: incident e loses energy, new e + ion created.

        Energy partition:
        - Incident electron loses E_threshold
        - Remaining energy split between incident and ejected electron
          (equal sharing is simplest; Opal-Peterson-Beaty for better physics)
        """
        n = len(indices)
        rng = np.random.default_rng()

        # Remaining energy after ionization, split equally
        remaining_ev = np.maximum(energy_ev - threshold_ev, 0.01)
        share_ev = remaining_ev * 0.5

        # Scattered incident electron
        new_speed = np.sqrt(2.0 * share_ev * E_CHARGE / self.projectile_mass)
        cos_theta = 2.0 * rng.random(n) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random(n)

        idx_gpu = cp.asarray(indices)
        particles.vr[idx_gpu] = cp.asarray(new_speed * sin_theta * np.cos(phi))
        particles.vz[idx_gpu] = cp.asarray(new_speed * cos_theta)
        particles.vtheta[idx_gpu] = cp.asarray(new_speed * sin_theta * np.sin(phi))

        # New ejected electron — born at same position, with remaining energy share
        ejected_speed = np.sqrt(2.0 * share_ev * E_CHARGE / self.projectile_mass)
        cos_theta2 = 2.0 * rng.random(n) - 1.0
        sin_theta2 = np.sqrt(1.0 - cos_theta2**2)
        phi2 = 2.0 * np.pi * rng.random(n)

        # Get positions and weights of ionizing electrons
        r_new = cp.asnumpy(particles.r[idx_gpu])
        z_new = cp.asnumpy(particles.z[idx_gpu])
        w_new = cp.asnumpy(particles.weight[idx_gpu])

        # Buffer new electrons (append, don't overwrite — supports multiple
        # ionization channels within one perform_collisions() call).
        electron_data = {
            "r": r_new,
            "z": z_new,
            "vr": ejected_speed * sin_theta2 * np.cos(phi2),
            "vz": ejected_speed * cos_theta2,
            "vtheta": ejected_speed * sin_theta2 * np.sin(phi2),
            "weight": w_new,
        }
        if not hasattr(self, "_new_electron_buffers"):
            self._new_electron_buffers = []
        self._new_electron_buffers.append(electron_data)

        # New ions born cold at collision site
        ion_data = {
            "r": r_new.copy(),
            "z": z_new.copy(),
            "vr": np.zeros(n),
            "vz": np.zeros(n),
            "vtheta": np.zeros(n),
            "weight": w_new.copy(),
        }
        if not hasattr(self, "_new_ion_buffers"):
            self._new_ion_buffers = []
        self._new_ion_buffers.append(ion_data)

    def _charge_exchange(
        self,
        particles,
        indices: np.ndarray,
    ) -> None:
        """Charge exchange: fast ion → slow ion (swap with cold neutral).

        The ion becomes a fast neutral (leaves simulation if we don't track neutrals)
        and a new cold ion is created at the collision site.

        Simplified model: ion velocity is set to thermal background velocity.
        """
        n = len(indices)
        rng = np.random.default_rng()

        # Replace ion velocity with thermal neutral velocity
        # v_th = sqrt(kT/m) ≈ sqrt(0.026 eV * e / M) for room temperature
        v_th = np.sqrt(0.026 * E_CHARGE / self.background_mass)

        cos_theta = 2.0 * rng.random(n) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * rng.random(n)

        idx_gpu = cp.asarray(indices)
        particles.vr[idx_gpu] = cp.asarray(v_th * sin_theta * np.cos(phi))
        particles.vz[idx_gpu] = cp.asarray(v_th * cos_theta)
        particles.vtheta[idx_gpu] = cp.asarray(v_th * sin_theta * np.sin(phi))

    def get_new_electrons(self) -> dict[str, np.ndarray] | None:
        """Retrieve newly created electrons from ionization events.

        Concatenates products from all ionization channels within the
        last perform_collisions() call, then clears the buffer.
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

    def get_new_ions(self) -> dict[str, np.ndarray] | None:
        """Retrieve newly created ions from ionization events.

        Concatenates products from all ionization channels within the
        last perform_collisions() call, then clears the buffer.
        """
        buffers = getattr(self, "_new_ion_buffers", None)
        if not buffers:
            self._new_ion_buffers = []
            return None
        result = _concat_particle_dicts(buffers)
        self._new_ion_buffers = []
        return result

    def update_background_density(self, n_bg: float) -> None:
        """Update background neutral density (e.g., for gas rarefaction)."""
        self.background_density = n_bg
        self._update_null_frequency()


def make_electron_ar_mcc(
    n_ar: float,
    sigma_elastic: CrossSectionTable | None = None,
    sigma_excitation: CrossSectionTable | None = None,
    sigma_ionization: CrossSectionTable | None = None,
) -> MCCHandler:
    """Create MCC handler for electron-Ar collisions.

    Args:
        n_ar: Argon neutral density [m^-3].
        sigma_elastic: Elastic cross-section table (e + Ar -> e + Ar).
        sigma_excitation: Excitation cross-section (e + Ar -> Ar* + e).
        sigma_ionization: Ionization cross-section (e + Ar -> Ar+ + 2e).

    Returns:
        Configured MCCHandler.
    """
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
