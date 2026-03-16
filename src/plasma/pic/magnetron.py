"""Magnetron boundary conditions: simplified SEE, sputtering, target geometry.

Prototype implementation of magnetron target surface physics.  The models
here are simplified closures suitable for early-stage PIC integration,
**not** research-grade surface interaction models:

    - **SEE**: Constant Bernoulli yield (gamma), independent of ion species,
      angle, and impact energy.  Real SEE depends on all three.
    - **Sputtering**: Energy-only Yamamura yield Y(E) with cosine angular
      emission of sputtered atoms.  This is *not* an angle-dependent yield
      law Y(E, theta) — the cosine applies to the emission distribution,
      not the incidence angle dependence of the yield itself.
    - **Impact energy**: Computed from particle kinetic energy only.
      Sheath potential contribution is not included; the caller is
      responsible for ensuring ions have been accelerated through the
      sheath before reaching the target boundary.

Race-track geometry constrains where sputtering occurs: between the inner
and outer erosion radii defined by the magnetic field topology.

References:
    - Gudmundsson et al. (2022), Surf. Coat. Technol. 442, 128189.
    - Depla & Mahieu, "Reactive Sputter Deposition", Ch. 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from numba import cuda

from plasma.core.constants import E_CHARGE
from plasma.data.sputtering import SputterYield


@dataclass
class MagnetronTarget:
    """Magnetron target surface properties.

    Attributes:
        z_target: Axial position of target surface [m] (typically 0).
        r_inner: Inner erosion race-track radius [m].
        r_outer: Outer erosion race-track radius [m].
        see_yield: Secondary electron emission yield (electrons per ion).
        see_energy_ev: Energy of emitted secondary electrons [eV].
        sputter_yield: Sputtering yield model Y(E).
        surface_binding_ev: Surface binding energy for sputtered atoms [eV].
        material_mass: Mass of target atom [kg].
    """

    z_target: float = 0.0
    r_inner: float = 0.015
    r_outer: float = 0.035
    see_yield: float = 0.1
    see_energy_ev: float = 3.0
    sputter_yield: SputterYield | None = None
    surface_binding_ev: float = 3.49  # Cu cohesive energy
    material_mass: float = 63.546 * 1.66053906660e-27  # Cu

    def is_on_target(self, r: float) -> bool:
        """Check if radial position is within the erosion zone."""
        return self.r_inner <= r <= self.r_outer

    def is_on_target_array(self, r: np.ndarray) -> np.ndarray:
        """Vectorized check for target zone."""
        return (r >= self.r_inner) & (r <= self.r_outer)


def process_target_impacts(
    target: MagnetronTarget,
    ion_particles,
    grid,
) -> dict[str, np.ndarray | None]:
    """Process ions hitting the target surface.

    For each ion that has crossed z=0 and lies within the erosion zone:
    1. Compute impact energy from particle kinetic energy (no sheath
       potential correction — assumes ions were already accelerated
       through the sheath by the self-consistent field solve).
    2. Generate secondary electrons (constant Bernoulli yield).
    3. Generate sputtered neutral atoms (energy-only Y(E), cosine emission).

    This function re-detects hits from particle arrays rather than consuming
    the hit_target_flags returned by apply_magnetron_boundaries.  It should
    be called *before* the boundary kernel kills the particles (or on a copy).

    Args:
        target: Magnetron target configuration.
        ion_particles: ParticleArray of ions hitting the target.
        grid: CylindricalGrid (for domain info).

    Returns:
        Dict with keys:
            "see_electrons": dict of arrays for new secondary electrons
            "sputtered_neutrals": dict of arrays for new sputtered neutrals
            "impact_positions": dict of arrays for target impact sites
            "n_impacts": number of ion impacts on target
    """
    n = ion_particles.count
    if n == 0:
        return {"see_electrons": None, "sputtered_neutrals": None, "impact_positions": None, "n_impacts": 0}

    # Find ions that are at or below the target surface
    z_cpu = cp.asnumpy(ion_particles.z[:n])
    alive_cpu = cp.asnumpy(ion_particles.alive[:n])
    r_cpu = cp.asnumpy(ion_particles.r[:n])
    vr_cpu = cp.asnumpy(ion_particles.vr[:n])
    vz_cpu = cp.asnumpy(ion_particles.vz[:n])
    vt_cpu = cp.asnumpy(ion_particles.vtheta[:n])
    w_cpu = cp.asnumpy(ion_particles.weight[:n])

    # Mask: alive, at target (z <= z_target), and within erosion zone
    hit_mask = (
        (alive_cpu == 1)
        & (z_cpu <= target.z_target)
        & target.is_on_target_array(r_cpu)
    )
    hit_idx = np.where(hit_mask)[0]
    n_hits = len(hit_idx)

    if n_hits == 0:
        return {"see_electrons": None, "sputtered_neutrals": None, "impact_positions": None, "n_impacts": 0}

    # Impact energies [eV] from kinetic energy
    v2 = vr_cpu[hit_idx]**2 + vz_cpu[hit_idx]**2 + vt_cpu[hit_idx]**2
    impact_energy_ev = 0.5 * ion_particles.species.mass * v2 / E_CHARGE

    rng = np.random.default_rng()

    # --- Secondary Electron Emission ---
    see_data = None
    if target.see_yield > 0:
        # Probabilistic SEE: each impact has probability = yield
        # For yield < 1, each ion randomly emits 0 or 1 electron
        see_rand = rng.random(n_hits)
        see_mask = see_rand < target.see_yield
        n_see = int(np.sum(see_mask))

        if n_see > 0:
            see_r = r_cpu[hit_idx][see_mask]
            see_z = np.full(n_see, target.z_target + 1e-6)  # Just above target

            # SEE electrons emitted with low energy, random direction into half-space
            v_see = np.sqrt(2.0 * target.see_energy_ev * E_CHARGE / 9.109e-31)
            cos_theta = rng.random(n_see)  # Half-space: cos(theta) in [0, 1]
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            phi = 2.0 * np.pi * rng.random(n_see)

            see_data = {
                "r": see_r,
                "z": see_z,
                "vr": v_see * sin_theta * np.cos(phi),
                "vz": v_see * cos_theta,  # Away from target (+z)
                "vtheta": v_see * sin_theta * np.sin(phi),
                "weight": w_cpu[hit_idx][see_mask],
            }

    # --- Sputtering ---
    sputter_data = None
    if target.sputter_yield is not None:
        # Compute yield for each impact energy
        yields = target.sputter_yield(impact_energy_ev)

        # Probabilistic: each impact creates floor(Y) + maybe 1 more atom
        n_sputtered_per_ion = np.floor(yields).astype(int)
        frac = yields - n_sputtered_per_ion
        extra = (rng.random(n_hits) < frac).astype(int)
        n_sputtered_per_ion += extra

        total_sputtered = int(np.sum(n_sputtered_per_ion))

        if total_sputtered > 0:
            # Allocate arrays for sputtered atoms
            sp_r = np.empty(total_sputtered)
            sp_z = np.empty(total_sputtered)
            sp_vr = np.empty(total_sputtered)
            sp_vz = np.empty(total_sputtered)
            sp_vt = np.empty(total_sputtered)
            sp_w = np.empty(total_sputtered)

            idx = 0
            for i in range(n_hits):
                ns = n_sputtered_per_ion[i]
                if ns == 0:
                    continue

                # Position: at target surface, same r as impact
                sp_r[idx:idx + ns] = r_cpu[hit_idx[i]]
                sp_z[idx:idx + ns] = target.z_target + 1e-6

                # Thompson energy distribution: f(E) ∝ E / (E + E_b)^3
                # where E_b = surface binding energy
                # Sample using inverse CDF
                e_b = target.surface_binding_ev
                u = rng.random(ns)
                # Approximate: E = E_b * u / (1 - u) capped at impact energy
                sp_energy = e_b * u / (1.0 - u + 1e-10)
                sp_energy = np.minimum(sp_energy, impact_energy_ev[i])
                sp_energy = np.maximum(sp_energy, 0.01)

                sp_speed = np.sqrt(2.0 * sp_energy * E_CHARGE / target.material_mass)

                # Cosine distribution into half-space (away from target)
                cos_theta = np.sqrt(rng.random(ns))  # cos-weighted
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                phi = 2.0 * np.pi * rng.random(ns)

                sp_vr[idx:idx + ns] = sp_speed * sin_theta * np.cos(phi)
                sp_vz[idx:idx + ns] = sp_speed * cos_theta  # +z away from target
                sp_vt[idx:idx + ns] = sp_speed * sin_theta * np.sin(phi)
                sp_w[idx:idx + ns] = w_cpu[hit_idx[i]]

                idx += ns

            sputter_data = {
                "r": sp_r[:idx],
                "z": sp_z[:idx],
                "vr": sp_vr[:idx],
                "vz": sp_vz[:idx],
                "vtheta": sp_vt[:idx],
                "weight": sp_w[:idx],
            }

    return {
        "see_electrons": see_data,
        "sputtered_neutrals": sputter_data,
        "impact_positions": {
            "r": r_cpu[hit_idx].copy(),
            "z": np.full(n_hits, target.z_target, dtype=np.float64),
        },
        "n_impacts": n_hits,
    }


@cuda.jit
def apply_magnetron_boundaries_kernel(
    r, z, vr, vz, vtheta, alive,
    r_max, z_max, z_target, r_inner, r_outer,
    hit_target_flag,
    n_particles,
):
    """Mark particles as dead and flag target impacts.

    hit_target_flag[idx] = 1 if particle hits the target erosion zone.
    Note: process_target_impacts currently re-detects hits from particle
    arrays rather than consuming these flags.  The flags are returned to
    the caller for optional diagnostics (e.g., counting erosion-zone hits).
    """
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    # Outer radial wall
    if r[idx] >= r_max:
        alive[idx] = 0
        return

    # Substrate (z = z_max) — absorbing
    if z[idx] >= z_max:
        alive[idx] = 0
        return

    # Target surface (z <= z_target)
    if z[idx] <= z_target:
        # Check if on erosion zone
        if r[idx] >= r_inner and r[idx] <= r_outer:
            hit_target_flag[idx] = 1
        alive[idx] = 0
        return

    # Axis reflection
    if r[idx] < 0.0:
        r[idx] = -r[idx]
        vr[idx] = -vr[idx]
        vtheta[idx] = -vtheta[idx]


def apply_magnetron_boundaries(
    grid,
    particles,
    target: MagnetronTarget,
) -> tuple[int, cp.ndarray]:
    """Apply magnetron-specific boundary conditions.

    Args:
        grid: CylindricalGrid.
        particles: ParticleArray.
        target: MagnetronTarget.

    Returns:
        (n_absorbed, hit_target_flags): Number absorbed and per-particle flags.
    """
    n = particles.count
    if n == 0:
        return 0, cp.zeros(0, dtype=cp.int32)

    n_alive_before = particles.n_alive
    hit_flags = cp.zeros(n, dtype=cp.int32)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    apply_magnetron_boundaries_kernel[blocks, threads_per_block](
        particles.r, particles.z,
        particles.vr, particles.vz, particles.vtheta,
        particles.alive,
        grid.r_max, grid.z_max,
        target.z_target, target.r_inner, target.r_outer,
        hit_flags,
        n,
    )

    n_alive_after = particles.n_alive
    return n_alive_before - n_alive_after, hit_flags


@dataclass
class MagnetronGeometry:
    """Complete magnetron geometry for PIC simulation.

    Combines the grid, target, and magnetic field configuration.
    """

    r_target: float         # Total target radius [m]
    r_inner_erosion: float  # Inner erosion zone [m]
    r_outer_erosion: float  # Outer erosion zone [m]
    z_target: float = 0.0   # Target axial position [m]
    z_substrate: float = 0.1  # Substrate position [m]
    r_max: float = 0.06     # Simulation domain outer radius [m]

    # Magnetic field parameters
    inner_magnet_r: float = 0.012   # Inner magnet pole radius [m]
    outer_magnet_r: float = 0.038   # Outer magnet pole radius [m]
    magnet_z: float = -0.005        # Magnet assembly position [m]
    magnet_current_inner: float = 1000.0  # Equivalent current [A]
    magnet_current_outer: float = -600.0  # Opposite polarity

    def make_target(
        self,
        see_yield: float = 0.1,
        sputter_yield: SputterYield | None = None,
        material_mass: float = 63.546 * 1.66053906660e-27,
        surface_binding_ev: float = 3.49,
    ) -> MagnetronTarget:
        """Create a MagnetronTarget from this geometry."""
        return MagnetronTarget(
            z_target=self.z_target,
            r_inner=self.r_inner_erosion,
            r_outer=self.r_outer_erosion,
            see_yield=see_yield,
            sputter_yield=sputter_yield,
            material_mass=material_mass,
            surface_binding_ev=surface_binding_ev,
        )

    def make_bfield(self, grid):
        """Compute magnetic field for this magnetron geometry."""
        from plasma.pic.magnetic import magnetron_field

        return magnetron_field(
            grid,
            inner_loop_r=self.inner_magnet_r,
            outer_loop_r=self.outer_magnet_r,
            loop_z=self.magnet_z,
            current_inner=self.magnet_current_inner,
            current_outer=self.magnet_current_outer,
        )
