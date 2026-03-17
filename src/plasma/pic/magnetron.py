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

from dataclasses import dataclass, field

import numpy as np

from plasma.core.constants import E_CHARGE
from plasma.data.sputtering import SputterYield
from plasma.runtime.cupy_compat import cp
from plasma.runtime.numba_compat import cuda
from plasma.runtime.random import SimulationRNG, uniform_cpu


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
    species_see_yields: dict[str, float] = field(default_factory=dict)
    species_sputter_yields: dict[str, SputterYield] = field(default_factory=dict)

    def is_on_target(self, r: float) -> bool:
        """Check if radial position is within the erosion zone."""
        return self.r_inner <= r <= self.r_outer

    def is_on_target_array(self, r: np.ndarray) -> np.ndarray:
        """Vectorized check for target zone."""
        return (r >= self.r_inner) & (r <= self.r_outer)

    def see_yield_for_species(self, species_name: str) -> float:
        """Return the SEE yield for one impacting ion species."""

        return float(self.species_see_yields.get(species_name, self.see_yield))

    def sputter_yield_for_species(self, species_name: str) -> SputterYield | None:
        """Return the sputter-yield model for one impacting ion species."""

        return self.species_sputter_yields.get(species_name, self.sputter_yield)


def process_target_impacts(
    target: MagnetronTarget,
    ion_particles,
    grid,
    *,
    rng=None,
) -> dict[str, np.ndarray | None]:
    """Process ions hitting the target surface.

    For each ion that has crossed z=0 and lies within the erosion zone:
    1. Compute impact energy from particle kinetic energy (no sheath
       potential correction — assumes ions were already accelerated
       through the sheath by the self-consistent field solve).
    2. Generate secondary electrons (constant Bernoulli yield).
    3. Generate sputtered neutral atoms (energy-only Y(E), cosine emission).

    GPU pre-filter: hit detection runs on GPU to avoid transferring all
    particle data.  Only the compact subset of hitting particles is
    transferred to CPU for the surface physics calculations.

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

    # GPU pre-filter: find hits on device to avoid full-array transfer
    z_gpu = ion_particles.z[:n]
    alive_gpu = ion_particles.alive[:n]
    r_gpu = ion_particles.r[:n]

    hit_mask_gpu = (
        (alive_gpu == 1)
        & (z_gpu <= target.z_target)
        & (r_gpu >= target.r_inner)
        & (r_gpu <= target.r_outer)
    )
    hit_idx_gpu = cp.where(hit_mask_gpu)[0]
    n_hits = len(hit_idx_gpu)

    if n_hits == 0:
        return {"see_electrons": None, "sputtered_neutrals": None, "impact_positions": None, "n_impacts": 0}

    # Transfer only the compact hit subset to CPU
    r_cpu = cp.asnumpy(r_gpu[hit_idx_gpu])
    vr_cpu = cp.asnumpy(ion_particles.vr[:n][hit_idx_gpu])
    vz_cpu = cp.asnumpy(ion_particles.vz[:n][hit_idx_gpu])
    vt_cpu = cp.asnumpy(ion_particles.vtheta[:n][hit_idx_gpu])
    w_cpu = cp.asnumpy(ion_particles.weight[:n][hit_idx_gpu])

    # Impact energies [eV] from kinetic energy (compact arrays — already hit subset)
    v2 = vr_cpu**2 + vz_cpu**2 + vt_cpu**2
    impact_energy_ev = 0.5 * ion_particles.species.mass * v2 / E_CHARGE

    if rng is None:
        rng = SimulationRNG()
    species_name = ion_particles.species.name
    see_yield = target.see_yield_for_species(species_name)
    sputter_yield = target.sputter_yield_for_species(species_name)

    # --- Secondary Electron Emission ---
    see_data = None
    if see_yield > 0:
        see_rand = uniform_cpu(rng, n_hits)
        see_mask = see_rand < see_yield
        n_see = int(np.sum(see_mask))

        if n_see > 0:
            see_r = r_cpu[see_mask]
            see_z = np.full(n_see, target.z_target + 1e-6)

            v_see = np.sqrt(2.0 * target.see_energy_ev * E_CHARGE / 9.109e-31)
            cos_theta = uniform_cpu(rng, n_see)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            phi = 2.0 * np.pi * uniform_cpu(rng, n_see)

            see_data = {
                "r": see_r,
                "z": see_z,
                "vr": v_see * sin_theta * np.cos(phi),
                "vz": v_see * cos_theta,
                "vtheta": v_see * sin_theta * np.sin(phi),
                "weight": w_cpu[see_mask],
            }

    # --- Sputtering ---
    sputter_data = None
    if sputter_yield is not None:
        yields = sputter_yield(impact_energy_ev)

        n_sputtered_per_ion = np.floor(yields).astype(int)
        frac = yields - n_sputtered_per_ion
        extra = (uniform_cpu(rng, n_hits) < frac).astype(int)
        n_sputtered_per_ion += extra

        total_sputtered = int(np.sum(n_sputtered_per_ion))

        if total_sputtered > 0:
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

                sp_r[idx:idx + ns] = r_cpu[i]
                sp_z[idx:idx + ns] = target.z_target + 1e-6

                e_b = target.surface_binding_ev
                u = uniform_cpu(rng, ns)
                sp_energy = e_b * u / (1.0 - u + 1e-10)
                sp_energy = np.minimum(sp_energy, impact_energy_ev[i])
                sp_energy = np.maximum(sp_energy, 0.01)

                sp_speed = np.sqrt(2.0 * sp_energy * E_CHARGE / target.material_mass)

                cos_theta = np.sqrt(uniform_cpu(rng, ns))
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                phi = 2.0 * np.pi * uniform_cpu(rng, ns)

                sp_vr[idx:idx + ns] = sp_speed * sin_theta * np.cos(phi)
                sp_vz[idx:idx + ns] = sp_speed * cos_theta
                sp_vt[idx:idx + ns] = sp_speed * sin_theta * np.sin(phi)
                sp_w[idx:idx + ns] = w_cpu[i]

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
            "r": r_cpu.copy(),
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
