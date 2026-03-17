"""Boris particle pusher for 2D axisymmetric (r, z) PIC.

Implements the standard Boris algorithm (leapfrog integration) for charged
particles in combined electric and magnetic fields:

    m * dv/dt = q * (E + v x B)
    dr/dt = v

The Boris scheme splits into:
    1. Half electric acceleration: v^- = v^{n-1/2} + (q*dt/2m) * E
    2. Magnetic rotation: v^- -> v^+ (exact rotation preserving |v_perp|)
    3. Half electric acceleration: v^{n+1/2} = v^+ + (q*dt/2m) * E
    4. Position update: r^{n+1} = r^n + dt * v^{n+1/2}

In axisymmetric geometry with 3V (vr, vz, vtheta) and 2D position (r, z),
the magnetic field B = (Br, Bz) creates rotation in the (vr, vtheta) plane.

References:
    - Boris, J.P. (1970). Proc. Fourth Conf. on Numerical Simulation of Plasmas.
    - Birdsall & Langdon, "Plasma Physics via Computer Simulation", Ch. 4.
    - Taccogna et al. (2023), J. Appl. Phys. 134, Eq. 3.
"""

from __future__ import annotations

from plasma.runtime.cupy_compat import cp
from plasma.runtime.numba_compat import cuda


@cuda.jit(fastmath=True)
def boris_push_kernel(
    r, z, vr, vz, vtheta, alive,
    Er, Ez, Br, Bz,
    qm, dt,
    n_particles,
):
    """Numba CUDA kernel: Boris push for one species.

    Each thread handles one particle. Fields (Er, Ez, Br, Bz) are
    pre-interpolated to particle positions.

    Args:
        r, z: Particle positions [m]. Shape (n,).
        vr, vz, vtheta: Velocity components [m/s]. Shape (n,).
        alive: Alive flags (1=active). Shape (n,).
        Er, Ez: Electric field at particle positions [V/m]. Shape (n,).
        Br, Bz: Magnetic field at particle positions [T]. Shape (n,).
        qm: Charge-to-mass ratio q/m [C/kg]. Scalar.
        dt: Timestep [s]. Scalar.
        n_particles: Number of particles.
    """
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    # Pre-compute half-step constants
    half_qm_dt = 0.5 * qm * dt

    # --- Step 1: Half electric acceleration ---
    vr_minus = vr[idx] + half_qm_dt * Er[idx]
    vz_minus = vz[idx] + half_qm_dt * Ez[idx]
    vt_minus = vtheta[idx]  # No E_theta in axisymmetric

    # --- Step 2: Magnetic rotation ---
    # t-vector: t = (q/m) * B * dt/2
    # In axisymmetric (r, z) with B = (Br, Bz), the rotation is in
    # the plane perpendicular to B. With vtheta as the third component,
    # v x B has components in (r, z, theta).
    #
    # v x B for (vr, vz, vtheta) x (Br, 0_theta, Bz):
    #   (v x B)_r     = vz * B_theta - vtheta * Bz  → -vtheta * Bz (B_theta = 0)
    #   (v x B)_z     = vtheta * Br - vr * B_theta   → vtheta * Br
    #   (v x B)_theta = vr * Bz - vz * Br
    #
    # But in axisymmetric geometry B_theta = 0, so:
    tr = half_qm_dt * Br[idx]
    tz = half_qm_dt * Bz[idx]

    # s = 2t / (1 + |t|^2)
    t_mag2 = tr * tr + tz * tz
    sr = 2.0 * tr / (1.0 + t_mag2)
    sz = 2.0 * tz / (1.0 + t_mag2)

    # v' = v^- + v^- x t
    # Cross product: v^- x t (where t has components tr, 0, tz in r,theta,z)
    vr_prime = vr_minus + (vz_minus * tz - vt_minus * 0.0)  # no t_theta
    # Actually, let's be precise about the cross product in cylindrical coords:
    # v x t where v = (vr, vtheta, vz) and t = (tr, 0, tz)
    # (v x t)_r = vtheta * tz - vz * 0 = vtheta * tz
    # (v x t)_theta = vz * tr - vr * tz
    # (v x t)_z = vr * 0 - vtheta * tr = -vtheta * tr
    vr_prime = vr_minus + vt_minus * tz
    vt_prime = vt_minus + (vz_minus * tr - vr_minus * tz)
    vz_prime = vz_minus + (-vt_minus * tr)

    # v^+ = v^- + v' x s
    vr_plus = vr_minus + (vt_prime * sz + (-vz_prime) * 0.0)
    # Same cross product pattern with s:
    vr_plus = vr_minus + vt_prime * sz
    vt_plus = vt_minus + (vz_prime * sr - vr_prime * sz)
    vz_plus = vz_minus + (-vt_prime * sr)

    # --- Step 3: Second half electric acceleration ---
    vr[idx] = vr_plus + half_qm_dt * Er[idx]
    vz[idx] = vz_plus + half_qm_dt * Ez[idx]
    vtheta[idx] = vt_plus

    # --- Step 4: Position update ---
    r[idx] = r[idx] + dt * vr[idx]
    z[idx] = z[idx] + dt * vz[idx]

    # Handle axis reflection: if particle crosses r=0
    if r[idx] <= 0.0:
        r_reflected = -r[idx]
        if r_reflected == 0.0:
            r_reflected = 1e-30
        r[idx] = r_reflected
        vr[idx] = -vr[idx]
        vtheta[idx] = -vtheta[idx]


def boris_push(
    particles,
    Er: cp.ndarray,
    Ez: cp.ndarray,
    Br: cp.ndarray,
    Bz: cp.ndarray,
    dt: float,
) -> None:
    """Push particles using Boris algorithm.

    Args:
        particles: ParticleArray to update in-place.
        Er, Ez: Electric field at particle positions [V/m].
        Br, Bz: Magnetic field at particle positions [T].
        dt: Timestep [s].
    """
    n = particles.count
    if n == 0:
        return

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    boris_push_kernel[blocks, threads_per_block](
        particles.r, particles.z,
        particles.vr, particles.vz, particles.vtheta,
        particles.alive,
        Er, Ez, Br, Bz,
        particles.species.qm_ratio,
        dt,
        n,
    )


@cuda.jit(fastmath=True)
def electrostatic_push_kernel(
    r, z, vr, vz, alive,
    Er, Ez,
    qm, dt,
    n_particles,
):
    """Simplified push without magnetic field (for 1D/electrostatic tests)."""
    idx = cuda.grid(1)
    if idx >= n_particles:
        return
    if alive[idx] == 0:
        return

    # Leapfrog: velocity at half-steps, position at integer steps
    vr[idx] += qm * dt * Er[idx]
    vz[idx] += qm * dt * Ez[idx]

    r[idx] += dt * vr[idx]
    z[idx] += dt * vz[idx]

    if r[idx] <= 0.0:
        r_reflected = -r[idx]
        if r_reflected == 0.0:
            r_reflected = 1e-30
        r[idx] = r_reflected
        vr[idx] = -vr[idx]


def electrostatic_push(
    particles,
    Er: cp.ndarray,
    Ez: cp.ndarray,
    dt: float,
) -> None:
    """Push particles in electric field only (no B)."""
    n = particles.count
    if n == 0:
        return

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    electrostatic_push_kernel[blocks, threads_per_block](
        particles.r, particles.z,
        particles.vr, particles.vz,
        particles.alive,
        Er, Ez,
        particles.species.qm_ratio,
        dt,
        n,
    )
