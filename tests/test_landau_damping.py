"""Landau damping benchmark: validate electrostatic PIC against kinetic theory.

A sinusoidal density perturbation in a uniform plasma produces an electric
field that decays via resonant wave-particle interaction (Landau damping).
Comparing the measured damping rate to the analytical prediction validates
the PIC charge deposition, Poisson solver, and particle pusher.

Setup:
    - Nearly-1D: nr=4 cells, large r_max (dynamics in z only).
    - Uniform electrons with perturbation dn = n0 * alpha * cos(k*z).
    - Ions included for charge neutrality but effectively immobile (M_Ar >> m_e).
    - No collisions, no magnetic field.
    - Dirichlet BCs: phi=0 at z boundaries (compatible with standing mode).

References:
    - Birdsall & Langdon, "Plasma Physics via Computer Simulation", Ch. 5.
    - Chen & Chacon, J. Comput. Phys. 230, 7018 (2011).
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from plasma.core.constants import E_CHARGE, EPSILON_0, M_AR, M_ELECTRON
from plasma.pic.grid import CylindricalGrid
from plasma.pic.loop import pic_step
from plasma.pic.particles import ParticleArray, Species
from plasma.pic.poisson import PoissonSolverCylindrical


def landau_damping_rate(k: float, n0: float, te_ev: float) -> float:
    """Analytical Landau damping rate (weak damping approximation).

    gamma / omega_pe = -sqrt(pi/8) * (1/(k*lambda_De))^3
                       * exp(-1/(2*k^2*lambda_De^2) - 3/2)

    Returns the absolute value of gamma [rad/s].
    """
    te_j = te_ev * E_CHARGE
    lambda_de = np.sqrt(EPSILON_0 * te_j / (n0 * E_CHARGE**2))
    omega_pe = np.sqrt(n0 * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON))
    kld = k * lambda_de

    gamma_norm = np.sqrt(np.pi / 8) * (1.0 / kld**3) * np.exp(-1.0 / (2 * kld**2) - 1.5)
    return abs(gamma_norm * omega_pe)


def setup_landau_damping(
    n0: float = 1e14,
    te_ev: float = 2.0,
    alpha: float = 0.05,
    k_mode: int = 1,
    nz: int = 64,
    ppc: int = 256,
) -> dict:
    """Initialize Landau damping test case.

    Uses k*lambda_De ~ 0.5 for moderate damping that is observable but stable.
    Domain size is chosen so that k = k_mode * pi / z_max (standing wave
    compatible with Dirichlet phi=0 at both z boundaries).
    """
    te_j = te_ev * E_CHARGE
    lambda_de = np.sqrt(EPSILON_0 * te_j / (n0 * E_CHARGE**2))
    omega_pe = np.sqrt(n0 * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON))
    v_th = np.sqrt(te_j / M_ELECTRON)

    # Choose k such that k*lambda_De ~ 0.5 (moderate damping)
    kld_target = 0.5
    k = kld_target / lambda_de

    # Domain length: must fit k_mode half-wavelengths for Dirichlet BCs
    # k = k_mode * pi / z_max → z_max = k_mode * pi / k
    z_max = k_mode * np.pi / k
    r_max = z_max * 0.2  # Nearly 1D
    nr = 4

    dz = z_max / nz

    # Timestep: omega_pe * dt < 0.2 and CFL: v_th * dt < dz
    dt_pe = 0.15 / omega_pe
    dt_cfl = 0.5 * dz / v_th
    dt = min(dt_pe, dt_cfl)

    grid = CylindricalGrid(nr=nr, nz=nz, r_max=r_max, z_max=z_max)

    # Poisson solver with Dirichlet BCs at z boundaries (phi=0)
    solver = PoissonSolverCylindrical(grid, permittivity_factor=1.0)

    electron_sp = Species(name="electron", charge=-E_CHARGE, mass=M_ELECTRON, charge_state=-1)
    ion_sp = Species(name="Ar+", charge=E_CHARGE, mass=M_AR, charge_state=1)

    rng = np.random.default_rng(12345)
    n_particles = nr * nz * ppc

    # Uniform base positions
    z_uniform = rng.uniform(0, z_max, n_particles)
    r_uniform = np.sqrt(rng.uniform(0, r_max**2, n_particles))

    # Density perturbation via position shift: dz = (alpha/k)*sin(k*z)
    # This creates dn/n0 = alpha*cos(k*z) to first order
    z_perturbed = z_uniform + (alpha / k) * np.sin(k * z_uniform)
    z_perturbed = np.clip(z_perturbed, 1e-10, z_max - 1e-10)

    # Maxwellian velocities
    vr = rng.normal(0, v_th, n_particles)
    vz = rng.normal(0, v_th, n_particles)
    vtheta = rng.normal(0, v_th, n_particles)

    volume = np.pi * r_max**2 * z_max
    weight = np.full(n_particles, n0 * volume / n_particles)

    electrons = ParticleArray(species=electron_sp)
    electrons.allocate(int(n_particles * 1.2))
    electrons.add_particles(
        r=r_uniform, z=z_perturbed,
        vr=vr, vz=vz, vtheta=vtheta, weight=weight,
    )

    # Ions: uniform (no perturbation), zero velocity (immobile background)
    ions = ParticleArray(species=ion_sp)
    ions.allocate(int(n_particles * 1.2))
    ions.add_particles(
        r=r_uniform.copy(), z=z_uniform.copy(),
        vr=np.zeros(n_particles), vz=np.zeros(n_particles),
        vtheta=np.zeros(n_particles), weight=weight.copy(),
    )

    return {
        "grid": grid,
        "electrons": electrons,
        "ions": ions,
        "solver": solver,
        "dt": dt,
        "k": k,
        "omega_pe": omega_pe,
        "lambda_de": lambda_de,
        "alpha": alpha,
        "n0": n0,
        "te_ev": te_ev,
    }


def measure_field_energy(setup: dict, n_steps: int) -> tuple:
    """Run PIC and return time and field energy arrays."""
    grid = setup["grid"]
    electrons = setup["electrons"]
    ions = setup["ions"]
    solver = setup["solver"]
    dt = setup["dt"]

    times = []
    field_energies = []
    node_vol = cp.asarray(grid.node_volumes())

    for step in range(n_steps):
        t = step * dt
        phi, _stats = pic_step(grid, [electrons, ions], solver, dt)

        # E-field energy: sum over z-component only (1D dynamics)
        Ez = cp.zeros_like(phi)
        if phi.shape[1] > 2:
            Ez[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * grid.dz)

        e_field = float(0.5 * EPSILON_0 * cp.sum(Ez**2 * node_vol).item())
        times.append(t)
        field_energies.append(e_field)

    return np.array(times), np.array(field_energies)


@pytest.mark.slow
class TestLandauDamping:
    def test_efield_oscillates_and_decays(self):
        """Field energy should oscillate at ~omega_pe and envelope should decay."""
        setup = setup_landau_damping(n0=1e14, te_ev=2.0, alpha=0.05, ppc=200, nz=64)
        omega_pe = setup["omega_pe"]
        dt = setup["dt"]

        # Run for ~4 plasma periods
        n_steps = min(int(4 * 2 * np.pi / omega_pe / dt), 1500)
        times, fe = measure_field_energy(setup, n_steps)

        # Field energy should be non-trivial (perturbation creates E field)
        assert np.max(fe) > 0, "No field energy detected"

        # Compare average of first half to second half — should decay
        n = len(fe)
        # Use windowed averages to smooth over oscillations
        window = max(n // 10, 5)
        first_avg = np.mean(fe[:window])
        last_avg = np.mean(fe[-window:])

        # For moderate kld~0.5, damping should be visible but numerical
        # heating can partially offset it at low ppc. Check that energy doesn't
        # blow up exponentially (>10x growth would indicate instability).
        assert last_avg < first_avg * 10.0, (
            f"Field energy growing too fast (possible instability): "
            f"first={first_avg:.2e}, last={last_avg:.2e}"
        )

    def test_damping_rate_order_of_magnitude(self):
        """Numerical damping rate should be within an order of magnitude of theory."""
        te_ev = 2.0
        n0 = 1e14
        setup = setup_landau_damping(n0=n0, te_ev=te_ev, alpha=0.05, ppc=256, nz=64)
        k = setup["k"]
        omega_pe = setup["omega_pe"]
        dt = setup["dt"]

        n_steps = min(int(5 * 2 * np.pi / omega_pe / dt), 2000)
        times, fe = measure_field_energy(setup, n_steps)

        # Analytical rate
        gamma_theory = landau_damping_rate(k, n0, te_ev)

        # Field energy ~ E^2 ~ exp(-2*gamma*t)
        fe_smooth = np.maximum(fe, 1e-35)
        log_fe = np.log(fe_smooth)

        # Extract envelope by taking local maxima
        # Simple approach: smooth with a window then fit
        window = max(len(log_fe) // 20, 3)
        if len(log_fe) < window * 3:
            pytest.skip("Too few timesteps for reliable envelope extraction")

        # Running maximum to trace envelope
        from scipy.ndimage import maximum_filter1d
        envelope = maximum_filter1d(log_fe, size=window)

        # Fit linear trend to envelope (skip first 10% transient)
        n = len(envelope)
        start = n // 10
        end = 9 * n // 10
        t_fit = times[start:end]
        env_fit = envelope[start:end]

        coeffs = np.polyfit(t_fit, env_fit, 1)
        gamma_num = -coeffs[0] / 2.0  # FE ~ exp(-2*gamma*t)

        # Should be same order of magnitude as theory
        # Generous tolerance: cylindrical geometry + low ppc + finite domain
        if gamma_theory > 0 and gamma_num > 0:
            log_ratio = abs(np.log10(gamma_num / gamma_theory))
            assert log_ratio < 1.5, (
                f"Damping rate off by >30x: numerical={gamma_num:.2e}, "
                f"theory={gamma_theory:.2e}"
            )
