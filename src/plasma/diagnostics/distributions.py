"""Energy and velocity distribution functions from PIC particle data."""

from __future__ import annotations

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from plasma.core.constants import E_CHARGE


def compute_iedf(
    particles,
    z_plane: float,
    dz_capture: float,
    n_bins: int = 100,
    e_max_ev: float = 500.0,
) -> tuple[NDArray, NDArray]:
    """Ion Energy Distribution Function at a given z-plane.

    Selects alive ions within z_plane +/- dz_capture and histograms their
    kinetic energy weighted by macro-particle weight.

    Returns:
        (energy_ev, counts): Bin centers [eV] and weighted counts.
    """
    n = particles.count
    if n == 0:
        edges = np.linspace(0, e_max_ev, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, np.zeros(n_bins)

    z = cp.asnumpy(particles.z[:n])
    alive = cp.asnumpy(particles.alive[:n])

    mask = (alive == 1) & (np.abs(z - z_plane) < dz_capture)
    if not np.any(mask):
        edges = np.linspace(0, e_max_ev, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, np.zeros(n_bins)

    vr = cp.asnumpy(particles.vr[:n][mask])
    vz = cp.asnumpy(particles.vz[:n][mask])
    vt = cp.asnumpy(particles.vtheta[:n][mask])
    w = cp.asnumpy(particles.weight[:n][mask])

    v2 = vr**2 + vz**2 + vt**2
    ke_ev = 0.5 * particles.species.mass * v2 / E_CHARGE

    edges = np.linspace(0, e_max_ev, n_bins + 1)
    counts, _ = np.histogram(ke_ev, bins=edges, weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def compute_eedf(
    particles,
    n_bins: int = 100,
    e_max_ev: float = 50.0,
) -> tuple[NDArray, NDArray]:
    """Electron Energy Distribution Function (volume-averaged).

    Computes f(E) = E^{-1/2} * dN/dE, the standard EEDF normalization
    where integral f(E) dE = n_e.

    Returns:
        (energy_ev, f_e): Bin centers [eV] and EEDF values.
    """
    n = particles.count
    edges = np.linspace(0, e_max_ev, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if n == 0:
        return centers, np.zeros(n_bins)

    alive = cp.asnumpy(particles.alive[:n])
    mask = alive == 1
    if not np.any(mask):
        return centers, np.zeros(n_bins)

    vr = cp.asnumpy(particles.vr[:n][mask])
    vz = cp.asnumpy(particles.vz[:n][mask])
    vt = cp.asnumpy(particles.vtheta[:n][mask])
    w = cp.asnumpy(particles.weight[:n][mask])

    v2 = vr**2 + vz**2 + vt**2
    ke_ev = 0.5 * particles.species.mass * v2 / E_CHARGE

    dn_de, _ = np.histogram(ke_ev, bins=edges, weights=w)
    de = edges[1] - edges[0]
    dn_de = dn_de / de

    # f(E) = E^{-1/2} * dN/dE
    sqrt_e = np.sqrt(np.maximum(centers, 1e-30))
    f_e = dn_de / sqrt_e
    return centers, f_e


def compute_velocity_histogram(
    particles,
    component: str = "vz",
    n_bins: int = 100,
) -> tuple[NDArray, NDArray]:
    """Velocity histogram for a single component.

    Args:
        component: One of "vr", "vz", "vtheta".

    Returns:
        (v_bins, counts): Bin centers [m/s] and weighted counts.
    """
    n = particles.count
    if n == 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    alive = cp.asnumpy(particles.alive[:n])
    mask = alive == 1
    v_arr = cp.asnumpy(getattr(particles, component)[:n][mask])
    w = cp.asnumpy(particles.weight[:n][mask])

    if len(v_arr) == 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    counts, edges = np.histogram(v_arr, bins=n_bins, weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts
