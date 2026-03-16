"""Spatial profiles: density, temperature, potential snapshots."""

from __future__ import annotations

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from plasma.core.constants import E_CHARGE
from plasma.pic.deposit import deposit_charge_kernel, deposit_number_density


def electron_density_profile(grid, electrons) -> NDArray:
    """Electron number density n_e(r,z) on grid nodes [m^-3].

    Reuses the CIC deposition kernel from deposit.py.
    """
    n_e = deposit_number_density(grid, electrons)
    return cp.asnumpy(n_e)


def electron_temperature_profile(grid, electrons) -> NDArray:
    """Electron temperature T_e(r,z) on grid nodes [eV].

    Deposits sum(w * v^2) and sum(w) separately, then computes
    T_e = (m_e / 3) * <v^2> / e  (equipartition: 3 DOF).

    Returns zero where no electrons are present.
    """
    nr, nz = grid.nr, grid.nz
    n = electrons.count
    if n == 0:
        return np.zeros((nr + 1, nz + 1))

    # Compute v^2 per particle on GPU
    v2 = (
        electrons.vr[:n] ** 2
        + electrons.vz[:n] ** 2
        + electrons.vtheta[:n] ** 2
    )

    # Deposit w * v^2
    wv2 = electrons.weight[:n] * v2
    raw_wv2 = cp.zeros((nr + 1, nz + 1), dtype=cp.float64)
    threads = 256
    blocks = (n + threads - 1) // threads
    deposit_charge_kernel[blocks, threads](
        raw_wv2,
        electrons.r, electrons.z,
        wv2, electrons.alive,
        1.0,
        grid.dr, grid.dz, nr, nz, n,
    )

    # Deposit weights
    raw_w = cp.zeros((nr + 1, nz + 1), dtype=cp.float64)
    deposit_charge_kernel[blocks, threads](
        raw_w,
        electrons.r, electrons.z,
        electrons.weight, electrons.alive,
        1.0,
        grid.dr, grid.dz, nr, nz, n,
    )

    # T_e = (m/3) * <v^2> / e = (m/3) * (sum(w*v2) / sum(w)) / e
    w_safe = cp.maximum(raw_w, 1e-30)
    mean_v2 = raw_wv2 / w_safe
    te_ev = electrons.species.mass * mean_v2 / (3.0 * E_CHARGE)

    # Zero out where no electrons
    te_ev[raw_w < 1e-30] = 0.0

    return cp.asnumpy(te_ev)


def potential_snapshot(phi: cp.ndarray) -> NDArray:
    """Copy potential from GPU to CPU numpy array."""
    return cp.asnumpy(phi)
