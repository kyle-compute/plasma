"""Reduced field builders for live PIC viewing."""

from __future__ import annotations

import cupy as cp
import numpy as np

from plasma.live.contracts import LiveField2D
from plasma.live.pic_window import EventWindow
from plasma.pic.deposit import deposit_charge, deposit_number_density


def field_bundle(
    values: np.ndarray,
    *,
    x,
    y,
    unit: str,
    label: str,
    max_side: int = 128,
) -> LiveField2D:
    """Convert a dense 2D field into a decimated live payload."""

    field = np.asarray(values, dtype=np.float64)
    y_axis = np.asarray(y, dtype=np.float64)
    x_axis = np.asarray(x, dtype=np.float64)

    y_stride = max(int(np.ceil(field.shape[0] / max_side)), 1)
    x_stride = max(int(np.ceil(field.shape[1] / max_side)), 1)
    field = field[::y_stride, ::x_stride]
    y_axis = y_axis[::y_stride]
    x_axis = x_axis[::x_stride]

    return LiveField2D(
        x=[float(value) for value in x_axis],
        y=[float(value) for value in y_axis],
        values=[[float(value) for value in row] for row in field],
        unit=unit,
        label=label,
    )


def number_density_view(grid, particles) -> np.ndarray:
    """Best-effort number density for live viewing."""

    try:
        return cp.asnumpy(deposit_number_density(grid, particles))
    except Exception:
        return number_density_cpu(grid, particles.to_numpy())


def charge_density_view(grid, particles_list: list) -> np.ndarray:
    """Best-effort charge density for live viewing."""

    try:
        return cp.asnumpy(deposit_charge(grid, particles_list))
    except Exception:
        rho = np.zeros((grid.nr + 1, grid.nz + 1), dtype=np.float64)
        for particles in particles_list:
            data = particles.to_numpy()
            charge = float(particles.species.charge)
            rho += density_cpu(grid, data["r"], data["z"], data["weight"] * charge)
        return rho


def electric_field_components(grid, phi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute electric-field components and magnitude from potential."""

    phi_np = np.asarray(cp.asnumpy(phi) if isinstance(phi, cp.ndarray) else phi, dtype=np.float64)
    er = np.zeros_like(phi_np)
    ez = np.zeros_like(phi_np)

    er[1:-1, :] = -(phi_np[2:, :] - phi_np[:-2, :]) / (2.0 * grid.dr)
    er[0, :] = -(phi_np[1, :] - phi_np[0, :]) / grid.dr
    er[-1, :] = -(phi_np[-1, :] - phi_np[-2, :]) / grid.dr

    ez[:, 1:-1] = -(phi_np[:, 2:] - phi_np[:, :-2]) / (2.0 * grid.dz)
    ez[:, 0] = -(phi_np[:, 1] - phi_np[:, 0]) / grid.dz
    ez[:, -1] = -(phi_np[:, -1] - phi_np[:, -2]) / grid.dz
    return er, ez, np.hypot(er, ez)


def magnetic_field_magnitude(br_grid, bz_grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return magnetic field components and magnitude."""

    br = np.asarray(cp.asnumpy(br_grid) if isinstance(br_grid, cp.ndarray) else br_grid, dtype=np.float64)
    bz = np.asarray(cp.asnumpy(bz_grid) if isinstance(bz_grid, cp.ndarray) else bz_grid, dtype=np.float64)
    return br, bz, np.hypot(br, bz)


def event_map(grid, window: EventWindow | None, names: tuple[str, ...]) -> np.ndarray:
    """Deposit one or more live event clouds onto the node grid."""

    if not window:
        return np.zeros((grid.nr + 1, grid.nz + 1), dtype=np.float64)

    field = np.zeros((grid.nr + 1, grid.nz + 1), dtype=np.float64)
    for name in names:
        cloud = window.get(name)
        if cloud is None:
            continue
        field += density_cpu(
            grid,
            np.asarray(cloud.get("r", np.empty(0)), dtype=np.float64),
            np.asarray(cloud.get("z", np.empty(0)), dtype=np.float64),
            np.ones_like(np.asarray(cloud.get("r", np.empty(0)), dtype=np.float64)),
        )
    return field


def emissivity_proxy(
    electron_density: np.ndarray | None,
    e_mag: np.ndarray | None,
    excitation_map: np.ndarray,
    ionization_map: np.ndarray,
    see_map: np.ndarray,
) -> np.ndarray:
    """Build a physics-first glow layer for live viewing."""

    if np.any(excitation_map) or np.any(ionization_map) or np.any(see_map):
        return normalize_field(excitation_map + 1.5 * ionization_map + 0.75 * see_map)
    if electron_density is None or e_mag is None:
        shape = excitation_map.shape if excitation_map.size else (0, 0)
        return np.zeros(shape, dtype=np.float64)
    return normalize_field(np.maximum(electron_density, 0.0) * np.maximum(e_mag, 0.0))


def normalize_field(values: np.ndarray) -> np.ndarray:
    """Normalize a positive field into a stable 0..1 display range."""

    field = np.asarray(values, dtype=np.float64)
    if field.size == 0:
        return field
    max_value = float(np.max(field))
    if max_value <= 0.0:
        return np.zeros_like(field)
    return np.clip(field / max_value, 0.0, 1.0)


def substrate_flux_proxy(grid, substrate, *, band_nodes: int = 4) -> np.ndarray:
    """Project the substrate collector into a thin axial band near the substrate."""

    field = np.zeros((grid.nr + 1, grid.nz + 1), dtype=np.float64)
    if substrate is None or getattr(substrate, "total_count", 0) <= 0:
        return field

    profile = np.asarray(substrate.radial_flux_profile(grid.r_edges), dtype=np.float64)
    if profile.size == 0 or not np.any(profile > 0.0):
        return field

    radial_nodes = np.zeros(grid.nr + 1, dtype=np.float64)
    radial_nodes[:-1] = profile
    radial_nodes[-1] = radial_nodes[-2]
    field[:, -band_nodes:] = radial_nodes[:, None]
    return normalize_field(field)


def density_cpu(grid, r: np.ndarray, z: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """CPU fallback for depositing weighted node densities."""

    field = np.zeros((grid.nr + 1, grid.nz + 1), dtype=np.float64)
    for r_p, z_p, w_p in zip(r, z, weight, strict=False):
        ri = r_p / grid.dr
        zi = z_p / grid.dz
        i = min(max(int(ri), 0), grid.nr - 1)
        j = min(max(int(zi), 0), grid.nz - 1)
        wr = ri - i
        wz = zi - j
        field[i, j] += w_p * (1.0 - wr) * (1.0 - wz)
        field[i + 1, j] += w_p * wr * (1.0 - wz)
        field[i, j + 1] += w_p * (1.0 - wr) * wz
        field[i + 1, j + 1] += w_p * wr * wz

    node_vol = grid.node_volumes()
    return field / np.maximum(node_vol, 1e-30)


def number_density_cpu(grid, particle_data: dict[str, np.ndarray]) -> np.ndarray:
    """CPU fallback for number density."""

    return density_cpu(grid, particle_data["r"], particle_data["z"], particle_data["weight"])
