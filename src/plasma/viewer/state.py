"""Pure helpers for live viewer state and geometry transforms."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from plasma.live.contracts import LiveParticleCloud, LiveSnapshot


def field_matrix(snapshot: LiveSnapshot, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return one live field as x, y, value arrays."""

    field = snapshot.fields.get(name)
    if field is None:
        return None
    x = np.asarray(field.x, dtype=np.float64)
    y = np.asarray(field.y, dtype=np.float64)
    values = np.asarray(field.values, dtype=np.float64)
    if x.size == 0 or y.size == 0 or values.size == 0:
        return None
    return x, y, values


def emissive_points(snapshot: LiveSnapshot, field_name: str = "emissivity_arb", n_theta: int = 36, threshold: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Revolve an axisymmetric field into a pseudo-3D point cloud."""

    field = field_matrix(snapshot, field_name)
    if field is None:
        return np.empty((0, 3)), np.empty(0)
    x, y, values = field
    mask = values > threshold
    if not np.any(mask):
        return np.empty((0, 3)), np.empty(0)

    r_idx, z_idx = np.where(mask)
    radii = y[r_idx]
    axial = x[z_idx]
    intensity = values[mask]
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phase = np.mod(0.71 * r_idx + 1.37 * z_idx, 2.0 * np.pi)

    radii = np.repeat(radii, n_theta)
    axial = np.repeat(axial, n_theta)
    intensity = np.repeat(intensity, n_theta)
    angles = np.tile(theta, len(r_idx)) + np.repeat(phase, n_theta)
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), axial))
    return points, intensity


def particle_ring_points(cloud: LiveParticleCloud, theta_offset: float = 0.0) -> np.ndarray:
    """Spread axisymmetric particles around the cylinder for the pseudo-3D view."""

    if not cloud.r or not cloud.z:
        return np.empty((0, 3))
    radii = np.asarray(cloud.r, dtype=np.float64)
    axial = np.asarray(cloud.z, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = np.mod(theta_offset + np.arange(len(radii), dtype=np.float64) * golden_angle, 2.0 * np.pi)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles), axial))


@dataclass
class TrailBuffer:
    """Short local history for drawing particle streaks."""

    length: int = 12
    trails: dict[str, deque[np.ndarray]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=12)))

    def set_length(self, length: int) -> None:
        """Resize all trail buffers."""

        self.length = length
        self.trails = {
            name: deque(points, maxlen=length)
            for name, points in self.trails.items()
        }

    def push(self, snapshot: LiveSnapshot) -> None:
        """Append the current particle positions for each species."""

        for name, cloud in snapshot.particles.items():
            points = np.column_stack((np.asarray(cloud.z), np.asarray(cloud.r)))
            trail = self.trails.setdefault(name, deque(maxlen=self.length))
            trail.append(points)
