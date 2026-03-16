"""Data collectors: accumulate per-step data for post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from plasma.core.constants import E_CHARGE


@dataclass
class SubstrateCollector:
    """Accumulates ion data at a substrate z-plane across multiple steps.

    Records radial positions, energies, and weights of ions arriving at the
    substrate for post-run IEDF analysis.
    """

    z_plane: float
    dz_capture: float = 1e-4

    _r: list[NDArray] = field(default_factory=list)
    _energy_ev: list[NDArray] = field(default_factory=list)
    _weight: list[NDArray] = field(default_factory=list)
    _times: list[float] = field(default_factory=list)
    _counts: list[int] = field(default_factory=list)
    _step_mean_energy_ev: list[float] = field(default_factory=list)

    def record_absorbed(self, particles, t: float = 0.0) -> int:
        """Record ions near the substrate plane.

        Returns count of ions captured this step.
        """
        n = particles.count
        if n == 0:
            self._counts.append(0)
            self._times.append(t)
            self._step_mean_energy_ev.append(0.0)
            return 0

        z = cp.asnumpy(particles.z[:n])
        alive = cp.asnumpy(particles.alive[:n])
        mask = (alive == 1) & (np.abs(z - self.z_plane) < self.dz_capture)
        count = int(np.sum(mask))

        if count > 0:
            r = cp.asnumpy(particles.r[:n][mask])
            vr = cp.asnumpy(particles.vr[:n][mask])
            vz = cp.asnumpy(particles.vz[:n][mask])
            vt = cp.asnumpy(particles.vtheta[:n][mask])
            w = cp.asnumpy(particles.weight[:n][mask])

            v2 = vr**2 + vz**2 + vt**2
            ke_ev = 0.5 * particles.species.mass * v2 / E_CHARGE

            self._r.append(r)
            self._energy_ev.append(ke_ev)
            self._weight.append(w)
            total_weight = float(np.sum(w))
            mean_energy_ev = float(np.average(ke_ev, weights=w)) if total_weight > 0.0 else float(np.mean(ke_ev))
        else:
            mean_energy_ev = 0.0

        self._counts.append(count)
        self._times.append(t)
        self._step_mean_energy_ev.append(mean_energy_ev)
        return count

    def iedf(self, n_bins: int = 100, e_max_ev: float = 500.0) -> tuple[NDArray, NDArray]:
        """Aggregated IEDF from all recorded steps."""
        edges = np.linspace(0, e_max_ev, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        if not self._energy_ev:
            return centers, np.zeros(n_bins)

        all_e = np.concatenate(self._energy_ev)
        all_w = np.concatenate(self._weight)
        counts, _ = np.histogram(all_e, bins=edges, weights=all_w)
        return centers, counts

    @property
    def total_count(self) -> int:
        return sum(self._counts)

    def latest_count(self) -> int:
        """Latest recorded arrival count."""

        return self._counts[-1] if self._counts else 0

    def latest_mean_energy_ev(self) -> float:
        """Mean arrival energy from the most recently recorded step."""

        return self._step_mean_energy_ev[-1] if self._step_mean_energy_ev else 0.0

    def mean_energy_ev(self) -> float:
        """Overall mean arrival energy across all recorded steps."""

        if not self._energy_ev:
            return 0.0
        all_e = np.concatenate(self._energy_ev)
        all_w = np.concatenate(self._weight)
        total_weight = float(np.sum(all_w))
        if total_weight <= 0.0:
            return float(np.mean(all_e))
        return float(np.average(all_e, weights=all_w))

    def radial_flux_profile(self, r_edges: NDArray[np.float64]) -> NDArray[np.float64]:
        """Weighted radial arrival profile across the substrate."""

        if not self._r:
            return np.zeros(len(r_edges) - 1, dtype=np.float64)
        all_r = np.concatenate(self._r)
        all_w = np.concatenate(self._weight)
        counts, _ = np.histogram(all_r, bins=r_edges, weights=all_w)
        return counts.astype(np.float64)


@dataclass
class CollisionTracker:
    """Accumulates collision count dicts from MCC per step."""

    _history: list[dict[str, int]] = field(default_factory=list)

    def record(self, counts: dict[str, int]) -> None:
        self._history.append(counts.copy())

    def totals(self) -> dict[str, int]:
        """Sum all collision counts across all recorded steps."""
        result: dict[str, int] = {}
        for step_counts in self._history:
            for k, v in step_counts.items():
                result[k] = result.get(k, 0) + v
        return result

    @property
    def n_steps(self) -> int:
        return len(self._history)
