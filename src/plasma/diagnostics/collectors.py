"""Data collectors: accumulate per-step data for post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plasma.core.constants import E_CHARGE
from plasma.runtime.cupy_compat import cp


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
    _step_weight_sum: list[float] = field(default_factory=list)
    _step_energy_weighted_sum: list[float] = field(default_factory=list)
    _counts_by_species: dict[str, int] = field(default_factory=dict)
    _energy_ev_by_species: dict[str, list[NDArray]] = field(default_factory=dict)
    _weight_by_species: dict[str, list[NDArray]] = field(default_factory=dict)
    _r_by_species: dict[str, list[NDArray]] = field(default_factory=dict)

    def record_absorbed(self, particles, t: float = 0.0) -> int:
        """Record ions near the substrate plane.

        Returns count of ions captured this step.
        """
        n = particles.count
        if n == 0:
            self._append_step_stats(t=t, count=0, weight_sum=0.0, energy_weighted_sum=0.0)
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
            weight_sum = float(np.sum(w))
            energy_weighted_sum = float(np.sum(ke_ev * w)) if weight_sum > 0.0 else float(np.sum(ke_ev))
            if weight_sum <= 0.0:
                weight_sum = float(count)
        else:
            weight_sum = 0.0
            energy_weighted_sum = 0.0
        self._append_step_stats(
            t=t,
            count=count,
            weight_sum=weight_sum,
            energy_weighted_sum=energy_weighted_sum,
        )
        return count

    def record_incident(self, particles, *, t: float = 0.0, species_name: str | None = None) -> int:
        """Record positive ions that have crossed into the substrate boundary.

        This is the research-facing path used by the PIC runtime. Unlike
        `record_absorbed`, which samples a capture band near the substrate plane,
        this method records particles whose positions have crossed the substrate
        boundary and are about to be absorbed by the boundary condition.
        """

        n = particles.count
        if n == 0:
            self._counts.append(0)
            self._times.append(t)
            self._step_mean_energy_ev.append(0.0)
            return 0

        z = cp.asnumpy(particles.z[:n])
        alive = cp.asnumpy(particles.alive[:n])
        mask = (alive == 1) & (z >= self.z_plane)
        return self._record_masked_particles(
            particles,
            mask=mask,
            t=t,
            species_name=species_name or particles.species.name,
        )

    def record_particle_data(
        self,
        *,
        r: NDArray[np.float64],
        vr: NDArray[np.float64],
        vz: NDArray[np.float64],
        vtheta: NDArray[np.float64],
        weight: NDArray[np.float64],
        mass_kg: float,
        t: float = 0.0,
        species_name: str | None = None,
    ) -> int:
        """Record a pre-filtered set of absorbed particle data."""

        r_arr = np.asarray(r, dtype=np.float64)
        if r_arr.size == 0:
            self._append_step_stats(t=t, count=0, weight_sum=0.0, energy_weighted_sum=0.0)
            return 0

        v2 = (
            np.asarray(vr, dtype=np.float64) ** 2
            + np.asarray(vz, dtype=np.float64) ** 2
            + np.asarray(vtheta, dtype=np.float64) ** 2
        )
        ke_ev = 0.5 * mass_kg * v2 / E_CHARGE
        w = np.asarray(weight, dtype=np.float64)
        self._append_capture(
            r=r_arr,
            energy_ev=ke_ev,
            weight=w,
            t=t,
            species_name=species_name,
        )
        return int(r_arr.size)

    def iedf(
        self,
        n_bins: int = 100,
        e_max_ev: float = 500.0,
        *,
        species_name: str | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Aggregated IEDF from all recorded steps."""
        edges = np.linspace(0, e_max_ev, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        energies, weights = self._aggregated_species_arrays(species_name=species_name)
        if energies.size == 0:
            return centers, np.zeros(n_bins)

        counts, _ = np.histogram(energies, bins=edges, weights=weights)
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

    def mean_energy_ev(self, *, species_name: str | None = None) -> float:
        """Overall mean arrival energy across all recorded steps."""

        all_e, all_w = self._aggregated_species_arrays(species_name=species_name)
        if all_e.size == 0:
            return 0.0
        total_weight = float(np.sum(all_w))
        if total_weight <= 0.0:
            return float(np.mean(all_e))
        return float(np.average(all_e, weights=all_w))

    def radial_flux_profile(
        self,
        r_edges: NDArray[np.float64],
        *,
        species_name: str | None = None,
    ) -> NDArray[np.float64]:
        """Weighted radial arrival profile across the substrate."""

        all_r, all_w = self._aggregated_radial_arrays(species_name=species_name)
        if all_r.size == 0:
            return np.zeros(len(r_edges) - 1, dtype=np.float64)
        counts, _ = np.histogram(all_r, bins=r_edges, weights=all_w)
        return counts.astype(np.float64)

    def species_totals(self) -> dict[str, int]:
        """Return total absorbed counts by ion species."""

        return {name: int(value) for name, value in self._counts_by_species.items()}

    def _record_masked_particles(
        self,
        particles,
        *,
        mask: NDArray[np.bool_],
        t: float,
        species_name: str | None,
    ) -> int:
        count = int(np.sum(mask))
        if count == 0:
            self._append_step_stats(t=t, count=0, weight_sum=0.0, energy_weighted_sum=0.0)
            return 0

        r = cp.asnumpy(particles.r[: particles.count][mask])
        vr = cp.asnumpy(particles.vr[: particles.count][mask])
        vz = cp.asnumpy(particles.vz[: particles.count][mask])
        vt = cp.asnumpy(particles.vtheta[: particles.count][mask])
        w = cp.asnumpy(particles.weight[: particles.count][mask])
        v2 = vr**2 + vz**2 + vt**2
        ke_ev = 0.5 * particles.species.mass * v2 / E_CHARGE
        self._append_capture(
            r=r,
            energy_ev=ke_ev,
            weight=w,
            t=t,
            species_name=species_name,
        )
        return count

    def _append_capture(
        self,
        *,
        r: NDArray[np.float64],
        energy_ev: NDArray[np.float64],
        weight: NDArray[np.float64],
        t: float,
        species_name: str | None,
    ) -> None:
        self._r.append(r)
        self._energy_ev.append(energy_ev)
        self._weight.append(weight)
        total_weight = float(np.sum(weight))
        energy_weighted_sum = (
            float(np.sum(energy_ev * weight))
            if total_weight > 0.0
            else float(np.sum(energy_ev))
        )
        self._append_step_stats(
            t=t,
            count=int(r.size),
            weight_sum=total_weight if total_weight > 0.0 else float(r.size),
            energy_weighted_sum=energy_weighted_sum,
        )

        if species_name is None:
            return
        self._counts_by_species[species_name] = self._counts_by_species.get(species_name, 0) + int(r.size)
        self._energy_ev_by_species.setdefault(species_name, []).append(energy_ev)
        self._weight_by_species.setdefault(species_name, []).append(weight)
        self._r_by_species.setdefault(species_name, []).append(r)

    def _aggregated_species_arrays(self, *, species_name: str | None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if species_name is None:
            if not self._energy_ev:
                return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
            return np.concatenate(self._energy_ev), np.concatenate(self._weight)

        energies = self._energy_ev_by_species.get(species_name, [])
        if not energies:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        return np.concatenate(energies), np.concatenate(self._weight_by_species[species_name])

    def _aggregated_radial_arrays(self, *, species_name: str | None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if species_name is None:
            if not self._r:
                return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
            return np.concatenate(self._r), np.concatenate(self._weight)

        radii = self._r_by_species.get(species_name, [])
        if not radii:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        return np.concatenate(radii), np.concatenate(self._weight_by_species[species_name])

    def _append_step_stats(
        self,
        *,
        t: float,
        count: int,
        weight_sum: float,
        energy_weighted_sum: float,
    ) -> None:
        if self._times and self._times[-1] == t:
            self._counts[-1] += int(count)
            self._step_weight_sum[-1] += float(weight_sum)
            self._step_energy_weighted_sum[-1] += float(energy_weighted_sum)
            if self._step_weight_sum[-1] > 0.0:
                self._step_mean_energy_ev[-1] = self._step_energy_weighted_sum[-1] / self._step_weight_sum[-1]
            else:
                self._step_mean_energy_ev[-1] = 0.0
            return

        self._times.append(t)
        self._counts.append(int(count))
        self._step_weight_sum.append(float(weight_sum))
        self._step_energy_weighted_sum.append(float(energy_weighted_sum))
        if weight_sum > 0.0:
            self._step_mean_energy_ev.append(float(energy_weighted_sum / weight_sum))
        else:
            self._step_mean_energy_ev.append(0.0)


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
