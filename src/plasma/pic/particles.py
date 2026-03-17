"""GPU-resident particle data using Structure-of-Arrays (SoA) layout.

Each species has separate arrays for position (r, z), velocity (vr, vz, vtheta),
and macro-particle weight. All arrays live on GPU via CuPy for zero-copy
access from CUDA kernels.

In axisymmetric (r, z) geometry, particles have 3 velocity components (vr, vz,
vtheta) but only 2 position components (r, z). The azimuthal velocity vtheta is
needed for the v x B force and contributes to kinetic energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Type alias for GPU arrays
from plasma.runtime.cupy_compat import cp

GpuArray = cp.ndarray


@dataclass
class Species:
    """Particle species definition."""

    name: str
    charge: float       # [C] (signed: negative for electrons)
    mass: float         # [kg]
    charge_state: int   # Integer charge state (1 for Ar+, 2 for Ar2+, etc.)

    @property
    def qm_ratio(self) -> float:
        """Charge-to-mass ratio [C/kg]."""
        return self.charge / self.mass


@dataclass
class ParticleArray:
    """SoA particle storage on GPU.

    All position/velocity arrays have shape (capacity,) and only the first
    `count` entries are valid. This avoids frequent reallocations — we
    over-allocate and compact periodically.

    Attributes:
        species: Particle species definition.
        capacity: Total allocated slots.
        count: Number of active particles.
        r, z: Radial and axial positions [m].
        vr, vz, vtheta: Velocity components [m/s].
        weight: Macro-particle weight (number of physical particles per macro).
    """

    species: Species
    capacity: int = 0
    count: int = 0

    # GPU arrays — initialized by allocate()
    r: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.float64))
    z: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.float64))
    vr: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.float64))
    vz: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.float64))
    vtheta: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.float64))
    weight: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.float64))

    # Alive flags (1 = active, 0 = dead / absorbed)
    alive: GpuArray = field(default_factory=lambda: cp.empty(0, dtype=cp.int32))

    def allocate(self, capacity: int) -> None:
        """Allocate GPU memory for given capacity."""
        self.capacity = capacity
        self.count = 0
        self.r = cp.zeros(capacity, dtype=cp.float64)
        self.z = cp.zeros(capacity, dtype=cp.float64)
        self.vr = cp.zeros(capacity, dtype=cp.float64)
        self.vz = cp.zeros(capacity, dtype=cp.float64)
        self.vtheta = cp.zeros(capacity, dtype=cp.float64)
        self.weight = cp.zeros(capacity, dtype=cp.float64)
        self.alive = cp.zeros(capacity, dtype=cp.int32)

    def add_particles(
        self,
        r: NDArray | GpuArray,
        z: NDArray | GpuArray,
        vr: NDArray | GpuArray,
        vz: NDArray | GpuArray,
        vtheta: NDArray | GpuArray,
        weight: NDArray | GpuArray,
    ) -> int:
        """Add particles to the array. Returns number actually added.

        If capacity is exceeded, grows the array by 2x.
        """
        r = cp.asarray(r, dtype=cp.float64)
        n_new = int(r.shape[0])

        if n_new == 0:
            return 0

        # Grow if needed
        needed = self.count + n_new
        if needed > self.capacity:
            self._grow(max(needed, self.capacity * 2))

        s = self.count
        e = s + n_new
        self.r[s:e] = cp.asarray(r, dtype=cp.float64)
        self.z[s:e] = cp.asarray(z, dtype=cp.float64)
        self.vr[s:e] = cp.asarray(vr, dtype=cp.float64)
        self.vz[s:e] = cp.asarray(vz, dtype=cp.float64)
        self.vtheta[s:e] = cp.asarray(vtheta, dtype=cp.float64)
        self.weight[s:e] = cp.asarray(weight, dtype=cp.float64)
        self.alive[s:e] = 1

        self.count = e
        return n_new

    def _grow(self, new_capacity: int) -> None:
        """Grow all arrays to new_capacity, preserving existing data."""
        for attr in ("r", "z", "vr", "vz", "vtheta", "weight"):
            old = getattr(self, attr)
            new = cp.zeros(new_capacity, dtype=cp.float64)
            if self.count > 0:
                new[: self.count] = old[: self.count]
            setattr(self, attr, new)

        old_alive = self.alive
        new_alive = cp.zeros(new_capacity, dtype=cp.int32)
        if self.count > 0:
            new_alive[: self.count] = old_alive[: self.count]
        self.alive = new_alive
        self.capacity = new_capacity

    def compact(self) -> None:
        """Remove dead particles by compacting arrays.

        Uses stream compaction: gather alive particles to front of arrays.
        This should be called periodically (not every timestep) to reclaim
        memory from absorbed particles.
        """
        if self.count == 0:
            return

        mask = self.alive[: self.count] == 1
        n_alive = int(cp.sum(mask).item())

        if n_alive == self.count:
            return  # Nothing to compact

        for attr in ("r", "z", "vr", "vz", "vtheta", "weight"):
            arr = getattr(self, attr)
            arr[:n_alive] = arr[: self.count][mask]

        self.alive[:n_alive] = 1
        self.alive[n_alive : self.count] = 0
        self.count = n_alive

    def kill(self, indices: GpuArray) -> None:
        """Mark particles at given indices as dead."""
        if indices.size > 0:
            self.alive[indices] = 0

    def kinetic_energy(self) -> float:
        """Total kinetic energy of all alive particles [J]."""
        if self.count == 0:
            return 0.0
        mask = self.alive[: self.count] == 1
        v2 = (
            self.vr[: self.count][mask] ** 2
            + self.vz[: self.count][mask] ** 2
            + self.vtheta[: self.count][mask] ** 2
        )
        w = self.weight[: self.count][mask]
        return float(0.5 * self.species.mass * cp.sum(w * v2).item())

    def mean_energy_ev(self) -> float:
        """Mean kinetic energy per physical particle [eV]."""
        if self.count == 0:
            return 0.0
        from plasma.core.constants import E_CHARGE

        mask = self.alive[: self.count] == 1
        v2 = (
            self.vr[: self.count][mask] ** 2
            + self.vz[: self.count][mask] ** 2
            + self.vtheta[: self.count][mask] ** 2
        )
        w = self.weight[: self.count][mask]
        total_w = cp.sum(w)
        if total_w == 0:
            return 0.0
        mean_v2 = float(cp.sum(w * v2).item() / total_w.item())
        return 0.5 * self.species.mass * mean_v2 / E_CHARGE

    @property
    def n_alive(self) -> int:
        """Number of alive particles."""
        if self.count == 0:
            return 0
        return int(cp.sum(self.alive[: self.count]).item())

    def to_numpy(self) -> dict[str, NDArray]:
        """Copy alive particle data to CPU numpy arrays."""
        if self.count == 0:
            return {k: np.empty(0) for k in ("r", "z", "vr", "vz", "vtheta", "weight")}

        mask = self.alive[: self.count] == 1
        return {
            "r": cp.asnumpy(self.r[: self.count][mask]),
            "z": cp.asnumpy(self.z[: self.count][mask]),
            "vr": cp.asnumpy(self.vr[: self.count][mask]),
            "vz": cp.asnumpy(self.vz[: self.count][mask]),
            "vtheta": cp.asnumpy(self.vtheta[: self.count][mask]),
            "weight": cp.asnumpy(self.weight[: self.count][mask]),
        }

    def memory_usage_bytes(self) -> int:
        """Estimate GPU memory used by this particle array."""
        # 6 float64 arrays + 1 int32 array
        return self.capacity * (6 * 8 + 4)
