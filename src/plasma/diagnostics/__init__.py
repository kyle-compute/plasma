"""Diagnostics: energy distributions, spatial profiles, data collectors."""

from plasma.diagnostics.bundles import (
    DiagnosticSeriesBundle,
    DiagnosticsBundle,
    DistributionBundle,
    bundle_from_irm_state,
)

try:
    from plasma.diagnostics.collectors import CollisionTracker, SubstrateCollector
except ModuleNotFoundError:  # pragma: no cover - depends on optional GPU stack
    CollisionTracker = None
    SubstrateCollector = None

try:
    from plasma.diagnostics.distributions import (
        compute_eedf,
        compute_iedf,
        compute_velocity_histogram,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on optional GPU stack
    compute_eedf = None
    compute_iedf = None
    compute_velocity_histogram = None

try:
    from plasma.diagnostics.spatial import (
        electron_density_profile,
        electron_temperature_profile,
        potential_snapshot,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on optional GPU stack
    electron_density_profile = None
    electron_temperature_profile = None
    potential_snapshot = None

__all__ = [
    "CollisionTracker",
    "DiagnosticSeriesBundle",
    "DiagnosticsBundle",
    "DistributionBundle",
    "SubstrateCollector",
    "bundle_from_irm_state",
    "compute_eedf",
    "compute_iedf",
    "compute_velocity_histogram",
    "electron_density_profile",
    "electron_temperature_profile",
    "potential_snapshot",
]
