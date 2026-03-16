"""Pure reporting helpers for PIC runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma.diagnostics.bundles import DiagnosticSeriesBundle, DiagnosticsBundle, DistributionBundle


def bundle_from_pic_run(
    diag: Any,
    waveform,
    collector: Any | None = None,
    *,
    collision_provenance: str = "synthetic",
) -> DiagnosticsBundle:
    """Convert PIC loop diagnostics into a stable reporting bundle."""

    time_s = [float(value) for value in diag.time]
    series: dict[str, DiagnosticSeriesBundle] = {
        "target_voltage_v": _series(
            [float(waveform.V(t)) for t in diag.time],
            "V",
            waveform.provenance,
            "Applied target voltage waveform sampled on diagnostic steps.",
        ),
        "field_energy_j": _series(diag.field_energy, "J", "heuristic", "Electrostatic field energy."),
        "total_energy_j": _series(diag.total_energy, "J", "heuristic", "Total recorded PIC energy."),
        "n_target_impacts_step": _series(
            diag.n_target_impacts,
            "count",
            "heuristic",
            "Target impacts recorded at each diagnostics step.",
        ),
        "n_sputtered_step": _series(
            diag.n_sputtered_total,
            "count",
            "heuristic",
            "Sputtered neutral injections recorded at each diagnostics step.",
        ),
        "n_see_step": _series(
            diag.n_see_total,
            "count",
            "heuristic",
            "Secondary electrons recorded at each diagnostics step.",
        ),
        "collisions_per_sample": _series(
            [_collision_sum(step_counts) for step_counts in diag.collision_counts],
            "count",
            collision_provenance,
            "Collision events recorded at each diagnostics step.",
        ),
    }

    for name, counts in diag.n_particles.items():
        series[f"{name}_particles"] = _series(
            counts,
            "count",
            "heuristic",
            f"Alive macro-particles for {name}.",
        )
    for name, values in diag.kinetic_energy.items():
        series[f"{name}_kinetic_energy_j"] = _series(
            values,
            "J",
            "heuristic",
            f"Kinetic energy carried by {name}.",
        )

    summary = {
        "peak_target_voltage_v": float(np.max(np.abs(series["target_voltage_v"].values))),
        "peak_field_energy_j": float(np.max(series["field_energy_j"].values)),
        "peak_total_energy_j": float(np.max(series["total_energy_j"].values)),
        "total_collisions": float(np.sum(series["collisions_per_sample"].values)),
        "total_target_impacts": float(np.sum(series["n_target_impacts_step"].values)),
        "total_sputtered_atoms": float(np.sum(series["n_sputtered_step"].values)),
        "total_secondary_electrons": float(np.sum(series["n_see_step"].values)),
    }
    for name, counts in diag.n_particles.items():
        summary[f"final_{name}_particles"] = float(counts[-1]) if counts else 0.0

    distributions: dict[str, DistributionBundle] = {}
    if collector is not None and getattr(collector, "total_count", 0) > 0:
        energy_ev, counts = collector.iedf()
        distributions["substrate_iedf"] = DistributionBundle(
            axis=[float(value) for value in energy_ev],
            values=[float(value) for value in counts],
            axis_unit="eV",
            value_unit="count",
            provenance="heuristic",
            description="Aggregated ion energy distribution captured near the substrate plane.",
        )
        summary["substrate_ion_samples"] = float(collector.total_count)

    return DiagnosticsBundle(
        model="pic",
        time_s=time_s,
        series=series,
        summary=summary,
        distributions=distributions,
    )


def _series(values: list[float] | Any, unit: str, provenance: str, description: str) -> DiagnosticSeriesBundle:
    return DiagnosticSeriesBundle(
        values=[float(value) for value in values],
        unit=unit,
        provenance=provenance,
        description=description,
    )


def _collision_sum(step_counts: dict[str, int]) -> float:
    return float(sum(step_counts.values()))
