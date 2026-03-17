"""Pure reporting helpers for PIC runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma.diagnostics.bundles import DiagnosticsBundle, DiagnosticSeriesBundle, DistributionBundle


def bundle_from_pic_run(
    diag: Any,
    waveform,
    collector: Any | None = None,
    *,
    collision_provenance: str = "synthetic",
    collision_channel_provenance: dict[str, str] | None = None,
    background_state: dict[str, float] | None = None,
) -> DiagnosticsBundle:
    """Convert PIC loop diagnostics into a stable reporting bundle."""

    time_s = [float(value) for value in diag.time]
    overall_collision_provenance = _overall_collision_provenance(
        default=collision_provenance,
        channel_provenance=collision_channel_provenance,
    )
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
            overall_collision_provenance,
            "Collision events recorded at each diagnostics step.",
        ),
    }

    collision_names = sorted({name for step_counts in diag.collision_counts for name in step_counts})
    for name in collision_names:
        series[f"collision_{name}_step"] = _series(
            [float(step_counts.get(name, 0)) for step_counts in diag.collision_counts],
            "count",
            collision_provenance if collision_channel_provenance is None else collision_channel_provenance.get(name),
            f"Collision counts for channel {name} on each diagnostic sample.",
        )

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
    for name in collision_names:
        summary[f"total_collision_{name}"] = float(np.sum(series[f"collision_{name}_step"].values))
    if background_state is not None:
        for name, density in background_state.items():
            summary[f"background_{name}_density_m3"] = float(density)

    distributions: dict[str, DistributionBundle] = {}
    if collector is not None and getattr(collector, "total_count", 0) > 0:
        energy_ev, counts = collector.iedf()
        distributions["substrate_iedf"] = DistributionBundle(
            axis=[float(value) for value in energy_ev],
            values=[float(value) for value in counts],
            axis_unit="eV",
            value_unit="count",
            provenance="model-derived",
            description="Aggregated absorbed positive-ion energy distribution at the substrate boundary.",
        )
        summary["substrate_ion_samples"] = float(collector.total_count)
        if hasattr(collector, "species_totals"):
            for species_name, total in collector.species_totals().items():
                summary[f"substrate_{species_name}_samples"] = float(total)
                species_energy_ev, species_counts = collector.iedf(species_name=species_name)
                if np.any(species_counts > 0.0):
                    distributions[f"substrate_iedf_{species_name}"] = DistributionBundle(
                        axis=[float(value) for value in species_energy_ev],
                        values=[float(value) for value in species_counts],
                        axis_unit="eV",
                        value_unit="count",
                        provenance="model-derived",
                        description=f"Absorbed substrate ion-energy distribution for {species_name}.",
                    )

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


def _overall_collision_provenance(
    *,
    default: str,
    channel_provenance: dict[str, str] | None,
) -> str:
    if not channel_provenance:
        return default
    rank = {
        "synthetic": 0,
        "surrogate": 1,
        "heuristic": 2,
        "model-derived": 3,
        "literature-fit": 4,
        "public-download": 5,
        "measured": 6,
    }
    weakest = min(
        channel_provenance.values(),
        key=lambda provenance: rank.get(provenance, rank["model-derived"]),
    )
    return weakest
