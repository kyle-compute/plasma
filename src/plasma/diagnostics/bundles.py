"""Stable diagnostic bundles for downstream reporting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from plasma.contracts.cases import Provenance


class DiagnosticSeriesBundle(BaseModel):
    """One time-series with metadata for plotting and reporting."""

    values: list[float]
    unit: str | None = None
    provenance: Provenance | None = None
    description: str | None = None


class DistributionBundle(BaseModel):
    """One histogram or distribution for post-run analysis."""

    axis: list[float]
    values: list[float]
    axis_unit: str | None = None
    value_unit: str | None = None
    provenance: Provenance | None = None
    description: str | None = None


class DiagnosticsBundle(BaseModel):
    """Model-agnostic diagnostics payload."""

    model: Literal["global", "pic"]
    time_s: list[float]
    series: dict[str, DiagnosticSeriesBundle] = Field(default_factory=dict)
    summary: dict[str, float] = Field(default_factory=dict)
    distributions: dict[str, DistributionBundle] = Field(default_factory=dict)


def bundle_from_irm_state(state) -> DiagnosticsBundle:
    """Convert an IRMState into a stable diagnostics bundle."""

    series = {}
    for name, metric in state.diagnostics.series.items():
        series[name] = DiagnosticSeriesBundle(
            values=[float(value) for value in metric.values],
            unit=metric.unit,
            provenance=metric.provenance,
            description=metric.description,
        )

    return DiagnosticsBundle(
        model="global",
        time_s=[float(value) for value in state.time],
        series=series,
        summary=state.diagnostics.summary(),
    )
