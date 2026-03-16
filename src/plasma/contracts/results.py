"""Typed result contracts for saved run outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from plasma.contracts.cases import InputSource, Provenance


class MetricSummary(BaseModel):
    """Scalar metric emitted in a run manifest or validation report."""

    value: float
    unit: str | None = None
    provenance: Provenance | None = None
    description: str | None = None


class ValidationResult(BaseModel):
    """Pass/fail evaluation for one benchmark observable."""

    name: str
    metric: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    passed: bool = False
    citation: str | None = None
    notes: str | None = None


class ValidationReport(BaseModel):
    """Validation status for a run against configured targets."""

    case_name: str
    status: Literal["validated", "exploratory", "failed", "not_validated"]
    results: list[ValidationResult] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Serializable manifest for a single simulation run."""

    case_name: str
    model: Literal["global", "pic"]
    config_path: str
    input_sources: list[InputSource] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    validation_status: Literal["validated", "exploratory", "failed", "not_validated"]
