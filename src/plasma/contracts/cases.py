"""Typed case metadata and validation contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Provenance = Literal[
    "measured",
    "literature-fit",
    "public-download",
    "surrogate",
    "heuristic",
    "synthetic",
]


class InputSource(BaseModel):
    """Named input dataset with provenance information."""

    name: str
    kind: Literal[
        "waveform",
        "cross_sections",
        "magnetic_field",
        "reaction_set",
        "yield_model",
        "other",
    ] = "other"
    path: str | None = None
    provenance: Provenance = "synthetic"
    citation: str | None = None
    notes: str | None = None


class CaseMetadata(BaseModel):
    """Describes the benchmark or experiment a config is meant to represent."""

    benchmark: str | None = None
    material_system: str | None = None
    inputs: list[InputSource] = Field(default_factory=list)
    notes: str | None = None


class ValidationTarget(BaseModel):
    """Expected range for a post-run summary metric."""

    name: str
    metric: str
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    citation: str | None = None


class ValidationConfig(BaseModel):
    """Collection of validation targets for a case."""

    targets: list[ValidationTarget] = Field(default_factory=list)
