"""Typed contracts for live simulation viewing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LiveSeries(BaseModel):
    """One time-series shown in the live viewer."""

    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    x_unit: str | None = None
    y_unit: str | None = None
    label: str | None = None


class LiveField2D(BaseModel):
    """One 2D scalar field shown in the live viewer."""

    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    values: list[list[float]] = Field(default_factory=list)
    unit: str | None = None
    label: str | None = None


class LiveParticleCloud(BaseModel):
    """A decimated particle sample for realtime rendering."""

    r: list[float] = Field(default_factory=list)
    z: list[float] = Field(default_factory=list)
    energy_ev: list[float] = Field(default_factory=list)
    speed_m_s: list[float] = Field(default_factory=list)
    unit: str = "m"
    label: str | None = None
    species: str | None = None
    source_tag: str = "bulk"


class LiveHistogram(BaseModel):
    """A histogram shown in the live viewer."""

    axis: list[float] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    axis_unit: str | None = None
    value_unit: str | None = None
    label: str | None = None


class LiveGeometry(BaseModel):
    """Static geometry metadata for the current snapshot."""

    r_max: float | None = None
    z_max: float | None = None
    z_target: float | None = None
    z_substrate: float | None = None
    r_inner: float | None = None
    r_outer: float | None = None


class LiveSnapshot(BaseModel):
    """One full viewer refresh payload."""

    model: Literal["global", "pic"]
    state: Literal["running", "completed", "error"]
    title: str
    step: int | None = None
    time_s: float = 0.0
    updated_at_s: float
    message: str | None = None
    series: dict[str, LiveSeries] = Field(default_factory=dict)
    fields: dict[str, LiveField2D] = Field(default_factory=dict)
    particles: dict[str, LiveParticleCloud] = Field(default_factory=dict)
    histograms: dict[str, LiveHistogram] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    geometry: LiveGeometry | None = None


class LiveCommand(BaseModel):
    """One control command from the viewer to the simulation."""

    seq: int
    command: Literal["pause", "resume", "single_step"]
