"""Run-mode validation for simulation configs."""

from __future__ import annotations

from plasma.contracts.cases import EXPLORATORY_PROVENANCES, CaseMetadata, RunMode


def validate_run_mode(mode: RunMode, case: CaseMetadata | None) -> None:
    """Reject research runs that still depend on exploratory inputs."""

    if mode != "research":
        return
    if case is None:
        raise ValueError("Research mode requires case metadata with explicit input provenance.")

    exploratory_inputs = [
        source.name
        for source in case.inputs
        if source.provenance in EXPLORATORY_PROVENANCES
    ]
    if exploratory_inputs:
        names = ", ".join(exploratory_inputs)
        raise ValueError(
            "Research mode forbids surrogate, heuristic, or synthetic inputs; "
            f"found exploratory inputs: {names}.",
        )
