"""Tests for surrogate-prediction ranking helpers."""

from __future__ import annotations

from plasma.ml.optimization import OptimizationConstraint, OptimizationObjective, rank_candidates


def test_rank_candidates_prefers_feasible_high_score() -> None:
    ranked = rank_candidates(
        [
            {"metal_delivery_efficiency": 0.7, "peak_current_a": 80.0},
            {"metal_delivery_efficiency": 0.6, "peak_current_a": 55.0},
            {"metal_delivery_efficiency": 0.8, "peak_current_a": 52.0},
        ],
        objectives=[OptimizationObjective(metric="metal_delivery_efficiency", direction="maximize", weight=1.0)],
        constraints=[OptimizationConstraint(metric="peak_current_a", upper=60.0)],
    )

    assert ranked[0].index == 2
    assert ranked[0].feasible is True
    assert ranked[-1].index == 0
    assert ranked[-1].feasible is False
