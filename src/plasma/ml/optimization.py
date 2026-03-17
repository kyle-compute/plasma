"""Prediction ranking helpers for surrogate-driven optimization experiments."""

from __future__ import annotations

from pydantic import BaseModel


class OptimizationObjective(BaseModel):
    """One scalar objective over surrogate outputs."""

    metric: str
    direction: str = "maximize"
    weight: float = 1.0


class OptimizationConstraint(BaseModel):
    """One scalar box constraint over surrogate outputs."""

    metric: str
    lower: float | None = None
    upper: float | None = None


class CandidateRanking(BaseModel):
    """One ranked candidate and its predicted metrics."""

    index: int
    score: float
    feasible: bool
    predicted: dict[str, float]


def rank_candidates(
    predictions: list[dict[str, float]],
    *,
    objectives: list[OptimizationObjective],
    constraints: list[OptimizationConstraint] | None = None,
) -> list[CandidateRanking]:
    """Rank surrogate predictions by feasibility first, then weighted score."""

    constraints = constraints or []
    ranked: list[CandidateRanking] = []
    for index, predicted in enumerate(predictions):
        feasible = all(_satisfies_constraint(predicted, constraint) for constraint in constraints)
        score = 0.0
        for objective in objectives:
            value = predicted[objective.metric]
            if objective.direction == "maximize":
                score += objective.weight * value
            elif objective.direction == "minimize":
                score -= objective.weight * value
            else:
                raise ValueError(f"Unsupported objective direction '{objective.direction}'")
        ranked.append(
            CandidateRanking(
                index=index,
                score=score,
                feasible=feasible,
                predicted={name: float(value) for name, value in predicted.items()},
            )
        )
    return sorted(ranked, key=lambda item: (not item.feasible, -item.score, item.index))


def _satisfies_constraint(predicted: dict[str, float], constraint: OptimizationConstraint) -> bool:
    value = predicted[constraint.metric]
    if constraint.lower is not None and value < constraint.lower:
        return False
    return constraint.upper is None or value <= constraint.upper
