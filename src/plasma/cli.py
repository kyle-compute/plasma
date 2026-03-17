"""Command-line workflows for datasets, surrogates, and candidate ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasma.ml.dataset import build_scalar_dataset, load_scalar_dataset
from plasma.ml.optimization import OptimizationConstraint, OptimizationObjective, rank_candidates
from plasma.ml.surrogate import ScalarSurrogate, ScalarSurrogateConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plasma research workflow tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset-build", help="Build a scalar dataset from run directories")
    dataset_parser.add_argument("output_dir")
    dataset_parser.add_argument("run_dirs", nargs="+")
    dataset_parser.add_argument("--feature", dest="feature_specs", action="append", required=True)
    dataset_parser.add_argument("--target", dest="target_specs", action="append", required=True)
    dataset_parser.add_argument("--name", default="scalar_dataset")
    dataset_parser.add_argument("--include-exploratory", action="store_true")
    dataset_parser.set_defaults(handler=_handle_dataset_build)

    train_parser = subparsers.add_parser("surrogate-train", help="Train a scalar surrogate from a saved dataset")
    train_parser.add_argument("dataset_dir")
    train_parser.add_argument("output_dir")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--hidden", dest="hidden_layers", type=int, action="append")
    train_parser.set_defaults(handler=_handle_surrogate_train)

    predict_parser = subparsers.add_parser("surrogate-predict", help="Run a saved surrogate on feature rows")
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("features_json", help="JSON file containing a list of feature rows")
    predict_parser.add_argument("--output", help="Optional output JSON path")
    predict_parser.set_defaults(handler=_handle_surrogate_predict)

    rank_parser = subparsers.add_parser("rank-candidates", help="Rank candidate predictions with objectives/constraints")
    rank_parser.add_argument("predictions_json", help="JSON file containing a list of prediction dicts")
    rank_parser.add_argument("--objective", action="append", required=True, help="metric:direction[:weight]")
    rank_parser.add_argument("--constraint", action="append", default=[], help="metric[:lower][:upper]")
    rank_parser.add_argument("--output", help="Optional output JSON path")
    rank_parser.set_defaults(handler=_handle_rank_candidates)

    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_dataset_build(args) -> int:
    build_scalar_dataset(
        args.run_dirs,
        feature_specs=args.feature_specs,
        target_specs=args.target_specs,
        dataset_name=args.name,
        include_exploratory=args.include_exploratory,
        output_dir=args.output_dir,
    )
    return 0


def _handle_surrogate_train(args) -> int:
    dataset = load_scalar_dataset(args.dataset_dir)
    config = ScalarSurrogateConfig(
        hidden_layers=args.hidden_layers or [64, 64],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    surrogate = ScalarSurrogate(
        feature_names=dataset.manifest.feature_specs,
        target_names=dataset.manifest.target_specs,
        config=config,
    )
    try:
        surrogate.fit(dataset.features, dataset.targets)
        surrogate.save(args.output_dir)
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 2
    return 0


def _handle_surrogate_predict(args) -> int:
    try:
        surrogate = ScalarSurrogate.load(args.model_dir)
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 2
    features = json.loads(Path(args.features_json).read_text())
    predictions = surrogate.predict(features).tolist()
    payload = {"feature_names": surrogate.feature_names, "target_names": surrogate.target_names, "predictions": predictions}
    _write_payload(payload, args.output)
    return 0


def _handle_rank_candidates(args) -> int:
    predictions = json.loads(Path(args.predictions_json).read_text())
    ranked = rank_candidates(
        predictions,
        objectives=[_parse_objective(item) for item in args.objective],
        constraints=[_parse_constraint(item) for item in args.constraint],
    )
    payload = [entry.model_dump() for entry in ranked]
    _write_payload(payload, args.output)
    return 0


def _parse_objective(raw: str) -> OptimizationObjective:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("Objectives must use metric:direction[:weight]")
    metric, direction = parts[0], parts[1]
    weight = 1.0 if len(parts) == 2 or not parts[2] else float(parts[2])
    return OptimizationObjective(metric=metric, direction=direction, weight=weight)


def _parse_constraint(raw: str) -> OptimizationConstraint:
    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError("Constraints must use metric[:lower][:upper]")
    while len(parts) < 3:
        parts.append("")
    metric, lower_raw, upper_raw = parts
    lower = None if not lower_raw else float(lower_raw)
    upper = None if not upper_raw else float(upper_raw)
    return OptimizationConstraint(metric=metric, lower=lower, upper=upper)


def _write_payload(payload, output: str | None) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if output:
        Path(output).write_text(text)
    else:
        print(text, end="")
