"""JSON helpers for run manifests and validation reports."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


def save_json_model(path: str | Path, model: BaseModel) -> None:
    """Persist a pydantic model as pretty JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(model.model_dump_json(indent=2) + "\n")
