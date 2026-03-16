"""Interactive backend selection for the local live viewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def is_uv_managed_python(executable: str | None = None) -> bool:
    """Detect uv-managed Python runtimes that currently break TkAgg."""

    target = Path(executable or sys.executable)
    try:
        resolved = target.resolve()
    except OSError:
        resolved = target
    return "/.local/share/uv/python/" in str(resolved)


def backend_candidates(executable: str | None = None) -> tuple[str, ...]:
    """Prefer WebAgg on uv-managed Python because TkAgg is unreliable there."""

    if is_uv_managed_python(executable):
        return ("WebAgg", "TkAgg")
    return ("TkAgg", "WebAgg")


def configure_interactive_backend(matplotlib_module: Any, executable: str | None = None) -> str:
    """Switch away from Agg when an interactive backend is available."""

    current = str(matplotlib_module.get_backend()).lower()
    if current != "agg":
        return str(matplotlib_module.get_backend())

    for backend in backend_candidates(executable):
        try:
            if backend == "TkAgg":
                import tkinter  # noqa: F401
            elif backend == "WebAgg":
                import tornado  # noqa: F401
            matplotlib_module.use(backend, force=True)
            return str(matplotlib_module.get_backend())
        except Exception:
            continue
    return str(matplotlib_module.get_backend())


def backend_failure_message(executable: str | None = None) -> str:
    """Explain how to recover when no interactive backend is available."""

    base = [
        "No interactive Matplotlib backend is available.",
        "Install the project viewer dependencies and rerun the viewer.",
    ]
    if is_uv_managed_python(executable):
        base.extend(
            [
                "This uv-managed Python build prefers WebAgg because TkAgg is unreliable here.",
                "Repair the current venv with:",
                "  uv pip install --python .venv/bin/python tornado",
                "Or use the system Python as a fallback:",
                "  python3 scripts/live_viewer.py output/live/pic_quicklook",
            ]
        )
    else:
        base.extend(
            [
                "Install one of the supported backends:",
                "  pip install tornado",
                "or ensure Tk is available for your Python interpreter.",
            ]
        )
    return "\n".join(base)
