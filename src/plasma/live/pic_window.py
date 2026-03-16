"""Helpers for accumulating reduced live PIC event windows."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

EventCloud: TypeAlias = dict[str, np.ndarray]
EventWindow: TypeAlias = dict[str, EventCloud]


def merge_event_clouds(target: EventWindow, incoming: EventWindow | None) -> None:
    """Append one event window into another."""

    if not incoming:
        return

    for name, cloud in incoming.items():
        r = np.asarray(cloud.get("r", np.empty(0)), dtype=np.float64)
        z = np.asarray(cloud.get("z", np.empty(0)), dtype=np.float64)
        if r.size == 0 or z.size == 0:
            continue
        if name not in target:
            target[name] = {"r": r.copy(), "z": z.copy()}
            continue
        target[name]["r"] = np.concatenate((target[name]["r"], r))
        target[name]["z"] = np.concatenate((target[name]["z"], z))


def count_event_cloud(cloud: EventCloud | None) -> int:
    """Return the number of points in one event cloud."""

    if cloud is None:
        return 0
    return int(np.asarray(cloud.get("r", np.empty(0))).size)


def event_counts(window: EventWindow | None) -> dict[str, int]:
    """Count events by name within one accumulated window."""

    if not window:
        return {}
    return {name: count_event_cloud(cloud) for name, cloud in window.items()}


def clear_event_clouds(window: EventWindow) -> None:
    """Reset an event window in place."""

    window.clear()
