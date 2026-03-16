"""Quick-look visual summaries for PIC runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plasma.diagnostics.bundles import DiagnosticsBundle


def save_pic_quicklook(bundle: DiagnosticsBundle, path: str | Path, *, title: str) -> Path:
    """Persist a compact visual summary for a PIC run."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    t_us = np.asarray(bundle.time_s) * 1e6

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    _plot_series(axes[0, 0], t_us, bundle, ["target_voltage_v"], "Waveform", "Voltage [V]")
    _plot_matching(axes[0, 1], t_us, bundle, suffix="_particles", title="Particle Counts", ylabel="count")
    _plot_matching(axes[1, 0], t_us, bundle, suffix="_kinetic_energy_j", title="Kinetic Energy", ylabel="J")
    _plot_series(
        axes[1, 1],
        t_us,
        bundle,
        ["field_energy_j", "total_energy_j"],
        "Global Energy",
        "J",
    )
    _plot_series(
        axes[2, 0],
        t_us,
        bundle,
        ["n_target_impacts_step", "n_sputtered_step", "n_see_step", "collisions_per_sample"],
        "Surface / Collision Activity",
        "count",
    )
    _plot_distribution(axes[2, 1], bundle)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _plot_series(ax, t_us: np.ndarray, bundle: DiagnosticsBundle, names: list[str], title: str, ylabel: str) -> None:
    for name in names:
        series = bundle.series.get(name)
        if series is not None:
            ax.plot(t_us, series.values, label=name)
    ax.set(title=title, xlabel="Time [us]", ylabel=ylabel)
    if len(ax.lines) > 1:
        ax.legend()


def _plot_matching(ax, t_us: np.ndarray, bundle: DiagnosticsBundle, *, suffix: str, title: str, ylabel: str) -> None:
    names = [name for name in bundle.series if name.endswith(suffix)]
    _plot_series(ax, t_us, bundle, names, title, ylabel)


def _plot_distribution(ax, bundle: DiagnosticsBundle) -> None:
    dist = bundle.distributions.get("substrate_iedf")
    if dist is None:
        ax.text(0.5, 0.5, "No substrate IEDF", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.plot(dist.axis, dist.values)
    ax.set(title="Substrate IEDF", xlabel=f"Energy [{dist.axis_unit}]", ylabel=dist.value_unit or "count")
