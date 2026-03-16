"""Dark-mode Matplotlib fallback for live plasma viewing."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from plasma.live.publisher import FileLiveSession, LiveCommandWriteError
from plasma.viewer.theme import BACKGROUND, SPECIES_COLORS, SURFACE, TEXT


class LegacyLiveViewer:
    """Simple dark fallback when the Qt viewer dependencies are unavailable."""

    def __init__(self, live_dir: str, refresh_ms: int = 250):
        self.session = FileLiveSession(live_dir)
        self.refresh_ms = refresh_ms
        self.snapshot = None
        self.field_name = "emissivity_arb"
        self.field_colorbar = None
        self.command_error: str | None = None

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 9), facecolor=BACKGROUND)
        gs = self.fig.add_gridspec(2, 2, left=0.06, right=0.82, top=0.94, bottom=0.08, hspace=0.28, wspace=0.22)
        self.ax_field = self.fig.add_subplot(gs[0, 0], facecolor=SURFACE)
        self.ax_particles = self.fig.add_subplot(gs[0, 1], facecolor=SURFACE)
        self.ax_series = self.fig.add_subplot(gs[1, 0], facecolor=SURFACE)
        self.ax_hist = self.fig.add_subplot(gs[1, 1], facecolor=SURFACE)

        self._add_controls()
        self.animation = FuncAnimation(self.fig, self._refresh, interval=self.refresh_ms, cache_frame_data=False)

    def _add_controls(self) -> None:
        pause_ax = self.fig.add_axes([0.85, 0.82, 0.11, 0.05], facecolor=SURFACE)
        resume_ax = self.fig.add_axes([0.85, 0.75, 0.11, 0.05], facecolor=SURFACE)
        step_ax = self.fig.add_axes([0.85, 0.68, 0.11, 0.05], facecolor=SURFACE)
        self.pause_button = Button(pause_ax, "Pause", color=SURFACE, hovercolor="#1a2330")
        self.resume_button = Button(resume_ax, "Resume", color=SURFACE, hovercolor="#1a2330")
        self.step_button = Button(step_ax, "Step", color=SURFACE, hovercolor="#1a2330")
        for button in (self.pause_button, self.resume_button, self.step_button):
            button.label.set_color(TEXT)
        self.pause_button.on_clicked(lambda _event: self._send_command("pause"))
        self.resume_button.on_clicked(lambda _event: self._send_command("resume"))
        self.step_button.on_clicked(lambda _event: self._send_command("single_step"))
        self._update_command_buttons()

    def _refresh(self, _frame: int) -> None:
        snapshot = self.session.load_snapshot()
        self._update_command_buttons()
        if snapshot is None:
            self.fig.suptitle("Waiting for live snapshot...", fontsize=14, color=TEXT)
            return
        self.snapshot = snapshot
        if self.field_name not in snapshot.fields:
            self.field_name = "emissivity_arb" if "emissivity_arb" in snapshot.fields else next(iter(snapshot.fields))
        self.fig.suptitle(
            f"{snapshot.title} | state={snapshot.state} | step={snapshot.step} | t={snapshot.time_s * 1e6:.2f} us",
            fontsize=14,
            color=TEXT,
        )
        self._draw_field()
        self._draw_particles()
        self._draw_series()
        self._draw_histogram()

    def _draw_field(self) -> None:
        self.ax_field.clear()
        field = self.snapshot.fields[self.field_name]
        values = np.asarray(field.values, dtype=float)
        x = np.asarray(field.x, dtype=float)
        y = np.asarray(field.y, dtype=float)
        image = self.ax_field.imshow(values, origin="lower", aspect="auto", extent=[x.min(), x.max(), y.min(), y.max()], cmap="magma")
        self.ax_field.set_title(field.label or self.field_name, color=TEXT)
        self.ax_field.set_xlabel("z [m]")
        self.ax_field.set_ylabel("r [m]")
        if self.snapshot.geometry and self.snapshot.geometry.z_target is not None:
            self.ax_field.plot(
                [self.snapshot.geometry.z_target, self.snapshot.geometry.z_target],
                [self.snapshot.geometry.r_inner or 0.0, self.snapshot.geometry.r_outer or 0.0],
                color="#ffd166",
                linewidth=2.0,
            )
        if self.snapshot.geometry and self.snapshot.geometry.z_substrate is not None:
            self.ax_field.plot(
                [self.snapshot.geometry.z_substrate, self.snapshot.geometry.z_substrate],
                [0.0, self.snapshot.geometry.r_max or 0.0],
                color="#8ecae6",
                linewidth=1.5,
                alpha=0.8,
            )
        if self.field_colorbar is not None:
            self.field_colorbar.remove()
        self.field_colorbar = self.fig.colorbar(image, ax=self.ax_field, fraction=0.046, pad=0.04)

    def _draw_particles(self) -> None:
        self.ax_particles.clear()
        for name, cloud in self.snapshot.particles.items():
            if not cloud.r:
                continue
            self.ax_particles.scatter(cloud.z, cloud.r, s=6, alpha=0.65, color=SPECIES_COLORS.get(name), label=name)
        self.ax_particles.set_title("Particle Sample", color=TEXT)
        self.ax_particles.set_xlabel("z [m]")
        self.ax_particles.set_ylabel("r [m]")
        self.ax_particles.legend(loc="upper right")

    def _draw_series(self) -> None:
        self.ax_series.clear()
        for name, series in self.snapshot.series.items():
            x = np.asarray(series.x, dtype=float) * 1e6
            y = np.asarray(series.y, dtype=float)
            if x.size and y.size:
                self.ax_series.plot(x, y, label=series.label or name)
        self.ax_series.set_title("Live Diagnostics", color=TEXT)
        self.ax_series.set_xlabel("Time [us]")
        self.ax_series.grid(True, alpha=0.2)
        if self.ax_series.lines:
            self.ax_series.legend(loc="upper left", fontsize=8)

    def _draw_histogram(self) -> None:
        self.ax_hist.clear()
        if self.snapshot.histograms:
            _, histogram = next(iter(self.snapshot.histograms.items()))
            self.ax_hist.plot(histogram.axis, histogram.values, color="#57c7ff")
            self.ax_hist.set_title(histogram.label or "Histogram", color=TEXT)
            self.ax_hist.set_xlabel(f"Energy [{histogram.axis_unit or 'a.u.'}]")
            self.ax_hist.grid(True, alpha=0.2)
            return
        metrics = "\n".join(f"{name}: {value:.0f}" for name, value in sorted(self.snapshot.metrics.items()))
        self.ax_hist.text(0.03, 0.97, metrics or "No metrics", ha="left", va="top", color=TEXT)
        self.ax_hist.set_title("Run Metrics", color=TEXT)
        self.ax_hist.set_axis_off()

    def _update_command_buttons(self) -> None:
        state_error = None
        if self.snapshot is not None and self.snapshot.state in {"completed", "failed"}:
            state_error = f"live controls unavailable: run is {self.snapshot.state}"
        error = state_error or self.session.command_write_error()
        self.command_error = error
        enabled = error is None
        for button in (self.pause_button, self.resume_button, self.step_button):
            button.ax.set_visible(enabled)

    def _send_command(self, command: str) -> None:
        try:
            self.session.send_command(command)
            self.command_error = None
        except LiveCommandWriteError as exc:
            self.command_error = str(exc)
            self.fig.suptitle(self.command_error, fontsize=11, color=TEXT)

    def show(self) -> None:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dark fallback live viewer for file-backed plasma snapshots")
    parser.add_argument("live_dir")
    parser.add_argument("--refresh-ms", type=int, default=250)
    args = parser.parse_args(argv)
    LegacyLiveViewer(args.live_dir, refresh_ms=args.refresh_ms).show()
    return 0
