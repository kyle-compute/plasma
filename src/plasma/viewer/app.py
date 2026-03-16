"""Qt realtime viewer for dark, high-signal plasma monitoring."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyqtgraph as pg
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from plasma.live.hipims_monitor import PULSE_PHASES
from plasma.live.publisher import FileLiveSession, LiveCommandWriteError
from plasma.viewer.presentation import (
    FIELD_PRESETS,
    SERIES_GROUPS,
    format_metric,
    grouped_series,
    ordered_field_names,
    preferred_field_name,
    preset_field_name,
    tone_map_field,
)
from plasma.viewer.state import TrailBuffer, emissive_points, field_matrix, particle_ring_points
from plasma.viewer.theme import (
    BACKGROUND,
    FIELD_GRADIENTS,
    MUTED,
    QT_STYLE_SHEET,
    SERIES_COLORS,
    SPECIES_COLORS,
    TEXT,
)


class PlasmaViewerWindow(QtWidgets.QMainWindow):
    """Desktop viewer with synchronized 2D and pseudo-3D panels."""

    def __init__(self, live_dir: str, refresh_ms: int = 100) -> None:
        super().__init__()
        self.session = FileLiveSession(live_dir)
        self.snapshot = None
        self.field_name = "target_activity_arb"
        self.current_preset = "HiPIMS Monitor"
        self.trails = TrailBuffer(length=10)
        self._camera_seeded = False
        self._theta_offset = 0.0
        self._volume_error: str | None = None
        self._command_error: str | None = None
        self._lut_cache: dict[str, np.ndarray] = {}
        self._current_gradient = "glow"
        self._volume_enabled = self._detect_volume_support()
        self.command_buttons: list[QtWidgets.QPushButton] = []

        pg.setConfigOption("background", BACKGROUND)
        pg.setConfigOption("foreground", TEXT)
        pg.setConfigOption("imageAxisOrder", "row-major")
        self.setWindowTitle("Plasma Live Viewer")
        self.resize(1840, 1180)
        self.setStyleSheet(QT_STYLE_SHEET)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        self.setCentralWidget(central)

        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar())
        root.addWidget(self._build_body(), stretch=1)
        self._update_command_controls()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(refresh_ms)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def _detect_volume_support(self) -> bool:
        display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        headless = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen" or not display_available
        disable_3d = os.environ.get("PLASMA_VIEWER_DISABLE_3D", "").lower() in {"1", "true", "yes"}
        if headless:
            self._volume_error = "3D view disabled in headless mode"
        elif disable_3d:
            self._volume_error = "3D view disabled by PLASMA_VIEWER_DISABLE_3D"
        return not (headless or disable_3d)

    def _build_header(self) -> QtWidgets.QWidget:
        frame = self._panel_frame()
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)

        text_block = QtWidgets.QVBoxLayout()
        self.title_label = QtWidgets.QLabel("Plasma Live Viewer")
        self.title_label.setObjectName("title")
        self.subtitle_label = QtWidgets.QLabel("Waiting for snapshot...")
        self.subtitle_label.setObjectName("subtitle")
        text_block.addWidget(self.title_label)
        text_block.addWidget(self.subtitle_label)
        layout.addLayout(text_block, stretch=1)

        self.badge_labels: dict[str, QtWidgets.QLabel] = {}
        for key in ("state", "phase", "step", "time", "field", "capture"):
            label = QtWidgets.QLabel("--")
            label.setObjectName("badge")
            self.badge_labels[key] = label
            layout.addWidget(label)
        return frame

    def _build_toolbar(self) -> QtWidgets.QWidget:
        frame = self._panel_frame()
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)

        self.preset_selector = QtWidgets.QComboBox()
        self.preset_selector.addItems(FIELD_PRESETS.keys())
        self.preset_selector.setCurrentText(self.current_preset)
        self.preset_selector.currentTextChanged.connect(self._on_preset_selected)

        self.field_selector = QtWidgets.QComboBox()
        self.field_selector.currentTextChanged.connect(self._on_field_selected)

        self.trail_selector = QtWidgets.QSpinBox()
        self.trail_selector.setRange(2, 24)
        self.trail_selector.setValue(8)
        self.trail_selector.valueChanged.connect(self.trails.set_length)

        self.particles_toggle = QtWidgets.QCheckBox("Particles")
        self.particles_toggle.setChecked(True)
        self.trails_toggle = QtWidgets.QCheckBox("Trails")
        self.trails_toggle.setChecked(False)

        pause = QtWidgets.QPushButton("Pause")
        pause.clicked.connect(lambda _checked=False, command="pause": self._send_command(command))
        resume = QtWidgets.QPushButton("Resume")
        resume.clicked.connect(lambda _checked=False, command="resume": self._send_command(command))
        step = QtWidgets.QPushButton("Step")
        step.clicked.connect(lambda _checked=False, command="single_step": self._send_command(command))
        self.command_buttons = [pause, resume, step]

        layout.addWidget(QtWidgets.QLabel("Preset"))
        layout.addWidget(self.preset_selector)
        layout.addWidget(QtWidgets.QLabel("Field"))
        layout.addWidget(self.field_selector, stretch=1)
        layout.addWidget(QtWidgets.QLabel("Trail"))
        layout.addWidget(self.trail_selector)
        layout.addWidget(self.particles_toggle)
        layout.addWidget(self.trails_toggle)
        layout.addStretch(1)
        layout.addWidget(pause)
        layout.addWidget(resume)
        layout.addWidget(step)
        return frame

    def _build_body(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self._build_top_row())
        splitter.addWidget(self._build_bottom_row())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        return splitter

    def _build_top_row(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.field_plot = self._new_plot("Radial Field", bottom="z", left="r")
        self.field_image = pg.ImageItem()
        self.field_image.setLookupTable(self._field_lut("glow"))
        self.field_plot.addItem(self.field_image)
        splitter.addWidget(self._wrap_widget(self.field_plot))

        if self._volume_enabled:
            self.volume_view: QtInteractor | None = QtInteractor(self)
            self.volume_view.set_background(BACKGROUND)
            splitter.addWidget(self._wrap_widget(self.volume_view))
        else:
            self.volume_view = None
            notice = QtWidgets.QLabel(self._volume_error or "3D view unavailable")
            notice.setAlignment(QtCore.Qt.AlignCenter)
            notice.setWordWrap(True)
            splitter.addWidget(self._wrap_widget(notice))

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _build_bottom_row(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        diag_widget = self._panel_frame()
        diag_layout = QtWidgets.QVBoxLayout(diag_widget)
        diag_layout.setContentsMargins(8, 8, 8, 8)
        diag_layout.setSpacing(8)
        self.series_plots: dict[str, pg.PlotWidget] = {}
        for title, _ in SERIES_GROUPS:
            plot = self._new_plot(title, bottom="time", left="")
            plot.setMaximumHeight(170)
            self.series_plots[title] = plot
            diag_layout.addWidget(plot)
        splitter.addWidget(diag_widget)

        status_widget = self._panel_frame()
        status_layout = QtWidgets.QVBoxLayout(status_widget)
        status_layout.setContentsMargins(8, 8, 8, 8)
        status_layout.setSpacing(8)
        self.hist_plot = self._new_plot("Substrate IEDF", bottom="energy", left="count")
        self.hist_plot.setMaximumHeight(220)
        self.metrics_label = QtWidgets.QLabel("Waiting for snapshot...")
        self.metrics_label.setObjectName("metrics")
        self.metrics_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        status_layout.addWidget(self.hist_plot, stretch=3)
        status_layout.addWidget(self.metrics_label, stretch=2)
        splitter.addWidget(status_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _new_plot(self, title: str, *, bottom: str, left: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setTitle(f"<span style='color:{MUTED}; font-size:11pt'>{title}</span>")
        if bottom:
            plot.setLabel("bottom", bottom)
        if left:
            plot.setLabel("left", left)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setTextPen(pg.mkPen(MUTED))
            axis.setPen(pg.mkPen(MUTED))
        return plot

    def _panel_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        return frame

    def _wrap_widget(self, widget: QtWidgets.QWidget) -> QtWidgets.QFrame:
        frame = self._panel_frame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(widget)
        return frame

    def _field_lut(self, gradient_name: str) -> np.ndarray:
        if gradient_name not in self._lut_cache:
            gradient = FIELD_GRADIENTS[gradient_name]
            cmap = pg.ColorMap(
                np.asarray([stop for stop, _ in gradient], dtype=np.float64),
                np.asarray([color for _, color in gradient], dtype=np.ubyte),
            )
            self._lut_cache[gradient_name] = cmap.getLookupTable(0.0, 1.0, 256)
        return self._lut_cache[gradient_name]

    def _on_preset_selected(self, preset: str) -> None:
        if not self.snapshot:
            self.current_preset = preset
            return
        names = ordered_field_names(list(self.snapshot.fields))
        field_name = preset_field_name(names, preset, current=self.field_name)
        if field_name:
            self.current_preset = preset
            self.field_name = field_name
            self.field_selector.setCurrentText(field_name)
            self.draw()

    def _on_field_selected(self, name: str) -> None:
        if name:
            self.field_name = name
            self.draw()

    def refresh(self) -> None:
        snapshot = self.session.load_snapshot()
        if snapshot is None:
            return
        self.snapshot = snapshot
        self.trails.push(snapshot)
        self._sync_fields()
        self._update_command_controls()
        self.draw()

    def _sync_fields(self) -> None:
        if self.snapshot is None:
            return
        names = ordered_field_names(list(self.snapshot.fields))
        selected = preferred_field_name(names, current=self.field_name)
        current_items = [self.field_selector.itemText(i) for i in range(self.field_selector.count())]
        if current_items != names:
            self.field_selector.blockSignals(True)
            self.field_selector.clear()
            self.field_selector.addItems(names)
            self.field_selector.blockSignals(False)
        if selected and selected != self.field_name:
            self.field_name = selected
        if selected:
            self.field_selector.blockSignals(True)
            self.field_selector.setCurrentText(selected)
            self.field_selector.blockSignals(False)

    def draw(self) -> None:
        if self.snapshot is None:
            return
        self._update_header()
        self._draw_2d()
        self._draw_3d()
        self._draw_diagnostics()
        self._draw_histogram()
        self._draw_metrics()
        self.setWindowTitle(
            f"{self.snapshot.title} | {self.snapshot.state} | step={self.snapshot.step} | t={self.snapshot.time_s * 1e6:.2f} us"
        )

    def _update_header(self) -> None:
        metrics = self.snapshot.metrics
        phase_code = int(round(metrics.get("pulse_phase_code", 0.0)))
        phase_label = PULSE_PHASES[max(0, min(phase_code, len(PULSE_PHASES) - 1))]
        self.title_label.setText(self.snapshot.title)
        subtitle = self.snapshot.message or "Axisymmetric HiPIMS monitor"
        if self._command_error:
            subtitle = f"{subtitle} | {self._command_error}"
        self.subtitle_label.setText(subtitle)
        self.badge_labels["state"].setText(f"state  {self.snapshot.state}")
        self.badge_labels["phase"].setText(f"phase  {phase_label}")
        self.badge_labels["step"].setText(f"step   {self.snapshot.step}")
        self.badge_labels["time"].setText(f"time   {self.snapshot.time_s * 1e6:.2f} us")
        self.badge_labels["field"].setText(f"field  {self.field_name}")
        self.badge_labels["capture"].setText(
            f"capture {format_metric(metrics.get('substrate_capture_proxy', 0.0))}"
        )

    def _draw_2d(self) -> None:
        self.field_plot.clear()
        self.field_plot.addItem(self.field_image)
        field = field_matrix(self.snapshot, self.field_name)
        if field is None:
            return
        x, y, values = field
        render = tone_map_field(self.field_name, values)
        if render.gradient_name != self._current_gradient:
            self._current_gradient = render.gradient_name
            self.field_image.setLookupTable(self._field_lut(render.gradient_name))
        self.field_image.setImage(render.image, autoLevels=False, levels=(0.0, 1.0))
        self.field_image.setRect(QtCore.QRectF(float(x.min()), float(y.min()), float(np.ptp(x)), float(np.ptp(y))))
        self._draw_potential_contours()
        self._draw_monitor_overlays()
        self._draw_geometry()
        if self.trails_toggle.isChecked():
            self._draw_trails()
        if self.particles_toggle.isChecked():
            self._draw_particles()

    def _draw_potential_contours(self) -> None:
        contour_field = field_matrix(self.snapshot, "phi_v")
        if contour_field is None:
            return
        _, _, phi_values = contour_field
        phi_values = np.asarray(phi_values, dtype=np.float64)
        lower = float(np.min(phi_values))
        upper = float(np.max(phi_values))
        if np.isclose(lower, upper):
            return
        levels = np.linspace(lower, upper, 7)[1:-1]
        for level in levels:
            curve = pg.IsocurveItem(data=phi_values, level=float(level), pen=pg.mkPen((255, 255, 255, 42), width=0.8))
            curve.setParentItem(self.field_image)
            curve.setZValue(3)

    def _draw_monitor_overlays(self) -> None:
        if self.current_preset != "HiPIMS Monitor":
            return
        self._draw_field_isocurve("sputter_source_arb", (255, 177, 71, 115), level=0.46)
        self._draw_field_isocurve("see_source_arb", (155, 246, 255, 110), level=0.42)
        self._draw_field_isocurve("substrate_flux_proxy_arb", (122, 215, 255, 100), level=0.30)

    def _draw_field_isocurve(self, field_name: str, color: tuple[int, int, int, int], *, level: float) -> None:
        overlay_field = field_matrix(self.snapshot, field_name)
        if overlay_field is None:
            return
        _, _, values = overlay_field
        normalized = tone_map_field(field_name, values).image
        if normalized.size == 0 or not np.any(normalized > 0.0):
            return
        curve = pg.IsocurveItem(data=normalized, level=level, pen=pg.mkPen(color, width=1.2))
        curve.setParentItem(self.field_image)
        curve.setZValue(4)

    def _draw_geometry(self) -> None:
        geometry = self.snapshot.geometry
        if geometry is None:
            return
        if geometry.z_target is not None:
            self.field_plot.plot(
                [geometry.z_target, geometry.z_target],
                [geometry.r_inner or 0.0, geometry.r_outer or 0.0],
                pen=pg.mkPen("#ffd166", width=3),
            )
        if geometry.z_substrate is not None:
            self.field_plot.plot(
                [geometry.z_substrate, geometry.z_substrate],
                [0.0, geometry.r_max or 0.0],
                pen=pg.mkPen("#9ad7ff", width=2),
            )
        if geometry.z_target is not None and geometry.r_inner is not None and geometry.r_outer is not None:
            band_end = geometry.z_target + 0.006
            for radius in (geometry.r_inner, geometry.r_outer):
                self.field_plot.plot(
                    [geometry.z_target, band_end],
                    [radius, radius],
                    pen=pg.mkPen((255, 209, 102, 120), width=1, style=QtCore.Qt.DashLine),
                )

    def _draw_trails(self) -> None:
        for species, history in self.trails.trails.items():
            color = SPECIES_COLORS.get(species, "#ffffff")
            stride = 3 if species == "electron" else 2 if species == "Ar+" else 1
            width = 0.9 if species == "electron" else 1.15
            for idx, points in enumerate(history):
                if points.size == 0:
                    continue
                trail = points[::stride]
                alpha = int(18 + 110 * (idx + 1) / max(len(history), 1))
                pen_color = pg.mkColor(color)
                pen_color.setAlpha(alpha)
                self.field_plot.plot(trail[:, 0], trail[:, 1], pen=pg.mkPen(pen_color, width=width))

    def _draw_particles(self) -> None:
        for species, cloud in self.snapshot.particles.items():
            if not cloud.z:
                continue
            stride = 4 if species == "electron" else 2 if species == "Ar+" else 1
            z = np.asarray(cloud.z, dtype=np.float64)[::stride]
            r = np.asarray(cloud.r, dtype=np.float64)[::stride]
            brush = pg.mkColor(SPECIES_COLORS.get(species, "#ffffff"))
            brush.setAlpha(165 if species == "electron" else 185)
            scatter = pg.ScatterPlotItem(
                x=z,
                y=r,
                brush=pg.mkBrush(brush),
                pen=None,
                size=3.0 if species == "electron" else 4.2,
            )
            self.field_plot.addItem(scatter)

    def _draw_3d(self) -> None:
        if self.volume_view is None or self._volume_error:
            return
        try:
            self.volume_view.clear()
            self._draw_3d_source_layer("emissivity_arb", color="#5b2d91", threshold=0.18, stride_cap=50000, point_size=4.6, opacity=0.12)
            self._draw_3d_source_layer("ionization_source_arb", color="#ff66c4", threshold=0.22, stride_cap=18000, point_size=5.2, opacity=0.25)
            self._draw_3d_source_layer("sputter_source_arb", color="#ffb347", threshold=0.18, stride_cap=22000, point_size=6.2, opacity=0.35)
            self._draw_3d_source_layer("substrate_flux_proxy_arb", color="#7ad7ff", threshold=0.12, stride_cap=16000, point_size=5.6, opacity=0.32)
            self._draw_3d_geometry()
            self._draw_3d_particles()
            if not self._camera_seeded:
                self.volume_view.camera_position = [
                    (0.17, -0.22, 0.05),
                    (0.0, 0.0, 0.05),
                    (0.0, 0.0, 1.0),
                ]
                self._camera_seeded = True
            self.volume_view.camera.Azimuth(0.45)
            self.volume_view.reset_camera_clipping_range()
            self._theta_offset += 0.04
        except Exception as exc:
            self._volume_error = f"3D render disabled: {exc.__class__.__name__}"
            self.volume_view.clear()

    def _draw_3d_source_layer(
        self,
        field_name: str,
        *,
        color: str,
        threshold: float,
        stride_cap: int,
        point_size: float,
        opacity: float,
    ) -> None:
        if field_name not in self.snapshot.fields:
            return
        points, _ = emissive_points(self.snapshot, field_name=field_name, n_theta=56, threshold=threshold)
        if len(points) == 0:
            return
        stride = max(len(points) // stride_cap, 1)
        mesh = pv.PolyData(points[::stride])
        self.volume_view.add_mesh(
            mesh,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            opacity=opacity,
            show_scalar_bar=False,
        )

    def _draw_3d_geometry(self) -> None:
        geometry = self.snapshot.geometry
        if geometry is None:
            return
        if geometry.r_outer and geometry.z_target is not None:
            target = pv.Disc(inner=geometry.r_inner or 0.0, outer=geometry.r_outer, c_res=96)
            target.translate((0.0, 0.0, geometry.z_target), inplace=True)
            self.volume_view.add_mesh(target, color="#d98f43", opacity=0.92)
        if geometry.r_max and geometry.z_substrate is not None:
            substrate = pv.Disc(inner=0.0, outer=geometry.r_max, c_res=96)
            substrate.translate((0.0, 0.0, geometry.z_substrate), inplace=True)
            self.volume_view.add_mesh(substrate, color="#86c5ff", opacity=0.42)

    def _draw_3d_particles(self) -> None:
        if not self.particles_toggle.isChecked():
            return
        for idx, (species, cloud) in enumerate(self.snapshot.particles.items()):
            points = particle_ring_points(cloud, theta_offset=self._theta_offset + 0.2 * idx)
            if len(points) == 0:
                continue
            stride = 10 if species == "electron" else 6 if species == "Ar+" else 4
            mesh = pv.PolyData(points[::stride])
            self.volume_view.add_mesh(
                mesh,
                color=SPECIES_COLORS.get(species, "#ffffff"),
                point_size=4.2 if species == "electron" else 5.0,
                render_points_as_spheres=True,
                opacity=0.25,
                show_scalar_bar=False,
            )

    def _draw_diagnostics(self) -> None:
        for plot in self.series_plots.values():
            plot.clear()
            if plot.plotItem.legend is None:
                plot.addLegend(offset=(8, 8), labelTextSize="8pt")
        for title, entries in grouped_series(self.snapshot.series):
            plot = self.series_plots.get(title)
            if plot is None:
                continue
            for name, series in entries:
                x = np.asarray(series.x, dtype=np.float64) * 1e6
                y = np.asarray(series.y, dtype=np.float64)
                if x.size == 0 or y.size == 0:
                    continue
                color = SERIES_COLORS.get(name, SPECIES_COLORS.get(name.split("_")[0], "#8ecae6"))
                plot.plot(x, y, pen=pg.mkPen(color=color, width=1.8), name=series.label or name)

    def _draw_histogram(self) -> None:
        self.hist_plot.clear()
        if self.snapshot is None or not self.snapshot.histograms:
            return
        _, histogram = next(iter(self.snapshot.histograms.items()))
        self.hist_plot.plot(
            histogram.axis,
            histogram.values,
            fillLevel=0.0,
            brush=(87, 199, 255, 50),
            pen=pg.mkPen("#7ad7ff", width=2),
        )

    def _draw_metrics(self) -> None:
        metrics = self.snapshot.metrics
        phase_code = int(round(metrics.get("pulse_phase_code", 0.0)))
        phase_label = PULSE_PHASES[max(0, min(phase_code, len(PULSE_PHASES) - 1))]
        ordered_names = [
            "peak_target_activity_arb",
            "racetrack_peak_r_m",
            "see_yield_proxy",
            "sputter_yield_proxy",
            "substrate_capture_proxy",
            "source_activity_total_arb",
            "substrate_flux_total_arb",
            "substrate_mean_energy_ev",
            "max_e_field_v_m",
            "emissivity_total_arb",
            "mean_electron_energy_ev",
            "mean_ion_energy_ev",
            "e_Ar_excitation",
            "e_Ar_ionization",
            "Arplus_Ar_charge_exchange",
            "target_impacts",
            "secondary_electrons",
            "sputtered_neutrals",
        ]
        lines = [
            f"field        {self.field_name}",
            f"preset       {self.current_preset}",
            f"geometry     axisymmetric",
            f"pulse_phase  {phase_label}",
            f"step         {self.snapshot.step}",
            f"time_us      {self.snapshot.time_s * 1e6:.2f}",
        ]
        if self._volume_error:
            lines.append(f"3d_status    {self._volume_error}")
        for name in ordered_names:
            if name in metrics:
                lines.append(f"{name:<12} {format_metric(metrics[name])}")
        self.metrics_label.setText("\n".join(lines))

    def _update_command_controls(self) -> None:
        state_error = None
        if self.snapshot is not None and self.snapshot.state in {"completed", "failed"}:
            state_error = f"live controls unavailable: run is {self.snapshot.state}"
        self._command_error = state_error or self.session.command_write_error()
        for button in self.command_buttons:
            button.setEnabled(self._command_error is None)
            button.setToolTip(self._command_error or "Send a live control command to the simulation")

    def _send_command(self, command: str) -> None:
        try:
            self.session.send_command(command)
        except LiveCommandWriteError as exc:
            self._command_error = str(exc)
            for button in self.command_buttons:
                button.setEnabled(False)
                button.setToolTip(self._command_error)
            self.draw()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qt live viewer for file-backed plasma snapshots")
    parser.add_argument("live_dir")
    parser.add_argument("--refresh-ms", type=int, default=100)
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlasmaViewerWindow(args.live_dir, refresh_ms=args.refresh_ms)
    window.show()
    return int(app.exec())
