"""Qt desktop launcher for local PIC workflows."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from plasma.desktop_config import (
    DesktopDurationConfig,
    DesktopLauncherConfig,
    DesktopProfileConfig,
    default_desktop_config_path,
    load_desktop_config,
)
from plasma.desktop_runtime import (
    build_run_command,
    build_run_name,
    build_viewer_command,
    default_python_executable,
    find_duration,
    find_profile,
    project_root,
    write_launcher_config,
)


class PlasmaDesktopWindow(QtWidgets.QMainWindow):
    """Small Qt launcher for desktop PIC runs."""

    def __init__(self, launcher_config: DesktopLauncherConfig | None = None, *, root_dir: Path | None = None) -> None:
        super().__init__()
        self.root_dir = root_dir or project_root()
        self.launcher_config = launcher_config or load_desktop_config(default_desktop_config_path(self.root_dir))
        self.python_exe = default_python_executable(self.root_dir)
        self.generated_live_dir: Path | None = None
        self.run_process = QtCore.QProcess(self)
        self.viewer_processes: list[QtCore.QProcess] = []

        self.setWindowTitle(self.launcher_config.window_title)
        self.resize(920, 640)
        self._build_ui()
        self._connect_processes()
        self._sync_steps_from_duration()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Profile"))
        self.profile_selector = QtWidgets.QComboBox()
        self.profile_selector.addItems([profile.key for profile in self.launcher_config.profiles])
        self.profile_selector.setCurrentText(self.launcher_config.default_profile)
        self.profile_selector.currentTextChanged.connect(self._update_details)
        selector_row.addWidget(self.profile_selector)

        selector_row.addWidget(QtWidgets.QLabel("Duration"))
        self.duration_selector = QtWidgets.QComboBox()
        self.duration_selector.addItems([duration.key for duration in self.launcher_config.durations])
        self.duration_selector.setCurrentText(self.launcher_config.default_duration)
        self.duration_selector.currentTextChanged.connect(self._sync_steps_from_duration)
        selector_row.addWidget(self.duration_selector)

        selector_row.addWidget(QtWidgets.QLabel("Steps"))
        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(100, 100_000_000)
        self.steps_spin.setSingleStep(1000)
        self.steps_spin.valueChanged.connect(self._update_details)
        selector_row.addWidget(self.steps_spin)
        selector_row.addStretch(1)
        layout.addLayout(selector_row)

        self.details_label = QtWidgets.QLabel()
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        button_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run Preset")
        self.run_button.clicked.connect(self._start_run)
        self.viewer_button = QtWidgets.QPushButton("Open Viewer")
        self.viewer_button.clicked.connect(self._open_viewer)
        self.stop_button = QtWidgets.QPushButton("Stop Run")
        self.stop_button.clicked.connect(self._stop_run)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.viewer_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.status_label = QtWidgets.QLabel("Idle")
        layout.addWidget(self.status_label)

        self.command_label = QtWidgets.QLabel("")
        self.command_label.setWordWrap(True)
        self.command_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.command_label)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

    def _connect_processes(self) -> None:
        self.run_process.readyReadStandardOutput.connect(self._drain_run_output)
        self.run_process.readyReadStandardError.connect(self._drain_run_output)
        self.run_process.finished.connect(self._run_finished)

    def _selected_profile(self) -> DesktopProfileConfig:
        return find_profile(self.launcher_config, self.profile_selector.currentText())

    def _selected_duration(self) -> DesktopDurationConfig:
        return find_duration(self.launcher_config, self.duration_selector.currentText())

    def _sync_steps_from_duration(self) -> None:
        self.steps_spin.blockSignals(True)
        self.steps_spin.setValue(self._selected_duration().n_steps)
        self.steps_spin.blockSignals(False)
        self._update_details()

    def _update_details(self) -> None:
        profile = self._selected_profile()
        duration = self._selected_duration()
        n_steps = self.steps_spin.value() or duration.n_steps
        self.details_label.setText(
            f"{profile.title}: {profile.notes} Duration preset {duration.title}. "
            f"Current steps {n_steps:,}. Reference wall-clock {duration.est_wallclock} at {duration.n_steps:,} steps."
        )

    def _append_log(self, text: str) -> None:
        if text:
            self.log.appendPlainText(text.rstrip("\n"))

    def _start_run(self) -> None:
        if self.run_process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.information(self, "Run Active", "A run is already active.")
            return
        profile = self._selected_profile()
        n_steps = self.steps_spin.value()
        run_name = build_run_name(self.launcher_config, profile.key, n_steps)
        config_path, _output_dir, live_dir = write_launcher_config(
            self.root_dir,
            self.launcher_config,
            profile,
            n_steps=n_steps,
            run_name=run_name,
        )
        command = build_run_command(self.root_dir, self.python_exe, config_path, live_dir)
        self.generated_live_dir = live_dir
        self.command_label.setText(" ".join(command))
        self.status_label.setText(f"Running {profile.title} at {n_steps:,} steps")
        self._append_log(f"\n=== starting {profile.title} at {n_steps:,} steps ===")
        self.run_process.setWorkingDirectory(str(self.root_dir))
        self.run_process.setProgram(command[0])
        self.run_process.setArguments(command[1:])
        self.run_process.start()

    def _drain_run_output(self) -> None:
        stdout = bytes(self.run_process.readAllStandardOutput()).decode("utf-8", "replace")
        stderr = bytes(self.run_process.readAllStandardError()).decode("utf-8", "replace")
        self._append_log(stdout)
        self._append_log(stderr)

    def _run_finished(self, exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        self._drain_run_output()
        self.status_label.setText(f"Run exited with code {exit_code}")

    def _open_viewer(self) -> None:
        if self.generated_live_dir is None:
            QtWidgets.QMessageBox.information(self, "No Live Dir", "Start a preset first.")
            return
        command = build_viewer_command(self.root_dir, self.python_exe, self.generated_live_dir)
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(self.root_dir))
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.startDetached(command[0], command[1:], str(self.root_dir))
        self.viewer_processes.append(process)

    def _stop_run(self) -> None:
        if self.run_process.state() == QtCore.QProcess.NotRunning:
            return
        self.status_label.setText("Stopping run...")
        self.run_process.terminate()


def main() -> int:
    """Launch the Qt desktop app."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlasmaDesktopWindow()
    window.show()
    return app.exec()


__all__ = [
    "PlasmaDesktopWindow",
    "build_run_command",
    "build_viewer_command",
    "default_python_executable",
    "main",
    "project_root",
    "write_launcher_config",
]
