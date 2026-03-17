"""File-based live snapshot transport."""

from __future__ import annotations

import json
import os
from pathlib import Path

from plasma.live.contracts import LiveCommand, LiveSnapshot


class LiveCommandWriteError(RuntimeError):
    """Raised when a live control command cannot be written."""


class FileLiveSession:
    """Publish snapshots and exchange control commands through a directory."""

    def __init__(self, live_dir: str | Path):
        self.live_dir = Path(live_dir)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.live_dir / "latest.json"
        self.command_path = self.live_dir / "command.json"
        self._last_command_seq = 0

    def publish(self, snapshot: LiveSnapshot) -> None:
        """Write the latest live snapshot atomically."""
        _atomic_write(self.snapshot_path, snapshot.model_dump_json(indent=2) + "\n")

    def load_snapshot(self) -> LiveSnapshot | None:
        """Load the latest snapshot if present."""
        if not self.snapshot_path.exists():
            return None
        return LiveSnapshot.model_validate_json(self.snapshot_path.read_text())

    def clear_stale_commands(self) -> None:
        """Remove any leftover command file from a previous run."""
        if self.command_path.exists():
            self.command_path.unlink(missing_ok=True)
        self._last_command_seq = 0

    def send_command(self, command: str) -> LiveCommand:
        """Write a new control command for the simulation."""
        seq = self._next_command_seq()
        live_command = LiveCommand(seq=seq, command=command)
        try:
            _atomic_write(self.command_path, live_command.model_dump_json(indent=2) + "\n")
        except OSError as exc:
            raise LiveCommandWriteError(self._command_write_error_message()) from exc
        return live_command

    def command_write_error(self) -> str | None:
        """Return a concise reason why live controls are unavailable."""
        if os.access(self.live_dir, os.W_OK):
            return None
        return f"live controls unavailable: {self.live_dir} is not writable"

    def poll_command(self) -> LiveCommand | None:
        """Read the newest control command if it has not been consumed."""
        if not self.command_path.exists():
            return None
        try:
            command = LiveCommand.model_validate_json(self.command_path.read_text())
        except ValueError:
            return None
        if command.seq <= self._last_command_seq:
            return None
        self._last_command_seq = command.seq
        return command

    def _next_command_seq(self) -> int:
        if not self.command_path.exists():
            return 1
        try:
            payload = json.loads(self.command_path.read_text())
        except json.JSONDecodeError:
            return 1
        return int(payload.get("seq", 0)) + 1

    def _command_write_error_message(self) -> str:
        base_message = self.command_write_error()
        if base_message is None:
            base_message = f"failed to write live control command to {self.command_path}"
        return (
            f"{base_message}; if Docker created this run output, rerun the simulation with "
            "`docker compose run --user $(id -u):$(id -g)` or fix ownership with "
            f"`sudo chown -R $USER:$USER {self.live_dir}`"
        )


def _atomic_write(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)
