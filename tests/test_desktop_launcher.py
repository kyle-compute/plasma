"""Tests for the YAML-driven desktop launcher helpers."""

from __future__ import annotations

from plasma.desktop_config import load_desktop_config
from plasma.desktop_runtime import (
    build_run_command,
    build_run_name,
    default_python_executable,
    find_profile,
    write_launcher_config,
)


def test_load_desktop_config_from_repo_defaults() -> None:
    config = load_desktop_config("config/desktop/launcher.yaml")

    assert config.default_profile == "monitor"
    assert config.default_duration == "test_36m"
    assert [profile.key for profile in config.profiles] == ["quicklook", "monitor"]
    assert [duration.key for duration in config.durations] == ["smoke", "monitor", "test_36m", "overnight"]


def test_write_launcher_config_for_customized_run(tmp_path) -> None:
    root = tmp_path
    base_config = root / "config" / "hipims_cu_ar_pic_monitor.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text("name: base\nmodel: pic\n")

    config_path = root / "config" / "desktop" / "launcher.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                'default_profile: "monitor"',
                'default_duration: "test_36m"',
                "generation:",
                '  runtime_dir: "runtime/generated"',
                '  output_root: "runs/output"',
                '  live_root: "runs/live"',
                '  run_name_template: "launch_{profile}_{steps}"',
                "  diag_divisor: 50",
                "  diag_min: 10",
                "  compact_divisor: 20",
                "  compact_min: 25",
                "  checkpoint_interval: 7",
                "profiles:",
                '  - key: "monitor"',
                '    title: "Monitor"',
                '    base_config: "../hipims_cu_ar_pic_monitor.yaml"',
                '    notes: "monitor profile"',
                "durations:",
                '  - key: "test_36m"',
                '    title: "36 Min Test"',
                "    n_steps: 20000",
                '    est_wallclock: "~36 min"',
            ]
        )
        + "\n"
    )

    launcher_config = load_desktop_config(config_path)
    profile = find_profile(launcher_config, "monitor")
    run_name = build_run_name(launcher_config, profile.key, 20000)
    generated_path, output_dir, live_dir = write_launcher_config(
        root,
        launcher_config,
        profile,
        n_steps=20000,
        run_name=run_name,
    )

    payload = generated_path.read_text()
    assert "n_steps: 20000" in payload
    assert "diag_interval: 400" in payload
    assert "compact_interval: 1000" in payload
    assert "checkpoint_interval: 7" in payload
    assert str(output_dir) in payload
    assert output_dir.name == "launch_monitor_20000"
    assert live_dir.name == "launch_monitor_20000"


def test_build_run_command_uses_generated_config(tmp_path) -> None:
    root = tmp_path
    python_exe = default_python_executable(root)
    command = build_run_command(root, python_exe, tmp_path / "run.yaml", tmp_path / "live")

    assert command[0] == python_exe
    assert command[1].endswith("scripts/run_pic.py")
    assert command[-2] == "--live-dir"
    assert command[-1].endswith("live")
