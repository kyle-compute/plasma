#!/usr/bin/env python3
"""Run the 0D Ionization Region Model for a given configuration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plasma.core.runtime_env import ensure_runtime_environment

ensure_runtime_environment(ROOT)

import matplotlib.pyplot as plt
import numpy as np

from plasma.core.config import load_config
from plasma.diagnostics.bundles import bundle_from_irm_state
from plasma.global_model.irm import IRM, IRMState
from plasma.global_model.rate_equations import STATE_INDICES
from plasma.io.reports import save_json_model
from plasma.live.builders import build_global_live_snapshot
from plasma.live.publisher import FileLiveSession
from plasma.reporting import build_run_manifest, build_validation_report


def plot_results(result: IRMState, output_dir: Path, title: str) -> tuple[Path, Path | None]:
    """Plot key outputs and, when available, a benchmark overlay."""

    t_us = result.time * 1e6
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].semilogy(t_us, result.n_e)
    axes[0, 0].set(xlabel="Time [us]", ylabel="n_e [m^-3]", title="Electron Density")

    axes[0, 1].plot(t_us, result.te_ev)
    axes[0, 1].set(xlabel="Time [us]", ylabel="T_e [eV]", title="Electron Temperature")

    axes[1, 0].plot(t_us, result.metric("voltage_v"), label="Voltage")
    axes[1, 0].plot(t_us, result.metric("current_a"), label="Driven current")
    if "reference_current_a" in result.diagnostics.series:
        axes[1, 0].plot(t_us, result.metric("reference_current_a"), label="Reference current", linestyle="--")
    axes[1, 0].plot(t_us, result.metric("model_current_proxy_a"), label="Model current proxy")
    axes[1, 0].set(xlabel="Time [us]", ylabel="Waveform / Current", title="Pulse Inputs")
    axes[1, 0].legend()

    axes[1, 1].plot(t_us, result.metric("alpha_t"), label="alpha_t")
    axes[1, 1].plot(t_us, result.metric("beta_t"), label="beta_t")
    axes[1, 1].plot(t_us, result.metric("xi_t"), label="xi_t")
    axes[1, 1].set(xlabel="Time [us]", ylabel="Probability / Fraction", title="Transport Metrics")
    axes[1, 1].legend()

    for ion in ("Ar+", "Cu+", "Ar2+", "Cu2+"):
        if ion in STATE_INDICES:
            axes[2, 0].semilogy(t_us, np.maximum(result.density(ion), 1.0), label=ion)
    axes[2, 0].set(xlabel="Time [us]", ylabel="n_ion [m^-3]", title="Ion Densities")
    axes[2, 0].legend()

    axes[2, 1].plot(t_us, result.metric("deposition_flux_m2s"), label="Deposition flux")
    axes[2, 1].plot(t_us, result.metric("epsilon_ti_ev"), label="epsilon_ti")
    if "reference_current_a" in result.diagnostics.series:
        current_error = result.metric("model_current_proxy_a") - result.metric("reference_current_a")
        axes[2, 1].plot(t_us, current_error, label="current proxy error")
    axes[2, 1].set(xlabel="Time [us]", ylabel="Derived output", title="Usability Metrics")
    axes[2, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    benchmark_path = None

    if "reference_current_a" in result.diagnostics.series:
        benchmark_path = output_dir / "benchmark_overlay.png"
        overlay_fig, overlay_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        overlay_fig.suptitle(f"{title} Benchmark Overlay", fontsize=14)

        overlay_axes[0].plot(t_us, result.metric("reference_current_a"), label="Reference current", linewidth=2.0)
        overlay_axes[0].plot(t_us, result.metric("model_current_proxy_a"), label="Model current proxy")
        overlay_axes[0].set(ylabel="Current [A]", title="Current Calibration")
        overlay_axes[0].legend()

        current_error = result.metric("model_current_proxy_a") - result.metric("reference_current_a")
        overlay_axes[1].plot(t_us, current_error, label="Proxy - reference")
        overlay_axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        overlay_axes[1].set(xlabel="Time [us]", ylabel="Error [A]", title="Current Error")
        overlay_axes[1].legend()

        for ax in overlay_axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        overlay_fig.savefig(benchmark_path, dpi=150)
        plt.close(overlay_fig)
    return plot_path, benchmark_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 0D IRM simulation")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--live-dir", help="Directory used to publish live-view snapshots")
    parser.add_argument(
        "--live-delay-ms",
        type=int,
        default=40,
        help="Delay between replayed 0D live snapshots in milliseconds",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_output = cfg.output.dir if cfg.output is not None else f"output/global/{cfg.name}"
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    irm = IRM(cfg)
    result = irm.run()
    bundle = bundle_from_irm_state(result)
    report = build_validation_report(cfg, bundle)
    manifest = build_run_manifest(cfg, bundle, report)

    plot_path = None
    benchmark_plot_path = None
    if not args.no_plot:
        plot_path, benchmark_plot_path = plot_results(result, output_dir, title=cfg.name)

    bundle_path = output_dir / "diagnostics_bundle.json"
    report_path = output_dir / "validation_report.json"
    manifest_path = output_dir / "run_manifest.json"
    save_json_model(bundle_path, bundle)
    save_json_model(report_path, report)
    manifest.outputs = {
        "diagnostics_bundle": str(bundle_path),
        "validation_report": str(report_path),
    }
    if plot_path is not None:
        manifest.outputs["summary_plot"] = str(plot_path)
    if benchmark_plot_path is not None:
        manifest.outputs["benchmark_plot"] = str(benchmark_plot_path)
    save_json_model(manifest_path, manifest)
    if args.live_dir:
        session = FileLiveSession(args.live_dir)
        stride = max(len(result.time) // 200, 1)
        for end_index in range(stride, len(result.time) + 1, stride):
            session.publish(
                build_global_live_snapshot(
                    cfg.name,
                    result,
                    end_index=end_index,
                    state="running",
                    message="0D replay mode: the IRM is integrated in batch, then streamed to the viewer.",
                )
            )
            time.sleep(max(args.live_delay_ms, 0) / 1000.0)
        if (len(result.time) - 1) % stride != 0:
            session.publish(
                build_global_live_snapshot(
                    cfg.name,
                    result,
                    end_index=len(result.time),
                    state="running",
                    message="0D replay mode: the IRM is integrated in batch, then streamed to the viewer.",
                )
            )
        session.publish(
            build_global_live_snapshot(
                cfg.name,
                result,
                end_index=len(result.time),
                state="completed",
                message="0D replay complete.",
            )
        )

    print(f"Running IRM: {cfg.name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Final n_e = {result.n_e[-1]:.2e} m^-3")
    print(f"  Peak current = {bundle.summary['peak_current_a']:.2f} A")
    if "current_proxy_rmse_a" in bundle.summary:
        print(f"  Current proxy RMSE = {bundle.summary['current_proxy_rmse_a']:.2f} A")
    print(f"  Peak alpha_t = {bundle.summary['peak_alpha_t']:.3f}")
    print(f"  Validation status = {manifest.validation_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
