from __future__ import annotations

import numpy as np

from plasma.diagnostics.collectors import SubstrateCollector
from plasma.live.pic_fields import emissivity_proxy, event_map
from plasma.live.pic_window import clear_event_clouds, event_counts, merge_event_clouds
from plasma.pic.grid import CylindricalGrid
from plasma.pic.particles import ParticleArray, Species


def test_event_window_merges_and_clears() -> None:
    window: dict[str, dict[str, np.ndarray]] = {}
    merge_event_clouds(window, {"e_Ar_excitation": {"r": np.array([0.001]), "z": np.array([0.002])}})
    merge_event_clouds(window, {"e_Ar_excitation": {"r": np.array([0.003]), "z": np.array([0.004])}})

    assert event_counts(window)["e_Ar_excitation"] == 2
    clear_event_clouds(window)
    assert window == {}


def test_event_map_and_emissivity_proxy_from_events() -> None:
    grid = CylindricalGrid(8, 10, 0.02, 0.04)
    window = {
        "e_Ar_excitation": {"r": np.array([0.001, 0.0015]), "z": np.array([0.003, 0.004])},
        "e_Ar_ionization": {"r": np.array([0.0020]), "z": np.array([0.0050])},
    }

    excitation = event_map(grid, window, ("e_Ar_excitation",))
    ionization = event_map(grid, window, ("e_Ar_ionization",))
    glow = emissivity_proxy(None, None, excitation, ionization, np.zeros_like(excitation))

    assert np.max(excitation) > 0.0
    assert np.max(ionization) > 0.0
    assert np.max(glow) == 1.0


def test_substrate_flux_proxy_projects_collector_to_substrate_band() -> None:
    from plasma.live.pic_fields import substrate_flux_proxy

    grid = CylindricalGrid(8, 10, 0.02, 0.04)
    ion = ParticleArray(Species("Ar+", 1.0, 6.63e-26, 1))
    ion.allocate(8)
    ion.add_particles(
        r=np.array([0.004, 0.008]),
        z=np.array([0.0395, 0.0395]),
        vr=np.zeros(2),
        vz=np.array([8.0e3, 8.0e3]),
        vtheta=np.zeros(2),
        weight=np.ones(2),
    )
    collector = SubstrateCollector(z_plane=0.04, dz_capture=1e-3)
    collector.record_absorbed(ion, t=1e-6)

    flux = substrate_flux_proxy(grid, collector, band_nodes=3)

    assert np.max(flux) == 1.0
    assert np.any(flux[:, -3:] > 0.0)
