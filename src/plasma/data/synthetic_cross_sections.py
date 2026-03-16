"""Approximate analytical cross-sections for Ar when LXCat data is unavailable.

These are *not* research-grade cross-sections.  They reproduce the correct
order of magnitude, threshold behaviour, and peak locations for electron-Ar
and ion-Ar interactions so that the PIC-MCC loop can be exercised without
downloading external data files.

Shapes are loosely based on:
    - Ramsauer minimum for e-Ar elastic (~1 eV dip, ~5e-20 m^2 plateau)
    - BSR data for excitation (threshold 11.55 eV, peak ~3e-20 m^2)
    - BEB / Lotz for ionization (threshold 15.76 eV, peak ~2.7e-20 m^2 at 80-100 eV)
    - Charge-exchange: ~4e-19 m^2 slowly decreasing (Ar+ + Ar -> Ar + Ar+)
"""

from __future__ import annotations

import numpy as np

from plasma.data.cross_sections import CrossSectionTable


def constant_cross_section(sigma: float, name: str = "constant") -> CrossSectionTable:
    """Flat cross-section over a wide energy range."""
    e = np.array([0.01, 1e4])
    s = np.array([sigma, sigma])
    return CrossSectionTable(e, s, name=name)


def electron_ar_elastic() -> CrossSectionTable:
    """e + Ar elastic: Ramsauer minimum near 0.3 eV, plateau ~5e-20 m^2.

    Shape: dip at ~0.3 eV (Ramsauer-Townsend minimum) then flat plateau.
    """
    e = np.array([
        0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0,
        2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
        200.0, 500.0, 1000.0, 5000.0, 10000.0,
    ])
    # Ramsauer dip at ~0.3 eV, then plateau ~5e-20, slow decrease above 100 eV
    s = np.array([
        3e-20, 1.5e-20, 8e-21, 3e-21, 1e-21, 5e-21, 2e-20,
        4e-20, 6e-20, 7e-20, 6e-20, 4e-20, 2.5e-20,
        1.5e-20, 8e-21, 5e-21, 2e-21, 1e-21,
    ])
    return CrossSectionTable(e, s, name="e_Ar_elastic_synthetic")


def electron_ar_excitation() -> CrossSectionTable:
    """e + Ar -> Ar* + e: threshold 11.55 eV, peak ~3e-20 m^2 near 15 eV."""
    e = np.array([
        11.55, 11.6, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0,
        20.0, 25.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
    ])
    s = np.array([
        1e-25, 1e-22, 5e-21, 1.5e-20, 2.5e-20, 3e-20, 2.8e-20, 2.2e-20,
        1.8e-20, 1.2e-20, 9e-21, 5e-21, 2.5e-21, 1.2e-21, 5e-22, 2e-22,
    ])
    return CrossSectionTable(e, s, name="e_Ar_excitation_synthetic")


def electron_ar_ionization() -> CrossSectionTable:
    """e + Ar -> Ar+ + 2e: threshold 15.76 eV, peak ~2.7e-20 m^2 at ~80 eV."""
    e = np.array([
        15.76, 15.8, 16.0, 17.0, 18.0, 20.0, 25.0, 30.0,
        40.0, 50.0, 70.0, 80.0, 100.0, 150.0, 200.0,
        300.0, 500.0, 1000.0, 5000.0,
    ])
    s = np.array([
        1e-25, 1e-23, 5e-22, 3e-21, 6e-21, 1.1e-20, 1.8e-20, 2.2e-20,
        2.6e-20, 2.7e-20, 2.7e-20, 2.6e-20, 2.3e-20, 1.7e-20, 1.3e-20,
        9e-21, 6e-21, 3e-21, 8e-22,
    ])
    return CrossSectionTable(e, s, name="e_Ar_ionization_synthetic")


def ion_ar_charge_exchange() -> CrossSectionTable:
    """Ar+ + Ar -> Ar + Ar+: resonant CX, ~4e-19 m^2, slowly decreasing."""
    e = np.array([
        0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0,
        200.0, 500.0, 1000.0, 5000.0,
    ])
    s = np.array([
        5e-19, 4.8e-19, 4.5e-19, 4.0e-19, 3.8e-19, 3.0e-19, 2.5e-19,
        2.0e-19, 1.5e-19, 1.2e-19, 7e-20,
    ])
    return CrossSectionTable(e, s, name="ion_Ar_CX_synthetic")
