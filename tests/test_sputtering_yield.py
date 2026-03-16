"""Tests for Yamamura sputtering yield model."""

import numpy as np
import pytest

from plasma.data.sputtering import SputterYield, load_sputter_yields


@pytest.fixture
def ar_cu_yield():
    return SputterYield(
        ion="Ar+", target="Cu",
        a=0.1421, b=0.468,
        threshold_ev=17.0,
        cohesive_energy_ev=3.49,
    )


def test_yield_above_threshold(ar_cu_yield):
    """Y(300 eV) for Ar+ -> Cu should be ~1-2 atoms/ion."""
    y = ar_cu_yield(300.0).item()
    assert 0.5 < y < 5.0


def test_yield_below_threshold(ar_cu_yield):
    """Below sputter threshold, yield is zero."""
    y = ar_cu_yield(10.0).item()
    assert y == 0.0


def test_yield_increases_with_energy(ar_cu_yield):
    """Higher ion energy -> higher sputter yield."""
    y_100 = ar_cu_yield(100.0).item()
    y_500 = ar_cu_yield(500.0).item()
    y_1000 = ar_cu_yield(1000.0).item()
    assert y_100 < y_500 < y_1000


def test_yield_at_600ev(ar_cu_yield):
    """At 600 eV (typical HiPIMS), Y ~ 2-3 for Ar+ -> Cu."""
    y = ar_cu_yield(600.0).item()
    assert 1.0 < y < 5.0


def test_array_input(ar_cu_yield):
    energies = np.array([10, 50, 100, 300, 600])
    yields = ar_cu_yield(energies)
    assert yields.shape == (5,)
    assert yields[0] == 0.0  # below threshold
    assert all(yields[1:] > 0)


def test_load_from_yaml():
    yields = load_sputter_yields("data/sputtering/yamamura_yields.yaml")
    assert "Ar_Cu" in yields
    assert "Cu_Cu" in yields
    assert "Ar_Ti" in yields

    ar_cu = yields["Ar_Cu"]
    assert ar_cu.a == pytest.approx(0.1421)
    assert ar_cu.b == pytest.approx(0.468)


def test_self_sputtering_lower_yield():
    """Cu+ -> Cu self-sputtering should have lower yield than Ar+ -> Cu."""
    yields = load_sputter_yields("data/sputtering/yamamura_yields.yaml")
    y_ar = yields["Ar_Cu"](500.0).item()
    y_self = yields["Cu_Cu"](500.0).item()
    assert y_self < y_ar


def test_sputtered_energy_peak(ar_cu_yield):
    """Peak energy of sputtered atoms ≈ E_cohesive / 2."""
    e_peak = ar_cu_yield.sputtered_energy_peak()
    assert e_peak == pytest.approx(3.49 / 2.0)
