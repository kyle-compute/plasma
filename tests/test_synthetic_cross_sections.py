"""Tests for synthetic (analytical) cross-section tables."""

import numpy as np

from plasma.data.synthetic_cross_sections import (
    constant_cross_section,
    electron_ar_elastic,
    electron_ar_excitation,
    electron_ar_ionization,
    ion_ar_charge_exchange,
)


class TestSyntheticCrossSections:
    def test_constant_returns_flat(self):
        cs = constant_cross_section(1e-20, "test")
        vals = cs(np.array([1.0, 10.0, 100.0]))
        np.testing.assert_allclose(vals, 1e-20, rtol=0.01)

    def test_elastic_is_callable(self):
        cs = electron_ar_elastic()
        vals = cs(np.array([1.0, 10.0, 100.0]))
        assert vals.shape == (3,)
        assert np.all(vals > 0)

    def test_elastic_order_of_magnitude(self):
        cs = electron_ar_elastic()
        val_10ev = cs(np.array([10.0]))[0]
        assert 1e-21 < val_10ev < 1e-18

    def test_excitation_zero_below_threshold(self):
        cs = electron_ar_excitation()
        val_low = cs(np.array([5.0, 10.0, 11.0]))[0]
        # Below 11.55 eV threshold → 0 (CrossSectionTable returns 0 below e_min)
        assert val_low == 0.0

    def test_excitation_nonzero_above_threshold(self):
        cs = electron_ar_excitation()
        val = cs(np.array([15.0]))[0]
        assert val > 1e-22

    def test_ionization_zero_below_threshold(self):
        cs = electron_ar_ionization()
        val = cs(np.array([10.0]))[0]
        assert val == 0.0

    def test_ionization_peak_order(self):
        cs = electron_ar_ionization()
        val = cs(np.array([80.0]))[0]
        assert 1e-21 < val < 1e-18

    def test_charge_exchange_large(self):
        cs = ion_ar_charge_exchange()
        val = cs(np.array([1.0]))[0]
        # CX cross-section is large (~4e-19 m^2)
        assert val > 1e-19

    def test_charge_exchange_decreases(self):
        cs = ion_ar_charge_exchange()
        low = cs(np.array([1.0]))[0]
        high = cs(np.array([1000.0]))[0]
        assert high < low
