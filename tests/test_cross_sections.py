"""Tests for cross-section data loading and interpolation."""


import numpy as np
import pytest

from plasma.data.cross_sections import CrossSectionTable


@pytest.fixture
def sample_table(tmp_path):
    """Create a simple test cross-section file."""
    tsv = tmp_path / "test_cs.tsv"
    tsv.write_text(
        "# Test cross-section\n"
        "1.0  1.0e-20\n"
        "10.0  5.0e-20\n"
        "100.0  2.0e-20\n"
        "1000.0  5.0e-21\n"
    )
    return CrossSectionTable.from_file(tsv, name="test")


def test_load_from_file(sample_table):
    assert sample_table.name == "test"
    assert len(sample_table.energy_ev) == 4
    assert sample_table.e_min == 1.0
    assert sample_table.e_max == 1000.0


def test_interpolation_within_range(sample_table):
    """Cross-section at intermediate energy should be between neighbors."""
    sigma = sample_table(5.0)
    assert sigma > 1.0e-20
    assert sigma < 5.0e-20


def test_below_threshold_returns_zero(sample_table):
    """Energy below data range gives zero cross-section."""
    sigma = sample_table(0.1)
    assert sigma == pytest.approx(0.0)


def test_array_input(sample_table):
    """Should handle array of energies."""
    energies = np.array([0.5, 5.0, 50.0, 500.0])
    sigma = sample_table(energies)
    assert sigma.shape == (4,)
    assert sigma[0] == 0.0  # below range
    assert all(sigma[1:] > 0)  # within range


def test_max_sigma(sample_table):
    assert sample_table.max_sigma == pytest.approx(5.0e-20)


def test_log_log_interpolation_accuracy():
    """Log-log interpolation should be exact for power-law data."""
    # sigma = 1e-20 * E^(-0.5) (inverse square root)
    energies = np.array([1.0, 10.0, 100.0, 1000.0])
    sigmas = 1e-20 * energies ** (-0.5)
    table = CrossSectionTable(energies, sigmas, "power_law")

    # Check at intermediate points
    e_test = 31.62  # sqrt(1000)
    sigma_exact = 1e-20 * e_test ** (-0.5)
    sigma_interp = table(e_test).item()
    assert sigma_interp == pytest.approx(sigma_exact, rel=1e-6)
