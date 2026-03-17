"""Tests for the 0D Ionization Region Model."""

import numpy as np
import pytest

from plasma.core.config import load_config
from plasma.global_model.irm import IRM
from plasma.global_model.rate_equations import N_STATES, STATE_INDICES


@pytest.fixture
def irm():
    cfg = load_config("config/hipims_cu_ar.yaml")
    return IRM(cfg)


class TestIRMGeometry:
    def test_geometry_from_config(self, irm):
        g = irm.geom
        assert g.volume > 0
        assert g.area_target > 0
        assert g.area_loss > 0
        # Volume should be area * height
        assert g.volume == pytest.approx(g.area_target * g.z_ir, rel=0.01)

    def test_annular_area(self, irm):
        """Target erosion area = pi * (r_out^2 - r_in^2)."""
        g = irm.geom
        expected = np.pi * (g.r_outer**2 - g.r_inner**2)
        assert g.area_target == pytest.approx(expected, rel=1e-6)


class TestIRMInitialState:
    def test_state_vector_size(self, irm):
        y0 = irm.initial_state()
        assert len(y0) == N_STATES

    def test_quasineutrality(self, irm):
        """Initial state should satisfy n_e = sum(Z_i * n_i)."""
        y0 = irm.initial_state()
        n_e = y0[STATE_INDICES["e_cold"]] + y0[STATE_INDICES["e_hot"]]
        n_charge = (
            y0[STATE_INDICES["Ar+"]]
            + 2.0 * y0[STATE_INDICES["Ar2+"]]
            + y0[STATE_INDICES["Cu+"]]
            + 2.0 * y0[STATE_INDICES["Cu2+"]]
        )
        assert n_e == pytest.approx(n_charge, rel=1e-6)

    def test_positive_densities(self, irm):
        y0 = irm.initial_state()
        assert all(y0 >= 0)

    def test_positive_energy(self, irm):
        y0 = irm.initial_state()
        assert y0[STATE_INDICES["energy_cold"]] > 0
        assert y0[STATE_INDICES["energy_hot"]] > 0

    def test_hot_electron_seed_is_present(self, irm):
        y0 = irm.initial_state()
        assert y0[STATE_INDICES["e_hot"]] > 0
        assert y0[STATE_INDICES["current_circuit"]] >= 0.0


class TestIRMRHS:
    def test_rhs_returns_correct_size(self, irm):
        y0 = irm.initial_state()
        dydt = irm.rhs(0.0, y0)
        assert len(dydt) == N_STATES

    def test_rhs_finite(self, irm):
        """RHS should not produce NaN or Inf."""
        y0 = irm.initial_state()
        dydt = irm.rhs(0.0, y0)
        assert np.all(np.isfinite(dydt))

    def test_rhs_at_different_times(self, irm):
        """RHS should work during pulse and afterglow."""
        y0 = irm.initial_state()
        dydt_pulse = irm.rhs(20e-6, y0)  # During 40us pulse
        dydt_after = irm.rhs(100e-6, y0)  # During afterglow
        assert np.all(np.isfinite(dydt_pulse))
        assert np.all(np.isfinite(dydt_after))


class TestIRMSimulation:
    def test_simulation_completes(self, irm):
        """Full simulation should run without error."""
        result = irm.run()
        assert len(result.time) > 10
        assert result.time[-1] >= 200e-6  # Should reach at least 200 us

    def test_density_builds_during_pulse(self, irm):
        """Electron density should increase during the pulse."""
        result = irm.run()
        n_e_start = result.n_e[0]
        # Find density at roughly mid-pulse (~20 us)
        idx_mid = np.argmin(np.abs(result.time - 20e-6))
        n_e_mid = result.n_e[idx_mid]
        assert n_e_mid > n_e_start * 10  # Should grow significantly

    def test_cu_ions_produced(self, irm):
        """Cu+ should build up from sputtering and ionization."""
        result = irm.run()
        peak_cu = result.density("Cu+").max()
        assert peak_cu > 1e14  # Should produce meaningful Cu+

    def test_ar_depleted_during_pulse(self, irm):
        """Ar neutral density should decrease during pulse (gas rarefaction)."""
        result = irm.run()
        n_ar_start = result.density("Ar_c")[0]
        idx_end_pulse = np.argmin(np.abs(result.time - 40e-6))
        n_ar_pulse_end = result.density("Ar_c")[idx_end_pulse]
        assert n_ar_pulse_end < n_ar_start  # Gas rarefaction

    def test_hot_population_grows_during_pulse(self, irm):
        result = irm.run()
        peak_hot = result.density("e_hot").max()
        assert peak_hot > result.density("e_hot")[0]

    def test_circuit_current_builds_during_pulse(self, irm):
        result = irm.run()
        peak_current = result.current_a.max()
        assert peak_current > result.current_a[0]
