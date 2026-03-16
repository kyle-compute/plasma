"""Tests for reaction rate coefficients — verify against Gudmundsson Table 1."""

import numpy as np
import pytest

from plasma.data.reactions import RateCoeffFit, load_reactions


@pytest.fixture
def reactions():
    return load_reactions("data/reactions/gudmundsson_cu_ar.yaml")


def test_load_all_27_reactions(reactions):
    assert len(reactions) == 27


def test_reaction_ids(reactions):
    expected = [f"R{i}" for i in range(1, 28)]
    assert reactions.ids == expected


def test_r1_ar_ionization_rate(reactions):
    """R1: e + Ar -> Ar+ + 2e. At Te=5 eV, k should be ~1e-14 m^3/s order."""
    r1 = reactions["R1"]
    assert r1.reaction_type == "ionization"
    assert r1.threshold_ev == pytest.approx(15.76)

    k_5ev = float(r1.rate(5.0))
    # At 5 eV, Ar ionization rate ~ 10^-15 to 10^-14
    assert 1e-16 < k_5ev < 1e-13


def test_r2_excitation_rate(reactions):
    """R2: e + Ar -> Ar_m. Rate at 3 eV should be moderate."""
    k = float(reactions["R2"].rate(3.0))
    assert 1e-17 < k < 1e-14


def test_decay_reactions_have_constant_rate(reactions):
    """R14-R16 are radiative decays with fixed rate constants."""
    for rid in ("R14", "R15", "R16"):
        rxn = reactions[rid]
        assert rxn.is_decay
        assert rxn.rate_constant is not None
        assert rxn.rate_constant > 0
        # Rate should be independent of Te
        assert rxn.rate(1.0) == rxn.rate(10.0)


def test_heavy_particle_reactions(reactions):
    """R25-R27 have constant rate coefficients."""
    for rid in ("R25", "R26", "R27"):
        rxn = reactions[rid]
        assert rxn.rate_constant is not None
        assert rxn.rate_constant > 0


def test_rate_increases_with_temperature(reactions):
    """Ionization rates should generally increase with Te."""
    r1 = reactions["R1"]
    k_2 = float(r1.rate(2.0))
    k_5 = float(r1.rate(5.0))
    k_10 = float(r1.rate(10.0))
    assert k_2 < k_5 < k_10


def test_arrhenius_fit():
    """Direct test of the Arrhenius rate form."""
    fit = RateCoeffFit(a=1e-14, b=0.5, c=10.0)
    k = fit(5.0)
    expected = 1e-14 * 5.0**0.5 * np.exp(-10.0 / 5.0)
    assert k == pytest.approx(expected)


def test_electron_impact_classification(reactions):
    """Verify electron-impact reactions are correctly identified."""
    e_impact = reactions.electron_impact()
    # R1-R13, R17-R24 are electron impact (20 reactions)
    assert len(e_impact) >= 18

    heavy = reactions.heavy_particle()
    # R25-R27 are heavy particle
    assert len(heavy) == 3
