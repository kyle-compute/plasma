"""Tests for LXCat archive normalization."""

from __future__ import annotations

import json
from pathlib import Path

from plasma.data.cross_sections import CrossSectionTable
from plasma.data.lxcat_import import normalize_biagi_argon_archive


def test_normalize_biagi_archive(tmp_path) -> None:
    manifest = normalize_biagi_argon_archive(
        "data/cross_sections/LXCat_Biagi_e_Ar.zip",
        tmp_path / "normalized",
    )

    assert manifest.database is not None
    assert "Biagi" in manifest.database
    assert len(manifest.normalized_files) == 3

    manifest_path = tmp_path / "normalized" / "manifest.json"
    assert manifest_path.exists()
    loaded_manifest = json.loads(manifest_path.read_text())
    assert loaded_manifest["normalized_files"][0]["name"] == "electron_ar_elastic"

    elastic = CrossSectionTable.from_file(tmp_path / "normalized" / "electron_ar_elastic.tsv")
    excitation = CrossSectionTable.from_file(tmp_path / "normalized" / "electron_ar_excitation_total.tsv")
    ionization = CrossSectionTable.from_file(tmp_path / "normalized" / "electron_ar_ionization.tsv")

    assert elastic.e_min >= 0.0
    assert excitation.max_sigma > 0.0
    assert ionization.max_sigma > 0.0
