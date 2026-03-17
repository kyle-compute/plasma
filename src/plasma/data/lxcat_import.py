"""Normalize LXCat archives into explicit cross-section tables."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from plasma.data.cross_sections import CrossSectionTable
from plasma.data.lxcat_parser import parse_lxcat_text


class LXCatNormalizedFile(BaseModel):
    name: str
    process_names: list[str] = Field(default_factory=list)
    output_path: str


class LXCatImportManifest(BaseModel):
    archive_path: str
    text_member: str
    generated_on: str | None = None
    database: str | None = None
    permlink: str | None = None
    recommended_reference: str | None = None
    normalized_files: list[LXCatNormalizedFile] = Field(default_factory=list)


def normalize_biagi_argon_archive(
    archive_path: str | Path,
    output_dir: str | Path,
) -> LXCatImportManifest:
    """Normalize the downloaded Biagi e-Ar archive into reusable tables."""

    archive = Path(archive_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    text_member, text = _read_lxcat_text(archive)
    sections = parse_lxcat_text(text)
    metadata = _extract_metadata(text)

    elastic_name = _match_single(sections, suffix=", Elastic")
    ionization_name = _match_single(sections, suffix=", Ionization")
    excitation_names = sorted(name for name in sections if name.endswith(", Excitation"))
    if not excitation_names:
        raise ValueError("No excitation sections found in LXCat archive")

    outputs = [
        _write_table(target_dir / "electron_ar_elastic.tsv", [elastic_name], sections),
        _write_table(target_dir / "electron_ar_excitation_total.tsv", excitation_names, sections),
        _write_table(target_dir / "electron_ar_ionization.tsv", [ionization_name], sections),
    ]

    manifest = LXCatImportManifest(
        archive_path=str(archive.resolve()),
        text_member=text_member,
        generated_on=metadata.get("generated_on"),
        database=metadata.get("database"),
        permlink=metadata.get("permlink"),
        recommended_reference=metadata.get("recommended_reference"),
        normalized_files=outputs,
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def _read_lxcat_text(path: Path) -> tuple[str, str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            text_members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if not text_members:
                raise ValueError(f"No .txt member found in LXCat archive: {path}")
            text_member = text_members[0]
            return text_member, archive.read(text_member).decode("utf-8", "ignore")
    return path.name, path.read_text()


def _extract_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    patterns = {
        "generated_on": r"Generated on\s+([^\n]+)",
        "database": r"DATABASE:\s+([^\n]+)",
        "permlink": r"PERMLINK:\s+([^\n]+)",
        "recommended_reference": r"-\s+([^\n]*retrieved on[^\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metadata[key] = match.group(1).strip()
    return metadata


def _match_single(sections: dict[str, tuple[np.ndarray, np.ndarray]], *, suffix: str) -> str:
    matches = [name for name in sections if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match for suffix {suffix!r}, got {matches}")
    return matches[0]


def _write_table(
    path: Path,
    process_names: list[str],
    sections: dict[str, tuple[np.ndarray, np.ndarray]],
) -> LXCatNormalizedFile:
    tables = [CrossSectionTable(*sections[name], name=name) for name in process_names]
    energy_grid = np.unique(np.concatenate([table.energy_ev for table in tables]))
    sigma_total = np.zeros_like(energy_grid)
    for table in tables:
        sigma_total += table(energy_grid)
    content = [
        "# energy_eV sigma_m2",
        *[f"{e:.8e} {s:.8e}" for e, s in zip(energy_grid, sigma_total, strict=False)],
    ]
    path.write_text("\n".join(content) + "\n")
    return LXCatNormalizedFile(
        name=path.stem,
        process_names=process_names,
        output_path=str(path.resolve()),
    )
