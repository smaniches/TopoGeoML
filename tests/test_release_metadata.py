"""Cross-file contracts for release metadata and public version surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from topogeoml._version import __version__

ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _related_identifier(zenodo: dict[str, Any], relation: str) -> str:
    matches = [
        item["identifier"]
        for item in zenodo["related_identifiers"]
        if item["relation"] == relation
    ]
    assert len(matches) == 1, f"expected one {relation!r} identifier"
    return str(matches[0])


def test_structured_release_metadata_agrees() -> None:
    citation = yaml.safe_load(_text("CITATION.cff"))
    zenodo = json.loads(_text(".zenodo.json"))

    assert citation["version"] == __version__
    assert zenodo["version"] == __version__
    assert citation["doi"] == _related_identifier(zenodo, "isVersionOf")
    # A previous-version DOI is optional when the exact predecessor cannot
    # be independently verified. Never reuse the known 0.0.8 record for 0.0.10.
    prior = [
        item["identifier"]
        for item in zenodo["related_identifiers"]
        if item["relation"] == "isNewVersionOf"
    ]
    assert len(prior) <= 1
    if prior:
        assert prior[0] not in {citation["doi"], "10.5281/zenodo.23291234"}

    release_date = str(citation["date-released"])
    assert f"## [{__version__}] — {release_date}" in _text("CHANGELOG.md")


def test_public_version_surfaces_agree() -> None:
    surfaces = {
        "README.md": (
            f"version-{__version__}--beta",
            f"TopoGeoML {__version__} is beta scientific software",
            f"version = {{{__version__}}}",
        ),
        "LIMITATIONS.md": (
            f"Limitations of TopoGeoML v{__version__}",
            f"in v{__version__}",
            f"`{__version__}` is pre-stable",
        ),
        "STATUS.md": (f"TopoGeoML {__version__} is beta research software",),
        "docs/index.md": (f"version = {{{__version__}}}",),
        "topogeoml/__init__.py": (f"v{__version__} public surface",),
        "tests/test_feature_pipeline.py": (
            f'prov.pipeline_version == "{__version__}"',
        ),
    }

    for path, markers in surfaces.items():
        content = _text(path)
        for marker in markers:
            assert marker in content, f"{path} missing release marker {marker!r}"


def test_optional_filelock_security_floor() -> None:
    """Both PyTorch extras must reject the known vulnerable FileLock range."""
    import tomllib

    metadata = tomllib.loads(_text("pyproject.toml"))
    extras = metadata["project"]["optional-dependencies"]
    for extra in ("torch", "bench"):
        assert "filelock>=4.2.1" in extras[extra]
        assert "filelock==3.20.3" not in extras[extra]
