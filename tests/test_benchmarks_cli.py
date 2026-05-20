"""
Tests for the benchmark CLI entry point.

The CLI is a thin wrapper around ``benchmarks.runner.run`` plus
``benchmarks.report.render_markdown``; tests verify the wrapper
forwards arguments correctly and writes the expected output files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _has_torch_topological() -> bool:
    try:
        import torch_topological  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestCLI:
    def test_cli_writes_json_and_markdown(self, tmp_path: Path) -> None:
        from benchmarks.cli import main

        out_json = tmp_path / "result.json"
        out_md = tmp_path / "report.md"
        rc = main([
            "--backends", "topogeoml-diff-ph",
            "--datasets", "gaussian_blob",
            "--axes", "correctness",
            "--output", str(out_json),
            "--markdown", str(out_md),
        ])
        assert rc == 0
        assert out_json.exists()
        assert out_md.exists()

        # Validate JSON schema fingerprint.
        payload = json.loads(out_json.read_text())
        assert payload["provenance"]["schema_version"] == "1.0.0"
        assert payload["config"]["axis_names"] == ["correctness"]
        assert payload["config"]["backend_names"] == ["topogeoml-diff-ph"]

        # Validate markdown contains the correctness section.
        md = out_md.read_text()
        assert "Correctness" in md
        assert "Provenance" in md

    def test_cli_nonzero_exit_on_failed_cell(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a registered axis raises, the runner records it as failed and the CLI exits non-zero."""
        from benchmarks import runner
        from benchmarks.cli import main

        def _boom(backend, dataset):  # type: ignore[no-untyped-def]
            raise RuntimeError("synthetic failure for test")

        # Override an existing axis so we can drive the CLI through its
        # argparse `choices=` gate. ``correctness`` is the cheapest axis name
        # to stand in for our failing function.
        monkeypatch.setitem(runner.AXES, "correctness", _boom)

        out_json = tmp_path / "result.json"
        rc = main([
            "--backends", "topogeoml-diff-ph",
            "--datasets", "gaussian_blob",
            "--axes", "correctness",
            "--output", str(out_json),
        ])
        assert rc == 1

        payload = json.loads(out_json.read_text())
        failed = [c for c in payload["cells"] if not c["success"]]
        assert len(failed) == 1
        assert failed[0]["error_kind"] == "RuntimeError"
        assert "synthetic failure" in failed[0]["error_message"]
