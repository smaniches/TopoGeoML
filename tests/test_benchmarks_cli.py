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
            "--datasets", "mnist_mock_digit_1",
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

    def test_cli_returns_zero_when_only_skipped_non_differentiable_cells_fail(
        self, tmp_path: Path,
    ) -> None:
        """Regression: ``SkippedNonDifferentiable`` cells are expected
        behaviour for non-differentiable backends (gudhi-python) on
        autograd-required axes (stability, speed, optimization). The
        CLI must return exit code 0 in this case, otherwise CI fails
        on every workflow run that includes such a backend.

        Caught when the first ``benchmark.yml`` workflow finished under
        ``--quick`` and the previous "any non-success → exit 1" check
        wrongly counted the 18 expected skips as failures.
        """
        from benchmarks.cli import main

        # gudhi-python is non-differentiable; running it on stability
        # (which requires autograd through loss_longest_h1) produces a
        # ``SkippedNonDifferentiable`` cell. No real failures.
        out_json = tmp_path / "result.json"
        rc = main([
            "--backends", "gudhi-python",
            "--datasets", "mnist_mock_digit_1",
            "--axes", "stability",
            "--output", str(out_json),
        ])
        assert rc == 0
        payload = json.loads(out_json.read_text())
        skipped = [
            c for c in payload["cells"]
            if c["error_kind"] == "SkippedNonDifferentiable"
        ]
        assert len(skipped) >= 1, "expected at least one SkippedNonDifferentiable cell"

    def test_cli_returns_zero_when_backend_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: ``UnavailableBackend`` cells are expected when an
        optional dependency is not installed (e.g. CI image lacks one
        of the optional bench backends). The runner records the absence
        so reports flag missing comparison points, but it is not a
        *bug* — the CLI must not return exit code 1 just because a
        backend is missing in the runtime environment.

        Forces a backend to be unavailable by monkey-patching its
        ``available()`` method so the test runs deterministically
        regardless of which backends are actually installed in the
        test environment.
        """
        from benchmarks.backends import get_backend
        from benchmarks.cli import main

        backend_cls = get_backend("topogeoml-diff-ph")
        monkeypatch.setattr(backend_cls, "available", staticmethod(lambda: False))

        out_json = tmp_path / "result.json"
        rc = main([
            "--backends", "topogeoml-diff-ph",
            "--datasets", "mnist_mock_digit_1",
            "--axes", "correctness",
            "--output", str(out_json),
        ])
        assert rc == 0
        payload = json.loads(out_json.read_text())
        unavailable = [
            c for c in payload["cells"]
            if c["error_kind"] == "UnavailableBackend"
        ]
        assert len(unavailable) >= 1, "expected at least one UnavailableBackend cell"

    def test_cli_surfaces_real_failure_messages_on_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When a real (non-skip, non-unavailable) failure occurs, the
        CLI prints which cells failed on stderr so CI logs make the
        cause obvious without requiring the JSON artifact."""
        from benchmarks import runner
        from benchmarks.cli import main

        def _boom(backend, dataset, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("intentional failure to verify stderr surfacing")

        monkeypatch.setitem(runner.AXES, "correctness", _boom)

        out_json = tmp_path / "result.json"
        rc = main([
            "--backends", "topogeoml-diff-ph",
            "--datasets", "mnist_mock_digit_1",
            "--axes", "correctness",
            "--output", str(out_json),
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "unexpected cell failure" in captured.err
        assert "intentional failure" in captured.err

    def test_cli_quick_flag_thins_axis_kwargs(self) -> None:
        """``--quick`` populates the per-axis kwargs dict with shorter
        seed/repeat lists so the bench fits within CI's 30-minute budget.
        Regression for the CI-timeout fix that landed alongside the
        ``--quick`` flag."""
        from benchmarks.cli import _quick_axis_kwargs

        kwargs = _quick_axis_kwargs()
        # Every axis is thinned to ≤3 seeds.
        for axis_name, axis_overrides in kwargs.items():
            assert "seeds" in axis_overrides, axis_name
            assert len(axis_overrides["seeds"]) <= 3, axis_name
        # ``speed`` drops the heaviest n_points and tightens repeat × number.
        assert kwargs["speed"]["n_points_list"] == [30, 100]
        assert kwargs["speed"]["repeat"] == 3
        assert kwargs["speed"]["number"] == 10
        # ``optimization`` runs fewer steps.
        assert kwargs["optimization"]["n_steps"] == 60

    def test_cli_nonzero_exit_on_failed_cell(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a registered axis raises, the runner records it as failed and the CLI exits non-zero."""
        from benchmarks import runner
        from benchmarks.cli import main

        def _boom(backend, dataset, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("synthetic failure for test")

        # Override an existing axis so we can drive the CLI through its
        # argparse `choices=` gate. ``correctness`` is the cheapest axis name
        # to stand in for our failing function.
        monkeypatch.setitem(runner.AXES, "correctness", _boom)

        out_json = tmp_path / "result.json"
        rc = main([
            "--backends", "topogeoml-diff-ph",
            "--datasets", "mnist_mock_digit_1",
            "--axes", "correctness",
            "--output", str(out_json),
        ])
        assert rc == 1

        payload = json.loads(out_json.read_text())
        failed = [c for c in payload["cells"] if not c["success"]]
        assert len(failed) == 1
        assert failed[0]["error_kind"] == "RuntimeError"
        assert "synthetic failure" in failed[0]["error_message"]
