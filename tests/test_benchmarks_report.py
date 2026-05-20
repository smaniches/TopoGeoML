"""
Tests for the benchmark report renderer.

The renderer never makes a directional claim without a significance label;
these tests construct synthetic ``RunResult`` payloads and assert that
the rendered markdown obeys that rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.report import render_from_file, render_markdown


def _empty_run() -> dict[str, Any]:
    return {
        "provenance": {
            "schema_version": "1.0.0",
            "timestamp_utc": "2026-05-20T00:00:00Z",
            "git_sha": "deadbeef" * 5,
            "git_dirty": False,
            "python_version": "3.12.0",
            "torch_version": "2.0.0",
            "numpy_version": "1.26.0",
            "scipy_version": "1.13.0",
            "topogeoml_version": "0.0.1",
            "torch_topological_version": "0.1.9",
            "platform_string": "Linux-test",
            "cpu_count": 4,
            "process_memory_total_mb": 16000,
            "deterministic_algorithms_set": True,
        },
        "config": {"backend_names": [], "dataset_names": [], "axis_names": []},
        "cells": [],
    }


class TestProvenanceBlock:
    def test_renders_every_field(self) -> None:
        md = render_markdown(_empty_run())
        for needle in (
            "Schema: `1.0.0`",
            "Timestamp (UTC): 2026-05-20T00:00:00Z",
            "Python: 3.12.0",
            "PyTorch: 2.0.0",
            "TopoGeoML: 0.0.1",
            "torch-topological: 0.1.9",
            "Determinism: enabled",
        ):
            assert needle in md, f"missing {needle!r} in rendered output"

    def test_dirty_flag_surfaced(self) -> None:
        payload = _empty_run()
        payload["provenance"]["git_dirty"] = True
        md = render_markdown(payload)
        assert "(dirty)" in md

    def test_determinism_disabled_surfaced(self) -> None:
        payload = _empty_run()
        payload["provenance"]["deterministic_algorithms_set"] = False
        md = render_markdown(payload)
        assert "Determinism: disabled" in md


class TestCorrectnessSection:
    def _cell(self, *, success: bool, overall_pass: bool) -> dict[str, Any]:
        return {
            "backend_name": "test-backend",
            "dataset_name": "noisy_circle",
            "axis_name": "correctness",
            "success": success,
            "payload": {
                "n_points": 50,
                "atol": 1e-6,
                "per_seed": [{"max_abs_diff_h0": 1e-9, "max_abs_diff_h1": 5e-9}],
                "overall_pass": overall_pass,
            },
            "error_kind": None,
            "error_message": None,
            "error_traceback": None,
        }

    def test_pass_verdict(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell(success=True, overall_pass=True)]
        md = render_markdown(payload)
        assert "PASS" in md and "FAIL" not in md

    def test_fail_verdict(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell(success=True, overall_pass=False)]
        md = render_markdown(payload)
        assert "FAIL" in md

    def test_failed_cell_skipped_in_correctness_section(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell(success=False, overall_pass=False)]
        md = render_markdown(payload)
        assert "Correctness" not in md or "PASS" not in md


class TestStabilitySection:
    def _cell(self, *, violations: int, gradcheck_rate: float) -> dict[str, Any]:
        return {
            "backend_name": "test-backend",
            "dataset_name": "noisy_circle",
            "axis_name": "stability",
            "success": True,
            "payload": {
                "n_points": 50,
                "n_theorem_violations": violations,
                "lipschitz_median": 0.25,
                "lipschitz_ci95_low": 0.10,
                "lipschitz_ci95_high": 0.60,
                "gradcheck_pass_rate": gradcheck_rate,
            },
            "error_kind": None,
            "error_message": None,
            "error_traceback": None,
        }

    def test_violations_surfaced(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell(violations=3, gradcheck_rate=1.0)]
        md = render_markdown(payload)
        assert "| 3 |" in md

    def test_gradcheck_rate_as_percentage(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell(violations=0, gradcheck_rate=0.8)]
        md = render_markdown(payload)
        assert "80%" in md

    def test_ci_unavailable_when_single_seed(self) -> None:
        payload = _empty_run()
        cell = self._cell(violations=0, gradcheck_rate=1.0)
        cell["payload"]["lipschitz_ci95_low"] = float("nan")
        cell["payload"]["lipschitz_ci95_high"] = float("nan")
        payload["cells"] = [cell]
        md = render_markdown(payload)
        assert "CI not available" in md


class TestSpeedSection:
    def _cell(self, name: str, point_estimate_ms: float) -> dict[str, Any]:
        return {
            "backend_name": name,
            "dataset_name": "noisy_circle",
            "axis_name": "speed",
            "success": True,
            "payload": {
                "rows": [
                    {
                        "n_points": 30, "seed": 0, "operation": "forward",
                        "point_estimate_ms": point_estimate_ms,
                        "ci95_low_ms": point_estimate_ms * 0.95,
                        "ci95_high_ms": point_estimate_ms * 1.05,
                    },
                ],
            },
            "error_kind": None,
            "error_message": None,
            "error_traceback": None,
        }

    def test_single_backend_no_comparison(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell("only-backend", 1.0)]
        md = render_markdown(payload)
        assert "Single backend in this run" in md

    def test_two_backends_emit_comparison_table(self) -> None:
        payload = _empty_run()
        payload["cells"] = [self._cell("backend-a", 1.0), self._cell("backend-b", 2.0)]
        md = render_markdown(payload)
        # The single seed will fall into UNDERPOWERED territory (n=1).
        assert "underpowered" in md


class TestOptimizationSection:
    def test_ci_string_when_finite(self) -> None:
        payload = _empty_run()
        payload["cells"] = [{
            "backend_name": "test",
            "dataset_name": "noisy_circle",
            "axis_name": "optimization",
            "success": True,
            "payload": {
                "objective": "inflate_h1",
                "n_steps": 200,
                "learning_rate": 0.01,
                "final_loss_median": -2.0,
                "final_loss_ci95_low": -2.1,
                "final_loss_ci95_high": -1.9,
            },
            "error_kind": None,
            "error_message": None,
            "error_traceback": None,
        }]
        md = render_markdown(payload)
        assert "-2.0000 [-2.1000, -1.9000]" in md

    def test_ci_unavailable_falls_back_to_point_estimate(self) -> None:
        payload = _empty_run()
        payload["cells"] = [{
            "backend_name": "test",
            "dataset_name": "noisy_circle",
            "axis_name": "optimization",
            "success": True,
            "payload": {
                "objective": "inflate_h1",
                "n_steps": 200,
                "learning_rate": 0.01,
                "final_loss_median": -2.0,
                "final_loss_ci95_low": float("nan"),
                "final_loss_ci95_high": float("nan"),
            },
            "error_kind": None,
            "error_message": None,
            "error_traceback": None,
        }]
        md = render_markdown(payload)
        assert "CI n/a" in md


class TestFailuresSection:
    def test_failed_cell_is_surfaced(self) -> None:
        payload = _empty_run()
        payload["cells"] = [{
            "backend_name": "broken",
            "dataset_name": "noisy_circle",
            "axis_name": "stability",
            "success": False,
            "payload": None,
            "error_kind": "ValueError",
            "error_message": "synthetic test failure",
            "error_traceback": None,
        }]
        md = render_markdown(payload)
        assert "Failures" in md
        assert "ValueError" in md
        assert "synthetic test failure" in md

    def test_no_failures_section_when_all_successful(self) -> None:
        payload = _empty_run()
        md = render_markdown(payload)
        assert "## Failures" not in md


class TestRenderFromFile:
    def test_round_trip(self, tmp_path: Path) -> None:
        payload = _empty_run()
        path = tmp_path / "run.json"
        path.write_text(json.dumps(payload))
        md = render_from_file(path)
        # The provenance section is always rendered, so the output must
        # contain the schema-version banner.
        assert "Schema: `1.0.0`" in md
