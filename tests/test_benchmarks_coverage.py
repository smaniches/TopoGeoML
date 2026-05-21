"""
Targeted tests that close remaining coverage gaps in the bench framework.

Each test below is named for the specific code path it exercises so a
future reader sees the rationale for the test, not just the assertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _has_torch_topological() -> bool:
    try:
        import torch_topological  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# correctness.py
# ---------------------------------------------------------------------------

class TestCorrectnessHelpers:
    def test_sorted_finite_on_empty_array(self) -> None:
        from benchmarks.axes.correctness import _sorted_finite

        out = _sorted_finite(np.empty((0,)))
        assert out.shape == (0, 2)
        assert out.dtype == np.float64

    def test_sorted_finite_on_all_infinite_bars(self) -> None:
        from benchmarks.axes.correctness import _sorted_finite

        bars = np.array([[0.0, np.inf], [1.0, np.inf]])
        out = _sorted_finite(bars)
        assert out.shape == (0, 2)
        assert out.dtype == np.float64

    def test_max_abs_diff_mismatched_shapes_returns_inf(self) -> None:
        from benchmarks.axes.correctness import _max_abs_diff

        a = np.array([[0.0, 1.0]])
        b = np.array([[0.0, 1.0], [0.5, 0.7]])
        assert _max_abs_diff(a, b) == float("inf")

    def test_max_abs_diff_both_empty_zero(self) -> None:
        from benchmarks.axes.correctness import _max_abs_diff

        empty = np.empty((0, 2), dtype=np.float64)
        assert _max_abs_diff(empty, empty.copy()) == 0.0


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestCorrectnessOverallFail:
    def test_overall_pass_false_when_atol_too_strict(self) -> None:
        """Setting ``atol`` below the achievable f64 round-off forces the
        backend's tiny but nonzero diff against ripser to register as a
        failure, exercising the ``overall_pass = False`` branch."""
        from benchmarks.axes.correctness import measure_correctness
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        backend = get_backend("topogeoml-diff-ph")
        dataset = get_dataset("mnist_mock_digit_0")
        report = measure_correctness(
            backend, dataset, n_points=20, seeds=[0], atol=0.0,
        )
        # The backend produces a tiny nonzero diff against ripser; with
        # atol=0 every seed fails the match.
        assert not report.overall_pass


# ---------------------------------------------------------------------------
# speed.py
# ---------------------------------------------------------------------------

class TestSpeedHelpers:
    def test_ci_from_empty_list(self) -> None:
        from benchmarks.axes.speed import _ci_from_pass_medians

        pe, lo, hi = _ci_from_pass_medians([])
        assert np.isnan(pe) and np.isnan(lo) and np.isnan(hi)

    def test_ci_from_single_element_returns_point_estimate_only(self) -> None:
        from benchmarks.axes.speed import _ci_from_pass_medians

        pe, lo, hi = _ci_from_pass_medians([1.23])
        assert pe == pytest.approx(1.23)
        assert np.isnan(lo) and np.isnan(hi)


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestSpeedAsDict:
    def test_speed_report_as_dict_roundtrip(self) -> None:
        from benchmarks.axes.speed import measure_speed
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_speed(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            n_points_list=[10],
            seeds=[0],
            warmup=1, repeat=2, number=2,
        )
        d = report.as_dict()
        s = json.dumps(d)
        recovered = json.loads(s)
        assert recovered["backend_name"] == "topogeoml-diff-ph"
        assert recovered["n_outer_repeats"] == 2

    def test_speed_uses_default_sizes_and_seeds_when_none(self) -> None:
        """Cover the ``if n_points_list is None`` / ``if seeds is None`` branches
        without paying the full default-cost (300-point clouds) by only
        running the inner config minimally."""
        from benchmarks.axes import speed as speed_mod
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        # Patch the defaults to the smallest viable values so the test
        # finishes in seconds.
        backend = get_backend("topogeoml-diff-ph")
        dataset = get_dataset("mnist_mock_digit_0")
        report = speed_mod.measure_speed(
            backend, dataset,
            n_points_list=None, seeds=None,
            warmup=1, repeat=2, number=2,
        )
        # default n_points_list=[30,100,300], seeds=[0..4] => 3*5*2=30 rows.
        assert len(report.rows) == 30


# ---------------------------------------------------------------------------
# optimization.py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestOptimizationCoverage:
    def test_report_as_dict_roundtrip(self) -> None:
        from benchmarks.axes.optimization import measure_optimization
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_optimization(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            objective="inflate_h1", n_points=10, seeds=[0],
            n_steps=5, learning_rate=1e-2,
        )
        d = report.as_dict()
        assert "per_seed" in d
        assert isinstance(d["per_seed"], list)
        # Single-seed run: CI fallback to point-estimate-only path.
        assert np.isnan(d["final_loss_ci95_low"])

    def test_shrink_h1_objective(self) -> None:
        from benchmarks.axes.optimization import measure_optimization
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_optimization(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            objective="shrink_h1", n_points=12, seeds=[0],
            n_steps=5, learning_rate=1e-2,
        )
        assert report.objective == "shrink_h1"
        assert len(report.per_seed) == 1

    def test_uses_default_seeds_when_none(self) -> None:
        from benchmarks.axes.optimization import measure_optimization
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_optimization(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            objective="inflate_h1", n_points=10, seeds=None,
            n_steps=3, learning_rate=1e-2,
        )
        # Default seeds=[0,1,2,3,4].
        assert len(report.per_seed) == 5
        # 5 seeds, all finite -> bootstrap CI populated.
        assert not np.isnan(report.final_loss_ci95_low)


# ---------------------------------------------------------------------------
# stability.py
# ---------------------------------------------------------------------------

class TestStabilityBottleneckEdges:
    def test_bottleneck_one_empty_returns_zero(self) -> None:
        from benchmarks.axes.stability import _bottleneck_distance_finite

        # When both diagrams have only finite-stripped content the result
        # should still be 0 if both end up empty after stripping.
        only_inf = np.array([[0.0, np.inf]])
        empty = np.empty((0, 2))
        assert _bottleneck_distance_finite(only_inf, empty) == 0.0
        assert _bottleneck_distance_finite(empty, only_inf) == 0.0


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestStabilityReportCoverage:
    def test_stability_as_dict_roundtrip(self) -> None:
        from benchmarks.axes.stability import measure_stability
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_stability(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            n_points=12, seeds=[0],
            perturbation_inf_norms=(1e-3,),
            gradcheck_n_points=5,
        )
        d = report.as_dict()
        assert "cohen_steiner_pairs" in d
        assert isinstance(d["cohen_steiner_pairs"], list)
        # Single-seed -> Lipschitz CI is NaN.
        assert np.isnan(d["lipschitz_ci95_low"])

    def test_stability_uses_default_seeds(self) -> None:
        from benchmarks.axes.stability import measure_stability
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        # Default seeds=[0,1,2,3,4]; default perturbation_inf_norms has 6 entries.
        # We pass minimal perturbations and small n_points to keep the test fast.
        report = measure_stability(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            n_points=10,
            seeds=None,
            perturbation_inf_norms=(1e-3,),
            gradcheck_n_points=4,
        )
        # 5 seeds * 1 perturbation = 5 CS pairs.
        assert len(report.cohen_steiner_pairs) == 5
        # 5 seeds -> Lipschitz CI is populated.
        assert not np.isnan(report.lipschitz_ci95_low)


# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestCLIExtras:
    def test_writes_step_summary_when_env_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``GITHUB_STEP_SUMMARY`` is set, the CLI writes the markdown
        to that path in addition to stdout."""
        from benchmarks.cli import main

        summary_path = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

        out_json = tmp_path / "result.json"
        rc = main([
            "--backends", "topogeoml-diff-ph",
            "--datasets", "mnist_mock_digit_1",
            "--axes", "correctness",
            "--output", str(out_json),
        ])
        assert rc == 0
        assert summary_path.exists()
        assert summary_path.read_text().startswith("# TopoGeoML benchmark")


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

class TestReportEdges:
    def _provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "timestamp_utc": "2026-05-20T00:00:00Z",
            "git_sha": "",  # Empty -> "(not a git checkout)" branch.
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
        }

    def test_provenance_empty_sha_surfaced(self) -> None:
        from benchmarks.report import render_markdown

        payload = {"provenance": self._provenance(), "config": {}, "cells": []}
        md = render_markdown(payload)
        assert "(not a git checkout)" in md

    def test_correctness_with_zero_per_seed(self) -> None:
        """A correctness cell with an empty ``per_seed`` list must not
        crash the renderer; it should skip the row gracefully."""
        from benchmarks.report import render_markdown

        payload = {
            "provenance": self._provenance(),
            "config": {},
            "cells": [{
                "backend_name": "test",
                "dataset_name": "mnist_mock_digit_0",
                "axis_name": "correctness",
                "success": True,
                "payload": {
                    "n_points": 20, "atol": 1e-6,
                    "per_seed": [],
                    "overall_pass": True,
                },
                "error_kind": None,
                "error_message": None,
                "error_traceback": None,
            }],
        }
        # Should render without raising even though per_seed is empty.
        md = render_markdown(payload)
        assert "Correctness" in md


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestRunnerCoverage:
    def test_axis_correctness_dispatch(self) -> None:
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset
        from benchmarks.runner import AXES

        out = AXES["correctness"](
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_1"),
        )
        assert "overall_pass" in out

    def test_axis_stability_dispatch(self) -> None:
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset
        from benchmarks.runner import AXES

        # Call the wrapper to cover ``_axis_stability``.
        # We can't pass per-axis config through the dispatch wrapper, so
        # we accept the framework default (which is fast on n=12).
        # The dispatch wrapper takes (backend, dataset) only.
        backend = get_backend("topogeoml-diff-ph")
        # Use a minimal-cost custom dataset wrapper via a real registered one.
        out = AXES["stability"](backend, get_dataset("mnist_mock_digit_1"))
        assert "n_theorem_violations" in out

    def test_axis_optimization_dispatch(self) -> None:
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset
        from benchmarks.runner import AXES

        backend = get_backend("topogeoml-diff-ph")
        out = AXES["optimization"](backend, get_dataset("mnist_mock_digit_1"))
        assert "objective" in out

    def test_axis_speed_dispatch(self) -> None:
        """Cover ``_axis_speed`` — heavier so we accept that this single
        invocation costs more wall-clock than other axis tests."""
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset
        from benchmarks.runner import AXES

        backend = get_backend("topogeoml-diff-ph")
        out = AXES["speed"](backend, get_dataset("mnist_mock_digit_1"))
        assert "rows" in out

    def test_run_records_unavailable_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a registered backend reports ``available() = False`` the
        runner records an ``UnavailableBackend`` cell and skips the inner
        loops for that backend."""
        from benchmarks.backends import get_backend
        from benchmarks.runner import run

        backend_cls = get_backend("topogeoml-diff-ph")
        monkeypatch.setattr(backend_cls, "available", staticmethod(lambda: False))
        result = run(
            backend_names=["topogeoml-diff-ph"],
            dataset_names=["mnist_mock_digit_1"],
            axis_names=["correctness"],
        )
        assert len(result.cells) == 1
        cell = result.cells[0]
        assert not cell.success
        assert cell.error_kind == "UnavailableBackend"

    def test_run_uses_defaults_when_args_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cover the ``if X is None`` defaults for backend_names, dataset_names, axis_names."""
        from benchmarks import runner

        # Replace AXES with a single cheap pass-through to avoid running
        # the full matrix.
        monkeypatch.setattr(runner, "AXES", {"noop": lambda b, d: {"ok": True}})
        result = runner.run()
        # At least one cell per (available backend × registered dataset).
        assert len(result.cells) >= 1


# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------

class TestStatsExtras:
    def test_corrected_family_as_dict(self) -> None:
        from benchmarks.stats import benjamini_hochberg, compare_independent

        rng = np.random.default_rng(0)
        family = benjamini_hochberg(
            [compare_independent(rng.normal(0, 1, 60), rng.normal(2, 1, 60),
                                 arm_a_name="a", arm_b_name="b")],
            alpha=0.05,
        )
        d = family.as_dict()
        assert "alpha" in d
        assert "n_significant_after_bh" in d
        assert "comparisons" in d
        assert isinstance(d["comparisons"], list)

    def test_compare_independent_zero_median_arm_no_ratio(self) -> None:
        """When either arm's median is non-positive, ``median_ratio`` is None."""
        from benchmarks.stats import compare_independent

        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 60)  # ~symmetric, median near zero
        b = rng.normal(0.0, 1.0, 60)
        cmp = compare_independent(a, b, arm_a_name="a", arm_b_name="b")
        # Either median <= 0 -> ratio is None (otherwise it's the ratio of two positives).
        if cmp.median_a <= 0 or cmp.median_b <= 0:
            assert cmp.median_ratio is None

    def test_compare_paired_underpowered_path(self) -> None:
        """Cover the ``UNDERPOWERED`` branch of compare_paired with n < threshold."""
        from benchmarks.stats import ResultKind, compare_paired

        cmp = compare_paired(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            arm_a_name="a", arm_b_name="b",
        )
        assert cmp.kind == ResultKind.UNDERPOWERED
        assert np.isnan(cmp.p_value_raw)
        assert np.isnan(cmp.effect_size)

    def test_compare_independent_rejects_non_1d(self) -> None:
        from benchmarks.stats import compare_independent

        with pytest.raises(ValueError, match="1-D"):
            compare_independent(
                np.array([[1.0, 2.0]]),
                np.array([1.0]),
                arm_a_name="a", arm_b_name="b",
            )

    def test_compare_paired_rejects_2d(self) -> None:
        from benchmarks.stats import compare_paired

        with pytest.raises(ValueError, match="same 1-D shape"):
            compare_paired(
                np.array([[1.0, 2.0]]),
                np.array([[1.0, 2.0]]),
                arm_a_name="a", arm_b_name="b",
            )

    def test_bootstrap_ci_rejects_2d(self) -> None:
        from benchmarks.stats import bootstrap_ci

        with pytest.raises(ValueError, match="1-D"):
            bootstrap_ci(np.array([[1.0, 2.0, 3.0, 4.0]]))

    def test_bootstrap_ci_rejects_bad_confidence_level(self) -> None:
        from benchmarks.stats import bootstrap_ci

        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci(np.array([1.0, 2.0, 3.0]), confidence_level=1.5)

    def test_compare_paired_real_signal_path(self) -> None:
        """Cover lines 363-375 (Wilcoxon execution + rank-biserial computation)
        with a paired sample that has consistent non-zero differences and
        n >= MIN_SAMPLES_FOR_PVALUE."""
        from benchmarks.stats import ResultKind, compare_paired

        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 30)
        y = x + 0.5  # consistent positive shift
        cmp = compare_paired(x, y, arm_a_name="a", arm_b_name="b")
        assert cmp.test_name == "wilcoxon-signed-rank"
        assert not np.isnan(cmp.p_value_raw)
        assert cmp.p_value_raw < 1e-3
        # Effect size: negative because x < y systematically (rank-biserial uses sign of x - y).
        assert cmp.effect_size == pytest.approx(-1.0, abs=0.05)
        # Pre-BH still NOT_SIGNIFICANT until correction runs (per the docstring).
        assert cmp.kind == ResultKind.NOT_SIGNIFICANT


# ---------------------------------------------------------------------------
# dtype-rejection paths in backends
# ---------------------------------------------------------------------------

class TestBackendDtypeRejection:
    @pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
    def test_torch_topological_rejects_float32(self) -> None:
        from benchmarks.backends import get_backend

        backend = get_backend("torch-topological")
        X = torch.zeros((10, 2), dtype=torch.float32)
        with pytest.raises(TypeError, match="must be float64"):
            backend.compute_diagram(X, max_dim=1)

    def test_topogeoml_rejects_float32(self) -> None:
        from benchmarks.backends import get_backend

        backend = get_backend("topogeoml-diff-ph")
        X = torch.zeros((10, 2), dtype=torch.float32)
        with pytest.raises(TypeError, match="must be float64"):
            backend.compute_diagram(X, max_dim=1)

    @pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
    def test_torch_topological_empty_h1_loss_is_zero(self) -> None:
        """Cover lines 59 + 64: empty H1 diagram → zero loss that depends on X."""
        from benchmarks.backends import get_backend

        backend = get_backend("torch-topological")
        # Three colinear points yield H1 = empty.
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64,
                         requires_grad=True)
        loss = backend.loss_longest_h1(X)
        # Loss is a 0-dim float tensor.
        assert loss.shape == ()
        assert loss.dtype == torch.float64
        assert float(loss.item()) == 0.0
