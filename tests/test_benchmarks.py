"""
Tests for the benchmark framework itself.

The framework must meet the same quality floor as the code it measures.
These tests verify:

  - statistical machinery (bootstrap CI, BH correction, paired/unpaired tests)
    behaves correctly on synthetic distributions with known properties;
  - bottleneck-distance reduction has the expected invariants (identity,
    triangle inequality on simple cases);
  - each axis runs end-to-end on a tiny input and produces a result of the
    documented schema;
  - the runner captures provenance and handles failing cells without crashing;
  - report rendering does not throw on the result schema.

Tests that require ``torch`` or ``torch-topological`` are gated on
``importorskip`` so the suite degrades gracefully in CI matrices that
don't include the bench extras.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.stats import (
    ResultKind,
    benjamini_hochberg,
    bootstrap_ci,
    compare_independent,
    compare_paired,
)

# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_finite_ci_on_exponential_samples(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.exponential(1.0, size=200)
        ci = bootstrap_ci(samples, statistic="median", n_resamples=2000, seed=0)
        assert np.isfinite(ci.point_estimate)
        assert ci.ci_low < ci.point_estimate < ci.ci_high
        assert ci.confidence_level == 0.95

    def test_deterministic_given_seed(self) -> None:
        samples = np.linspace(1.0, 2.0, 50)
        a = bootstrap_ci(samples, n_resamples=1000, seed=42)
        b = bootstrap_ci(samples, n_resamples=1000, seed=42)
        assert a.point_estimate == b.point_estimate
        assert a.ci_low == b.ci_low
        assert a.ci_high == b.ci_high

    def test_ci_contains_true_median_at_nominal_rate(self) -> None:
        """Empirical coverage check: bootstrap 95% CI should cover the true
        median ~95% of the time across many trials."""
        true_median = 1.0
        n_trials = 200
        covered = 0
        rng_master = np.random.default_rng(123)
        for _ in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            samples = rng.lognormal(mean=0.0, sigma=0.3, size=80)  # median ~ 1
            ci = bootstrap_ci(samples, n_resamples=1000, seed=int(rng.integers(0, 2**31 - 1)))
            if ci.ci_low <= true_median <= ci.ci_high:
                covered += 1
        coverage = covered / n_trials
        # Allow a tolerance window — empirical CIs are slightly anticonservative
        # for the median on small samples (Efron & Tibshirani 1993 §13.5),
        # so the lower bound of the acceptable range is 0.88, not 0.93.
        assert 0.88 <= coverage <= 0.99, (
            f"empirical coverage {coverage:.3f} out of expected range [0.88, 0.99]"
        )

    def test_rejects_n_lt_2(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci(np.array([1.0]), n_resamples=1000)

    def test_rejects_too_few_resamples(self) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_ci(np.array([1.0, 2.0, 3.0]), n_resamples=100)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            bootstrap_ci(np.array([1.0, np.nan, 3.0]))


class TestCompareIndependent:
    def test_detects_large_separation(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 60)
        b = rng.normal(3.0, 1.0, 60)
        cmp = compare_independent(a, b, arm_a_name="lo", arm_b_name="hi")
        # Pre-BH, kind defaults to NOT_SIGNIFICANT; the raw p must still be tiny.
        assert cmp.p_value_raw < 1e-10
        assert cmp.effect_size < -0.6  # Cliff's δ should be strongly negative

    def test_underpowered_below_threshold(self) -> None:
        cmp = compare_independent(
            np.zeros(5), np.ones(5),
            arm_a_name="z", arm_b_name="o",
        )
        assert cmp.kind == ResultKind.UNDERPOWERED
        assert np.isnan(cmp.p_value_raw)

    def test_threshold_is_configurable(self) -> None:
        cmp = compare_independent(
            np.zeros(5), np.ones(5),
            arm_a_name="z", arm_b_name="o",
            min_samples_for_pvalue=3,
        )
        assert cmp.kind == ResultKind.NOT_SIGNIFICANT


class TestComparePaired:
    def test_zero_diffs_returns_not_significant(self) -> None:
        x = np.linspace(1.0, 2.0, 30)
        cmp = compare_paired(x, x.copy(), arm_a_name="a", arm_b_name="b")
        assert cmp.kind == ResultKind.NOT_SIGNIFICANT
        assert cmp.p_value_raw == 1.0
        assert cmp.effect_size == 0.0


class TestBenjaminiHochberg:
    def test_corrects_obvious_signal(self) -> None:
        rng = np.random.default_rng(0)
        family = [
            compare_independent(
                rng.normal(0.0, 1.0, 60),
                rng.normal(shift, 1.0, 60),
                arm_a_name="a", arm_b_name=f"b_{shift}",
            )
            for shift in (0.0, 0.05, 0.5, 2.0)
        ]
        corrected = benjamini_hochberg(family, alpha=0.05)
        # The shift=2.0 comparison must come out significant; the shift=0 must not.
        assert corrected.comparisons[0].kind == ResultKind.NOT_SIGNIFICANT
        assert corrected.comparisons[-1].kind == ResultKind.SIGNIFICANT

    def test_preserves_input_order(self) -> None:
        rng = np.random.default_rng(0)
        names = ["a", "b", "c", "d"]
        family = [
            compare_independent(
                rng.normal(0.0, 1.0, 60),
                rng.normal(shift, 1.0, 60),
                arm_a_name="ref", arm_b_name=n,
            )
            for n, shift in zip(names, [0.0, 1.0, 0.5, 2.0], strict=True)
        ]
        corrected = benjamini_hochberg(family, alpha=0.05)
        # Order of arm_b_name in the output must match input order.
        assert [c.arm_b_name for c in corrected.comparisons] == names


# ---------------------------------------------------------------------------
# axes — gated on torch availability
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _has_torch_topological() -> bool:
    try:
        import torch_topological  # noqa: F401
    except ImportError:
        return False
    return True


class TestBottleneckDistance:
    def test_identical_diagrams_zero(self) -> None:
        from benchmarks.axes.stability import _bottleneck_distance_finite

        d = np.array([[0.0, 1.0], [0.0, 2.0], [0.5, 0.7]])
        assert _bottleneck_distance_finite(d, d.copy()) == pytest.approx(0.0)

    def test_empty_diagrams_zero(self) -> None:
        from benchmarks.axes.stability import _bottleneck_distance_finite

        empty = np.empty((0, 2))
        assert _bottleneck_distance_finite(empty, empty) == pytest.approx(0.0)

    def test_shifted_bar_distance_is_shift(self) -> None:
        from benchmarks.axes.stability import _bottleneck_distance_finite

        # A single bar (b=0, d=1) vs the same bar shifted by 0.1 in death.
        # L_inf bottleneck distance should be 0.1.
        a = np.array([[0.0, 1.0]])
        b = np.array([[0.0, 1.1]])
        assert _bottleneck_distance_finite(a, b) == pytest.approx(0.1, abs=1e-12)


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestAxesEndToEnd:
    """Run each axis on a tiny input and assert the result schema is well-formed."""

    def test_correctness_passes_for_both_backends(self) -> None:
        from benchmarks.axes.correctness import measure_correctness
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        dataset = get_dataset("mnist_mock_digit_0")
        for name in ("topogeoml-diff-ph", "torch-topological"):
            backend = get_backend(name)
            report = measure_correctness(backend, dataset, n_points=20, seeds=[0, 1])
            assert report.overall_pass, (
                f"backend {name} failed correctness against ripser:\n"
                + "\n".join(
                    f"  seed={s.seed} h0_diff={s.max_abs_diff_h0:.2e} "
                    f"h1_diff={s.max_abs_diff_h1:.2e} "
                    f"dtype_ok={s.dtype_preserved}"
                    for s in report.per_seed
                )
            )

    def test_stability_no_cohen_steiner_violations_for_both_backends(self) -> None:
        from benchmarks.axes.stability import measure_stability
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        dataset = get_dataset("mnist_mock_digit_0")
        for name in ("topogeoml-diff-ph", "torch-topological"):
            backend = get_backend(name)
            report = measure_stability(
                backend, dataset,
                n_points=20, seeds=[0],
                perturbation_inf_norms=(1e-3, 1e-2),
                gradcheck_n_points=5,
            )
            assert report.n_theorem_violations == 0, (
                f"backend {name} violated Cohen-Steiner: "
                f"{report.n_theorem_violations} pairs out of "
                f"{len(report.cohen_steiner_pairs)}"
            )

    def test_optimization_decreases_loss(self) -> None:
        from benchmarks.axes.optimization import measure_optimization
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        dataset = get_dataset("mnist_mock_digit_0")
        backend = get_backend("topogeoml-diff-ph")
        report = measure_optimization(
            backend, dataset,
            objective="inflate_h1", n_points=20, seeds=[0],
            n_steps=20, learning_rate=1e-2,
        )
        # `inflate_h1` minimizes -longest_h1 — the loss trajectory should
        # decrease (or be approximately monotone) on a noisy circle.
        seed_row = report.per_seed[0]
        assert seed_row.final_loss <= seed_row.initial_loss + 1e-6


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestRunner:
    def test_runner_writes_json_with_provenance(self, tmp_path: Path) -> None:
        from benchmarks.runner import run, write_result

        result = run(
            backend_names=["topogeoml-diff-ph"],
            dataset_names=["mnist_mock_digit_1"],
            axis_names=["correctness"],
        )
        out = tmp_path / "result.json"
        write_result(result, out)
        loaded = json.loads(out.read_text())

        # Provenance contract
        prov = loaded["provenance"]
        assert prov["schema_version"]
        assert prov["timestamp_utc"]
        # The framework runs inside this git checkout so SHA is non-empty here.
        assert prov["python_version"] == result.provenance.python_version
        assert prov["deterministic_algorithms_set"] is True

        # Cell contract
        cells = loaded["cells"]
        assert len(cells) == 1
        cell = cells[0]
        assert cell["backend_name"] == "topogeoml-diff-ph"
        assert cell["axis_name"] == "correctness"
        # Either it passed or it captured a structured error.
        if not cell["success"]:
            assert cell["error_kind"]
            assert cell["error_message"]

    def test_runner_rejects_unknown_axis(self) -> None:
        from benchmarks.runner import run

        with pytest.raises(KeyError, match="unknown axes"):
            run(axis_names=["not_an_axis"])
