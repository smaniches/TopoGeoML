"""
Coverage extensions for ``benchmarks/stats.py``.

The main suite (``test_benchmarks.py``) covers the median bootstrap and
the BH machinery. These tests cover:

  - the ``mean`` and ``min`` reducers,
  - input-validation paths for compare_paired,
  - dataclass ``as_dict`` round-trips for JSON serialization,
  - underpowered comparisons pass through BH untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.stats import (
    ResultKind,
    StatisticName,
    benjamini_hochberg,
    bootstrap_ci,
    compare_independent,
    compare_paired,
)


class TestReducers:
    def test_mean_reducer(self) -> None:
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        ci = bootstrap_ci(samples, statistic="mean", n_resamples=2000, seed=0)
        assert ci.point_estimate == pytest.approx(2.5)
        assert ci.statistic_name == "mean"

    def test_min_reducer(self) -> None:
        samples = np.array([5.0, 3.0, 4.0, 2.0, 7.0])
        ci = bootstrap_ci(samples, statistic="min", n_resamples=2000, seed=0)
        assert ci.point_estimate == pytest.approx(2.0)
        assert ci.statistic_name == "min"

    def test_statistic_enum_accepted(self) -> None:
        samples = np.linspace(1.0, 2.0, 30)
        ci = bootstrap_ci(samples, statistic=StatisticName.MEDIAN, n_resamples=2000)
        assert ci.statistic_name == "median"

    def test_rejects_unknown_statistic(self) -> None:
        with pytest.raises(ValueError, match="not a valid StatisticName"):
            bootstrap_ci(np.array([1.0, 2.0, 3.0]), statistic="not_a_stat", n_resamples=2000)


class TestComparePairedValidation:
    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same 1-D shape"):
            compare_paired(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                arm_a_name="a", arm_b_name="b",
            )

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compare_paired(
                np.array([]),
                np.array([]),
                arm_a_name="a", arm_b_name="b",
            )


class TestCompareIndependentValidation:
    def test_rejects_empty_arm(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compare_independent(
                np.array([]),
                np.array([1.0, 2.0]),
                arm_a_name="a", arm_b_name="b",
            )


class TestDataclassRoundTrip:
    def test_comparison_as_dict_serializes_enum_value(self) -> None:
        rng = np.random.default_rng(0)
        cmp = compare_independent(
            rng.normal(0, 1, 30), rng.normal(0, 1, 30),
            arm_a_name="a", arm_b_name="b",
        )
        d = cmp.as_dict()
        assert isinstance(d["kind"], str)
        assert d["kind"] in {"significant", "not_significant", "underpowered"}

    def test_bootstrap_ci_as_dict_is_serializable(self) -> None:
        import json
        ci = bootstrap_ci(np.linspace(1.0, 2.0, 30), n_resamples=2000, seed=0)
        s = json.dumps(ci.as_dict())
        # Schema fingerprint: re-parsing recovers exactly the same fields.
        recovered = json.loads(s)
        assert recovered["point_estimate"] == ci.point_estimate
        assert recovered["n_samples"] == 30


class TestBHWithUnderpowered:
    def test_underpowered_passes_through_untouched(self) -> None:
        # Build a family of mixed (powered, underpowered) comparisons.
        rng = np.random.default_rng(0)
        powered = compare_independent(
            rng.normal(0, 1, 60), rng.normal(2.0, 1, 60),
            arm_a_name="a", arm_b_name="b_signif",
        )
        underpowered = compare_independent(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            arm_a_name="a", arm_b_name="b_underpow",
        )
        family = benjamini_hochberg([powered, underpowered], alpha=0.05)
        # Powered comparison: classification populated.
        assert family.comparisons[0].kind in (ResultKind.SIGNIFICANT, ResultKind.NOT_SIGNIFICANT)
        # Underpowered: passes through unchanged.
        assert family.comparisons[1].kind == ResultKind.UNDERPOWERED
        assert family.comparisons[1].p_value_bh_adjusted is None

    def test_empty_family_returns_zero_significant(self) -> None:
        family = benjamini_hochberg([], alpha=0.05)
        assert family.n_significant_after_bh == 0
        assert family.comparisons == []

    def test_only_underpowered_family(self) -> None:
        underpowered = compare_independent(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            arm_a_name="a", arm_b_name="b",
        )
        family = benjamini_hochberg([underpowered], alpha=0.05)
        assert family.n_significant_after_bh == 0
        assert family.comparisons[0].kind == ResultKind.UNDERPOWERED


class TestBHRejectsBadAlpha:
    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            benjamini_hochberg([], alpha=bad_alpha)


class TestComparisonAsDictRoundTrip:
    def test_underpowered_comparison_serializable(self) -> None:
        import json
        cmp = compare_independent(
            np.zeros(3), np.ones(3),
            arm_a_name="a", arm_b_name="b",
        )
        d = cmp.as_dict()
        s = json.dumps(d)
        assert "underpowered" in s


class TestRemainingCoverage:
    """Close the coverage gaps for paths reached only via specific inputs."""

    def test_corrected_family_as_dict_serializes_comparisons(self) -> None:
        """``CorrectedFamily.as_dict`` returns a dict that JSON-encodes
        every comparison inside it."""
        import json
        rng = np.random.default_rng(0)
        cmp = compare_independent(
            rng.normal(0, 1, 30), rng.normal(0, 1, 30),
            arm_a_name="a", arm_b_name="b",
        )
        family = benjamini_hochberg([cmp], alpha=0.05)
        d = family.as_dict()
        # JSON-serialisable end-to-end.
        s = json.dumps(d)
        recovered = json.loads(s)
        assert recovered["alpha"] == 0.05
        assert len(recovered["comparisons"]) == 1

    def test_bootstrap_rejects_2d_samples(self) -> None:
        with pytest.raises(ValueError, match="samples must be 1-D"):
            bootstrap_ci(
                np.zeros((4, 4)), n_resamples=2000,
            )

    def test_bootstrap_rejects_bad_confidence_level(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci(
                np.linspace(0.0, 1.0, 30), n_resamples=2000, confidence_level=1.5,
            )

    def test_compare_independent_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="arms must be 1-D"):
            compare_independent(
                np.zeros((2, 2)), np.ones(4),
                arm_a_name="a", arm_b_name="b",
            )

    def test_compare_paired_underpowered_below_floor(self) -> None:
        """Paired comparison with n < min_samples_for_pvalue returns an
        UNDERPOWERED kind. Distinct from the all-zero-diff path which
        returns NOT_SIGNIFICANT."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.1, 2.1, 3.1, 4.1])
        cmp = compare_paired(a, b, arm_a_name="a", arm_b_name="b")
        assert cmp.kind == ResultKind.UNDERPOWERED

    def test_compare_paired_significant_path(self) -> None:
        """Wilcoxon signed-rank actually executes (n above the floor,
        diffs not all zero). Exercises the Kerby rank-biserial branch."""
        rng = np.random.default_rng(0)
        n = 40
        a = rng.normal(0.0, 1.0, n)
        b = a + rng.normal(1.0, 0.1, n)  # consistent shift
        cmp = compare_paired(a, b, arm_a_name="a", arm_b_name="b_shifted")
        assert cmp.test_name == "wilcoxon-signed-rank"
        # With a real consistent shift the test should populate p_value_raw.
        assert not np.isnan(cmp.p_value_raw)
        # Rank-biserial r is in [-1, 1].
        assert -1.0 <= cmp.effect_size <= 1.0
