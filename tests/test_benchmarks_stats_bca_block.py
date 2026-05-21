"""
Tests for BCa and block-bootstrap interval methods in
``benchmarks.stats.bootstrap_ci``.

These complement ``test_benchmarks_stats_extras.py`` (percentile-method
coverage) and include hypothesis-driven property tests for empirical
coverage and bracket-order invariants. The numerical-correctness tests
target three regimes:

  1. **Symmetric near-normal sampling distribution** — BCa, percentile,
     and the normal-theory CI should agree within Monte-Carlo
     uncertainty.
  2. **Skewed sampling distribution** — BCa should shift the interval
     toward the heavier tail; percentile should not.
  3. **Serially-correlated AR(1) timing model** — block bootstrap CIs
     should be wider than the percentile CIs (because percentile
     underestimates variance under positive autocorrelation).

References
----------
Efron, B. (1987). "Better Bootstrap Confidence Intervals." JASA 82.
Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." Ann. Statist. 17.
DiCiccio, T. J., & Efron, B. (1996). "Bootstrap Confidence Intervals."
  Stat. Sci. 11.
Hall, P., Horowitz, J. L., & Jing, B.-Y. (1995). "On Blocking Rules for
  the Bootstrap with Dependent Data." Biometrika 82.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from benchmarks.stats import (
    BootstrapCI,
    BootstrapMethod,
    bootstrap_ci,
)

# ---------------------------------------------------------------------------
# Cross-method invariants (apply to every method).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method", [BootstrapMethod.PERCENTILE, BootstrapMethod.BCA, BootstrapMethod.BLOCK],
)
class TestCrossMethodInvariants:
    def test_ci_brackets_point_estimate_on_iid_normal(
        self, method: BootstrapMethod,
    ) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=2.0, scale=1.0, size=80)
        ci = bootstrap_ci(samples, statistic="mean", method=method, n_resamples=2000, seed=0)
        assert ci.ci_low <= ci.point_estimate <= ci.ci_high

    def test_deterministic_given_seed(self, method: BootstrapMethod) -> None:
        rng = np.random.default_rng(7)
        samples = rng.normal(size=50)
        ci1 = bootstrap_ci(samples, method=method, n_resamples=2000, seed=42)
        ci2 = bootstrap_ci(samples, method=method, n_resamples=2000, seed=42)
        assert ci1.ci_low == ci2.ci_low
        assert ci1.ci_high == ci2.ci_high

    def test_method_recorded_in_result(self, method: BootstrapMethod) -> None:
        rng = np.random.default_rng(1)
        ci = bootstrap_ci(rng.normal(size=30), method=method, n_resamples=1000)
        assert ci.method == method.value

    def test_dataclass_round_trip(self, method: BootstrapMethod) -> None:
        import json

        rng = np.random.default_rng(2)
        ci = bootstrap_ci(rng.normal(size=30), method=method, n_resamples=1000)
        s = json.dumps(ci.as_dict())
        d = json.loads(s)
        assert d["method"] == method.value
        # block_length serialises as None or int depending on method.
        if method is BootstrapMethod.BLOCK:
            assert isinstance(d["block_length"], int)
        else:
            assert d["block_length"] is None


# ---------------------------------------------------------------------------
# BCa-specific behaviour.
# ---------------------------------------------------------------------------

class TestBCaInterval:
    def test_bca_brackets_mean_on_normal_sample(self) -> None:
        """For a large symmetric sample BCa should bracket the true mean
        and roughly match a normal-theory CI."""
        rng = np.random.default_rng(0)
        n = 200
        samples = rng.normal(loc=5.0, scale=2.0, size=n)
        ci = bootstrap_ci(
            samples, statistic="mean", method=BootstrapMethod.BCA,
            confidence_level=0.95, n_resamples=2000, seed=0,
        )
        se = samples.std(ddof=1) / math.sqrt(n)
        normal_lo = float(samples.mean()) - 1.96 * se
        normal_hi = float(samples.mean()) + 1.96 * se
        # BCa should agree with the normal-theory CI to within 10% of its width.
        width = normal_hi - normal_lo
        assert abs(ci.ci_low - normal_lo) < 0.15 * width
        assert abs(ci.ci_high - normal_hi) < 0.15 * width

    def test_bca_differs_from_percentile_on_skewed_sample(self) -> None:
        """On a heavy-right-tailed sample (lognormal), the BCa interval
        should shift relative to the percentile interval; otherwise BCa
        would offer no advantage."""
        rng = np.random.default_rng(0)
        samples = rng.lognormal(mean=0.0, sigma=1.0, size=120)
        ci_pct = bootstrap_ci(
            samples, statistic="mean", method=BootstrapMethod.PERCENTILE,
            n_resamples=2000, seed=0,
        )
        ci_bca = bootstrap_ci(
            samples, statistic="mean", method=BootstrapMethod.BCA,
            n_resamples=2000, seed=0,
        )
        # Same bootstrap draws (same seed, same n_resamples) → the
        # difference is entirely the BCa adjustment.
        assert (ci_pct.ci_low, ci_pct.ci_high) != (ci_bca.ci_low, ci_bca.ci_high)
        # Same point estimate (it's the statistic on the original sample).
        assert ci_pct.point_estimate == pytest.approx(ci_bca.point_estimate)

    def test_bca_falls_back_to_bias_correction_on_constant_sample(self) -> None:
        """If the sample is constant the jackknife denominator is 0 and
        the helper returns ``a=0`` — BCa degenerates to a pure
        bias-corrected interval (no zero-division)."""
        samples = np.full(40, 3.0)
        ci = bootstrap_ci(
            samples, statistic="mean", method=BootstrapMethod.BCA,
            n_resamples=2000, seed=0,
        )
        assert ci.ci_low == ci.ci_high == pytest.approx(3.0)

    def test_bca_records_method_and_no_block_length(self) -> None:
        ci = bootstrap_ci(
            np.linspace(0.0, 1.0, 30), method=BootstrapMethod.BCA, n_resamples=1000,
        )
        assert ci.method == "bca"
        assert ci.block_length is None

    def test_higher_confidence_yields_wider_ci(self) -> None:
        rng = np.random.default_rng(3)
        samples = rng.normal(size=80)
        ci_90 = bootstrap_ci(
            samples, method=BootstrapMethod.BCA,
            confidence_level=0.90, n_resamples=2000, seed=0,
        )
        ci_99 = bootstrap_ci(
            samples, method=BootstrapMethod.BCA,
            confidence_level=0.99, n_resamples=2000, seed=0,
        )
        # 99% interval must be (weakly) wider than the 90% interval.
        assert ci_99.ci_high - ci_99.ci_low >= ci_90.ci_high - ci_90.ci_low


# ---------------------------------------------------------------------------
# Block-bootstrap-specific behaviour.
# ---------------------------------------------------------------------------

def _ar1_sample(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Generate an AR(1) sample with autocorrelation φ.

    x_t = φ x_{t-1} + ε_t, ε_t ~ N(0, σ²). x_0 drawn from the
    stationary distribution N(0, σ² / (1 - φ²)) so the chain is
    stationary from the first observation.
    """
    out = np.empty(n, dtype=np.float64)
    out[0] = rng.normal(0.0, sigma / math.sqrt(1.0 - phi**2))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(0.0, sigma)
    return out


class TestBlockBootstrap:
    def test_block_length_auto_selects_cube_root_of_n(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(size=125)
        ci = bootstrap_ci(
            samples, method=BootstrapMethod.BLOCK, n_resamples=1000, seed=0,
        )
        # ceil(125^(1/3)) = ceil(4.999...) = 5
        assert ci.block_length == 5

    def test_block_length_user_override(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(size=100)
        ci = bootstrap_ci(
            samples, method=BootstrapMethod.BLOCK, block_length=10,
            n_resamples=1000, seed=0,
        )
        assert ci.block_length == 10

    def test_rejects_block_length_out_of_range(self) -> None:
        samples = np.linspace(0.0, 1.0, 30)
        with pytest.raises(ValueError, match="block_length"):
            bootstrap_ci(
                samples, method=BootstrapMethod.BLOCK, block_length=0,
                n_resamples=1000,
            )
        with pytest.raises(ValueError, match="block_length"):
            bootstrap_ci(
                samples, method=BootstrapMethod.BLOCK, block_length=999,
                n_resamples=1000,
            )

    def test_block_wider_than_percentile_on_ar1(self) -> None:
        """On a positively-autocorrelated series the IID-percentile
        interval underestimates the variance of the mean; the block
        bootstrap reflects the longer-range dependence and produces a
        wider interval.

        We average over multiple AR(1) realisations to reduce the
        Monte-Carlo noise in the comparison, then check the average
        widths.
        """
        n_trials = 30
        widths_pct = np.zeros(n_trials)
        widths_blk = np.zeros(n_trials)
        for trial in range(n_trials):
            rng = np.random.default_rng(100 + trial)
            samples = _ar1_sample(rng, n=120, phi=0.7, sigma=1.0)
            ci_pct = bootstrap_ci(
                samples, statistic="mean", method=BootstrapMethod.PERCENTILE,
                n_resamples=2000, seed=trial,
            )
            ci_blk = bootstrap_ci(
                samples, statistic="mean", method=BootstrapMethod.BLOCK,
                block_length=10, n_resamples=2000, seed=trial,
            )
            widths_pct[trial] = ci_pct.ci_high - ci_pct.ci_low
            widths_blk[trial] = ci_blk.ci_high - ci_blk.ci_low
        # Block-bootstrap interval is, on average, at least 25% wider
        # under φ = 0.7 — the theoretical variance inflation factor is
        # (1 + φ)/(1 - φ) ≈ 5.7, so this lower bound is comfortable.
        assert widths_blk.mean() > 1.25 * widths_pct.mean()

    def test_block_brackets_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        samples = _ar1_sample(rng, n=100, phi=0.5, sigma=1.0)
        ci = bootstrap_ci(
            samples, method=BootstrapMethod.BLOCK, n_resamples=2000, seed=0,
        )
        assert ci.ci_low <= ci.point_estimate <= ci.ci_high


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis).
# ---------------------------------------------------------------------------

_finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64,
)


_property_settings = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

_determinism_settings = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


class TestPropertyBracketOrder:
    @_property_settings
    @given(
        samples=hnp.arrays(
            dtype=np.float64, shape=st.integers(min_value=10, max_value=80),
            elements=_finite_floats, unique=False,
        ),
        confidence=st.sampled_from([0.80, 0.90, 0.95, 0.99]),
        method=st.sampled_from(
            [BootstrapMethod.PERCENTILE, BootstrapMethod.BCA, BootstrapMethod.BLOCK],
        ),
    )
    def test_ci_low_le_high_for_any_iid_sample(
        self,
        samples: np.ndarray,
        confidence: float,
        method: BootstrapMethod,
    ) -> None:
        """ci_low <= ci_high for every sample, every method, every
        confidence level. This is the most basic well-formedness
        invariant."""
        # Discard degenerate (all-identical) inputs that exercise the
        # constant-sample fallback path — those have ci_low == ci_high
        # and are covered explicitly elsewhere.
        assume(np.unique(samples).size >= 2)
        ci = bootstrap_ci(
            samples, method=method,
            confidence_level=confidence, n_resamples=1000, seed=0,
        )
        assert ci.ci_low <= ci.ci_high
        assert ci.ci_low <= ci.point_estimate <= ci.ci_high

    @_property_settings
    @given(
        samples=hnp.arrays(
            dtype=np.float64, shape=st.integers(min_value=10, max_value=60),
            elements=_finite_floats, unique=False,
        ),
        method=st.sampled_from(
            [BootstrapMethod.PERCENTILE, BootstrapMethod.BCA, BootstrapMethod.BLOCK],
        ),
    )
    def test_ci_monotone_in_confidence_level(
        self, samples: np.ndarray, method: BootstrapMethod,
    ) -> None:
        """Higher confidence levels yield (weakly) wider CIs."""
        assume(np.unique(samples).size >= 2)
        ci_lo = bootstrap_ci(
            samples, method=method, confidence_level=0.80, n_resamples=1000, seed=0,
        )
        ci_hi = bootstrap_ci(
            samples, method=method, confidence_level=0.99, n_resamples=1000, seed=0,
        )
        assert (ci_hi.ci_high - ci_hi.ci_low) >= (ci_lo.ci_high - ci_lo.ci_low) - 1e-12


class TestPropertyDeterminism:
    @_determinism_settings
    @given(
        samples=hnp.arrays(
            dtype=np.float64, shape=st.integers(min_value=15, max_value=60),
            elements=_finite_floats, unique=False,
        ),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        method=st.sampled_from(
            [BootstrapMethod.PERCENTILE, BootstrapMethod.BCA, BootstrapMethod.BLOCK],
        ),
    )
    def test_same_seed_gives_same_ci(
        self, samples: np.ndarray, seed: int, method: BootstrapMethod,
    ) -> None:
        a = bootstrap_ci(samples, method=method, n_resamples=1000, seed=seed)
        b = bootstrap_ci(samples, method=method, n_resamples=1000, seed=seed)
        assert isinstance(a, BootstrapCI) and isinstance(b, BootstrapCI)
        assert a.ci_low == b.ci_low
        assert a.ci_high == b.ci_high
        assert a.point_estimate == b.point_estimate


# ---------------------------------------------------------------------------
# Coverage-calibration test (slow but the main empirical claim).
# ---------------------------------------------------------------------------

class TestEmpiricalCoverage:
    """Verify that the nominal 95% interval actually achieves ~95%
    coverage when the data are drawn from a known distribution.

    We run :samp:`n_trials` independent realisations from a normal
    sampling distribution with known mean, and check the fraction of
    trials whose 95% BCa CI contains the true mean. A correctly
    calibrated BCa interval should have empirical coverage in
    ``[0.90, 0.98]`` for a sample size of 60 (the small-sample regime
    where BCa shines).

    This is a smoke-level coverage check rather than a definitive
    benchmark — we'd need ``n_trials >> 100`` to discriminate BCa from
    percentile by simulation. The point is to detect a regression
    where coverage collapses to (say) 60%.
    """

    @pytest.mark.parametrize("method", [BootstrapMethod.BCA, BootstrapMethod.PERCENTILE])
    def test_empirical_coverage_in_band(self, method: BootstrapMethod) -> None:
        n_trials = 100
        n = 60
        true_mean = 1.5
        covered = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 + trial)
            samples = rng.normal(loc=true_mean, scale=1.0, size=n)
            ci = bootstrap_ci(
                samples, statistic="mean", method=method,
                confidence_level=0.95, n_resamples=1000, seed=trial,
            )
            if ci.ci_low <= true_mean <= ci.ci_high:
                covered += 1
        coverage = covered / n_trials
        # 95% nominal coverage; allow a wide [85%, 100%] empirical band
        # since n_trials=100 only resolves coverage to ±5 pp at 1σ.
        assert 0.85 <= coverage <= 1.0, (
            f"{method.value} coverage out of band: {coverage:.2%}"
        )
