"""
Statistical machinery for the benchmark.

Every numerical claim made by the framework — "backend A is faster than
backend B", "gradient norms are lower under perturbation", "optimization
converges more reliably" — must be backed by one of the procedures here.
Axes that make a comparison call ``compare_independent`` (or
``compare_paired`` for matched samples). Reports never say "X beats Y"
without surfacing the CI and the post-correction p-value.

Methods
-------
- ``bootstrap_ci``: nonparametric bootstrap confidence interval for any
  statistic of one sample (Efron 1979; Efron & Tibshirani 1993, ch. 13).
  Three interval methods are supported via ``BootstrapMethod``:

  * ``percentile`` (Efron 1979) — straight resample percentiles. Fast
    and asymptotically correct, but biased when the bootstrap
    distribution of the statistic is skewed.
  * ``bca`` (Efron 1987, JASA 82) — bias-corrected and accelerated
    interval. Adjusts the percentile cutoffs using a bias correction
    ``z_0`` (from the rank of the point estimate in the resample
    distribution) and an acceleration ``a`` (from the jackknife
    third moment). Second-order accurate and transformation-respecting
    under monotone reparameterization; the gold-standard
    nonparametric interval for non-normal sampling distributions.
  * ``block`` (Künsch 1989; Politis & Romano 1994) — overlapping
    moving-block bootstrap for serially-correlated samples (e.g.
    timing measurements with thermal drift or warm-cache effects).
    The block length defaults to ``ceil(n^{1/3})`` per Hall, Horowitz
    & Jing (1995, Biometrika 82), which is consistent for stationary
    weakly-dependent sequences.

- ``compare_independent``: Mann–Whitney U test for unpaired samples,
  with Cliff's δ as the effect-size statistic (Cliff 1996). Robust to
  non-normal timing distributions.
- ``compare_paired``: Wilcoxon signed-rank test for matched samples,
  with rank-biserial correlation as effect size (Kerby 2014).
- ``benjamini_hochberg``: BH (1995) step-up procedure for false discovery
  rate control across the family of axis-level p-values.

Design notes
------------
All routines return ``dataclass`` results so reports can render them
uniformly. Sample sizes below a configurable floor (default n=20) produce
``ResultKind.UNDERPOWERED`` rather than a misleading p-value; the report
will then label the comparison "preliminary".

References
----------
Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." Annals
  of Statistics, 7(1), 1-26.
Efron, B. (1987). "Better Bootstrap Confidence Intervals." JASA 82(397),
  171-185.
Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall. (BCa derivation: §14.3.)
Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." Annals of Statistics, 17(3), 1217-1241.
Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
  JASA 89(428), 1303-1313.
Hall, P., Horowitz, J. L., & Jing, B.-Y. (1995). "On Blocking Rules for
  the Bootstrap with Dependent Data." Biometrika 82(3), 561-574.
Cliff, N. (1996). *Ordinal Methods for Behavioral Data Analysis*. Erlbaum.
Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate:
  A Practical and Powerful Approach to Multiple Testing." JRSS-B, 57(1).
Kerby, D. S. (2014). "The simple difference formula: An approach to teaching
  nonparametric correlation." Comprehensive Psychology, 3, 11.IT.3.1.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu, norm, wilcoxon

#: Default per-arm sample count below which we refuse to report a p-value.
#: Below this floor the rank-test asymptotic null distribution is unreliable
#: (Conover 1999, p. 281); we surface ``ResultKind.UNDERPOWERED`` instead of
#: a number that someone might quote out of context. Configurable per-call.
DEFAULT_MIN_SAMPLES_FOR_PVALUE = 20


class StatisticName(StrEnum):
    """Bootstrap reductions supported by :func:`bootstrap_ci`."""

    MEDIAN = "median"
    MEAN = "mean"
    MIN = "min"


class BootstrapMethod(StrEnum):
    """Interval methods supported by :func:`bootstrap_ci`."""

    #: Plain quantile interval of the bootstrap distribution
    #: (Efron 1979). Fastest, but miscalibrated when the sampling
    #: distribution is skewed or biased.
    PERCENTILE = "percentile"

    #: Bias-corrected and accelerated interval (Efron 1987). Adjusts
    #: percentile cutoffs by a bias correction ``z_0`` and a jackknife
    #: acceleration ``a``. Second-order accurate.
    BCA = "bca"

    #: Overlapping moving-block bootstrap (Künsch 1989) for
    #: serially-correlated samples. Block length defaults to
    #: ``ceil(n^{1/3})`` per Hall–Horowitz–Jing 1995.
    BLOCK = "block"


class ResultKind(StrEnum):
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    UNDERPOWERED = "underpowered"


_REDUCERS = {
    StatisticName.MEDIAN: np.median,
    StatisticName.MEAN: np.mean,
    StatisticName.MIN: np.min,
}


@dataclass(frozen=True)
class BootstrapCI:
    """Confidence interval for a single-sample statistic.

    ``method`` records which interval procedure produced ``ci_low`` and
    ``ci_high``. For the block bootstrap, ``block_length`` records the
    block length used (either user-provided or auto-selected from
    ``ceil(n^{1/3})``); for other methods it is ``None``.
    """

    statistic_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float  # e.g. 0.95
    n_resamples: int
    n_samples: int
    method: str = BootstrapMethod.PERCENTILE.value
    block_length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Comparison:
    """Pairwise statistical comparison of two samples."""

    arm_a_name: str
    arm_b_name: str
    median_a: float
    median_b: float
    median_diff: float  # arm_a - arm_b
    median_ratio: float | None  # arm_a / arm_b when both positive, else None
    n_a: int
    n_b: int
    test_name: str  # e.g. "mann-whitney-u" or "wilcoxon-signed-rank"
    test_statistic: float
    p_value_raw: float
    effect_size_name: str  # e.g. "cliffs-delta" or "rank-biserial"
    effect_size: float
    kind: ResultKind
    # Filled in by ``benjamini_hochberg`` when comparisons are batched.
    p_value_bh_adjusted: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass(frozen=True)
class CorrectedFamily:
    """Result of applying BH FDR control to a family of comparisons."""

    comparisons: list[Comparison]
    alpha: float
    n_significant_after_bh: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "n_significant_after_bh": self.n_significant_after_bh,
            "comparisons": [c.as_dict() for c in self.comparisons],
        }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _percentile_interval(
    boot: NDArray[np.float64], confidence_level: float
) -> tuple[float, float]:
    """Two-sided quantile cutoffs of the bootstrap distribution."""
    alpha = 1.0 - confidence_level
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0, method="linear"))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0), method="linear"))
    return lo, hi


def _jackknife_acceleration(
    samples: NDArray[np.float64], reducer: Any
) -> float:
    """Jackknife estimate of the acceleration constant ``a`` (Efron 1987).

    .. math::

        a = \\frac{\\sum_i (\\bar\\theta_{(\\cdot)} - \\hat\\theta_{(i)})^3}
                  {6\\left[\\sum_i (\\bar\\theta_{(\\cdot)} -
                  \\hat\\theta_{(i)})^2\\right]^{3/2}}

    where ``θ̂_{(i)}`` is the statistic on the leave-one-out sample and
    ``θ̄_{(·)} = mean_i θ̂_{(i)}``. ``a`` measures the rate of change of
    the statistic's standard error w.r.t. the true parameter — what BCa
    corrects beyond plain bias.

    Returns 0.0 when the jackknife denominator is degenerate (a constant
    sample), which collapses BCa to a pure bias-corrected interval.
    """
    n = samples.size
    # Vectorised leave-one-out: index matrix of shape (n, n-1).
    full = np.arange(n)
    idx = np.tile(full, (n, 1))
    keep = idx[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    theta_minus_i = np.asarray(reducer(samples[keep], axis=1), dtype=np.float64)
    theta_bar = theta_minus_i.mean()
    diffs = theta_bar - theta_minus_i
    num = float((diffs**3).sum())
    denom = 6.0 * float((diffs**2).sum()) ** 1.5
    if denom == 0.0:
        return 0.0
    return num / denom


def _bca_interval(
    samples: NDArray[np.float64],
    boot: NDArray[np.float64],
    reducer: Any,
    confidence_level: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval (Efron 1987).

    Combines bias correction ``z_0`` from the rank of the point estimate
    in the bootstrap distribution with the jackknife acceleration ``a``
    to map a target two-sided level ``α`` to bootstrap-distribution
    quantiles.
    """
    theta_hat = float(reducer(samples))
    n_boot = boot.size
    # Bias correction: how often the bootstrap statistic falls below θ̂.
    n_less = int((boot < theta_hat).sum())
    # Continuity correction for ties at θ̂ (avoid z_0 → ±∞).
    n_eq = int((boot == theta_hat).sum())
    prop = (n_less + 0.5 * n_eq) / n_boot
    prop = min(max(prop, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = float(norm.ppf(prop))
    a = _jackknife_acceleration(samples, reducer)

    alpha = 1.0 - confidence_level
    z_lo = norm.ppf(alpha / 2.0)
    z_hi = norm.ppf(1.0 - alpha / 2.0)
    # Adjusted target percentiles α₁, α₂ (Efron 1987 eq. (2.2)).
    denom_lo = 1.0 - a * (z0 + z_lo)
    denom_hi = 1.0 - a * (z0 + z_hi)
    if denom_lo == 0.0 or denom_hi == 0.0:  # pragma: no cover  -- degenerate
        # Fall back to a pure bias-corrected interval if acceleration
        # makes the denominator vanish (a numerically singular sample).
        a = 0.0
        denom_lo = denom_hi = 1.0
    alpha_lo = float(norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha_hi = float(norm.cdf(z0 + (z0 + z_hi) / denom_hi))
    # Clamp to (0, 1) so np.percentile is well-defined.
    eps = 1.0 / (2.0 * n_boot)
    alpha_lo = min(max(alpha_lo, eps), 1.0 - eps)
    alpha_hi = min(max(alpha_hi, eps), 1.0 - eps)
    lo = float(np.percentile(boot, 100.0 * alpha_lo, method="linear"))
    hi = float(np.percentile(boot, 100.0 * alpha_hi, method="linear"))
    return lo, hi


def _block_bootstrap_distribution(
    samples: NDArray[np.float64],
    reducer: Any,
    n_resamples: int,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Overlapping moving-block bootstrap distribution (Künsch 1989).

    The sample is divided into ``n_blocks = ceil(n / block_length)``
    overlapping blocks, one starting at each valid position. Each
    resample draws ``n_blocks`` blocks with replacement, concatenates
    them, truncates to length ``n``, and applies the reducer.
    """
    n = samples.size
    n_blocks = (n + block_length - 1) // block_length
    n_starts = n - block_length + 1
    starts = rng.integers(0, n_starts, size=(n_resamples, n_blocks))
    block_offset = np.arange(block_length)
    # Fully vectorised: build (n_resamples, n_blocks * block_length) index
    # matrix in one allocation, trim to n along the last axis, then run the
    # reducer along axis=1. Memory cost is O(n_resamples · n) int64s —
    # ≤ 16 MB for the sample sizes the benchmark targets (n ≤ 200,
    # n_resamples ≤ 10000). The previous per-resample Python loop incurred
    # one ``reducer`` call per resample, which dominated wall time for
    # small ``n``.
    all_indices = (
        starts[:, :, None] + block_offset[None, None, :]
    ).reshape(n_resamples, -1)[:, :n]
    return np.asarray(reducer(samples[all_indices], axis=1), dtype=np.float64)


def _auto_block_length(n: int) -> int:
    """Heuristic block length ``ceil(n^{1/3})`` (Hall–Horowitz–Jing 1995).

    Optimal for the mean of stationary weakly-dependent sequences.
    Capped at ``n // 2`` so at least two blocks fit.
    """
    L = max(1, math.ceil(n ** (1.0 / 3.0)))
    return min(L, max(1, n // 2))


def bootstrap_ci(
    samples: NDArray[np.floating],
    *,
    statistic: StatisticName | str = StatisticName.MEDIAN,
    method: BootstrapMethod | str = BootstrapMethod.PERCENTILE,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    block_length: int | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """Nonparametric bootstrap CI for ``statistic`` on ``samples``.

    Parameters
    ----------
    samples : 1-D ndarray
        Observed sample. Must contain at least two finite values; ``n == 1``
        is degenerate (the only resample is the sample itself) and we raise.
    statistic : :class:`StatisticName` or one of ``"median"``, ``"mean"``, ``"min"``.
        Reduction to bootstrap. Median is the default because timing
        distributions are heavy-tailed and asymmetric.
    method : :class:`BootstrapMethod` or one of ``"percentile"``, ``"bca"``,
        ``"block"``.

        - ``percentile`` (Efron 1979): straight quantile interval.
        - ``bca`` (Efron 1987): bias-corrected and accelerated interval.
          Second-order accurate; preferred for skewed sampling
          distributions.
        - ``block`` (Künsch 1989): moving-block bootstrap with
          overlapping reseating, for serially-correlated samples
          (e.g. timing measurements with thermal drift).
    confidence_level : float in (0, 1)
        Two-sided interval coverage.
    n_resamples : int
        Number of bootstrap resamples. ``10_000`` is a common floor for
        2-decimal-place CI stability (Davidson & MacKinnon 2000, JEDC 24).
    block_length : int or None
        Only used by ``method=block``. ``None`` selects
        ``ceil(n^{1/3})`` per Hall–Horowitz–Jing 1995.
    seed : int
        Seeds a local ``numpy.random.default_rng``. Does not touch the
        global RNG state.

    Notes
    -----
    The percentile and BCa branches assume ``samples`` are i.i.d.. For
    serially-correlated data (warm caches, thermal drift, autoregressive
    timing) use the block-bootstrap branch — it preserves the
    short-range dependence within each block while still allowing the
    resampling to explore the sampling distribution.

    BCa accuracy
    ~~~~~~~~~~~~
    BCa is second-order accurate
    (coverage error ``O(n^{-1})`` vs ``O(n^{-1/2})`` for percentile;
    DiCiccio & Efron 1996, Stat. Sci. 11). On heavy-tailed or biased
    sampling distributions it can shift the interval by tens of
    percent relative to the plain percentile interval.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D, got shape {samples.shape}")
    if samples.size < 2:
        raise ValueError(
            f"samples must contain at least 2 values, got n={samples.size}"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples contains non-finite values")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0,1), got {confidence_level}")
    if n_resamples < 1000:
        raise ValueError(
            f"n_resamples={n_resamples} is too low for stable CIs; "
            "use >= 1000 (10000+ recommended; Davidson & MacKinnon 2000)"
        )

    stat = StatisticName(statistic) if not isinstance(statistic, StatisticName) else statistic
    bootstrap_method = (
        BootstrapMethod(method) if not isinstance(method, BootstrapMethod) else method
    )
    reducer: Any = _REDUCERS[stat]

    rng = np.random.default_rng(seed)
    n = samples.size
    block_length_used: int | None = None

    if bootstrap_method is BootstrapMethod.BLOCK:
        if block_length is None:
            block_length_used = _auto_block_length(n)
        else:
            if block_length < 1 or block_length > n:
                raise ValueError(
                    f"block_length={block_length} must satisfy 1 <= L <= n={n}"
                )
            block_length_used = int(block_length)
        boot = _block_bootstrap_distribution(
            samples, reducer, n_resamples, block_length_used, rng,
        )
    else:
        idx = rng.integers(0, n, size=(n_resamples, n))
        boot = reducer(samples[idx], axis=1)

    if bootstrap_method is BootstrapMethod.BCA:
        lo, hi = _bca_interval(samples, boot, reducer, confidence_level)
    else:
        lo, hi = _percentile_interval(boot, confidence_level)

    return BootstrapCI(
        statistic_name=stat.value,
        point_estimate=float(reducer(samples)),
        ci_low=lo,
        ci_high=hi,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_samples=n,
        method=bootstrap_method.value,
        block_length=block_length_used,
    )


# ---------------------------------------------------------------------------
# Two-sample comparisons
# ---------------------------------------------------------------------------

def _cliffs_delta(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cliff's δ: P(a > b) - P(a < b). Range [-1, 1]. Magnitude = effect size."""
    # Naive O(n_a n_b) is fine for the sample sizes here (typically ≤ 100).
    diff = a[:, None] - b[None, :]
    gt = float((diff > 0).sum())
    lt = float((diff < 0).sum())
    return (gt - lt) / float(a.size * b.size)


def compare_independent(
    arm_a: NDArray[np.floating],
    arm_b: NDArray[np.floating],
    *,
    arm_a_name: str,
    arm_b_name: str,
    min_samples_for_pvalue: int = DEFAULT_MIN_SAMPLES_FOR_PVALUE,
) -> Comparison:
    """Mann–Whitney U test + Cliff's δ for two independent samples.

    Use this when the two arms were generated from independent processes
    (e.g. timing measurements on different backends with different seeds).
    For matched samples (same seed, paired observations) use
    :func:`compare_paired` — it is strictly more powerful.

    Returns
    -------
    Comparison
        ``kind`` is ``UNDERPOWERED`` when ``min(n_a, n_b) < min_samples_for_pvalue``;
        otherwise ``NOT_SIGNIFICANT`` until :func:`benjamini_hochberg` is
        applied to the comparison family.
    """
    arm_a = np.asarray(arm_a, dtype=np.float64)
    arm_b = np.asarray(arm_b, dtype=np.float64)
    if arm_a.ndim != 1 or arm_b.ndim != 1:
        raise ValueError("arms must be 1-D")
    if arm_a.size == 0 or arm_b.size == 0:
        raise ValueError("arms must be non-empty")

    med_a = float(np.median(arm_a))
    med_b = float(np.median(arm_b))
    ratio: float | None = med_a / med_b if (med_a > 0 and med_b > 0) else None

    if min(arm_a.size, arm_b.size) < min_samples_for_pvalue:
        return Comparison(
            arm_a_name=arm_a_name, arm_b_name=arm_b_name,
            median_a=med_a, median_b=med_b,
            median_diff=med_a - med_b, median_ratio=ratio,
            n_a=int(arm_a.size), n_b=int(arm_b.size),
            test_name="mann-whitney-u",
            test_statistic=float("nan"), p_value_raw=float("nan"),
            effect_size_name="cliffs-delta",
            effect_size=_cliffs_delta(arm_a, arm_b),
            kind=ResultKind.UNDERPOWERED,
        )

    u_stat, p_raw = mannwhitneyu(arm_a, arm_b, alternative="two-sided")
    delta = _cliffs_delta(arm_a, arm_b)

    return Comparison(
        arm_a_name=arm_a_name, arm_b_name=arm_b_name,
        median_a=med_a, median_b=med_b,
        median_diff=med_a - med_b, median_ratio=ratio,
        n_a=int(arm_a.size), n_b=int(arm_b.size),
        test_name="mann-whitney-u",
        test_statistic=float(u_stat), p_value_raw=float(p_raw),
        effect_size_name="cliffs-delta",
        effect_size=float(delta),
        kind=ResultKind.NOT_SIGNIFICANT,
    )


def compare_paired(
    arm_a: NDArray[np.floating],
    arm_b: NDArray[np.floating],
    *,
    arm_a_name: str,
    arm_b_name: str,
    min_samples_for_pvalue: int = DEFAULT_MIN_SAMPLES_FOR_PVALUE,
) -> Comparison:
    """Wilcoxon signed-rank test + rank-biserial r for matched samples.

    Use this when arms share a seeding/index dimension (e.g. for each seed
    ``s`` in ``[0..n)``, measure both backends on the same input). Strictly
    more powerful than :func:`compare_independent` for matched data
    (Wilcoxon 1945; relative efficiency ≈ 0.955 of the paired t under
    normality, ≥ 1 under most non-normal alternatives — Conover 1999).

    Edge case
    ---------
    When *every* paired diff is exactly zero the Wilcoxon test is
    degenerate (scipy raises). We detect this and return a
    ``NOT_SIGNIFICANT`` comparison with ``p_value_raw = 1.0`` and
    ``effect_size = 0`` — the honest interpretation of "no signal".
    """
    arm_a = np.asarray(arm_a, dtype=np.float64)
    arm_b = np.asarray(arm_b, dtype=np.float64)
    if arm_a.shape != arm_b.shape or arm_a.ndim != 1:
        raise ValueError("paired arms must have the same 1-D shape")
    if arm_a.size == 0:
        raise ValueError("arms must be non-empty")

    med_a = float(np.median(arm_a))
    med_b = float(np.median(arm_b))
    ratio: float | None = med_a / med_b if (med_a > 0 and med_b > 0) else None

    if arm_a.size < min_samples_for_pvalue:
        return Comparison(
            arm_a_name=arm_a_name, arm_b_name=arm_b_name,
            median_a=med_a, median_b=med_b,
            median_diff=med_a - med_b, median_ratio=ratio,
            n_a=int(arm_a.size), n_b=int(arm_b.size),
            test_name="wilcoxon-signed-rank",
            test_statistic=float("nan"), p_value_raw=float("nan"),
            effect_size_name="rank-biserial",
            effect_size=float("nan"),
            kind=ResultKind.UNDERPOWERED,
        )

    diffs = arm_a - arm_b
    if np.all(diffs == 0.0):
        return Comparison(
            arm_a_name=arm_a_name, arm_b_name=arm_b_name,
            median_a=med_a, median_b=med_b,
            median_diff=0.0, median_ratio=ratio,
            n_a=int(arm_a.size), n_b=int(arm_b.size),
            test_name="wilcoxon-signed-rank",
            test_statistic=0.0, p_value_raw=1.0,
            effect_size_name="rank-biserial",
            effect_size=0.0,
            kind=ResultKind.NOT_SIGNIFICANT,
        )

    res = wilcoxon(arm_a, arm_b, alternative="two-sided", zero_method="wilcox")
    w_stat = float(res.statistic)
    p_raw = float(res.pvalue)

    # Rank-biserial r = (n_pos - n_neg) / n_nonzero (Kerby 2014).
    nz = diffs[diffs != 0.0]
    r_rb = (
        float(int(np.sum(diffs > 0)) - int(np.sum(diffs < 0))) / float(nz.size)
        if nz.size > 0
        else 0.0
    )

    return Comparison(
        arm_a_name=arm_a_name, arm_b_name=arm_b_name,
        median_a=med_a, median_b=med_b,
        median_diff=med_a - med_b, median_ratio=ratio,
        n_a=int(arm_a.size), n_b=int(arm_b.size),
        test_name="wilcoxon-signed-rank",
        test_statistic=w_stat, p_value_raw=p_raw,
        effect_size_name="rank-biserial",
        effect_size=r_rb,
        kind=ResultKind.NOT_SIGNIFICANT,
    )


# ---------------------------------------------------------------------------
# Multi-test correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(comparisons: list[Comparison], alpha: float = 0.05) -> CorrectedFamily:
    """Apply the BH (1995) step-up FDR control to a family of comparisons.

    Pure function: returns a new :class:`CorrectedFamily` whose
    ``comparisons`` list is in the **same positional order** as the input.
    Each non-underpowered comparison has its ``p_value_bh_adjusted`` and
    ``kind`` populated. ``UNDERPOWERED`` comparisons pass through unchanged
    — they are not part of the multi-test family because we have no valid
    p-value to correct.

    The BH step-up adjusted p-value at sorted rank ``k`` is
    ``min_{j >= k} p_(j) * m / j`` (Benjamini & Yekutieli 2001, monotone
    enforcement of the original BH 1995 critical values).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    powered = [c for c in comparisons if c.kind != ResultKind.UNDERPOWERED]
    underpowered = [c for c in comparisons if c.kind == ResultKind.UNDERPOWERED]

    if not powered:
        return CorrectedFamily(
            comparisons=list(comparisons), alpha=alpha, n_significant_after_bh=0,
        )

    m = len(powered)
    # Sort by raw p ascending; track original positions for write-back.
    sorted_with_idx = sorted(enumerate(powered), key=lambda kv: kv[1].p_value_raw)
    adjusted = [0.0] * m
    running_min = 1.0
    # BH adjusted p-value definition (Benjamini & Yekutieli 2001 step-up form):
    # p_bh_(k) = min_{j>=k} ( p_(j) * m / j )
    for rank_from_end in range(m, 0, -1):
        i = rank_from_end - 1  # 0-based index into sorted list
        _, comp = sorted_with_idx[i]
        candidate = comp.p_value_raw * m / float(rank_from_end)
        running_min = min(running_min, candidate)
        adjusted[i] = min(running_min, 1.0)

    # Write back into a new list of Comparison objects.
    rewritten: dict[int, Comparison] = {}
    n_significant = 0
    for sorted_pos, (orig_idx, comp) in enumerate(sorted_with_idx):
        p_adj = adjusted[sorted_pos]
        is_sig = p_adj < alpha
        kind = ResultKind.SIGNIFICANT if is_sig else ResultKind.NOT_SIGNIFICANT
        if is_sig:
            n_significant += 1
        rewritten[orig_idx] = Comparison(
            arm_a_name=comp.arm_a_name, arm_b_name=comp.arm_b_name,
            median_a=comp.median_a, median_b=comp.median_b,
            median_diff=comp.median_diff, median_ratio=comp.median_ratio,
            n_a=comp.n_a, n_b=comp.n_b,
            test_name=comp.test_name,
            test_statistic=comp.test_statistic,
            p_value_raw=comp.p_value_raw,
            effect_size_name=comp.effect_size_name,
            effect_size=comp.effect_size,
            kind=kind,
            p_value_bh_adjusted=p_adj,
        )

    # Reassemble in original order, mixing in the underpowered ones unchanged.
    out: list[Comparison] = []
    powered_iter = iter(rewritten[i] for i in range(m))
    underpowered_iter = iter(underpowered)
    for orig in comparisons:
        if orig.kind == ResultKind.UNDERPOWERED:
            out.append(next(underpowered_iter))
        else:
            out.append(next(powered_iter))

    return CorrectedFamily(comparisons=out, alpha=alpha, n_significant_after_bh=n_significant)


__all__ = [
    "DEFAULT_MIN_SAMPLES_FOR_PVALUE",
    "BootstrapCI",
    "Comparison",
    "CorrectedFamily",
    "ResultKind",
    "StatisticName",
    "benjamini_hochberg",
    "bootstrap_ci",
    "compare_independent",
    "compare_paired",
]
