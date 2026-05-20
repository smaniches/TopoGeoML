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
  statistic of one sample (Efron 1979; Efron & Tibshirani 1993, chapter 13).
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
Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
Cliff, N. (1996). *Ordinal Methods for Behavioral Data Analysis*. Erlbaum.
Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate:
  A Practical and Powerful Approach to Multiple Testing." JRSS-B, 57(1).
Kerby, D. S. (2014). "The simple difference formula: An approach to teaching
  nonparametric correlation." Comprehensive Psychology, 3, 11.IT.3.1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu, wilcoxon

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
    """Confidence interval for a single-sample statistic."""

    statistic_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float  # e.g. 0.95
    n_resamples: int
    n_samples: int

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

def bootstrap_ci(
    samples: NDArray[np.floating],
    *,
    statistic: StatisticName | str = StatisticName.MEDIAN,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Nonparametric percentile bootstrap CI for ``statistic`` on ``samples``.

    Parameters
    ----------
    samples : 1-D ndarray
        Observed sample. Must contain at least two finite values; ``n == 1``
        is degenerate (the only resample is the sample itself) and we raise.
    statistic : :class:`StatisticName` or one of ``"median"``, ``"mean"``, ``"min"``.
        Reduction to bootstrap. Median is the default because timing
        distributions are heavy-tailed and asymmetric.
    confidence_level : float in (0, 1)
        Two-sided percentile interval width.
    n_resamples : int
        Number of bootstrap resamples. ``10_000`` is a common floor for
        2-decimal-place CI stability (Davidson & MacKinnon 2000, JEDC 24).
    seed : int
        Seeds a local ``numpy.random.default_rng``. Does not touch the
        global RNG state.

    Notes
    -----
    Implements the **percentile** bootstrap. The percentile interval is
    not transformation-invariant and can be miscalibrated for highly
    skewed sampling distributions; the BCa interval (Efron 1987, JASA 82)
    addresses this and is a planned addition (item 2 in the framework
    self-rejection log). For timing distributions encountered here the
    percentile interval is adequate.

    The procedure assumes ``samples`` are i.i.d. Timing measurements
    drawn from the same process exhibit serial correlation (warm caches,
    thermal drift); the i.i.d. assumption is approximately satisfied
    when the measurement window contains a GC-disabled section, a
    process-isolated runner, and sufficient warm-up. Block-bootstrap
    (Künsch 1989, Annals 17) is appropriate when those conditions cannot
    be guaranteed and is also planned.
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
            f"n_resamples={n_resamples} is too low for stable percentile CIs; "
            "use >= 1000 (10000+ recommended; Davidson & MacKinnon 2000)"
        )

    stat = StatisticName(statistic) if not isinstance(statistic, StatisticName) else statistic
    reducer: Any = _REDUCERS[stat]

    rng = np.random.default_rng(seed)
    n = samples.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = reducer(samples[idx], axis=1)

    alpha = 1.0 - confidence_level
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0, method="linear"))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0), method="linear"))
    return BootstrapCI(
        statistic_name=stat.value,
        point_estimate=float(reducer(samples)),
        ci_low=lo,
        ci_high=hi,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_samples=n,
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
