"""Reproduce the investigation-wide multiple-testing summary from the artifacts.

This script pools every ``pairwise_comparisons`` entry across the JSON result
files in ``notebooks/results/`` and recomputes, using the repository's own
:func:`benchmarks.stats.benjamini_hochberg`, the investigation-wide statistics
reported in ``docs/STATISTICAL_SUMMARY.md`` §2:

  * the 59 distinct comparisons (primary) and the 76-with-re-reports pool,
  * the per-procedure counts (Benjamini-Hochberg vs Bonferroni),
  * the per-claim rank and Benjamini-Hochberg critical value used by the
    "Key claims under investigation-wide correction" table.

The rank is the comparison's position in the p-value-ascending ordering
(competition rank = number of strictly-smaller p-values + 1, i.e. the first
slot in a tie block). The Benjamini-Hochberg critical value at rank ``k`` over
``m`` comparisons is ``(k / m) * alpha`` (Benjamini & Hochberg 1995).

Run from the repository root::

    python -m scripts.investigation_wide_bh
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks.stats import Comparison, ResultKind, benjamini_hochberg

RESULTS_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "results"
ALPHA = 0.05


@dataclass(frozen=True)
class PooledComparison:
    """A single pooled comparison with its source file + dataset for provenance."""

    source: str
    dataset: str
    comparison: Comparison


def load_pool(results_dir: Path = RESULTS_DIR) -> list[PooledComparison]:
    """Pool every ``pairwise_comparisons`` entry across the result JSONs."""
    pool: list[PooledComparison] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("pairwise_comparisons", []):
            pool.append(
                PooledComparison(
                    source=path.name,
                    dataset=entry.get("dataset") or path.name,
                    comparison=Comparison(
                        arm_a_name=entry["arm_a_name"],
                        arm_b_name=entry["arm_b_name"],
                        median_a=entry["median_a"],
                        median_b=entry["median_b"],
                        median_diff=entry["median_diff"],
                        median_ratio=entry["median_ratio"],
                        n_a=entry["n_a"],
                        n_b=entry["n_b"],
                        test_name=entry["test_name"],
                        test_statistic=entry["test_statistic"],
                        p_value_raw=entry["p_value_raw"],
                        effect_size_name=entry["effect_size_name"],
                        effect_size=entry["effect_size"],
                        kind=ResultKind(entry["kind"]),
                    ),
                )
            )
    return pool


def rank_and_threshold(p_raw: float, p_values: list[float], alpha: float = ALPHA) -> tuple[int, float]:
    """Return the competition rank and Benjamini-Hochberg critical value for ``p_raw``.

    Rank = (number of strictly-smaller p-values) + 1; the critical value at
    rank ``k`` over ``m`` comparisons is ``(k / m) * alpha``.
    """
    m = len(p_values)
    rank = sum(1 for p in p_values if p < p_raw) + 1
    return rank, (rank / m) * alpha


def summarize(comparisons: list[Comparison], label: str) -> None:
    """Print BH / Bonferroni / non-significant counts for one comparison pool."""
    p_values = [c.p_value_raw for c in comparisons]
    m = len(comparisons)
    n_bh = benjamini_hochberg(comparisons, alpha=ALPHA).n_significant_after_bh
    bonferroni_threshold = ALPHA / m
    n_bonferroni = sum(1 for p in p_values if p < bonferroni_threshold)
    print(
        f"  {label}: m={m}; BH {n_bh}/{m}; "
        f"Bonferroni (alpha/{m}={bonferroni_threshold:.2e}) {n_bonferroni}/{m}; "
        f"non-significant {m - n_bh}/{m}"
    )


def main() -> None:
    pool = load_pool()
    comparisons = [pc.comparison for pc in pool]

    # De-duplicate: the same (dataset, unordered arm-pair, raw p-value) carried
    # into multiple hypothesis families is ONE distinct comparison. The
    # investigation-wide FDR is computed over the distinct set (primary); the
    # full pool with re-reports is reported alongside for transparency.
    # Key on the EXACT raw p-value: the re-reports are byte-identical floats, so
    # exact equality de-duplicates them without rounding fragility (verified: no
    # two distinct comparisons are within 1e-10 but unequal). The dataset falls
    # back to the source filename when absent, so a missing field never silently
    # merges comparisons across files.
    seen: set[tuple[str, tuple[str, ...], float]] = set()
    distinct: list[PooledComparison] = []
    for pc in pool:
        key = (
            pc.dataset,
            tuple(sorted([pc.comparison.arm_a_name, pc.comparison.arm_b_name])),
            pc.comparison.p_value_raw,
        )
        if key not in seen:
            seen.add(key)
            distinct.append(pc)
    distinct_comparisons = [pc.comparison for pc in distinct]
    distinct_p_values = [c.p_value_raw for c in distinct_comparisons]
    m = len(distinct_comparisons)

    n_files = len(list(RESULTS_DIR.glob("*.json")))
    print(f"Result files: {n_files}")
    print(
        f"Total computed: {len(comparisons)}; distinct: {m} "
        f"({len(comparisons) - m} exact re-reports removed)"
    )
    print("Investigation-wide FDR:")
    summarize(distinct_comparisons, "distinct (PRIMARY)")
    summarize(comparisons, "full (with re-reports)")
    print()

    # The four claims tabulated in docs/STATISTICAL_SUMMARY.md §2, identified
    # by (source file, arm A, arm B) so the lookup is exact. Ranks/thresholds
    # are computed over the DISTINCT (primary) pool.
    claims = [
        ("Hodge-residual > MLP on NCI1 (H003)", "nci1_hodge_ablation_30seeds.json", "hodge-mp-residual", "mlp-baseline"),
        ("gin-residual > MLP on NCI1 (H008-c)", "h008c_nci1_gin_residual_30seeds.json", "gin-residual", "mlp-baseline"),
        ("Hodge-residual > GIN on NCI1 (H008)", "h008_nci1_gin_gat_30seeds.json", "hodge-mp-residual", "gin-baseline"),
        ("gin-residual > gin-normalised on NCI1 (H008-c)", "h008c_nci1_gin_residual_30seeds.json", "gin-residual", "gin-normalised"),
    ]
    bonferroni_threshold = ALPHA / m
    largest_passing_rank = max(
        (k for k, p in enumerate(sorted(distinct_p_values), start=1) if p <= (k / m) * ALPHA),
        default=0,
    )

    print("Key claims under investigation-wide correction (ranks over the distinct pool):")
    header = f"{'Claim':<48} {'p_raw':>12} {'rank':>7} {'BH thr':>11} {'BH':>4} {'Bonf':>5}"
    print(header)
    for label, source, arm_a, arm_b in claims:
        match = next(
            pc for pc in pool
            if pc.source == source
            and pc.comparison.arm_a_name == arm_a
            and pc.comparison.arm_b_name == arm_b
        )
        p_raw = match.comparison.p_value_raw
        rank, threshold = rank_and_threshold(p_raw, distinct_p_values)
        survives_bh = "Yes" if rank <= largest_passing_rank else "No"
        survives_bonferroni = "Yes" if p_raw < bonferroni_threshold else "No"
        print(
            f"{label:<48} {p_raw:>12.2e} {f'{rank}/{m}':>7} {threshold:>11.2e} "
            f"{survives_bh:>4} {survives_bonferroni:>5}"
        )


if __name__ == "__main__":
    main()
