---
title: Statistical summary
nav_order: 4
---

# Investigation-Wide Statistical Summary

This document reports the statistical properties of the complete TopoGeoML investigation (H001-H011) at the investigation level, addressing multiple-testing burden, power, and the false discovery rate across all experiments.

## 1. Scope

- **Distinct pairwise comparisons (primary unit of analysis):** 59
- **Total comparisons computed:** 76 — the 59 distinct plus 17 exact re-reports (the same baseline comparison is carried into multiple hypothesis families; e.g. the NCI1 hodge-residual-vs-MLP test recurs across 7 result files)
- **Result files:** 22 JSON artifacts in `notebooks/results/`
- **Hypotheses:** 14 preregistered documents, 12 resolved
- **Sub-predictions:** 50+ falsifiable sub-hypotheses

## 2. Multiple-Testing Correction

### Per-hypothesis correction (as reported)

Each hypothesis applies Benjamini-Hochberg FDR at alpha=0.05 within its own comparison family (typically 3-10 comparisons per hypothesis). This controls the false discovery rate within each experiment but does not control the investigation-wide FDR.

### Investigation-wide BH-FDR

The investigation-wide FDR correction is computed over the **59 distinct comparisons** — the primary unit of analysis. The 76-total figure includes 17 exact re-reports (the same baseline comparison carried into multiple hypothesis families); a global FDR over those 76 double-counts re-reported baselines and inflates both the denominator and the significant count. Both pools are reported below for transparency.

| Investigation-wide correction (alpha=0.05) | Over 59 distinct (primary) | Over 76 with re-reports |
|---|---|---|
| Significant at investigation-wide BH | 31/59 (53%) | 47/76 (62%) |
| Surviving Bonferroni (alpha=0.05/m) | 16/59 (27%) | 22/76 (29%) |
| Non-significant | 28/59 (47%) | 29/76 (38%) |

Per-hypothesis BH — applied within each hypothesis's own comparison family, as each hypothesis document reports — yields 47 family-level significant results; that within-family correction is unaffected by investigation-wide de-duplication.

**The headline conclusions hold under either denominator:** all four key claims below survive investigation-wide BH identically over the 59 distinct and the 76 pools (only the ranks/thresholds shift). De-duplication moves the significant fraction down (62% -> 53%), the conservative direction. Not all primary claims survive Bonferroni: the H003 NCI1 Hodge-residual > MLP comparison (p_raw = 3.38 x 10^-3) survives investigation-wide BH but not Bonferroni, whereas the residual-placement finding (gin-residual > MLP, p_raw = 4.03 x 10^-4) survives both. See the key claims table below.

### Key claims under investigation-wide correction

| Claim | p_raw | Rank (of 59) | BH threshold | Survives global BH | Survives Bonferroni |
|---|---|---|---|---|---|
| Hodge-residual > MLP on NCI1 (H003) | 3.38 x 10^-3 | 22/59 | 1.86 x 10^-2 | Yes | No |
| gin-residual > MLP on NCI1 (H008-c) | 4.03 x 10^-4 | 14/59 | 1.19 x 10^-2 | Yes | Yes |
| Hodge-residual > GIN on NCI1 (H008) | 2.12 x 10^-6 | 5/59 | 4.24 x 10^-3 | Yes | Yes |
| gin-residual > gin-normalised on NCI1 (H008-c) | 1.73 x 10^-6 | 2/59 | 1.69 x 10^-3 | Yes | Yes |

Ranks are over the 59 distinct comparisons (primary). The same four verdicts hold over the 76-with-re-reports pool — only the ranks/thresholds shift (there H003 is rank 29/76, threshold 1.91 x 10^-2). The NCI1 Hodge-vs-MLP claim (p_raw = 3.38 x 10^-3) survives investigation-wide BH but NOT Bonferroni; under the most conservative correction it would require a lower alpha or more seeds to be confirmed. The residual-placement finding (gin-residual vs MLP, p = 4.03 x 10^-4) survives both.

## 3. Statistical Power

For paired Wilcoxon signed-rank at alpha=0.05, power=0.80:

| Seeds (n) | Minimum detectable effect (|r|) |
|---|---|
| 18 | 0.373 |
| 30 | 0.289 |

At n=30 seeds, the investigation has 80% power to detect effects with rank-biserial |r| >= 0.289. Effects smaller than this may produce non-significant results due to insufficient power rather than absence of effect.

**Implications for null results:** When we report "no significant difference" (e.g., Hodge matches MLP on PROTEINS, p_BH = 0.548), this is a failure to reject the null at the tested power level, not evidence of equality. An equivalence test (TOST procedure) would be needed to make a positive equality claim. We do not make such claims; all null results are reported as "not significantly different at the tested configuration."

## 4. Post-Hoc Hypothesis Generation

H001-H007 were designed sequentially: each hypothesis's design was informed by the prior hypothesis's outcome. H008-H010 were designed after seeing H001-H007 results, and H011 (with its COLLAB follow-up H011b) was designed after H001-H010 — H011 to test higher-order L₁ structure once H008-c had shown the L₀ operator choice was secondary, and H011b after H011's degenerate NCI1 result. While each hypothesis was preregistered (committed to git before execution), the hypothesis *selection* was data-driven.

This is legitimate sequential testing (Pocock 1977; O'Brien & Fleming 1979), not p-hacking, provided:
1. Each hypothesis was committed before its experiment ran (verified by git timestamps)
2. All results — positive and negative — are reported (verified: 28 of the 59 distinct comparisons, 47%, are non-significant under investigation-wide BH; 38% of the 76-with-re-reports pool)
3. The per-hypothesis BH correction accounts for the within-family multiplicity

The investigation-wide BH analysis in §2 provides the additional global correction.

## 5. Configuration Scope

All reported results are bounded to:

| Parameter | Value |
|---|---|
| Architecture depth | 1 layer (2 for deep-residual arm) |
| Hidden dimension | 32 |
| Training epochs | 10-20 |
| Optimiser | Adam, lr=1e-2, no scheduling |
| Batch normalisation | None |
| Seeds | 30 (18 for COLLAB) |
| Datasets | MUTAG (188), PROTEINS (1113), NCI1 (4110), COLLAB (5000, pending) |
| Split | Stratified 80/20 per seed |

Results at different configurations (deeper architectures, batch normalisation, learning rate schedules, larger hidden dimensions) may differ. No claim of generality beyond the tested configuration is made.

## 6. Summary of All Comparisons

Investigation-wide counts over the **59 distinct comparisons** (primary), with the 76-with-re-reports pool alongside:

| Category | Over 59 distinct (primary) | Over 76 with re-reports |
|---|---|---|
| Significant at investigation-wide BH | 31 (53%) | 47 (62%) |
| Significant at Bonferroni | 16 (27%) | 22 (29%) |
| Non-significant (null results reported) | 28 (47%) | 29 (38%) |
| **Total** | **59 (100%)** | **76 (100%)** |

Per-hypothesis BH (within each hypothesis family) yields 47 family-level significant results, unaffected by investigation-wide de-duplication. Within each pool the Bonferroni-significant set is a strict subset of the investigation-wide BH-significant set, and the partition is additive: **significant + non-significant = total** (31 + 28 = 59; 47 + 29 = 76).

No selective reporting. All 76 computed comparisons (59 distinct + 17 exact re-reports) are documented in their respective hypothesis documents and JSON artifacts; the investigation-wide FDR is computed over the 59 distinct. Negative results are given identical formatting and statistical treatment as positive results.
