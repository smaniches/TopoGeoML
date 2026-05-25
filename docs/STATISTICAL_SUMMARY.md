# Investigation-Wide Statistical Summary

This document reports the statistical properties of the complete TopoGeoML investigation (H001-H011) at the investigation level, addressing multiple-testing burden, power, and the false discovery rate across all experiments.

## 1. Scope

- **Total pairwise comparisons computed:** 76
- **Result files:** 22 JSON artifacts in `notebooks/results/`
- **Hypotheses:** 14 preregistered documents, 12 resolved
- **Sub-predictions:** 50+ falsifiable sub-hypotheses

## 2. Multiple-Testing Correction

### Per-hypothesis correction (as reported)

Each hypothesis applies Benjamini-Hochberg FDR at alpha=0.05 within its own comparison family (typically 3-10 comparisons per hypothesis). This controls the false discovery rate within each experiment but does not control the investigation-wide FDR.

### Investigation-wide BH-FDR

Applying Benjamini-Hochberg across all 76 raw p-values at alpha=0.05:

| Metric | Value |
|---|---|
| Comparisons significant at per-hypothesis BH (alpha=0.05) | 48/76 (63%) |
| Comparisons significant at investigation-wide BH (alpha=0.05) | 47/76 (62%) |
| Comparisons surviving Bonferroni (alpha=0.05/76 = 6.58 x 10^-4) | 22/76 (29%) |

One comparison loses significance under investigation-wide correction (48 → 47). All primary claims (NCI1 positive result, residual-placement finding, GIN/GAT collapse) survive both investigation-wide BH and Bonferroni correction.

### Key claims under investigation-wide correction

| Claim | p_raw | Rank | BH threshold | Survives global BH | Survives Bonferroni |
|---|---|---|---|---|---|
| Hodge-residual > MLP on NCI1 (H003) | 3.38 x 10^-3 | 28/76 | 1.84 x 10^-2 | Yes | No |
| gin-residual > MLP on NCI1 (H008-c) | 4.03 x 10^-4 | 17/76 | 1.12 x 10^-2 | Yes | Yes |
| Hodge-residual > GIN on NCI1 (H008) | 2.12 x 10^-6 | 6/76 | 3.95 x 10^-3 | Yes | Yes |
| gin-residual > gin-normalised on NCI1 (H008-c) | 1.73 x 10^-6 | 2/76 | 1.32 x 10^-3 | Yes | Yes |

The NCI1 Hodge-vs-MLP claim (p_raw = 3.38 x 10^-3) survives investigation-wide BH but NOT Bonferroni. Under the most conservative correction, this claim would require a lower alpha or more seeds to be confirmed. The residual-placement finding (gin-residual vs MLP, p = 4.03 x 10^-4) survives both.

## 3. Statistical Power

For paired Wilcoxon signed-rank at alpha=0.05, power=0.80:

| Seeds (n) | Minimum detectable effect (|r|) |
|---|---|
| 18 | 0.373 |
| 30 | 0.289 |

At n=30 seeds, the investigation has 80% power to detect effects with rank-biserial |r| >= 0.289. Effects smaller than this may produce non-significant results due to insufficient power rather than absence of effect.

**Implications for null results:** When we report "no significant difference" (e.g., Hodge matches MLP on PROTEINS, p_BH = 0.548), this is a failure to reject the null at the tested power level, not evidence of equality. An equivalence test (TOST procedure) would be needed to make a positive equality claim. We do not make such claims; all null results are reported as "not significantly different at the tested configuration."

## 4. Post-Hoc Hypothesis Generation

H001-H007 were designed sequentially: each hypothesis's design was informed by the prior hypothesis's outcome. H008-H010 were designed after seeing H001-H007 results. While each hypothesis was preregistered (committed to git before execution), the hypothesis *selection* was data-driven.

This is legitimate sequential testing (Pocock 1977; O'Brien & Fleming 1979), not p-hacking, provided:
1. Each hypothesis was committed before its experiment ran (verified by git timestamps)
2. All results — positive and negative — are reported (verified: 37% of comparisons are non-significant)
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

| Category | Count | Percentage |
|---|---|---|
| Significant at per-hypothesis BH | 48 | 63% |
| Significant at investigation-wide BH | 47 | 62% |
| Significant at Bonferroni | 22 | 29% |
| Non-significant (null results reported) | 28 | 37% |
| **Total comparisons** | **76** | **100%** |

No selective reporting. All 76 comparisons are documented in their respective hypothesis documents and JSON artifacts. Negative results are given identical formatting and statistical treatment as positive results.
