---
title: Statistical summary
nav_order: 4
---

# Investigation-Wide Statistical Summary

This document summarizes multiplicity, inferential scope, and the adaptive structure of the current TopoGeoML graph-classification investigation. It distinguishes the preregistered decision rules inside each experiment from retrospective analyses across the entire realized research program.

## 1. Scope

- **Distinct pairwise comparisons in the realized record:** 59
- **Total pairwise comparisons computed:** 76
- **Repeated comparisons:** 17 of the 76 are repeated baseline comparisons that also appear in other hypothesis families
- **Result files:** 22 JSON artifacts in `notebooks/results/`
- **Preregistered hypothesis documents:** 14
- **Resolved primary experiment documents:** 12
- **Partially resolved:** H011 on NCI1
- **Unresolved:** H011b on COLLAB
- **Falsifiable sub-predictions:** 53

The 59-comparison set is used when reporting a de-duplicated investigation-wide sensitivity analysis. It is not treated as a globally preregistered family because the research questions themselves were generated sequentially from earlier results.

## 2. Within-experiment inference

Each preregistered experiment defines a comparison family and a decision rule before its corresponding result is observed. Pairwise Wilcoxon p-values are adjusted with Benjamini-Hochberg within the declared family.

The decision threshold is not always 0.05. Several sub-hypotheses explicitly use 0.01. Therefore:

- a BH-adjusted p-value below 0.05 is not automatically a preregistered confirmation if that sub-hypothesis required 0.01;
- a result between 0.01 and 0.05 can be conventionally significant while remaining inconclusive under a stricter preregistered rule;
- the hypothesis documents, not a generic alpha, determine the confirmation or falsification label.

H009 H41 is the clearest example: sheaf-residual versus gin-residual has p_BH = 0.0137, but H41 preregistered p_BH < 0.01 as the strict underperformance threshold. H41 is therefore inconclusive under its own decision rule.

## 3. Retrospective investigation-wide multiplicity sensitivity

For transparency, the raw p-values from the realized program were also re-evaluated as one de-duplicated 59-comparison set. This produces the following retrospective sensitivity analysis:

| Retrospective correction | Over 59 distinct comparisons | Over 76 computed entries including repeats |
|---|---:|---:|
| BH-adjusted p < 0.05 | 31/59 (53%) | 47/76 (62%) |
| Bonferroni-adjusted p < 0.05 | 16/59 (27%) | 22/76 (29%) |
| Not significant under BH at 0.05 | 28/59 (47%) | 29/76 (38%) |

The 59-comparison version is the more meaningful of the two because it does not count the same raw comparison repeatedly. The 76-entry version is retained only to show what was actually computed in individual result files.

### Interpretation of the global BH table

The investigation-wide BH calculation is a **retrospective robustness check**, not a claim that the adaptive program has a formally guaranteed 5% global false-discovery rate. The reason is structural:

1. H001-H011b were not all specified before any data were observed.
2. Later hypotheses were chosen because of earlier outcomes.
3. The resulting global comparison family is therefore adaptively generated rather than fixed in advance.
4. Standard BH guarantees depend on assumptions about the tested family and dependence structure that have not been established for this adaptive selection process.

The global BH result is still useful. It asks whether important comparisons remain small relative to the total realized testing burden. It should be read as a sensitivity analysis, not as a substitute for the preregistered within-experiment decision rules.

Bonferroni over the realized 59-comparison set is also reported as a conservative sensitivity check. Because the global family was defined retrospectively, its adjusted thresholds likewise should not be presented as if they were prospectively specified confirmatory thresholds.

### Selected comparisons under the 59-comparison sensitivity analysis

| Comparison | p_raw | Rank of 59 | BH threshold at rank | Global BH < 0.05 | Bonferroni over 59 |
|---|---:|---:|---:|---|---|
| Hodge-residual vs MLP on NCI1 (H003) | 3.38 x 10^-3 | 22 | 1.86 x 10^-2 | Yes | No |
| gin-residual vs MLP on NCI1 (H008c) | 4.03 x 10^-4 | 14 | 1.19 x 10^-2 | Yes | Yes |
| Hodge-residual vs GIN on NCI1 (H008) | 2.12 x 10^-6 | 5 | 4.24 x 10^-3 | Yes | Yes |
| gin-residual vs gin-normalised on NCI1 (H008c) | 1.73 x 10^-6 | 2 | 1.69 x 10^-3 | Yes | Yes |

For H003, “survives global BH but not Bonferroni” means exactly what the table says for this retrospective 59-comparison sensitivity analysis. It does not mean that a lower alpha would help. A stricter alpha would make rejection harder. Stronger Bonferroni evidence would require a smaller observed p-value relative to the fixed 0.05/59 threshold.

## 4. Non-significance, equivalence, and sensitivity

A non-significant comparison is a failure to reject the null under the stated test and threshold. It is not evidence that two methods are equivalent.

The project therefore uses language such as “no significant difference detected” rather than “equal,” “equivalent,” or “statistically indistinguishable” unless an explicit equivalence design is performed.

The repository previously listed exact minimum-detectable-effect values for the paired Wilcoxon test without documenting a generative model or reproducible power calculation. Those values have been removed. Power for a signed-rank test depends on the distribution of paired differences, zero/tie structure, effect definition, and the planned decision threshold. A defensible prospective power statement should be produced by an explicit analytical model or simulation tied to the intended experiment.

For a future equivalence claim, preregister an equivalence margin and an appropriate equivalence procedure rather than interpreting p > alpha as equality.

## 5. Adaptive hypothesis generation

The research program is sequential and adaptive:

- H001-H007 were designed in response to preceding results.
- H008-H010 were designed after the earlier mechanism studies.
- H011 was motivated by the conclusion that `L_0` does not provide a unique Hodge advantage.
- H011b was created after the NCI1 triangle census showed that H011 was a poor test of a triangle-rich `L_1` mechanism.

Each new hypothesis was committed before the corresponding new experiment ran. That protects the experiment from changing its prediction after seeing its own result.

It does **not** make the entire program equivalent to a single pre-specified group-sequential clinical-style design, and classical Pocock or O'Brien-Fleming stopping rules do not automatically apply to this adaptive hypothesis-generation process.

The correct interpretation is:

- individual hypothesis tests can be evaluated against their preregistered local rules;
- the sequence of scientific questions is adaptive and should be reported as such;
- the 59-comparison BH calculation is a retrospective multiplicity sensitivity analysis across the realized program;
- independent replication on new data is the strongest next step for claims intended to generalize beyond this research sequence.

## 6. Configuration scope

The main graph experiments are bounded to the following regime:

| Parameter | Current scope |
|---|---|
| Architecture depth | Primarily 1 layer; 2 layers for the deep-residual H001 arm |
| Hidden dimension | Primarily 32 |
| Training epochs | 10 to 20 |
| Optimiser | Adam, lr=1e-2, no scheduler in the reported graph experiments |
| Batch normalisation | None |
| Seeds | Usually 30 for confirmatory graph experiments |
| Datasets | MUTAG (188), PROTEINS (1113), NCI1 (4110); COLLAB H011b remains incomplete |
| Split | Stratified 80/20 per seed in the reported graph-classification experiments |

H011b preregisters a 30-seed COLLAB design. A separate 18-seed compute attempt exceeded the GitHub Actions six-hour limit. That timed-out attempt is not a completed substitute for the preregistered 30-seed design and produces no statistical claim.

Results at different depths, normalization schemes, training budgets, hidden dimensions, optimizers, or datasets can differ. No claim of generality beyond the tested configuration is made.

## 7. Current comparison counts

For the retrospective realized-family analysis:

| Category | 59 distinct comparisons | 76 computed entries including repeats |
|---|---:|---:|
| BH-adjusted p < 0.05 | 31 | 47 |
| Bonferroni-adjusted p < 0.05 | 16 | 22 |
| Not significant under BH at 0.05 | 28 | 29 |
| Total | 59 | 76 |

These counts describe statistical significance under the stated retrospective corrections. They are not counts of “true discoveries,” and they are not counts of preregistered hypotheses confirmed, because individual hypotheses can use stricter decision thresholds or contain multiple comparisons.

All computed comparisons remain in the public record, including regressions, non-significant results, and results that are inconclusive under their preregistered rules.

## 8. Recommended confirmatory next step

For any result intended to support a broad external claim, the preferred next step is an independent confirmatory experiment with:

1. a fixed hypothesis and comparison family specified before data analysis;
2. a pre-specified primary endpoint and effect direction;
3. a justified sample-size or simulation-based sensitivity analysis;
4. a fixed multiplicity procedure;
5. a new dataset, held-out benchmark, or independent reproduction not used to generate the hypothesis.

That design would separate hypothesis generation from confirmation and provide a cleaner basis for generalization.

## References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.
