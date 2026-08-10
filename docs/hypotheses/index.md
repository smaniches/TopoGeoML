---
title: Hypotheses (H001-H011b)
nav_order: 3
has_children: true
description: "The preregistered hypothesis series and its resolved outcomes."
---

# Preregistered hypotheses (H001-H011b)
{: .no_toc }

Each hypothesis below was written with falsifiable sub-predictions and a pre-specified outcome decision tree, then committed to git before its experiment ran. The commit history is the preregistration timestamp. Results were appended afterwards, with the original predictions preserved for audit.

{: .note }
> The primary conclusion of this series is negative and specific: at the tested configurations, no unique `L_0` Hodge advantage is supported once the Hodge and normalized-adjacency arms use the same external-residual architecture. H008c shows that the tested external-residual adjacency formulation recovers NCI1 performance after the internal-self normalized formulation does not; that result is scoped to the tested self-path formulations and is not a universal causal claim about residual connections. Null results, refutations, positive differences, and unresolved tests are kept distinct below.

## Outcomes at a glance

| # | Question | Current outcome |
|---|---|---|
| **H001** | Why does minimal Hodge MP lose on MUTAG; what closes the gap? | Symmetric normalisation removes the detected deficit relative to MLP; the residual variant underperforms MLP at this configuration |
| **H002** | Does the selected architecture beat MLP on PROTEINS? | No significant improvement detected; the strict superiority hypothesis is refuted |
| **H003** | Does the NCI1 experiment produce a positive difference from MLP? | Narrow positive difference: +8.6 pp for Hodge-residual versus the matched-capacity MLP, regime-bound |
| **H004** | Is sample size the mechanism behind the cross-dataset sign change? | Sample size alone does not reproduce the MUTAG sign reversal |
| **H005** | Is feature dimensionality the mechanism? | Feature dimensionality alone does not explain the result; NCI1 retains a graph-aware advantage after projection |
| **H006** | Is graph-structural classification signal present under constant features? | Yes on the three tested datasets; the cross-dataset ordering is inverted relative to the full-feature Hodge-versus-MLP differences |
| **H007** | Does one tested structural proxy explain the full-feature pattern? | None of the five tested proxies tracks the full-feature cross-dataset pattern |
| **H008** | Does the NCI1 result hold against GIN and GAT at matched capacity? | In this deliberately constrained regime, GIN and GAT are at class prior while Hodge-residual is not; H008 alone does not isolate the mechanism |
| **H008b** | Does degree normalisation explain the GIN-Hodge gap? | Refuted; normalisation with the tested internal-self formulation does not recover the result |
| **H008c** | Does the matched external-residual adjacency formulation recover performance and remove a unique Hodge advantage? | Yes on NCI1. gin-residual reaches 0.629 versus Hodge 0.609; the result identifies the successful tested self-path formulation, not a universal residual-only mechanism |
| **H009** | Does a learned scalar sheaf Laplacian beat the fixed operators? | Invalidated by implementation audit; the historical artifact is provenance only. H39-H41 were resolved by the corrective replication H009-R |
| **H009-R** | Corrective replication: does the repaired scalar sheaf Laplacian resolve H39-H41? | H39 supported (sheaf 0.604 above MLP 0.523, p_BH = 4.74 x 10^-3); H40 not supported (sheaf versus Hodge p_BH = 0.428); H41 falsification condition not met (sheaf below gin-residual at p_BH = 0.0342, which does not cross the preregistered 0.01 threshold; inconclusive, not equivalence) |
| **H010** | Does the Hodge-versus-adjacency gap follow a dataset-level operator mechanism? | No unique Hodge advantage is detected; H44 remains inconclusive because no dataset-level correlate was established |
| **H011** | Does `L_1` edge propagation capture signal that `L_0` cannot on NCI1? | H47-H49 are refuted under their stated rules; the higher-order question remains unresolved because 96% of NCI1 graphs contain no triangles |
| **H011b** | Does `L_1` help on triangle-rich COLLAB? | Unresolved; one-seed directional smoke result only, full run compute-constrained |

A non-significant test is not an equivalence result. The individual preregistration files preserve the original hypotheses and appended outcomes; current inferential terminology is summarized in [`../STATISTICAL_SUMMARY.md`](../STATISTICAL_SUMMARY.md).
