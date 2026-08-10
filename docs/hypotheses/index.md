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
> The primary conclusion of this series is negative and specific: at the tested configurations, the Hodge `L_0` operator does not provide a unique advantage once an external residual is present. H008c identifies the external residual as the operative architectural factor in that experiment. Null results, refutations, positive differences, and unresolved tests are kept distinct below.

## Outcomes at a glance

| # | Question | Current outcome |
|---|---|---|
| **H001** | Why does minimal Hodge MP lose on MUTAG; what closes the gap? | Symmetric normalisation removes the detected deficit relative to MLP; the residual variant underperforms MLP at this configuration |
| **H002** | Does the selected architecture beat MLP on PROTEINS? | No significant improvement detected; the strict superiority hypothesis is refuted |
| **H003** | Does the NCI1 experiment produce a positive difference from MLP? | Narrow positive difference: +8.6 pp for Hodge-residual versus the matched-capacity MLP, regime-bound |
| **H004** | Is sample size the mechanism behind the cross-dataset sign change? | Sample size alone does not reproduce the MUTAG sign reversal |
| **H005** | Is feature dimensionality the mechanism? | Feature dimensionality alone does not explain the result; NCI1 retains a graph-aware advantage after projection |
| **H006** | Is graph-structural classification signal present under constant features? | Yes on the three tested datasets; the cross-dataset ordering is inverted relative to the full-feature Hodge-versus-MLP differences |
| **H007** | Does one tested structural proxy explain the full-feature pattern? | No tested proxy explains it individually |
| **H008** | Does the NCI1 result hold against GIN and GAT at matched capacity? | In this deliberately constrained regime, GIN and GAT collapse to class prior while Hodge-residual does not |
| **H008b** | Does degree normalisation explain the GIN-Hodge gap? | Refuted; normalisation alone does not recover the result |
| **H008c** | Is the external residual the operative factor? | Confirmed in the tested NCI1 ablation; normalized adjacency with the same residual performs comparably to or better than Hodge |
| **H009** | Does a learned sheaf Laplacian beat the fixed operators? | No supported improvement in the tested configuration |
| **H010** | Does high-pass Hodge propagation beat low-pass adjacency cross-dataset? | No supported cross-dataset advantage for the Hodge operator |
| **H011** | Does `L_1` edge propagation capture signal that `L_0` cannot on NCI1? | Inconclusive for the higher-order question because NCI1 is almost triangle-free |
| **H011b** | Does `L_1` help on triangle-rich COLLAB? | Unresolved; one-seed directional smoke result only, full run compute-constrained |

A non-significant test is not an equivalence result. The individual preregistration files preserve the original hypotheses and appended outcomes; current inferential terminology is summarized in [`../STATISTICAL_SUMMARY.md`](../STATISTICAL_SUMMARY.md).
