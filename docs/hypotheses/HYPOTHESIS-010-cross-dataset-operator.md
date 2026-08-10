---
title: H010 · Cross-dataset operator
parent: Hypotheses (H001–H011b)
nav_order: 12
---

# Hypothesis 010: Does the high-pass vs low-pass operator distinction predict cross-dataset performance?

**Status.** Resolved 2026-05-25, with one preregistered sub-hypothesis left inconclusive. H42 confirmed: gin-residual exceeds Hodge on MUTAG at p_BH = 7.44 x 10^-3. H43 detects no significant Hodge-versus-gin difference on PROTEINS. H44 is not confirmed: the gap magnitude varies across datasets, but the preregistered prediction required association with a dataset-level property and no such association was established. H45 and H46 are refuted. See §6.

**Falsification target.** Whether the choice between Hodge Laplacian propagation L_tilde and normalised adjacency propagation I - L_tilde produces dataset-dependent classification differences when both arms use the same external residual. H008-c had already removed any supported claim of a unique Hodge advantage on NCI1. This experiment tests the operator comparison on MUTAG and PROTEINS as well.

**Prior results motivating this hypothesis.**

1. H008-c: On NCI1, gin-residual has a +0.0195 median difference over Hodge, with p_BH = 0.0101 in that comparison family.
2. H006: The constant-feature Hodge signal has the rank ordering MUTAG (+0.098) > PROTEINS (+0.088) > NCI1 (+0.071), the inverse of the full-feature Hodge-versus-MLP differences.
3. H001: On MUTAG, Hodge-residual underperforms MLP by 4 pp (p_BH = 0.019). The gin-residual arm had not yet been tested on MUTAG.

**Theoretical context.** For the normalised graph Laplacian L_tilde, an eigenmode with eigenvalue lambda is multiplied by lambda under L_tilde and by 1 - lambda under I - L_tilde. The preregistration uses “high-pass” for L_tilde and “low-pass” for I - L_tilde as spectral shorthand. That language should not be read as an ideal signal-processing filter classification: the adjacency response 1 - lambda becomes negative above lambda = 1, so its magnitude is not a monotone low-pass response over the full spectrum.

The original mechanism proposal was that the relative response to smooth and oscillatory graph signals might interact with dataset structure. H010 tests the empirical operator comparison; it does not independently establish a homophily mechanism.

---

## 1. Design

Run `hodge-mp-residual` and `gin-residual` on all three datasets with the same external residual. Use MLP as the no-message-passing control.

| Dataset | Hodge | gin-residual | MLP | New data needed? |
|---|---|---|---|---|
| NCI1 | 0.609 | 0.629 | 0.523 | No (reuse H008-c) |
| MUTAG | ? | ? | 0.789 (H001) | Yes |
| PROTEINS | ? | ? | 0.675 (H002) | Yes |

## 2. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H42** | gin-residual outperforms Hodge on MUTAG | gin-residual > Hodge (p_BH < 0.05) | MUTAG was hypothesised to favour adjacency-style smoothing over Laplacian differencing | p_BH >= 0.05 or Hodge >= gin-residual |
| **H43** | gin-residual outperforms Hodge on PROTEINS | Uncertain | PROTEINS may occupy an intermediate operator regime | Hodge strictly beats gin-residual at p_BH < 0.05 |
| **H44** | The gin-residual vs Hodge gap is dataset-dependent | The operator advantage (gin-residual median - Hodge median) correlates with some dataset-level property | H006 showed graph-structural separability differs across datasets; the operator preference may track this | All three datasets show the same direction and magnitude |
| **H45** | Both gin-residual and Hodge outperform MLP on MUTAG with external residual | p_BH < 0.05 for both | External residual might rescue message passing relative to MLP | Either arm <= MLP |
| **H46** | Both gin-residual and Hodge outperform MLP on PROTEINS with external residual | p_BH < 0.05 for both | External residual might produce a positive difference relative to MLP | Either arm <= MLP |

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H42 confirmed (gin-residual > Hodge on MUTAG), consistent with NCI1 | The tested adjacency operator is preferred over the Hodge operator on more than one dataset at this capacity; this does not by itself establish a universal mechanism. |
| H42 refuted (Hodge >= gin-residual on MUTAG), opposite of NCI1 | The operator preference changes direction across datasets and requires a dataset-level explanation. |
| H45 refuted | External residual is not sufficient to guarantee a positive difference from MLP on MUTAG. |
| No significant Hodge-versus-gin difference on all three datasets | The experiment would provide no evidence that the operator choice materially separates performance at the tested configuration. |

## 4. Experimental design

- **Datasets:** MUTAG (188 graphs, 20 epochs) and PROTEINS (1113 graphs, 10 epochs). NCI1 results reused from H008-c.
- **Models:** `hodge-mp-residual`, `gin-residual`, `mlp-baseline`.
- **Seeds:** 30, matched to prior experiments.
- **Epochs:** MUTAG: 20 (matched to H001); PROTEINS: 10 (matched to H002).
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon, BH-FDR at alpha=0.05.

## 5. Reproduction

```bash
# MUTAG (20 epochs, matched to H001)
python -m benchmarks.hodge \
  --datasets mutag \
  --models hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 20 \
  --output notebooks/results/h010_mutag_operator_30seeds.json \
  --markdown notebooks/results/h010_mutag_operator_30seeds.md

# PROTEINS (10 epochs, matched to H002)
python -m benchmarks.hodge \
  --datasets proteins \
  --models hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h010_proteins_operator_30seeds.json \
  --markdown notebooks/results/h010_proteins_operator_30seeds.md
```

## 6. Resolved outcome (2026-05-25, 30 seeds, MUTAG 20 epochs / PROTEINS 10 epochs)

Per-arm reports in `notebooks/results/h010_{mutag,proteins}_operator_30seeds.{json,md}`.

### Cross-dataset summary (with NCI1 from H008-c)

| Dataset | Hodge | gin-residual | MLP | Hodge vs gin-residual p_BH | Result |
|---|---|---|---|---|---|
| MUTAG (188) | 0.750 [0.724, 0.789] | 0.789 [0.763, 0.816] | 0.789 [0.763, 0.816] | 7.44 x 10^-3 | Significant gin-residual advantage |
| PROTEINS (1113) | 0.686 [0.670, 0.717] | 0.675 [0.657, 0.709] | 0.675 [0.596, 0.706] | 0.292 | No significant operator difference detected |
| NCI1 (4110) | 0.609 [0.581, 0.625] | 0.629 [0.607, 0.641] | 0.523 [0.513, 0.566] | 1.01 x 10^-2 | Significant gin-residual advantage at alpha=0.05 |

### Sub-hypotheses resolved

- **H42** (gin-residual > Hodge on MUTAG): **CONFIRMED.** gin-residual 0.789 > Hodge 0.750, p_BH = 7.44 x 10^-3. The preregistered directional prediction is satisfied. The proposed homophily rationale was not independently tested by H010.
- **H43** (gin-residual vs Hodge on PROTEINS): **NO SIGNIFICANT DIFFERENCE DETECTED.** Hodge has the higher median point estimate (0.686 vs 0.675), but p_BH = 0.292. H43 did not preregister a positive confirmation threshold and its stated falsification condition, significant Hodge superiority, is not met.
- **H44** (dataset-dependent gap correlated with a dataset-level property): **INCONCLUSIVE.** Gap magnitude varies across the three datasets, but no preregistered dataset-level property is shown to explain that variation. The result does not satisfy the substantive correlation prediction. The stated falsification condition is also not met because the magnitudes are not the same.
- **H45** (both arms beat MLP on MUTAG): **REFUTED.** Hodge 0.750 underperforms MLP 0.789 at p_BH = 8.61 x 10^-3. gin-residual 0.789 has no significant difference from MLP (p_BH = 0.438).
- **H46** (both arms beat MLP on PROTEINS): **REFUTED.** Neither arm has a significant positive difference from MLP (Hodge vs MLP p_BH = 0.29; gin-residual vs MLP p_BH = 0.78).

### Interpretation

H010 provides no evidence of a dataset on which the Hodge operator significantly outperforms the matched gin-residual operator. The reverse comparison is significant on MUTAG and NCI1 at the H010 alpha=0.05 family threshold, while PROTEINS detects no significant operator difference and has a slightly higher Hodge median.

The defensible cross-dataset statement is therefore narrower than “low-pass is always better”: in this experiment, no unique Hodge advantage is detected on any of the three datasets, and adjacency propagation is significantly higher on two of them under the declared H010 family threshold. The experiment does not establish the preregistered dataset-property mechanism in H44.

### What the H001-H010 investigation supports

1. On NCI1 at the tested matched-capacity configuration, message passing with an external residual produces positive differences from MLP for several operator choices.
2. H008-c identifies the external residual as the operative architectural factor in its NCI1 ablation; H009 does not show an improvement from replacing the fixed operator with the tested scalar learned sheaf operator.
3. No tested dataset provides evidence of a unique `L_0` Hodge advantage over the matched normalised-adjacency operator.
4. On MUTAG, Hodge-residual underperforms MLP and gin-residual under the H010 comparison family.
5. On PROTEINS, the tested arms do not produce significant positive differences from MLP at this configuration.

---

The preregistered question about genuinely higher-order Hodge information is taken up by H011 using `L_1` edge propagation.
