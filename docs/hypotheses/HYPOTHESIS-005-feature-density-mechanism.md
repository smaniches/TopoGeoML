# Hypothesis 005: Is feature density the mechanism behind the residual-scale effect? Cross-dataset feature-projection study

**Status.** Preregistered 2026-05-22, before hypothesis 004's full result lands. Conditional on H13 being refuted (which preliminary 3-of-4-sizes evidence already suggests).
**Falsification target.** Whether the residual-vs-MLP effect tracks input *feature dimensionality / sparsity* rather than dataset identity.
**Prior result motivating this hypothesis.** Hypothesis 004 preliminary (n=188, 1113, 2000 NCI1 subsamples already in): the residual variant *does not* lose at MUTAG-sized NCI1 subsamples. Sample size is not the mechanism. The remaining explanations are dataset-specific feature properties — most plausibly *feature density* (NCI1's 37-dim atom-type one-hot vs MUTAG's 7-dim atom one-hot vs PROTEINS' 3-dim secondary-structure one-hot).

---

## 1. Design: orthogonal feature manipulation

The cleanest test of feature-density is to **project node features to a different dimensionality** while holding everything else constant. Two complementary directions:

**Direction A: Dim-reduce NCI1's features.** Take NCI1's 37-dim atom one-hot, project to 7-dim via random projection (deterministic per seed using a fixed Gaussian projection matrix). This makes NCI1's *feature density and dimensionality* match MUTAG's, while leaving NCI1's sample size, graph statistics, and label distribution intact.

**Direction B: Dim-expand MUTAG's features.** Take MUTAG's 7-dim atom one-hot, project to 37-dim via random expansion + ReLU + linear (deterministic per seed). This makes MUTAG's *feature space* match NCI1's, while leaving MUTAG's sample size + graph statistics intact.

These are *opposite* manipulations of the same axis. If the residual is feature-density-driven:
- **A**: residual loses on NCI1-7d (matching its MUTAG behaviour)
- **B**: residual wins on MUTAG-37d (matching its NCI1 behaviour)

If both directions confirm, feature-density is the mechanism. If A confirms but B doesn't, then NCI1 has *some other* dataset-specific property (e.g. graph topology) that interacts with feature dim. If neither confirms, there's a third unidentified mechanism.

## 2. Preregistered sub-hypotheses (verbatim)

| ID | Sub-hypothesis | Predicted | Falsified if |
|---|---|---|---|
| **H18** | Residual loses on NCI1-7d (direction A) | median Δ < 0, p_BH < 0.05 | median Δ ≥ 0 OR p_BH ≥ 0.05 |
| **H19** | Residual wins on MUTAG-37d (direction B) | median Δ > 0, p_BH < 0.05 | median Δ ≤ 0 OR p_BH ≥ 0.05 |
| **H20** | Both H18 and H19 confirmed (full mechanism) | yes | either falsified above |
| **H21** | NCI1-7d residual median ≤ MUTAG residual median (continuity) | yes | NCI1-7d residual > full MUTAG residual |

## 3. Experimental design

- **Datasets**: NCI1-7d (random projection) and MUTAG-37d (random expansion). Implementation note: the projection matrices are deterministic per seed via `np.random.default_rng(seed).normal(...)`, applied to features as a one-shot transform before the existing `_proj_in` layer.
- **Arms**: `hodge-mp-residual` and `mlp-baseline` only (the two arms that tell us the headline).
- **Seeds**: 30 per (dataset, arm).
- **Epochs**: 10 (matched to hypothesis 004 for direct comparison).
- **Statistical procedure**: Paired Wilcoxon at each direction, BH-FDR across the 2-comparison family.

## 4. Wall-clock budget

| Dataset variant | Original wall time | Estimated for variant |
|---|---|---|
| NCI1-7d (project 37→7) | NCI1 full ~60 min | ~60 min (projection is O(N·D)) |
| MUTAG-37d (expand 7→37) | MUTAG full ~5 min | ~5 min |
| **Total** | — | **~65 min** |

The projection / expansion is cheap; the dominating cost is the training loop, which is unchanged.

## 5. Outcome decision tree

| Outcome | Interpretation | Next step |
|---|---|---|
| H18 + H19 both confirmed | Feature density IS the mechanism. The framework can claim "residual helps when input feature dim ≥ some threshold, regardless of sample size". | Hypothesis 006 sweeps the projection dim from 3 to 37 to locate the threshold. |
| H18 confirmed, H19 refuted | NCI1 has additional dataset-specific structure beyond features. Random expansion isn't enough; the *semantic content* of features matters. | Hypothesis 006 examines graph topology / degree distribution. |
| H18 refuted | Feature dim alone is not the mechanism. Some other property (label noise, graph density, etc.) drives the residual-scale effect. | Investigate next via held-out factor isolation. |

## 6. Resolved outcome

*Pending — run starts after hypothesis 004's n=4110 control reproduces hypothesis 003.*

---

## 7. What this hypothesis deliberately does NOT do

- **Does not use learned embeddings**. A learnable feature transform would entangle the test (the residual would help learn the embedding, conflating the mechanism). The random projection / expansion is a *fixed* (per-seed deterministic) transform applied BEFORE training, so the architecture's capacity to learn the embedding is removed from the comparison.
- **Does not vary architecture or hyperparameters**. Holding everything else constant is the matched-design discipline.
- **Does not commit to a specific projection technique** (random Gaussian vs PCA vs hashing) — the design choice is locked to "random Gaussian with fixed seed" per the preregistration but the underlying signal should not depend on the projection method if feature density is the right mechanism.