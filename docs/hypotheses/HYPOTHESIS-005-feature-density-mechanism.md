# Hypothesis 005: Is feature density the mechanism behind the residual-scale effect? Cross-dataset feature-projection study

**Status.** Resolved 2026-05-22. **H18 REFUTED, H19 REFUTED — feature density is NOT the mechanism either.** The striking sub-finding: NCI1-7d MLP collapses to chance (0.500) while Hodge-residual still gets 0.58 — the graph topology carries classification signal that survives feature-space corruption. See §6.
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

## 6. Resolved outcome (2026-05-22, 30 seeds × 10 epochs, both directions, corrected Johnson-Lindenstrauss scaling)

The first run of this experiment used a buggy projection scaling (`σ = 1/√src_dim` instead of `σ = 1/√target_dim`) which Gemini caught on PR #21 review. That run was killed; the corrected scaling preserves the feature-vector norm in expectation (`E[|x @ P|² = |x|²`) in both directions, so the experiment isolates dimensionality from feature magnitude. Results below are from the corrected run.

| Direction | Setup | hodge-mp-residual (median, BCa 95%) | mlp-baseline (median, BCa 95%) | median Δ | paired Wilcoxon p_BH | Predicted | Outcome |
|---|---|---|---|---|---|---|---|
| **A**: NCI1-7d | project NCI1's 37 → 7 | 0.581 [0.525, 0.598] | **0.500 [0.500, 0.518]** | **+0.081** | **4.93 × 10⁻⁴** | residual LOSES (H18) | **residual WINS** |
| **B**: MUTAG-37d | project MUTAG's 7 → 37 | 0.776 [0.737, 0.816] | 0.789 [0.763, 0.829] | -0.013 | 0.246 | residual WINS (H19) | matches |

### Sub-hypotheses, resolved (under the tested configuration)

- **H18** (residual loses on NCI1-7d): **REFUTED.** Observed median Δ = +0.081, p_BH = 4.93 × 10⁻⁴. The MLP baseline observed median is 0.500 [0.500, 0.518] under the 7-dim Gaussian projection of NCI1's features; Hodge-residual's observed median is 0.581 [0.525, 0.598]. Under this tested configuration, the Hodge-residual arm achieves higher accuracy than the MLP baseline arm when feature dimensionality is reduced to MUTAG's.
- **H19** (residual wins on MUTAG-37d): **REFUTED.** Observed median Δ = -0.013, p_BH = 0.246 — no significant difference between arms at this configuration.
- **H20** (both directions confirmed → feature density is the mechanism): **REFUTED** (since H18 + H19 both refuted).
- **H21** (NCI1-7d residual median ≤ MUTAG residual median, continuity): observed inequality holds (0.581 ≤ 0.750); not directly informative about the mechanism question.

### Observed pattern (descriptive, scoped to H005 only)

Under H005's NCI1-7d configuration, the MLP-baseline observed median is 0.500 [0.500, 0.518] — at the NCI1 class prior. The Hodge-residual observed median is 0.581 [0.525, 0.598] with paired Wilcoxon p_BH = 4.93 × 10⁻⁴. The arms' observed medians differ at this configuration. **This is a descriptive observation about the H005 setup; it does not establish a general causal mechanism, and it does NOT imply that only NCI1 carries graph-structural signal.** Hypothesis 006 (constant-feature ablation) is the appropriate test of that broader question; under H006's preliminary results PROTEINS also shows above-prior signal under constant features, so the H005 NCI1-7d → MLP-collapses-to-chance pattern is a finding about the projected-feature regime, not about NCI1's uniqueness.

### Mechanism candidates after H004 + H005

Two of the three leading mechanism candidates from hypothesis 003 §6 have now been **falsified at this configuration**:

| Candidate | Hypothesis | Outcome | Falsified by |
|---|---|---|---|
| Sample size | (b) | NOT consistent with this data | H004 |
| Feature dimensionality | (a) | NOT consistent with this data | H005 |
| Graph structural signal (constant-feature ablation) | (c) — new | tested by H006 | — |

H006 picks the cheapest operationalisation — **constant-feature ablation** — which directly measures whether the Hodge model retains predictive signal when node-feature information is removed.

### Scoped statement of the H005 finding

The statement consistent with the H005 data is:

> *Under this configuration (matched-capacity 1378-param arms, 30 seeds × 10 epochs × stratified 80/20 split), on NCI1 with node features replaced by a 7-dim Gaussian projection, the Hodge-residual arm's observed median test accuracy exceeds the MLP baseline arm's at paired Wilcoxon p_BH = 4.93 × 10⁻⁴. The same configuration on MUTAG with 37-dim expansion shows no significant difference between arms. This is evidence consistent with an architecture × data interaction at this configuration; whether the interaction generalises across datasets is the question H006 was designed to answer.*

No claim is made of generality beyond the configurations explicitly enumerated above. **In particular, this section does not assert that only NCI1 carries graph-structural classification signal — that broader question is H006's, not H005's.**

---

## 7. What this hypothesis deliberately does NOT do

- **Does not use learned embeddings**. A learnable feature transform would entangle the test (the residual would help learn the embedding, conflating the mechanism). The random projection / expansion is a *fixed* (per-seed deterministic) transform applied BEFORE training, so the architecture's capacity to learn the embedding is removed from the comparison.
- **Does not vary architecture or hyperparameters**. Holding everything else constant is the matched-design discipline.
- **Does not commit to a specific projection technique** (random Gaussian vs PCA vs hashing) — the design choice is locked to "random Gaussian with fixed seed" per the preregistration but the underlying signal should not depend on the projection method if feature density is the right mechanism.