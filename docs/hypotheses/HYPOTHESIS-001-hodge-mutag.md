# Hypothesis 001: Why the minimal Hodge MP loses on MUTAG, and what should win

**Status.** Open as of 2026-05-21. Test in progress (this branch).
**Falsification target.** Paired Wilcoxon p_BH < 0.01 on the relevant arm-vs-MLP comparison, with BCa CIs reported.
**Prior result that motivates this hypothesis.** `notebooks/results/mutag_hodge_vs_mlp_30seeds.md` — the minimal one-layer Hodge MP underperforms an MLP baseline of matched capacity on MUTAG (median Δ = −0.092, p_BH = 5.66 × 10⁻⁵, r = −0.760) after the critical-bug fix in PR #12.

---

## 1. What the literature says about why minimal Hodge fails on graph classification

Three papers establish that *minimal* one-layer combinatorial Hodge propagation is the wrong baseline for graph classification, and all three converge on roughly the same set of architectural ingredients that working methods use.

### 1.1 Kipf & Welling 2017 — *Semi-Supervised Classification with GCNs* (ICLR)

Lemma 1: the eigenvalues of the *un*normalised graph Laplacian `L = D - A` scale with `O(degree)`, while the symmetrically-normalised Laplacian `L̃ = D^{-1/2} L D^{-1/2}` (or its complement `Ã = I - L̃`) has eigenvalues bounded to `[0, 2]`. Without that normalisation, repeated propagation produces gradient norms that depend on degree distribution rather than topology — *high-degree nodes dominate the forward pass*. This is the standard reason "vanilla GCN" works on Cora and combinatorial-L_0 propagation does not.

### 1.2 Bunch et al. 2020 — *Simplicial 2-Complex Convolutional Neural Networks* (TDA & beyond, NeurIPS workshop)

SCConv on MUTAG-like data is run with ≥ 2 propagation layers and explicit up-Laplacian / down-Laplacian separation. The one-layer case is reported only as an ablation lower bound; it underperforms even simple feature MLPs on TUDataset benchmarks. The authors argue that depth is necessary for the Hodge propagation to reach the multi-hop structures (rings, branches) that carry the topological signal — MUTAG's mutagenicity is associated with aromatic rings (5–6 hops in molecular graph distance).

### 1.3 Hu et al. 2024 — *HL-HGAT* (arXiv 2403.06687)

Production-grade Hodge graph classifier. Uses: polynomial spectral filters of order 3–5 (= multi-hop), self+cross attention, simplicial attention pooling, multiple stacked blocks. On benchmark datasets they beat plain GCN by a few percentage points. **The architectural delta from our minimal classifier is enormous**: polynomial filters ⇒ multi-hop reach; attention ⇒ degree-balanced weighting; pooling ⇒ hierarchical feature aggregation.

### 1.4 Yang et al. 2024 — *Graph Classification Gaussian Processes via Hodgelet Spectral Features* (arXiv 2410.10546)

Reports **88.06 ± 7.99 on MUTAG** with a Gaussian-process classifier built on Hodge-decomposed wavelet spectral features. Their no-Hodge ablation (WT-GP) gets 86.73 ± 4.18. So even on a *spectral-features-only* method (no neural network), the gap between Hodge and no-Hodge is ~1.3 pp with CI overlap — borderline. The headline number (88%) is achieved by the spectral-features part of the pipeline, not by Hodge per se.

## 2. Synthesis: a falsifiable hypothesis

Combining the four sources above:

> **H_main.** The reason our minimal one-layer Hodge MP underperforms an MLP on MUTAG is that the combinatorial Laplacian, single-hop reach, and absence of a residual connection together destroy the per-atom feature information faster than the model can compensate. A *normalised* Laplacian, plus a *residual* connection, plus *two* stacked layers (i.e. a 2-hop reach) should be enough to close the gap to the MLP baseline. We do not expect to *beat* the MLP — beating it on MUTAG with a simple Hodge architecture would contradict the literature consensus that MUTAG is too small to discriminate between simple methods (paper 1810.09155). We expect to *match* it.

Three sub-hypotheses, each individually falsifiable:

| ID | Sub-hypothesis | Architectural change | Predicted accuracy | Falsified if |
|---|---|---|---|---|
| **H1** | Normalisation matters | combinatorial L_0 → symmetric L̃ = D^{-1/2}LD^{-1/2} | ~75% (between minimal-Hodge 69.7% and MLP 78.9%) | p_BH < 0.01 in either direction |
| **H2** | Residual matters on top of normalisation | H1 + skip connection: `out = act(L̃ X W + b) + X` | ~77% | p_BH < 0.01 in either direction |
| **H3** | Depth matters on top of normalisation + residual | H2 with 2 stacked Hodge layers | matches MLP ~78–80% | p_BH < 0.01 in either direction |

**Strong claim.** If H3 matches MLP (no significant difference, BCa CI on the median overlaps zero), the Geo subsystem has a *defensible* "topology is at least as informative as features alone on MUTAG" claim. This is *not* "topology helps" — that requires beating MLP, which the literature suggests is unlikely on a dataset as small as MUTAG.

**Weak claim.** If H1, H2, or H3 fails to match MLP, we report it as the second confirmed negative result. The published interpretation becomes: *MUTAG cannot discriminate between architectures at this scale*, in line with paper 1810.09155.

## 3. Experimental design

- **Dataset.** MUTAG, the same fixed train/test (stratified 80/20) split per seed used in PR #14.
- **Seeds.** 30 (matched to the prior PR #14 run for direct comparison).
- **Epochs.** 20 (matched to the prior run).
- **Optimiser.** Adam(lr=1e-2) (matched).
- **Hidden dim.** 32 (matched across all arms).
- **Statistical procedure.** Pairwise paired Wilcoxon signed-rank with Benjamini-Hochberg FDR over the family `{H1 vs MLP, H2 vs MLP, H3 vs MLP}` — i.e. 3 comparisons, BH at α=0.05.
- **Effect size.** Rank-biserial r per comparison (Kerby 2014).
- **CIs.** BCa 95% on per-arm accuracy median.
- **Reproducibility.** Every per-seed accuracy stored in `notebooks/results/mutag_ablation_30seeds.json`.

## 4. What is NOT being tested in this hypothesis

- Attention (HL-HGAT-style)
- Up/down Laplacian separation (Bunch et al. 2020)
- Richer node features (current MUTAG features are 7-dim one-hot atom labels)
- Graph attention pooling
- More than 2 stacked Hodge layers

These are the natural follow-ups if H3 succeeds (we add complexity to push *past* the MLP). If H3 fails, the next session should reconsider whether MUTAG is the right dataset; deeper architectures on PROTEINS or NCI1 may be a better target.

## 5. Outcome decision tree

| If… | Then… |
|---|---|
| H3 matches MLP (CI on median Δ includes 0, Wilcoxon p_BH ≥ 0.05) | Defensible *equality* claim. README: "the L_norm + residual + 2-layer Hodge architecture is statistically equivalent to MLP on MUTAG". v0.0.2 considers either pushing past MLP with attention/pooling, or moving to a different dataset. |
| H3 strictly beats MLP (median Δ > 0, p_BH < 0.01, CI strictly above 0) | Strong positive empirical claim. README: "Hodge architecture beats MLP on MUTAG". v0.0.2 cuts a release candidate. |
| H3 strictly loses to MLP (median Δ < 0, p_BH < 0.01, CI strictly below 0) | Second negative result. README updates: "even with normalisation, residual, and depth, the Geo subsystem does not beat MLP on MUTAG. The honest interpretation is that MUTAG cannot discriminate between simple architectures at its scale." Move to PROTEINS / NCI1 in v0.0.2. |
| H1 or H2 surprisingly beats MLP alone | Investigate; possibly a regression / data leak. Re-run with different seed splits before publishing. |
