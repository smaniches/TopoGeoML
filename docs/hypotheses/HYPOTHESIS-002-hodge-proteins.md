# Hypothesis 002: Does the H1-winning Hodge architecture beat MLP on PROTEINS, where MUTAG's discrimination ceiling doesn't apply?

**Status.** Open as of 2026-05-21. Test in progress (this branch). Result will fill §6 when the 30-seed run completes.
**Falsification target.** Paired Wilcoxon p_BH < 0.01 on the H1-vs-MLP comparison, with BCa CIs reported. This is a *strict* positive-difference test — equality is no longer enough to license v0.0.2.
**Prior result that motivates this hypothesis.** `notebooks/results/mutag_hodge_ablation_30seeds.md` and `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md` §6 — the symm-normalised one-layer Hodge MP (arm H1) matches MLP on MUTAG (p_BH = 0.714). Errica et al. 2020 argue MUTAG's 188-graph size makes it incapable of discriminating between simple architectures; a larger dataset is the only way to convert the equality claim into a strict win.

---

## 1. Why PROTEINS

The H1 architecture beats every other Hodge variant on MUTAG but ties the MLP baseline. Three possible explanations for the tie:

1. **The architecture is correct but MUTAG is too small to discriminate.** (Errica et al. 2020 finding.) A larger dataset would show the gap.
2. **The architecture is approximately as informative as the MLP for MUTAG-like inputs.** Both models capture the relevant signal; topology adds no new information beyond what the per-atom features already encode.
3. **The architecture has the right inductive bias for some graph structures but not others.** PROTEINS, with larger and more heterogeneous graphs than MUTAG (average 39 nodes, 73 edges vs 18/19), might be a regime where topology starts to matter.

This hypothesis tests (1) and (3) on PROTEINS. If H1 strictly beats MLP on PROTEINS at p_BH < 0.01 with BCa CI above zero, the equality result on MUTAG is the discrimination-ceiling failure and topology *does* help once the dataset can show it. If H1 still ties MLP on PROTEINS, the tie is structural rather than dataset-scale-driven and the next hypothesis needs to test richer architectures (attention, polynomial filters) on the same datasets.

**Dataset characteristics.**

| Property | MUTAG | PROTEINS |
|---|---|---|
| n_graphs | 188 | 1113 (5.9×) |
| Avg nodes / graph | 18 | 39 (2.2×) |
| Avg edges / graph | 19 | 73 (3.8×) |
| Node feature dim | 7 (one-hot atom) | 3 (helix/sheet/turn) or 32 (continuous) |
| Classes | 2 (mutagenic) | 2 (enzyme vs non-enzyme) |
| Citation | Debnath et al. 1991 | Borgwardt et al. 2005, Dobson & Doig 2003 |

The 5.9× larger sample size shifts the bootstrap CI on per-arm accuracy from MUTAG's ±2-3% to PROTEINS' expected ±1-2%, doubling the statistical power for a paired Wilcoxon at the same effect size.

## 2. Preview from a 3-seed × 5-epoch smoke

A quick smoke run (intended only to time the wall-clock cost) produced these *preliminary* numbers, which we explicitly do not use as the empirical claim — Wilcoxon is underpowered at n=3:

| Arm | Smoke median acc | vs combinatorial |
|---|---|---|
| `hodge-mp-classifier` (combinatorial) | ~0.70 | — |
| `hodge-mp-normalised` (H1) | 0.673 | **−0.027** |
| `hodge-mp-residual` (H2) | 0.682 | −0.018 |
| `hodge-mp-deep-residual` (H3) | 0.709 | +0.009 |
| `mlp-baseline` | 0.731 | +0.031 |

**Provisional observation that the 30-seed run will either confirm or refute.** On the smoke, `hodge-mp-normalised` is the *worst* Hodge arm — the opposite of MUTAG, where it was the *best*. If this directional pattern holds at 30 seeds, the H1 architecture is dataset-dependent: it wins on MUTAG (small, low average degree, sparse feature one-hot) and loses on PROTEINS (larger, higher average degree, dense secondary-structure features). That would be a meaningful negative empirical finding — and one with a clean mechanistic interpretation: PROTEINS' higher average degree means more weight on the diagonal of L, which symmetric normalisation rescales by D^{-1} relative to the off-diagonal smoothing; on a graph where the diagonal *is* the signal (per-residue features carry the class), normalisation washes it out.

## 3. Sub-hypotheses (preregistered for the 30-seed run)

| ID | Sub-hypothesis | Predicted at 30 seeds | Falsified if |
|---|---|---|---|
| **H4** | Combinatorial Hodge < MLP on PROTEINS, same direction as MUTAG | p_BH < 0.05, median Δ < 0 | p_BH ≥ 0.05 or median Δ ≥ 0 |
| **H5** | H1 (symm-normalised) ≥ MLP on PROTEINS | p_BH ≥ 0.05 OR (median Δ > 0 AND p_BH < 0.01) | median Δ < 0 with p_BH < 0.05 |
| **H6** (strong) | H1 strictly beats MLP at p_BH < 0.01 | as stated | p_BH ≥ 0.01 |
| **H7** (depth at scale) | H3 (deep-residual) ≥ H1 — depth matters more at PROTEINS scale | p_BH ≥ 0.05 or median Δ > 0 | median Δ < 0 with p_BH < 0.05 |

These are deliberately preregistered (written *before* the 30-seed result lands) so the falsification record cannot be massaged after the fact.

## 4. Experimental design

- **Dataset.** PROTEINS, full 1113-graph collection from PyG's TUDataset cache. Stratified 80/20 train/test split per seed.
- **Models.** Same 5 arms as hypothesis 001 (combinatorial / normalised / residual / deep-residual / mlp-baseline), same matched-capacity discipline (1378-1442 trainable params at hidden_dim=32 or 24 for the deep arm).
- **Seeds.** 30 (matched to hypothesis 001 for direct cross-dataset comparison).
- **Epochs.** 10 (rather than MUTAG's 20). PROTEINS smoke shows convergence by epoch 5; 10 leaves headroom without wasting compute.
- **Optimiser.** Adam(lr=1e-2) (matched).
- **Statistical procedure.** Pairwise paired Wilcoxon signed-rank with Benjamini-Hochberg FDR across the full family of C(5, 2) = 10 comparisons at α = 0.05. Bonferroni-equivalent floor is α = 0.005 per test, comfortably above the H6 target of 0.01.
- **CIs.** BCa 95% on per-arm accuracy median.
- **Reproducibility.** Every per-seed accuracy stored in `notebooks/results/mutag_hodge_proteins_30seeds.json`.

## 5. Outcome decision tree (preregistered)

| Outcome on H1 (`hodge-mp-normalised`) vs MLP | Interpretation | v0.0.2 implication |
|---|---|---|
| Strictly beats MLP (median Δ > 0, p_BH < 0.01, CI > 0) | **First strict positive-difference claim.** Topology helps on PROTEINS. | v0.0.2 release candidate cut |
| Strictly loses to MLP (median Δ < 0, p_BH < 0.01, CI < 0) | **Architecture is dataset-dependent.** Wins on MUTAG, loses on PROTEINS. Symm-normalisation is the *right* choice for low-degree graphs and the *wrong* choice for higher-degree ones. Hypothesis 003 tests degree-aware normalisation. | No release; pivot the Geo subsystem narrative |
| Matches MLP (p_BH ≥ 0.05) | Two-dataset *equality* claim. Topology with symm-normalisation is competitive on small and medium TUDatasets. | v0.0.2 ships the equality claim as the headline, deferring strict-positive to hypothesis 003 |
| Mixed (H4 confirmed but H5 ambiguous) | Combinatorial L is harmful but normalisation alone is not enough at PROTEINS scale. Need attention or polynomial filters. | Document and move to richer architectures in hypothesis 003 |

## 6. Resolved outcome (filled in when the 30-seed run completes)

*Pending — 30-seed run started 2026-05-21, expected wall time ~25 min.*

---

## 7. What this hypothesis deliberately does NOT test

- **Attention-based propagation** (HL-HGAT-style polynomial filters of order 3-5). Reserved for hypothesis 003 if PROTEINS shows topology isn't enough.
- **Up-down Laplacian separation** (SCConv-style). Same.
- **NCI1, ENZYMES, DD**. PROTEINS is the natural first scale-up from MUTAG; further datasets are hypothesis 004+.
- **Richer node features**. PROTEINS' default 3-dim secondary-structure one-hot is what we test; experiments with the 32-dim continuous features go to a future hypothesis.
- **Hyperparameter tuning**. The optimiser, lr, hidden_dim, and epochs are fixed at MUTAG values to keep the comparison clean. A separate "what's the best architecture" sweep would conflate the topology-vs-no-topology signal with the optimiser-tuning signal.
