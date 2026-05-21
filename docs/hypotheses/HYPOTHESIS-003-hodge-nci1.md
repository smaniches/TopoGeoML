# Hypothesis 003: Does scale alone lift the Hodge MP = MLP ceiling? NCI1 (4110 graphs, 22× MUTAG)

**Status.** Open as of 2026-05-21. Test in progress (this branch). Result will fill §6 when the 30-seed × 10-epoch ablation completes.
**Falsification target.** Paired Wilcoxon p_BH < 0.01 on the H1-vs-MLP comparison, with BCa CIs reported. As in hypothesis 002, this is a *strict* positive-difference test.
**Prior results that motivate this hypothesis.**

| Dataset | Graphs | H1 vs MLP | Outcome |
|---|---|---|---|
| MUTAG | 188 | median Δ = +0.000, p_BH = 0.714 | matches (PR #15) |
| PROTEINS | 1113 | median Δ = +0.014, p_BH = 0.548 | matches (PR #16) |
| **NCI1** | **4110** | **TBD** | **this hypothesis** |

The minimal Hodge architectures (combinatorial / symm-normalised / residual / 2-stacked) plateau at MLP performance on both MUTAG and PROTEINS. Two competing explanations remain (hypothesis 002 §6):

1. **Discrimination ceiling.** Both datasets are below the scale at which simple architectures separate (Errica et al. 2020 finding extended to PROTEINS). NCI1 at 4110 graphs is 3.7× PROTEINS' size and the largest standard TUDataset readily available in PyG; if it *also* shows equality, the architecture is the bottleneck and hypothesis 004 needs HL-HGAT-style attention.
2. **Architectural insufficiency irrespective of scale.** The minimal Hodge MP is genuinely no better than MLP for this task family. Scale doesn't change the answer. NCI1 confirms this if it shows equality.

This hypothesis discriminates between (1) and (2) with the most leverage available without writing new architecture code.

---

## 1. NCI1 specifics

| Property | MUTAG | PROTEINS | NCI1 |
|---|---|---|---|
| n_graphs | 188 | 1113 | **4110** |
| Avg nodes / graph | 18 | 39 | 30 |
| Avg edges / graph | 19 | 73 | 32 |
| Node feature dim | 7 (atom one-hot) | 3 (helix/sheet/turn) | **37 (atom one-hot)** |
| Classes | 2 (mutagenic) | 2 (enzyme) | 2 (anti-cancer activity) |
| Citation | Debnath 1991 | Borgwardt 2005 | Wale et al. 2008 |
| Best published Hodge | n/a directly | n/a | n/a |
| GIN / GCN reference | 89-91% | 73-76% | 78-80% |

NCI1's 37-dim atom features are much richer than MUTAG's 7-dim or PROTEINS' 3-dim. The MLP baseline has more information to work with — it might actually beat Hodge if the topology adds nothing on top of well-encoded chemical features. Or it might plateau at the same rate that Hodge does, in which case both architectures saturate.

The 4110-graph sample size triples the effective statistical power compared to PROTEINS: per-fold 80/20 splits give ~822 test graphs vs PROTEINS' ~222. The BCa CIs on per-arm accuracy should be ~2× narrower.

## 2. Preregistered sub-hypotheses (written BEFORE the result lands)

| ID | Sub-hypothesis | Predicted at 30 seeds × 10 epochs | Falsified if |
|---|---|---|---|
| **H8** | Combinatorial Hodge ≈ MLP on NCI1 (replicating PROTEINS, not MUTAG) | p_BH ≥ 0.05 | p_BH < 0.05 in either direction |
| **H9** | H1 (symm-normalised) ≥ MLP on NCI1 — replicates two-dataset equality | p_BH ≥ 0.05 OR (median Δ > 0 AND p_BH < 0.01) | median Δ < 0 with p_BH < 0.05 |
| **H10** (strong, the headline test) | H1 strictly beats MLP at p_BH < 0.01 on NCI1 | as stated | p_BH ≥ 0.01 |
| **H11** (depth at scale) | H3 (deep-residual) > H1 — depth matters at NCI1's bigger scale where PROTEINS already showed depth at least doesn't hurt | median Δ > 0 with p_BH < 0.05 | p_BH ≥ 0.05 |
| **H12** (effect-size convergence) | Combinatorial-vs-normalised gap on NCI1 is closer to PROTEINS (2.9 pp) than MUTAG (9 pp), confirming the "small-graph phenomenon" interpretation | gap ≤ 5 pp | gap > 5 pp (would re-open the small-graph explanation) |

These five sub-claims discriminate cleanly between the two competing explanations from hypothesis 002 §6:

- **If H10 confirmed (H1 strictly beats MLP)**: explanation (1) holds — MUTAG and PROTEINS were both discrimination-ceiling cases, and scale alone closes the gap. The simplest Hodge architecture is genuinely informative for graph classification once the dataset can show it. v0.0.2 ships the strict positive claim.
- **If H10 refuted (H1 = MLP on NCI1 too)**: explanation (2) holds — three-dataset equality. Minimal Hodge architectures saturate at MLP regardless of scale; the next hypothesis needs attention / polynomial filters / SCConv up-down. Strong "topology helps graph classification" claim ruled out at this architectural class.

Either outcome closes a major epistemological question. The hypothesis is genuinely informative either way.

## 3. Experimental design

- **Dataset.** NCI1, full 4110-graph collection from PyG's TUDataset cache.
- **Models.** Same 5 arms as hypotheses 001 and 002 (combinatorial / normalised / residual / deep-residual / mlp-baseline), same matched-capacity discipline.
- **Seeds.** 30 (matched).
- **Epochs.** 10 (matched to PROTEINS).
- **Optimiser.** Adam(lr=1e-2) (matched).
- **Statistical procedure.** Pairwise paired Wilcoxon signed-rank with Benjamini-Hochberg FDR across the full family of C(5, 2) = 10 comparisons at α = 0.05.
- **CIs.** BCa 95% on per-arm accuracy median.
- **Reproducibility.** Every per-seed accuracy stored in `notebooks/results/nci1_hodge_ablation_30seeds.json`.
- **Wall-clock budget.** NCI1 is 3.7× PROTEINS' graph count; expected wall time ~90 min on the CPU container (1.5 hours).

## 4. What this hypothesis does NOT test

- **Attention / polynomial filters / SCConv up-down.** Reserved for hypothesis 004 if NCI1 shows equality.
- **Larger TUDatasets (DD, COLLAB, REDDIT-12K).** Conditional on hypothesis 003's outcome; if NCI1 confirms the strict positive, hypothesis 004 escalates the architecture. If NCI1 refutes it, hypothesis 004 must escalate both architecture AND dataset.
- **Hyperparameter sweep.** Fixed Adam(lr=1e-2), hidden_dim=32, n_epochs=10. The point is the architectural comparison under matched config, not "best Hodge accuracy".

## 5. Outcome decision tree (preregistered, before the result lands)

| Outcome on H1 (`hodge-mp-normalised`) vs MLP on NCI1 | Interpretation | v0.0.2 implication |
|---|---|---|
| Strictly beats MLP (median Δ > 0, p_BH < 0.01, CI > 0) | **First strict positive-difference claim.** Scale lifts the ceiling; topology helps when the dataset can show it. | v0.0.2 release candidate; README + LIMITATIONS rewritten with the strong claim |
| Strictly loses to MLP (median Δ < 0, p_BH < 0.01, CI < 0) | Architectural failure mode that scales — the minimal Hodge architecture is actively worse than MLP on richer-feature datasets. | No release; hypothesis 004 = attention/polynomial filters required, scale alone is not enough |
| Matches MLP (p_BH ≥ 0.05) | **Three-dataset equality** confirmed. Minimal Hodge architectures saturate at MLP regardless of dataset scale. Strong "topology helps" claim ruled out at this architectural class. | v0.0.2 ships the three-dataset equality claim; hypothesis 004 = architectural escalation |
| Mixed (H8/H9 ambiguous) | Document; investigate further per-seed to understand whether the result is dataset-pathology or genuine. | No release until clarified |

## 6. Resolved outcome (filled in when the 30-seed run completes)

*Pending — 30-seed × 10-epoch run started 2026-05-21, expected wall time ~90 min.*
