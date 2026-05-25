# Hypothesis 010: Does the high-pass vs low-pass operator distinction predict cross-dataset performance?

**Status.** Resolved 2026-05-25. H42 directionally confirmed but for the wrong reason (gin-residual wins on MUTAG AND NCI1, not MUTAG-specifically); H43 neither arm distinguishable on PROTEINS; H44 partially confirmed (gap is dataset-dependent in magnitude but not direction); H45 refuted (Hodge loses to MLP on MUTAG even with external residual); H46 refuted (neither arm significantly outperforms MLP on PROTEINS). See §6.

**Falsification target.** Whether the choice between high-pass (Hodge Laplacian L_tilde) and low-pass (normalised adjacency I - L_tilde) propagation operators produces dataset-dependent classification differences when both arms use external residual. H008-c showed the operators are interchangeable on NCI1 (gin-residual 0.629 vs Hodge 0.609). This experiment tests whether the same holds on MUTAG and PROTEINS, or whether the high-pass/low-pass distinction interacts with dataset-level structural properties.

**Prior results motivating this hypothesis.**

1. H008-c: On NCI1, gin-residual (low-pass + external residual) slightly outperforms Hodge (high-pass + external residual) at p_BH = 0.010.
2. H006: The constant-feature Hodge signal has the rank ordering MUTAG (+0.098) > PROTEINS (+0.088) > NCI1 (+0.071) — the inverse of the full-feature Hodge-vs-MLP gain.
3. H001: On MUTAG, Hodge-residual (high-pass + external residual) underperforms MLP by 4 pp (p_BH = 0.019). The low-pass variant (gin-residual) has not been tested on MUTAG.

**Theoretical context.** The normalised Laplacian L_tilde acts as a high-pass filter in the spectral domain: its eigenvalues are in [0, 2], with 0 corresponding to the constant eigenvector (DC component) and 2 corresponding to the maximally oscillating eigenvector. Propagation via L_tilde @ h attenuates low-frequency (smooth) signals and amplifies high-frequency (varying) signals across the graph. Conversely, (I - L_tilde) @ h is a low-pass filter that smooths features across neighbours.

On homophilic graphs (connected nodes share labels/features), low-pass smoothing reinforces the class signal. On heterophilic graphs (connected nodes differ), high-pass filtering preserves the class signal while low-pass smoothing destroys it. The TUDataset benchmarks have varying degrees of structural homophily, which may interact with the filter choice.

---

## 1. Design

Run `hodge-mp-residual` (high-pass) and `gin-residual` (low-pass) on all three datasets (MUTAG, PROTEINS, NCI1) with external residual on both. MLP baseline as control.

| Dataset | Hodge (H008-c) | gin-residual (H008-c) | MLP | New data needed? |
|---|---|---|---|---|
| NCI1 | 0.609 | 0.629 | 0.523 | No (reuse H008-c) |
| MUTAG | ? | ? | 0.789 (H001) | Yes |
| PROTEINS | ? | ? | 0.675 (H002) | Yes |

## 2. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H42** | gin-residual outperforms Hodge on MUTAG | gin-residual > Hodge (p_BH < 0.05) | MUTAG is strongly homophilic (aromatic rings, functional groups share atom types); low-pass averaging should be more effective than high-pass differencing | p_BH >= 0.05 or Hodge >= gin-residual |
| **H43** | gin-residual outperforms Hodge on PROTEINS | Uncertain — PROTEINS may have intermediate homophily | Protein secondary-structure elements (helix/sheet/turn) may or may not be homophilically connected | Hodge strictly beats gin-residual at p_BH < 0.05 |
| **H44** | The gin-residual vs Hodge gap is dataset-dependent | The operator advantage (gin-residual median - Hodge median) correlates with some dataset-level property | H006 showed graph-structural separability differs across datasets; the operator preference may track this | All three datasets show the same direction and magnitude |
| **H45** | Both gin-residual and Hodge outperform MLP on MUTAG with external residual | p_BH < 0.05 for both | The external residual should rescue the Hodge-residual arm's failure on MUTAG (H001 used 20 epochs; this uses 10, but the residual architecture is the same) | Either arm <= MLP |
| **H46** | Both gin-residual and Hodge outperform MLP on PROTEINS with external residual | p_BH < 0.05 for both | H002 showed all arms matched MLP without the operator comparison; the external residual may or may not help | Either arm <= MLP |

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H42 confirmed (gin-residual > Hodge on MUTAG), consistent with NCI1 | Low-pass is universally better than high-pass at this capacity. The Hodge Laplacian's high-pass filtering is a disadvantage, not an advantage. The operator distinction exists but favours adjacency averaging. |
| H42 refuted (Hodge >= gin-residual on MUTAG), opposite of NCI1 | **The operator preference is dataset-dependent.** High-pass helps on some graphs, low-pass on others. This would be a genuinely novel finding — identifying conditions under which each operator is preferred. The H006 rank-inversion may have a mechanistic explanation. |
| H45 refuted (both arms <= MLP on MUTAG with external residual) | The external residual is necessary but not sufficient on MUTAG. Dataset size (188 graphs) and/or feature dimensionality (7-dim) create a regime where topology-aware message passing does not help even with optimal residual architecture. |
| All three datasets show gin-residual ≈ Hodge | The high-pass/low-pass distinction does not matter at any tested dataset. The operator is truly irrelevant once the residual is present — a stronger version of the H008-c conclusion. |

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

| Dataset | Hodge (high-pass) | gin-residual (low-pass) | MLP | Hodge vs gin-residual p_BH | Direction |
|---|---|---|---|---|---|
| MUTAG (188) | 0.750 [0.724, 0.789] | 0.789 [0.763, 0.816] | 0.789 [0.763, 0.816] | 7.44 x 10^-3 | low-pass wins |
| PROTEINS (1113) | 0.686 [0.670, 0.717] | 0.675 [0.657, 0.709] | 0.675 [0.596, 0.706] | 0.292 | no difference |
| NCI1 (4110) | 0.609 [0.581, 0.625] | 0.629 [0.607, 0.641] | 0.523 [0.513, 0.566] | 1.01 x 10^-2 | low-pass wins |

### Sub-hypotheses resolved

- **H42** (gin-residual > Hodge on MUTAG): **CONFIRMED directionally** (gin-residual 0.789 > Hodge 0.750, p_BH = 7.44 x 10^-3). However, the prediction that this would be MUTAG-specific due to homophily is not supported — the same direction holds on NCI1.
- **H43** (gin-residual vs Hodge on PROTEINS): Neither arm is distinguishable from the other or from MLP (all p_BH > 0.29). PROTEINS does not discriminate between operators at this configuration.
- **H44** (dataset-dependent gap): **PARTIALLY CONFIRMED.** The gap magnitude varies (significant on MUTAG and NCI1, null on PROTEINS), but the direction never reverses. The low-pass operator is consistently equal or better than the high-pass operator across all three datasets.
- **H45** (both arms beat MLP on MUTAG): **REFUTED.** Hodge (0.750) strictly underperforms MLP (0.789) at p_BH = 8.61 x 10^-3. gin-residual matches MLP (p_BH = 0.438). The external residual does not rescue the Hodge arm on MUTAG — high-pass filtering actively harms classification on this dataset.
- **H46** (both arms beat MLP on PROTEINS): **REFUTED.** Neither arm significantly outperforms MLP (Hodge vs MLP: p_BH = 0.29; gin-residual vs MLP: p_BH = 0.78).

### Interpretation

The high-pass (Hodge Laplacian) vs low-pass (normalised adjacency) operator distinction does not produce a dataset-dependent advantage that favours the Hodge Laplacian on any tested dataset. The low-pass operator is consistently equal or superior:

- **MUTAG:** low-pass matches MLP; high-pass loses by 4 pp. The Laplacian's high-pass filtering attenuates the smooth class signal that MLP captures directly from atom-type features.
- **PROTEINS:** neither operator adds measurable value over MLP. The dataset does not discriminate between architectures at this capacity (consistent with H002).
- **NCI1:** both operators beat MLP with external residual; low-pass slightly ahead (+2 pp over Hodge, p_BH = 0.010).

### What the full investigation establishes (H001-H010)

The complete preregistered investigation, comprising 13 hypotheses and 46 falsifiable sub-predictions across three datasets, converges on the following:

1. **Topology-aware message passing with external residual outperforms MLP on NCI1** (+8-10 pp, robust across operators). This is the one positive claim that survives the full ablation series.
2. **The operative architectural factor is the external residual connection**, not the propagation operator. Without external residual, all message-passing architectures (GIN, GAT, normalised GIN) collapse to class prior on NCI1.
3. **The Hodge Laplacian does not confer a unique advantage on any tested dataset.** The normalised adjacency operator (low-pass) matches or outperforms the Hodge Laplacian (high-pass) on all three datasets when both use external residual.
4. **The high-pass Hodge Laplacian is actively harmful on MUTAG**, where it attenuates the class signal that the MLP captures from features alone.
5. **The NCI1 advantage does not transfer to MUTAG or PROTEINS** at this capacity and epoch budget.
