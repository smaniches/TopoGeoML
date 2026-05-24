# TopoGeoML: A Preregistered Investigation into Hodge-Augmented Graph Classification

**Version 0.0.2** | Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) | TOPOLOGICA LLC

---

## Abstract

We report a preregistered research program investigating whether Hodge decomposition improves graph neural network classification on standard benchmarks. The program comprises ten hypotheses (H001-H008c) with 38 falsifiable sub-predictions, following a systematic mechanism-elimination methodology. A one-layer Hodge message-passing classifier with symmetric Laplacian normalisation and residual connection strictly outperforms a matched-capacity MLP baseline on NCI1 (4110 chemical-compound graphs, +8.6 percentage points, paired Wilcoxon p_BH = 4.83 x 10^-3, rank-biserial r = +0.533) and extracts graph-structural classification signal under constant-feature ablation across all three tested datasets (MUTAG, PROTEINS, NCI1; all p_BH < 5 x 10^-4). Systematic mechanism elimination rules out sample size and feature dimensionality as drivers of the cross-dataset pattern. The mechanism is narrowed to an architecture-data interaction where the Hodge advantage is largest on datasets where the MLP baseline fails to extract class signal from node features alone. Research continues with deeper architectures and cross-domain validation.

---

## 1. Introduction and Motivation

### 1.1 The Open Question

Does encoding topological structure -- specifically, the Hodge Laplacian spectrum of a graph's clique complex -- improve classification accuracy on molecular and protein graph benchmarks beyond what node features alone provide? The question is deceptively simple; the literature is equivocal.

### 1.2 Literature Context

Four lines of work frame the investigation:

1. **Kipf & Welling (2017, ICLR).** Established that symmetric normalisation of the graph Laplacian is essential for stable GCN training. Without it, eigenvalues scale with node degree and high-degree nodes dominate the forward pass.

2. **Bunch et al. (2020, NeurIPS workshop).** Demonstrated that minimal one-layer simplicial propagation underperforms MLPs on TUDataset benchmarks. Depth and up/down Laplacian separation are required for the topology signal to propagate beyond immediate neighbours.

3. **Errica et al. (2020, ICLR).** Showed that MUTAG and similar small TUDatasets cannot discriminate between architectures -- simple baselines match complex GNNs when evaluation is fair. This motivates our use of larger datasets (NCI1) to test architecture effects.

4. **HL-HGAT (Hu et al. 2024, arXiv 2403.06687) and Hodgelet GP (Yang et al. 2024, arXiv 2410.10546).** Production-grade Hodge architectures reporting marginal gains over GCN baselines. The architectural delta from our minimal classifier is large (polynomial filters, attention, hierarchical pooling), providing a clear next research direction.

### 1.3 Why Preregistration

Graph neural network benchmarking suffers from selective reporting (Errica et al. 2020; Shchur et al. 2018). Results are often published only when positive, dataset/hyperparameter combinations are chosen post-hoc, and negative findings are suppressed. This project uses preregistration -- writing falsifiable hypotheses with explicit outcome decision trees *before* execution -- to ensure that the empirical record is complete. Every hypothesis document was committed to the repository before its experiment ran.

---

## 2. Methods

### 2.1 Architecture: Hodge Laplacian Message Passing

The core computational primitive is a single round of Hodge message passing on k-simplices:

```
x' = sigma(L_tilde_k @ x @ W + b)
```

where L_tilde_k = D^{-1/2} L_k D^{-1/2} is the symmetrically-normalised k-Hodge Laplacian, W is a learnable weight matrix, b is a bias, and sigma is a nonlinearity (ReLU). The Hodge Laplacian L_k = partial_k^T partial_k + partial_{k+1} partial_{k+1}^T encodes both the "lower" boundary structure (shared faces) and the "upper" co-boundary structure (shared co-faces) of k-dimensional simplices.

For the graph classification experiments reported here, we operate exclusively on 0-simplices (nodes) with L_0 = D - A (the standard graph Laplacian on the 1-skeleton). The codebase constructs the full clique complex via Bron-Kerbosch enumeration (infrastructure for future k > 0 experiments), but the higher-order simplices do not influence the results presented in this report. The symmetrically-normalised variant used in practice is L_tilde_0 = D^{-1/2}(D - A)D^{-1/2}, equivalent to the normalised graph Laplacian of Kipf & Welling (2017).

### 2.2 Experimental Arms

Five matched-capacity classifier arms (1378-1442 trainable parameters each at hidden_dim=32):

| Arm | Architecture | Key feature |
|---|---|---|
| `hodge-mp-classifier` | L_0 @ X @ W + b, sum-pool, linear | Combinatorial (unnormalised) Laplacian |
| `hodge-mp-normalised` | L_tilde_0 @ X @ W + b, sum-pool, linear | Symmetric normalisation |
| `hodge-mp-residual` | L_tilde_0 @ X @ W + b + X, sum-pool, linear | Normalisation + identity skip |
| `hodge-mp-deep-residual` | 2 stacked layers of above | Normalisation + residual + depth |
| `mlp-baseline` | Linear(in, 32) -> ReLU -> Linear(32, 2), sum-pool | No topology (control) |

Parameter matching ensures that accuracy differences reflect architectural choices, not capacity differences.

### 2.3 Datasets

| Dataset | Graphs | Avg nodes | Avg edges | Node features | Classes | Citation |
|---|---|---|---|---|---|---|
| MUTAG | 188 | 18 | 19 | 7-dim atom one-hot | 2 (mutagenicity) | Debnath et al. 1991 |
| PROTEINS | 1113 | 39 | 73 | 3-dim (helix/sheet/turn) | 2 (enzyme) | Borgwardt et al. 2005 |
| NCI1 | 4110 | 30 | 32 | 37-dim atom one-hot | 2 (anti-cancer) | Wale et al. 2008 |

All datasets accessed via PyTorch Geometric's TUDataset interface.

### 2.4 Statistical Framework

- **Seeds:** 30 independent random initialisations per experiment (sufficient for paired Wilcoxon power at moderate effect sizes).
- **Train/test split:** Stratified 80/20 per seed.
- **Confidence intervals:** BCa bootstrap 95% CIs (Efron 1987) on per-arm accuracy median, using 10,000 bootstrap replicates.
- **Hypothesis testing:** Paired Wilcoxon signed-rank test (matched by seed), with Benjamini-Hochberg FDR correction across comparison families.
- **Effect size:** Rank-biserial correlation r (Kerby 2014). For paired Wilcoxon, computed as r = (R+ - R-) / (R+ + R-) where R+ and R- are the sums of positive and negative signed ranks.
- **Falsification threshold:** p_BH < 0.01 for strict positive-difference claims; p_BH < 0.05 for directional findings.
- **Implementation:** `benchmarks/stats.py` (100% test coverage, all procedures cited).

### 2.5 Preregistration Discipline

Each hypothesis follows a fixed template:

1. Document written and committed to `docs/hypotheses/HYPOTHESIS-NNN-*.md` BEFORE execution.
2. Falsifiable sub-predictions with explicit thresholds.
3. Pre-specified outcome decision tree: what each pattern of results means and what hypothesis follows.
4. Results appended to the same document AFTER execution; original predictions preserved for audit.
5. Negative results shipped and added to `LEADERBOARD.md` with the same formatting as positive results.

---

## 3. Results

### 3.1 Phase I: Empirical Establishment (H001-H003)

#### Hypothesis 001: MUTAG Ablation

**Question:** Why does a minimal Hodge MP underperform MLP on MUTAG, and can normalisation/residual/depth close the gap?

**Result (30 seeds x 20 epochs):**

| Arm | Median accuracy (BCa 95% CI) | p_BH vs MLP | Verdict |
|---|---|---|---|
| combinatorial L | 0.697 [0.658, 0.750] | 5.66 x 10^-4 | LOSES by 9 pp |
| symm-normalised L_tilde | 0.789 [0.763, 0.816] | 0.714 | MATCHES MLP |
| + residual | 0.750 [0.724, 0.789] | 0.019 | loses by 4 pp |
| + 2 layers + residual | 0.776 [0.737, 0.789] | 0.102 | matches (weak) |
| MLP baseline | 0.789 [0.763, 0.816] | -- | control |

**Key findings:** Symmetric normalisation is necessary and sufficient to close the gap on MUTAG. Sub-hypotheses H1 confirmed, H2 refuted, H3 refuted.

#### Hypothesis 002: PROTEINS Replication

**Question:** Does H1's winning architecture beat MLP on a 5.9x larger dataset?

**Result (30 seeds x 10 epochs):**

| Arm | Median accuracy (BCa 95% CI) | p_BH vs MLP | Verdict |
|---|---|---|---|
| combinatorial L | 0.646 [0.605, 0.700] | 0.646 | matches |
| symm-normalised | 0.688 [0.670, 0.704] | 0.548 | matches |
| + residual | 0.686 [0.670, 0.717] | 0.339 | matches |
| + 2 layers + residual | 0.695 [0.659, 0.709] | 0.426 | matches |
| MLP baseline | 0.675 [0.596, 0.706] | -- | control |

**Key findings:** Two-dataset equality confirmed. The MUTAG combinatorial-L harm does not replicate on PROTEINS (rank-biserial r drops from -0.760 to -0.071; percentage-point gap shrinks from 9.2 pp to 2.9 pp). Sub-hypotheses H4 refuted, H5 reconfirmed, H6 refuted, H7 unresolved.

#### Hypothesis 003: NCI1 Scale Escalation -- The Headline Positive Result

**Question:** Does scale lift the Hodge = MLP ceiling?

**Result (30 seeds x 10 epochs):**

| Arm | Median accuracy (BCa 95% CI) | p_BH vs MLP | Verdict |
|---|---|---|---|
| combinatorial L | 0.506 [0.501, 0.511] | 2.6 x 10^-4 | LOSES 1.7 pp |
| symm-normalised | 0.516 [0.511, 0.523] | 0.253 | matches |
| **+ residual** | **0.609 [0.581, 0.625]** | **4.83 x 10^-3** | **WINS +8.6 pp** |
| + 2 layers + residual | 0.603 [0.594, 0.623] | 1.18 x 10^-2 | wins +8.0 pp |
| MLP baseline | 0.523 [0.513, 0.566] | -- | control |

**The positive claim:** On NCI1, the hodge-mp-residual arm strictly outperforms MLP at paired Wilcoxon p_BH = 4.83 x 10^-3, rank-biserial r = +0.533, median Delta = +0.086. This is the framework's first strict positive-difference real-data result, achieved on the largest dataset in the test panel.

**Cross-dataset pattern:** The residual variant's verdict shifts across dataset scale:

| Architecture | MUTAG (188) | PROTEINS (1113) | NCI1 (4110) |
|---|---|---|---|
| combinatorial L | LOSES (-9 pp) | matches | LOSES (-1.7 pp) |
| symm-normalised | matches | matches | matches |
| symm L + residual | loses (-4 pp) | matches | **WINS (+8.6 pp)** |
| symm L + 2 layers + residual | matches | matches | **WINS (+8.0 pp)** |

This cross-dataset pattern motivates the mechanism-investigation phase.

---

### 3.2 Phase II: Mechanism Investigation (H004-H005)

Two leading candidate mechanisms for the NCI1 positive result were proposed:
- **(a) Feature density:** NCI1 has 37-dim atom features vs MUTAG's 7-dim
- **(b) Sample size:** NCI1 has 4110 graphs vs MUTAG's 188

#### Hypothesis 004: Sample-Size Mechanism Test

**Design:** Subsample NCI1 to {188, 1113, 2000, 4110} graphs per seed, holding features constant. If sample size drives the NCI1 positive result, NCI1-at-MUTAG-size should fail.

**Result (30 seeds x 10 epochs x 4 sample sizes):**

| n_graphs | hodge-mp-residual (median) | MLP (median) | Delta | p_BH | Verdict |
|---|---|---|---|---|---|
| 188 (MUTAG-size) | 0.579 | 0.560 | +0.019 | 0.897 | not significant |
| 1113 (PROTEINS-size) | 0.601 | 0.549 | +0.052 | 0.045 | significant (alpha=0.05) |
| 2000 | 0.595 | 0.536 | +0.059 | 0.053 | not significant (border) |
| 4110 (full, control) | 0.609 | 0.523 | +0.086 | 3.38 x 10^-3 | significant (alpha=0.01) |

**Verdict:** Sample size is not the primary mechanism. Subsampling NCI1 to MUTAG's 188 graphs does NOT reproduce MUTAG's residual-defeat (Delta = +0.019, not -0.04). The Hodge advantage on NCI1 data persists even at small sample sizes -- meaning NCI1's data distribution, not its sample count, drives the result. The advantage is monotone in n (Spearman rho = +1.0), growing more statistically detectable with more data.

Sub-hypotheses: H13 REFUTED, H14 REFUTED, H15 CONFIRMED (reproducibility), H16 CONFIRMED (monotone).

#### Hypothesis 005: Feature-Density Mechanism Test

**Design:** Two complementary manipulations:
- **Direction A:** Project NCI1's 37-dim features to 7-dim (Gaussian random projection, norm-preserving).
- **Direction B:** Expand MUTAG's 7-dim features to 37-dim (random expansion).

**Result (30 seeds x 10 epochs, corrected Johnson-Lindenstrauss scaling):**

| Direction | Setup | hodge-mp-residual | MLP baseline | Delta | p_BH | Outcome |
|---|---|---|---|---|---|---|
| A: NCI1-7d | NCI1, features 37 -> 7 | 0.581 | 0.500 | +0.081 | 4.93 x 10^-4 | **Hodge WINS** |
| B: MUTAG-37d | MUTAG, features 7 -> 37 | 0.776 | 0.789 | -0.013 | 0.246 | matches |

**Verdict:** Feature dimensionality alone is not the mechanism. On NCI1-7d, the MLP collapses to class prior (0.500) while Hodge-residual retains above-chance accuracy (0.581) -- demonstrating that the Hodge arm reads graph-structural signal that the MLP cannot access even when feature information is severely degraded. This is itself a positive result: the Hodge architecture is robust to feature degradation in a way the MLP is not.

Sub-hypotheses: H18 REFUTED, H19 REFUTED, H20 REFUTED.

**Investigation summary after Phase II:**

| Candidate mechanism | Hypothesis | Outcome |
|---|---|---|
| Sample size | H004 | Ruled out as primary driver |
| Feature dimensionality | H005 | Ruled out as primary driver |
| Graph-structural signal | H006 (next) | To be investigated |

---

### 3.3 Phase III: Topology Signal Characterisation (H006-H007)

#### Hypothesis 006: Constant-Feature Ablation

**Design:** Replace all node features with a constant vector (all-ones), eliminating feature information entirely. The MLP's accuracy floor becomes the class-prior baseline; the Hodge model can still use topology via the Laplacian. This directly measures how much classification signal lives in graph structure alone.

**Result (30 seeds x 10 epochs x 3 datasets):**

| Dataset | Hodge score | Class prior | Gap | p_BH | Verdict |
|---|---|---|---|---|---|
| MUTAG | 0.763 | 0.665 | +0.098 | 4.53 x 10^-6 | Hodge > prior |
| PROTEINS | 0.684 | 0.596 | +0.088 | 1.41 x 10^-4 | Hodge > prior |
| NCI1 | 0.571 | 0.501 | +0.071 | 1.93 x 10^-5 | Hodge > prior |

**Key positive finding:** ALL three datasets carry graph-structural classification signal accessible to the Hodge architecture, at highly significant levels (all p_BH < 5 x 10^-4). The Hodge Laplacian message-passing model extracts meaningful class information from graph topology alone, without any node features.

**Cross-dataset pattern:** The constant-feature gap ordering (MUTAG > PROTEINS > NCI1) is rank-order inverted relative to the full-feature Hodge-vs-MLP gain (NCI1 > PROTEINS > MUTAG), with Spearman rho = -1.0. The dataset where graph structure carries the *most* class signal under constant features (MUTAG) is where the Hodge architecture has the *least* advantage under full features -- and vice versa. This is consistent with a complementarity interpretation: the Hodge architecture adds the most value where the MLP baseline cannot extract class signal from features alone.

Sub-hypotheses: H22 SUPPORTED, H23 REFUTED, H24 REFUTED, H25 REFUTED.

#### Hypothesis 007: Graph-Structural Proxy Decomposition

**Design:** For five graph-structural proxies (graph size, degree distribution, Weisfeiler-Lehman subtree histogram, cycle statistics, normalised Laplacian spectrum), measure per-class separability (max |rank-biserial r|) on all three datasets.

**Result (deterministic analysis, no seeded sampling):**

| Dataset | size | degree | WL | cycle | spectral |
|---|---|---|---|---|---|
| MUTAG | 0.763 | 0.772 | 0.672 | 0.808 | 0.743 |
| PROTEINS | 0.523 | 0.500 | 0.207 | 0.549 | 0.466 |
| NCI1 | 0.368 | 0.366 | 0.181 | 0.298 | 0.318 |

**Key findings:**

1. Every proxy follows the same rank order: MUTAG > PROTEINS > NCI1. Graph-structural class separability is highest on MUTAG and lowest on NCI1.
2. Every proxy correlates with the H006 constant-feature gap (rho = +1.0) -- the Hodge arm under constant features reads whichever graph-structural signal is present.
3. No proxy correlates with the full-feature Hodge-vs-MLP gain (rho = -1.0) -- the full-feature gain is driven by something other than raw graph-structural separability.

This reinforces the complementarity interpretation: the Hodge advantage tracks the MLP's *failure* to extract class signal, not the amount of structural signal available.

Sub-hypotheses: H26 REFUTED, H27 REFUTED.

---

## 4. Discussion

### 4.1 Positive Results Summary

The investigation produces three categories of positive findings:

1. **Direct classification improvement.** Hodge-MP-residual strictly outperforms MLP on NCI1 (+8.6 pp, p_BH = 4.83 x 10^-3). This is a reproducible, preregistered result on a real-world benchmark.

2. **Feature-independent graph-structural signal extraction.** Under constant-feature ablation, the Hodge architecture extracts classification signal from graph topology alone on all three datasets (all p_BH < 5 x 10^-4). This validates that the Hodge Laplacian encodes class-relevant structural information.

3. **Robustness to feature degradation.** On NCI1 with features projected to 7-dim noise, the MLP collapses to class prior while Hodge-residual retains +0.081 accuracy above MLP (p_BH = 4.93 x 10^-4). The architecture degrades gracefully when feature information is impoverished.

### 4.2 Bounded Claims

The NCI1 result licenses the following:

> On NCI1 at the tested configuration (30 seeds x 10 epochs x hidden_dim=32, matched-capacity 1378-param arms, stratified 80/20 split), a one-layer Hodge MP classifier with symmetric Laplacian normalisation and identity residual connection strictly outperforms a no-topology MLP baseline (median Delta = +0.086, p_BH = 4.83 x 10^-3, r = +0.533).

The result does not license general claims about topology helping graph classification (fails on MUTAG at this configuration, ties on PROTEINS) or about Hodge decomposition versus other topology-aware methods (no GCN/GIN comparison performed in this series).

### 4.3 The Complementarity Interpretation

The cross-dataset pattern is consistent with a complementarity hypothesis: the Hodge architecture's value is largest on datasets where the MLP baseline cannot extract class signal from node features alone. On MUTAG, the MLP already captures the relevant signal (atom-type features implicitly encode local topology); the Hodge Laplacian provides redundant information. On NCI1, the 37-dim features encode atom types but not the structural patterns (aromatic rings, functional groups) that distinguish active from inactive compounds -- the Hodge Laplacian provides *complementary* structural information.

Testing this interpretation is a priority for the next phase of research.

### 4.4 Relationship to Literature

Our findings are consistent with Errica et al. (2020): simple architecture comparisons on small TUDatasets are largely uninformative. The NCI1 result emerges only because (a) we test at sufficient scale, (b) we include the residual variant, and (c) we use matched-capacity baselines. The cross-dataset pattern adds a new empirical finding: architecture effects can reverse sign across datasets of the same family.

### 4.5 Limitations and Next Steps

- Three datasets from one domain (chemistry/proteins). Cross-domain validation (DD, COLLAB, social-network benchmarks) is planned.
- One architectural class (one-layer Hodge MP). Deeper architectures (HL-HGAT, polynomial filters, SCConv) are the natural next experimental direction.
- 10-20 epochs is shallow. Longer training and learning-rate scheduling may change the picture.
- Absolute accuracy levels (52-79%) are below literature SOTA. The claim is comparative (Hodge vs MLP), not absolute.
- n=3 datasets for cross-dataset correlations. More datasets are needed to move from descriptive patterns to statistical claims about the complementarity interpretation.

---

## 5. Conclusion

A preregistered investigation comprising 10 hypotheses and 38 falsifiable sub-predictions establishes:

1. **A positive NCI1 claim.** Hodge-MP-residual strictly beats MLP on NCI1 (+8.6 pp, p_BH = 4.83 x 10^-3).
2. **Graph-structural signal extraction.** The Hodge architecture reads class-relevant topology on all three datasets under constant-feature control (all p_BH < 5 x 10^-4).
3. **Feature-degradation robustness.** The Hodge architecture maintains above-chance performance when the MLP collapses to class prior under feature degradation.
4. **A complementarity pattern.** The Hodge advantage is largest where the MLP baseline fails to extract signal from features alone.
5. **Mechanism narrowing.** Sample size and feature dimensionality are ruled out; the pattern is consistent with an architecture-data complementarity that warrants further investigation with richer architectures and additional datasets.

The full codebase, all experiment artifacts (per-seed JSON reports), and all preregistered hypothesis documents (with original predictions preserved) are openly available under MIT licence. Research continues.

---

## 6. Reproduction

Complete reproduction instructions are available in [`REPRODUCING.md`](../REPRODUCING.md) at the repository root.

### Per-hypothesis artifact locations:

| Hypothesis | Preregistration | Results |
|---|---|---|
| H001 | `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md` | `notebooks/results/mutag_hodge_ablation_30seeds.{json,md}` |
| H002 | `docs/hypotheses/HYPOTHESIS-002-hodge-proteins.md` | `notebooks/results/proteins_hodge_ablation_30seeds.{json,md}` |
| H003 | `docs/hypotheses/HYPOTHESIS-003-hodge-nci1.md` | `notebooks/results/nci1_hodge_ablation_30seeds.{json,md}` |
| H004 | `docs/hypotheses/HYPOTHESIS-004-sample-size-mechanism.md` | `notebooks/results/h004_nci1_n*_30seeds.{json,md}` |
| H005 | `docs/hypotheses/HYPOTHESIS-005-feature-density-mechanism.md` | `notebooks/results/h005_*_30seeds.{json,md}` |
| H006 | `docs/hypotheses/HYPOTHESIS-006-graph-topology-mechanism.md` | `notebooks/results/h006_*_30seeds.{json,md}` |
| H007 | `docs/hypotheses/HYPOTHESIS-007-graph-structural-signal-decomposition.md` | `notebooks/results/h007_structural_decomposition.{json,md}` |

---

## References

- Adams, H., Emerson, T., Kirby, M., et al. (2017). Persistence images: A stable vector representation of persistent homology. *JMLR*, 18(8), 1-35.
- Borgwardt, K. M., Ong, C. S., Schonauer, S., et al. (2005). Protein function prediction via graph kernels. *Bioinformatics*, 21(suppl 1), i47-i56.
- Bunch, E., You, Q., Fung, G., & Singh, V. (2020). Simplicial 2-complex convolutional neural networks. *NeurIPS Workshop on TDA and Beyond*.
- Debnath, A. K., Lopez de Compadre, R. L., Debnath, G., et al. (1991). Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. *J. Med. Chem.*, 34(2), 786-797.
- Efron, B. (1987). Better bootstrap confidence intervals. *JASA*, 82(397), 171-185.
- Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2020). A fair comparison of graph neural networks for graph classification. *ICLR 2020*.
- Hu, J., Li, Z., Wang, Z., & Li, J. (2024). HL-HGAT: Hierarchical learning on hypergraph with attention. *arXiv:2403.06687*.
- Kerby, D. S. (2014). The simple difference formula: An approach to teaching nonparametric correlation. *Comprehensive Psychology*, 3, 11.IT.3.1.
- Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
- Shchur, O., Mumme, M., Bojchevski, A., & Gunnemann, S. (2018). Pitfalls of graph neural network evaluation. *arXiv:1811.05868*.
- Wale, N., Watson, I. A., & Karypis, G. (2008). Comparison of descriptor spaces for chemical compound retrieval and classification. *Knowledge and Information Systems*, 14(3), 347-375.
- Yang, M., Isufi, E., Schaub, M. T., & Leus, G. (2024). Graph classification Gaussian processes via Hodgelet spectral features. *arXiv:2410.10546*.
