# TopoGeoML — Empirical Leaderboard

Single navigable artefact aggregating every empirical claim the framework makes. Each row points to (a) the per-seed report in `notebooks/results/`, (b) the preregistered hypothesis (when applicable) in `docs/hypotheses/`, and (c) the reproduction command. **A claim is not in this table unless a paired Wilcoxon p-value (BH-corrected where appropriate) and a BCa 95% CI have been computed from a seeded run; smoke runs, theoretical arguments, and aspirational results do not appear.**

The discipline of the table:

- **Positive** = strict improvement; p_BH < 0.01 AND BCa CI on the difference strictly above zero.
- **Equality** = p_BH ≥ 0.05 OR CI overlaps zero; method is *not significantly worse* but not significantly better either.
- **Negative** = strict regression; the comparison method underperforms the baseline at p_BH < 0.05 with CI strictly below zero.
- **Exploratory** = a directional signal exists but the design cannot yet support a positive/negative conclusion (e.g. a censored magnitude with no negative control). Reported for transparency; not counted as a confirmed claim.
- **Pending** = experiment is running or queued; results will land in a future PR.

> **Important — read before citing: capacity regime.** Every accuracy number below is obtained under a deliberately constrained *matched-capacity* protocol (1 layer, hidden_dim=32, 10–20 epochs, no batch normalisation, ~1.4–2.3k parameters per arm). This isolates architectural *mechanism* at fixed capacity. It is **not** a benchmark-performance comparison: absolute accuracies (0.50–0.79) sit well below literature SOTA (e.g. properly-trained GNNs reach ~0.80+ on NCI1), and under this protocol the standard GNN baselines (GIN, GAT) collapse to the class prior (0.500) on NCI1. Phrases like "outperforms GIN/GAT" mean "at equal, severely-limited capacity" — **not** "is a better graph classifier." The investigation's *primary* finding is negative: the Hodge Laplacian confers no unique advantage over a normalised-adjacency operator once an external residual is present (Claim 11 / H008c).

---

## Claim 1 — Topology-divergence score detects overfitting no later than a val-loss watchdog

| Field | Value |
|---|---|
| Status | **Exploratory (directional only — floor-limited, no negative control)** |
| Domain | Training-loop monitoring |
| Setup | 200-sample `sklearn.load_digits`, 64-hidden MLP, Adam(lr=1e-2), 600 steps, 30 independent seeds |
| Comparison | `ShapeOfLearningCallback.divergence_score` (`topogeoml.training`) vs textbook val-loss-ratio watchdog (val_loss > 1.10 × running_min) |
| Headline numbers | Direction count: 14 topology earlier / 16 tie / 0 loss earlier; rank-biserial r = +1.000; paired Wilcoxon p_raw = **5.77 × 10⁻⁴**; BCa 95% CI on median advantage = [+0.0, +10.0] steps |
| Why exploratory, not positive | The topology watchdog fired at step 30 — its *earliest possible* step given the 3-snapshot baseline window — in every one of the 30 seeds, and all 30 runs overfit (train loss → 0). The data therefore establish only that topology is never *slower* than the loss watchdog; they do **not** establish that it *anticipates* divergence. No no-overfitting (negative) control has been run, so a genuine falsification test does not yet exist. Reported here for transparency, not counted as a confirmed claim. |
| Per-seed report | `notebooks/results/topology_predicts_divergence_30seeds.md` |
| Preregistered? | No (PR #11 was opportunistic) |
| Reproduce | `python notebooks/topology_predicts_divergence.py --n-seeds 30` |
| First shipped in | PR #11 |

---

## Claim 2 — Symmetrically-normalised Hodge MP matches MLP baseline on MUTAG

| Field | Value |
|---|---|
| Status | **Equality (matched-capacity, BH-corrected family of 10)** |
| Domain | Graph classification |
| Setup | MUTAG (188 molecular graphs, 2 classes, Debnath 1991 via PyG TUDataset), 30 seeds × 20 epochs × hidden_dim=32, stratified 80/20 split per seed |
| Comparison | 5 matched-capacity arms (1378–1442 trainable params): combinatorial L, symm L̃ = D⁻¹/² L D⁻¹/², symm L̃ + residual, symm L̃ + 2 stacked + residual, MLP baseline |
| Headline numbers | `hodge-mp-normalised` 0.789 [0.763, 0.816] vs `mlp-baseline` 0.789 [0.763, 0.816]; median Δ = +0.000; paired Wilcoxon p_BH = **0.714**; rank-biserial r = +0.130 |
| Sub-finding 1 | Combinatorial L underperforms MLP by **9 pp** (p_BH = 5.66 × 10⁻⁴, r = -0.760). Symmetric normalisation closes the entire gap. |
| Sub-finding 2 (H2 refuted) | Adding a residual on top of normalisation *hurts* — `hodge-mp-residual` underperforms MLP at p_BH = 0.019. |
| Sub-finding 3 (H3 refuted) | Two stacked layers + residual: not significantly different from H1 (p_BH = 0.21). |
| Per-seed report | `notebooks/results/mutag_hodge_ablation_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md` (H1/H2/H3) |
| Reproduce | `python -m benchmarks.hodge --datasets mutag --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 20` |
| First shipped in | PR #15 |

---

## Claim 3 — Two-dataset equality holds on PROTEINS; strict positive-difference refuted

| Field | Value |
|---|---|
| Status | **Equality (replicates Claim 2 on a 5.9× larger dataset); strong "topology beats MLP" hypothesis refuted** |
| Domain | Graph classification |
| Setup | PROTEINS (1113 protein graphs, 2 classes, Borgwardt 2005 via PyG TUDataset), 30 seeds × 10 epochs × hidden_dim=32 |
| Comparison | Same 5 arms as Claim 2 |
| Headline numbers | `hodge-mp-normalised` 0.688 [0.670, 0.704] vs `mlp-baseline` 0.675 [0.596, 0.706]; median Δ = +0.014; paired Wilcoxon p_BH = **0.548** |
| Sub-finding (H4 refuted) | The combinatorial-L harm from MUTAG (9 pp, p_BH = 5.66 × 10⁻⁴) does not replicate on PROTEINS (2.9 pp, p_BH = 0.65, r = -0.07). The normalisation effect is dataset-dependent. |
| Cross-dataset claim | The symmetrically-normalised one-layer Hodge MP matches MLP on **both** MUTAG (p_BH = 0.714) and PROTEINS (p_BH = 0.548). Strong "topology helps graph classification" claim ruled out at this architectural class on two TUDatasets. |
| Per-seed report | `notebooks/results/proteins_hodge_ablation_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-002-hodge-proteins.md` (H4/H5/H6/H7) |
| Reproduce | `python -m benchmarks.hodge --datasets proteins --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |
| First shipped in | PR #16 |

---

## Claim 4 — NCI1 scale-escalation: one narrow strict positive-difference claim (matched-capacity regime)

| Field | Value |
|---|---|
| Status | **Positive (strict positive-difference), regime-bound.** See the capacity-regime caveat at the top: this is a matched-capacity *mechanism* result (best arm 0.609 vs MLP 0.523, both ~20 pp below SOTA), not a benchmark-performance claim. H008c (Claim 11) shows the Hodge Laplacian is not the operative factor. |
| Domain | Graph classification |
| Setup | NCI1 (4110 chemical-compound graphs, 2 classes, Wale et al. 2008 via PyG TUDataset), 30 seeds × 10 epochs × hidden_dim=32 |
| Comparison | Same 5 arms as Claims 2 and 3 |
| Headline numbers | `hodge-mp-residual` 0.609 [0.581, 0.625] vs `mlp-baseline` 0.523 [0.513, 0.566]; median Δ = +0.086; paired Wilcoxon p_BH = **4.83 × 10⁻³**; rank-biserial r = +0.533 |
| Sub-finding 1 | Combinatorial L still underperforms MLP (Δ = −0.017, p_BH = 2.6 × 10⁻⁴) |
| Sub-finding 2 | The residual variant — which *lost* on MUTAG and *matched* on PROTEINS — **wins** on NCI1. The residual's contribution scales positively with dataset size at this architectural class. |
| Cross-dataset pattern | Architecture effects invert across datasets: same architecture underperforms MLP on MUTAG, matches on PROTEINS, outperforms on NCI1 |
| Per-seed report | `notebooks/results/nci1_hodge_ablation_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-003-hodge-nci1.md` (H8/H9/H10/H11/H12) |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |
| First shipped in | PR #19 |

---

## Claim 5 — Sample size is NOT the mechanism (H004)

| Field | Value |
|---|---|
| Status | **Negative (mechanism ruled out)** |
| Domain | Mechanism investigation |
| Setup | NCI1 subsampled to {188, 1113, 2000, 4110} graphs per seed, 30 seeds × 10 epochs, `hodge-mp-residual` vs `mlp-baseline` |
| Headline numbers | NCI1@188: Δ = +0.019, p_BH = 0.897 (not significant); NCI1@4110: Δ = +0.086, p_BH = 3.38 × 10⁻³ (control reproduces). Hodge advantage monotone in n (Spearman ρ = +1.0) but never crosses zero — MUTAG's residual-defeat is NOT replicated by sample-size matching. |
| Per-seed report | `notebooks/results/h004_nci1_n{188,1113,2000,4110}_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-004-sample-size-mechanism.md` (H13/H14/H15/H16/H17) |
| Reproduce | See `REPRODUCING.md` §H004 |
| First shipped in | PR #21 |

---

## Claim 6 — Feature dimensionality is NOT the mechanism (H005)

| Field | Value |
|---|---|
| Status | **Negative (mechanism ruled out); secondary positive (feature-degradation robustness)** |
| Domain | Mechanism investigation |
| Setup | NCI1 features projected 37→7 dim (direction A); MUTAG features expanded 7→37 dim (direction B). 30 seeds × 10 epochs, `hodge-mp-residual` vs `mlp-baseline` |
| Headline numbers | NCI1-7d: Hodge 0.581 vs MLP 0.500 (class prior), Δ = +0.081, p_BH = 4.93 × 10⁻⁴ — MLP collapses, Hodge retains signal. MUTAG-37d: Δ = −0.013, p_BH = 0.246 — no difference. |
| Per-seed report | `notebooks/results/h005_{nci1_7d,mutag_37d}_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-005-feature-density-mechanism.md` (H18/H19/H20/H21) |
| Reproduce | See `REPRODUCING.md` §H005 |
| First shipped in | PR #21 |

---

## Claim 7 — Graph-structural signal is universal but rank-inverted vs full-feature gain (H006)

| Field | Value |
|---|---|
| Status | **Positive (graph-structural signal on all 3 datasets); negative (simple topology-predicts-gain hypothesis refuted)** |
| Domain | Mechanism investigation |
| Setup | All node features replaced with constant vector (all-ones). 30 seeds × 10 epochs × 3 datasets, `hodge-mp-residual` vs class prior |
| Headline numbers | MUTAG: +0.098 over class prior (p_BH = 4.53 × 10⁻⁶); PROTEINS: +0.088 (p_BH = 1.41 × 10⁻⁴); NCI1: +0.071 (p_BH = 1.93 × 10⁻⁵). All significant. But constant-feature gap is rank-inverted vs full-feature gain (Spearman ρ = −1.0). |
| Per-seed report | `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-006-graph-topology-mechanism.md` (H22/H23/H24/H25) |
| Reproduce | See `REPRODUCING.md` §H006 |
| First shipped in | PR #22 |

---

## Claim 8 — No single structural proxy explains the full-feature gain (H007)

| Field | Value |
|---|---|
| Status | **Negative (no proxy is predictive of full-feature gain)** |
| Domain | Mechanism investigation (analysis-only, no model training) |
| Setup | Five graph-structural proxies (size, degree, WL subtree, cycle, spectral) × 3 datasets. Per-class separability measured by max |rank-biserial r|. |
| Headline numbers | All five proxies rank MUTAG > PROTEINS > NCI1 (ρ = +1.0 vs constant-feature gap; ρ = −1.0 vs full-feature gain). No single proxy explains where Hodge helps under full features. |
| Per-seed report | `notebooks/results/h007_structural_decomposition.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-007-graph-structural-signal-decomposition.md` (H26/H27) |
| Reproduce | `python -m benchmarks.hodge.h007_analysis` |
| First shipped in | PR #23 |

---

## Claim 9 — Under matched capacity, GIN and GAT collapse to class prior on NCI1; Hodge-MP-residual does not (H008)

| Field | Value |
|---|---|
| Status | **Regime-bound, NOT an expressiveness or SOTA claim.** Under the matched-capacity protocol, GIN (0.500) and GAT (0.500) collapse to the class prior on NCI1 while Hodge-MP-residual reaches 0.609. This is a *training-stability-at-fixed-capacity* finding — properly-trained GIN reaches ~0.80+ on NCI1, so "outperforms GIN/GAT" here means only "at equal, severely-limited capacity." H008-c (Claim 11) subsequently showed a normalised-adjacency arm with an external residual matches or exceeds Hodge, so the operative factor is the residual, not the operator. |
| Domain | Architecture comparison |
| Setup | NCI1 (4110 graphs), 30 seeds × 10 epochs × hidden_dim=32, 4 arms: `hodge-mp-residual`, `gin-baseline`, `gat-baseline`, `mlp-baseline`. All arms matched to ~2339 params. |
| Headline numbers | Hodge 0.609 [0.581, 0.625] vs GIN 0.500 [0.500, 0.505]: p_BH = 6.36 × 10⁻⁶, r = +0.933. Hodge vs GAT 0.500 [0.500, 0.500]: p_BH = 6.36 × 10⁻⁶, r = +1.000. GIN and GAT both strictly underperform MLP 0.523 [0.513, 0.566]. |
| Interpretation | The Hodge arm's symmetric Laplacian normalisation provides training stability that unnormalised GIN/GAT aggregation lacks under the tested capacity constraints. This is an architectural interaction finding, not a theoretical expressiveness claim. |
| Per-seed report | `notebooks/results/h008_nci1_gin_gat_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-008-gin-gat-comparison.md` (H28/H29/H30/H31/H32) |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-residual gin-baseline gat-baseline mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |
| First shipped in | This commit |

---

## Claim 10 — Degree normalisation does not close the GIN-Hodge gap (H008-b)

| Field | Value |
|---|---|
| Status | **Negative (candidate explanation refuted)** |
| Domain | Architecture comparison / mechanism ablation |
| Setup | NCI1 (4110 graphs), 30 seeds × 10 epochs × hidden_dim=32, 4 arms: `hodge-mp-residual`, `gin-normalised` (D^{-1/2}AD^{-1/2}), `gin-baseline` (raw A), `mlp-baseline`. |
| Headline numbers | gin-normalised: 0.500 [0.500, 0.500] — still at class prior. Hodge vs gin-normalised: p_BH = 6.36 × 10⁻⁶, r = +1.000 (perfect rank separation). The candidate explanation from H008 (normalisation accounts for the gap) is refuted. |
| Interpretation | The Hodge advantage on NCI1 is not attributable to degree normalisation alone. The operative architectural difference involves the spectral operator (Laplacian vs adjacency), the weight-propagation interaction order, or the residual placement. |
| Per-seed report | `notebooks/results/h008b_nci1_gin_normalised_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-008b-gin-normalised.md` (H33/H34/H35) |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-residual gin-normalised gin-baseline mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 11 — The external residual, not the Hodge Laplacian, is the operative factor (H008-c) — *primary finding*

| Field | Value |
|---|---|
| Status | **Primary finding of the investigation (refutes "topology helps").** Once an external residual is added, a normalised-*adjacency* operator (low-pass, `I − L̃`) matches or slightly exceeds the Hodge Laplacian (high-pass, `L̃`). The Hodge Laplacian confers no unique advantage. |
| Domain | Architecture comparison / mechanism isolation |
| Setup | NCI1 (4110 graphs), 30 seeds × 10 epochs × hidden_dim=32, 4 arms: `gin-residual` (`I−L̃` + external residual), `hodge-mp-residual` (`L̃` + external residual), `gin-normalised` (internal self-loop), `mlp-baseline`. All differ only in operator/residual placement. |
| Headline numbers | `gin-residual` 0.629 [0.607, 0.641] vs MLP 0.523: p_BH = 6.05 × 10⁻⁴, r = +0.600 (WINS +10.6 pp). `gin-residual` vs `hodge-mp-residual`: Δ = +0.0195, p_BH = 1.01 × 10⁻², r = +0.400 — adjacency *slightly beats* Hodge. `gin-normalised` (no external residual) 0.500 — class-prior collapse. |
| Interpretation | The operative architectural element is the external residual (`act(prop @ W + b) + h`), which preserves projected features through propagation. The choice of spectral operator (high-pass Hodge vs low-pass adjacency) is secondary. This refutes the strong "topology helps graph classification" hypothesis at the tested configuration. |
| Per-seed report | `notebooks/results/h008c_nci1_gin_residual_30seeds.md` |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-008c-gin-residual.md` (H36/H37/H38) |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-residual gin-residual gin-normalised mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Deferred — DRIVE retinal-vessel segmentation with `CubicalTopologyLoss`

| Field | Value |
|---|---|
| Status | **Deferred** (infrastructure exists; experiment not yet run) |
| Domain | Image segmentation training-loss regularisation |
| Setup | DRIVE (40 retinal fundus images, binary vessel segmentation, Staal 2004), 5–10 seeds × 50 epochs, small 3-level U-Net, Dice+BCE baseline vs Dice+BCE + λ·`CubicalTopologyLoss` |
| Comparison | Per-seed paired IoU on the test split (matched seed, same model architecture, only the loss term differs) |
| Preregistered? | Not yet — script exists in `notebooks/drive_unet_topology_loss.py`, hypothesis doc to be written before the run |
| Reproduce | `python notebooks/drive_unet_topology_loss.py --seeds 0 1 2 3 4 --n-epochs 50 --topo-weight 0.1 --topo-resolution 64` (requires DRIVE downloaded to `~/.cache/topogeoml/drive/`) |

---

## Quality-floor metrics (not claims, just discipline)

| Metric | Value |
|---|---|
| Total tests | 500 |
| Coverage | **100% line and 100% branch** on the `topogeoml/` package (full deps), gated in CI (`--cov-branch --cov-fail-under=100`); `benchmarks/` harness ~93% (cross-backend tests need the `bench` extra), outside the gated scope |
| Ruff clean across `topogeoml tests benchmarks scripts notebooks` | Yes |
| Mypy strict on `topogeoml/` | **0 errors** |
| CI workflows | 8 (4 test matrix + 2 CodeQL + benchmark-hodge + experiment runner) — all green on main |
| Registered model arms | 11 (4 Hodge + MLP + GIN + GIN-normalised + GIN-residual + GAT + sheaf-residual + L1-Hodge-residual) |
| Lockfile / Dockerfile | None — deliberate; library is a research toolkit, not a deployment artefact |
| DOI | [10.5281/zenodo.20365816](https://doi.org/10.5281/zenodo.20365816) |

---

## How to add a new claim

1. Write a preregistered hypothesis doc in `docs/hypotheses/HYPOTHESIS-NNN-…md` with falsifiable sub-predictions BEFORE running the experiment.
2. Run the ablation with ≥ 20 seeds (the `min_samples_for_pvalue` floor in `benchmarks.stats`) so paired Wilcoxon has power.
3. Save the JSON + Markdown report to `notebooks/results/`.
4. Add a row to this leaderboard with status (positive / equality / negative / pending), headline numbers, and the reproduce command.
5. Update `LIMITATIONS.md` if the claim refutes a previously-listed unvalidated hypothesis.
6. Open a PR; do not merge without (a) CI green and (b) the row in this file matching the per-seed report.

Negative results count and are shipped. Selective reporting is the failure mode this leaderboard exists to prevent.
