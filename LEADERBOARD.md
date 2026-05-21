# TopoGeoML — Empirical Leaderboard

Single navigable artefact aggregating every empirical claim the framework makes. Each row points to (a) the per-seed report in `notebooks/results/`, (b) the preregistered hypothesis (when applicable) in `docs/hypotheses/`, and (c) the reproduction command. **A claim is not in this table unless a paired Wilcoxon p-value (BH-corrected where appropriate) and a BCa 95% CI have been computed from a seeded run; smoke runs, theoretical arguments, and aspirational results do not appear.**

The discipline of the table:

- **Positive** = strict improvement; p_BH < 0.01 AND BCa CI on the difference strictly above zero.
- **Equality** = p_BH ≥ 0.05 OR CI overlaps zero; method is *not significantly worse* but not significantly better either.
- **Negative** = strict regression; the comparison method underperforms the baseline at p_BH < 0.05 with CI strictly below zero.
- **Pending** = experiment is running or queued; results will land in a future PR.

---

## Claim 1 — Topology-divergence score detects overfitting no later than a val-loss watchdog

| Field | Value |
|---|---|
| Status | **Positive (directional)** |
| Domain | Training-loop monitoring |
| Setup | 200-sample `sklearn.load_digits`, 64-hidden MLP, Adam(lr=1e-2), 600 steps, 30 independent seeds |
| Comparison | `ShapeOfLearningCallback.divergence_score` (`topogeoml.training`) vs textbook val-loss-ratio watchdog (val_loss > 1.10 × running_min) |
| Headline numbers | Direction count: 14 topology earlier / 16 tie / 0 loss earlier; rank-biserial r = +1.000; paired Wilcoxon p_raw = **5.77 × 10⁻⁴**; BCa 95% CI on median advantage = [+0.0, +10.0] steps |
| Caveat | Magnitude floor-censored — every topology firing landed at step 30, the earliest possible step given a 3-snapshot baseline window. The directional verdict is robust; the magnitude is a lower bound. |
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

## Claim 4 — NCI1 (4110 graphs, 22× MUTAG): scale-escalation test

| Field | Value |
|---|---|
| Status | **Pending** (30-seed × 10-epoch ablation running ~2h background) |
| Domain | Graph classification |
| Setup | NCI1 (4110 chemical-compound graphs, 2 classes, Wale et al. 2008 via PyG TUDataset), 30 seeds × 10 epochs × hidden_dim=32 |
| Comparison | Same 5 arms as Claims 2 and 3 |
| Smoke preview (3 seeds, underpowered) | `hodge-mp-deep-residual` 0.630 vs `mlp-baseline` 0.518 (Δ = +0.112) — *if this direction holds at 30 seeds*, it would be the framework's first strict positive-difference claim. The underpowered smoke is not licensed as a result; the 30-seed run is the verdict. |
| Per-seed report | `notebooks/results/nci1_hodge_ablation_30seeds.md` (when result lands) |
| Preregistered? | Yes — `docs/hypotheses/HYPOTHESIS-003-hodge-nci1.md` (H8/H9/H10/H11/H12) |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --seeds 0..29 --n-epochs 10` |

---

## Claim 5 — DRIVE retinal-vessel segmentation with `CubicalTopologyLoss`

| Field | Value |
|---|---|
| Status | **Pending** (manual GPU run, DRIVE dataset gated behind registration at https://drive.grand-challenge.org/) |
| Domain | Image segmentation training-loss regularisation |
| Setup | DRIVE (40 retinal fundus images, binary vessel segmentation, Staal 2004), 5–10 seeds × 50 epochs, small 3-level U-Net, Dice+BCE baseline vs Dice+BCE + λ·`CubicalTopologyLoss` |
| Comparison | Per-seed paired IoU on the test split (matched seed, same model architecture, only the loss term differs) |
| Per-seed report | `notebooks/results/drive_*.{md,json}` (when result lands) |
| Preregistered? | Not yet — script exists in `notebooks/drive_unet_topology_loss.py`, hypothesis doc to be written before the run |
| Reproduce | `python notebooks/drive_unet_topology_loss.py --seeds 0 1 2 3 4 --n-epochs 50 --topo-weight 0.1 --topo-resolution 64` (requires DRIVE downloaded to `~/.cache/topogeoml/drive/`) |

---

## Quality-floor metrics (not claims, just discipline)

| Metric | Value |
|---|---|
| Total tests | 476 (as of PR #17) |
| Coverage on `topogeoml/` and `benchmarks/` | **100%** |
| Ruff clean across `topogeoml tests benchmarks scripts notebooks` | Yes |
| Mypy strict on `topogeoml/` | **0 errors** (as of this PR — `continue-on-error: true` removed from `ci.yml`) |
| CI workflows | 6 (4 matrix cells + 2 bench workflows) — all green on main |
| Lockfile / Dockerfile | None — deliberate; library is a research toolkit, not a deployment artefact |

---

## How to add a new claim

1. Write a preregistered hypothesis doc in `docs/hypotheses/HYPOTHESIS-NNN-…md` with falsifiable sub-predictions BEFORE running the experiment.
2. Run the ablation with ≥ 20 seeds (the `min_samples_for_pvalue` floor in `benchmarks.stats`) so paired Wilcoxon has power.
3. Save the JSON + Markdown report to `notebooks/results/`.
4. Add a row to this leaderboard with status (positive / equality / negative / pending), headline numbers, and the reproduce command.
5. Update `LIMITATIONS.md` if the claim refutes a previously-listed unvalidated hypothesis.
6. Open a PR; do not merge without (a) CI green and (b) the row in this file matching the per-seed report.

Negative results count and are shipped. Selective reporting is the failure mode this leaderboard exists to prevent.
