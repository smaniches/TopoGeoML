# TopoGeoML Empirical Evidence Index

The filename is retained for stable links, but this document is an evidence index, not a competitive model leaderboard. It summarizes the current empirical record, points to the corresponding preregistration and result artifact, and states what each experiment does and does not establish.

## Status vocabulary

- **Positive difference:** the declared directional comparison crosses its preregistered significance threshold.
- **Significant regression:** the tested method is significantly lower than the comparator under the declared rule.
- **No significant difference detected:** the comparison does not cross the declared threshold. This is not statistical equivalence.
- **Refuted:** a preregistered falsification condition is satisfied.
- **Inconclusive:** neither the preregistered confirmation condition nor the falsification condition is satisfied, or the data do not test the intended mechanism cleanly.
- **Exploratory:** the design is intentionally nonconfirmatory.
- **Invalidated:** a later implementation audit shows that the executed experiment did not instantiate the mathematical or computational object stated by the hypothesis. The historical artifact is retained for provenance but is not current evidence for that claim.
- **Pending:** the preregistered experiment is incomplete and licenses no statistical claim.

Seeded model comparisons use paired Wilcoxon tests with Benjamini-Hochberg correction within their declared comparison family unless the source document states otherwise. Investigation-wide multiplicity is reported separately in [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md). Deterministic analyses are labeled as such rather than being forced into a seeded-test template.

> **Capacity regime.** The graph-classification experiments are deliberately constrained mechanism studies, primarily one layer, `hidden_dim=32`, 10 to 20 epochs, no batch normalisation, and matched parameter budgets. They are not competitive benchmark submissions. Absolute NCI1 accuracies in this record are substantially below well-tuned literature baselines. Any phrase such as “outperforms” refers only to the stated matched experimental configuration.

---

## Claim 1: Topology-divergence trigger is not later than the loss trigger in the published overfitting experiment

| Field | Value |
|---|---|
| Status | **Exploratory** |
| Domain | Training-loop monitoring |
| Setup | 200 `sklearn.load_digits` samples, 64-hidden MLP, Adam(lr=1e-2), 600 steps, 30 seeds |
| Result | 14 topology earlier / 16 ties / 0 loss earlier; rank-biserial r = +1.000; paired Wilcoxon p_raw = 5.77 x 10^-4; BCa 95% CI on median step advantage = [0.0, 10.0] |
| Limitation | The topology trigger fires at its earliest possible step in every seed and no non-overfitting negative control was run. The experiment does not establish anticipatory prediction of divergence. |
| Artifact | `notebooks/results/topology_predicts_divergence_30seeds.md` |
| Preregistered | No |
| Reproduce | `python notebooks/topology_predicts_divergence.py --n-seeds 30` |

---

## Claim 2: MUTAG normalization removes the detected combinatorial-L deficit, but does not establish superiority over MLP

| Field | Value |
|---|---|
| Status | **Mixed: regression plus no significant difference findings** |
| Domain | Graph classification, H001 |
| Setup | MUTAG, 188 graphs, 30 seeds, 20 epochs, hidden_dim=32 |
| Normalized Hodge vs MLP | 0.789 [0.763, 0.816] vs 0.789 [0.763, 0.816]; median Delta = 0.000; p_BH = 0.714. **No significant difference detected.** |
| Combinatorial Hodge vs MLP | About -9 pp; p_BH = 5.66 x 10^-4. **Significant regression.** |
| Hodge residual vs MLP | About -4 pp; p_BH = 0.019. **Significant regression at alpha=0.05.** |
| Deep residual arm | No significant improvement over the selected normalized arm at this experiment's comparison threshold |
| Artifact | `notebooks/results/mutag_hodge_ablation_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md` |
| Reproduce | `python -m benchmarks.hodge --datasets mutag --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 20` |

---

## Claim 3: PROTEINS does not show a significant normalized-Hodge improvement over MLP

| Field | Value |
|---|---|
| Status | **No significant difference detected; strict superiority hypothesis refuted** |
| Domain | Graph classification, H002 |
| Setup | PROTEINS, 1113 graphs, 30 seeds, 10 epochs, hidden_dim=32 |
| Result | normalized Hodge 0.688 [0.670, 0.704] vs MLP 0.675 [0.596, 0.706]; median Delta = +0.014; p_BH = 0.548 |
| Interpretation | The experiment does not support a positive-difference claim for normalized Hodge over MLP. It is not an equivalence test. |
| Artifact | `notebooks/results/proteins_hodge_ablation_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-002-hodge-proteins.md` |
| Reproduce | `python -m benchmarks.hodge --datasets proteins --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 4: Hodge-residual has a narrow positive difference from MLP on NCI1

| Field | Value |
|---|---|
| Status | **Positive difference, regime-bound** |
| Domain | Graph classification, H003 |
| Setup | NCI1, 4110 graphs, 30 seeds, 10 epochs, hidden_dim=32 |
| Result | Hodge-residual 0.609 [0.581, 0.625] vs MLP 0.523 [0.513, 0.566]; median Delta = +0.086; p_BH = 4.83 x 10^-3; r = +0.533 |
| Investigation-wide correction | The previously reported 59-comparison retrospective sensitivity analysis included invalidated H009 comparisons and remains withdrawn until regenerated from the validated comparison set, which the corrected H009-R rerun (Claim 12b) now completes. |
| Scope | Later H008c/H010 operator controls show that this positive difference is not unique to the Hodge `L_0` operator. |
| Artifact | `notebooks/results/nci1_hodge_ablation_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-003-hodge-nci1.md` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 5: Sample size alone does not explain the MUTAG/NCI1 sign change

| Field | Value |
|---|---|
| Status | **Mechanism hypothesis not supported** |
| Domain | H004 sample-size ablation |
| Setup | NCI1 subsampled to 188, 1113, 2000, and 4110 graphs per seed; 30 seeds |
| NCI1 at 188 | Hodge-residual vs MLP median Delta = +0.019; p_BH = 0.897 |
| NCI1 at 4110 | median Delta = +0.086; p_BH = 3.38 x 10^-3 |
| Interpretation | Matching MUTAG's sample count does not reproduce MUTAG's negative Hodge-residual difference. Sample size alone is therefore insufficient to explain the cross-dataset sign change. The increasing NCI1 effect with sample count is descriptive and does not identify a causal mechanism by itself. |
| Artifact | `notebooks/results/h004_nci1_n{188,1113,2000,4110}_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-004-sample-size-mechanism.md` |
| Reproduce | See `REPRODUCING.md` §H004 |

---

## Claim 6: Feature dimensionality alone does not explain the cross-dataset result

| Field | Value |
|---|---|
| Status | **Mechanism hypothesis not supported; one positive degradation result** |
| Domain | H005 feature ablation |
| NCI1 37 to 7 dimensions | Hodge-residual 0.581 vs MLP 0.500; median Delta = +0.081; p_BH = 4.93 x 10^-4 |
| MUTAG 7 to 37 dimensions | median Delta = -0.013; p_BH = 0.246; no significant difference detected |
| Interpretation | Dimensionality manipulation does not transfer the dataset behavior in either direction. The NCI1 projected-feature result shows robustness of the tested graph-aware arm relative to MLP under that perturbation, not a general theorem about feature degradation. |
| Artifact | `notebooks/results/h005_{nci1_7d,mutag_37d}_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-005-feature-density-mechanism.md` |
| Reproduce | See `REPRODUCING.md` §H005 |

---

## Claim 7: Graph-structural classification signal is detected on all three tested datasets under constant features

| Field | Value |
|---|---|
| Status | **Positive differences from class-prior controls on three tested datasets** |
| Domain | H006 constant-feature ablation |
| MUTAG | +0.098 over class prior; p_BH = 4.53 x 10^-6 |
| PROTEINS | +0.088; p_BH = 1.41 x 10^-4 |
| NCI1 | +0.071; p_BH = 1.93 x 10^-5 |
| Interpretation | The tested graph-aware architecture extracts class-relevant graph structure when node features are constant on these datasets. The result does not imply that the Hodge operator is uniquely responsible or that the result generalizes beyond the three datasets. |
| Artifact | `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-006-graph-topology-mechanism.md` |
| Reproduce | See `REPRODUCING.md` §H006 |

---

## Claim 8: None of the five tested structural proxies tracks the full-feature cross-dataset pattern

| Field | Value |
|---|---|
| Status | **Deterministic descriptive mechanism result** |
| Domain | H007 structural proxy analysis |
| Proxies | graph size, degree distribution, WL subtree histogram, cycle statistics, normalized-Laplacian spectrum |
| Result | All five proxies rank MUTAG > PROTEINS > NCI1, matching the H006 constant-feature ordering and reversing the full-feature Hodge-versus-MLP ordering |
| Scope | There are only three datasets. The rank pattern is descriptive; it does not support a population-level correlation claim or prove that no untested structural variable could explain the effect. |
| Artifact | `notebooks/results/h007_structural_decomposition.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-007-graph-structural-signal-decomposition.md` |
| Reproduce | `python -m benchmarks.hodge.h007_analysis` |

---

## Claim 9: Hodge-residual separates from the tested GIN/GAT baselines under the matched-capacity NCI1 protocol

| Field | Value |
|---|---|
| Status | **Positive pairwise differences, mechanism not isolated by H008 alone** |
| Domain | H008 architecture comparison |
| Result | Hodge-residual 0.609 vs GIN 0.500 and GAT 0.500; Hodge vs each p_BH = 6.36 x 10^-6 |
| Scope | The standard baselines are intentionally tiny and short-trained here. This is not an expressiveness or state-of-the-art comparison. H008-b and H008c are required to interpret the architectural cause. |
| Artifact | `notebooks/results/h008_nci1_gin_gat_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-008-gin-gat-comparison.md` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-residual gin-baseline gat-baseline mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 10: Degree normalization with the tested internal-self GIN formulation does not recover NCI1 performance

| Field | Value |
|---|---|
| Status | **Candidate mechanism refuted** |
| Domain | H008b |
| Result | `gin-normalised` median = 0.500; Hodge-residual vs gin-normalised p_BH = 6.36 x 10^-6 |
| Interpretation | Normalizing adjacency while retaining the tested internal-self GIN update is insufficient. This experiment does not by itself identify which later architectural change is causal. |
| Artifact | `notebooks/results/h008b_nci1_gin_normalised_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-008b-gin-normalised.md` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-residual gin-normalised gin-baseline mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 11: The matched external-residual adjacency formulation recovers NCI1 performance and removes a unique Hodge advantage

| Field | Value |
|---|---|
| Status | **Primary architecture finding, scoped** |
| Domain | H008c |
| gin-residual vs gin-normalised | 0.629 vs 0.500; p_BH = 5.20 x 10^-6 |
| gin-residual vs Hodge | median Delta = +0.0195; p_BH = 1.01 x 10^-2; r = +0.400 |
| gin-residual vs MLP | median Delta = +0.1058; p_BH = 6.05 x 10^-4; r = +0.600 |
| Interpretation | The tested external-residual adjacency formulation is sufficient to recover performance and the matched Hodge arm has no unique advantage. `gin-normalised` and `gin-residual` differ in placement and parameterization of the self path, so this is not a universal proof that residual connections alone are the sole causal mechanism. |
| H37 decision-rule note | The preregistration encoded “match” as p_BH >= 0.05. The observed result is significant in the favorable gin-residual direction, not an equivalence result. It nevertheless directly rules out the preregistered Hodge-superiority falsification condition. |
| Artifact | `notebooks/results/h008c_nci1_gin_residual_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-008c-gin-residual.md` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-residual gin-residual gin-normalised mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 12: H009 is invalidated as evidence for a learned cellular-sheaf Laplacian

| Field | Value |
|---|---|
| Status | **Invalidated by implementation audit; H39-H41 resolved separately by the corrective replication H009-R (Claim 12b)** |
| Domain | H009 |
| Defect | The historical `sheaf-residual` implementation processed both orientations of each undirected edge independently. The resulting matrix was not guaranteed symmetric or positive semidefinite and was not, in general, the claimed `delta.T @ delta` sheaf Laplacian. |
| Capacity error | `Linear(64, 2)` has 130 trainable parameters, not 65. On NCI1 the historical sheaf arm has 2,468 parameters versus 2,338 for the fixed-operator controls, about 5.56% larger and slightly outside the stated 5% tolerance. |
| Historical artifact | The archived run reports sheaf median 0.604 and its pairwise statistics. Those numbers describe the historical implementation only and have no current inferential status for a cellular-sheaf claim. |
| Current scientific status | The rerun requirements were satisfied: the operator was repaired and invariant-tested (merge `b89d196`), capacity was accounted exactly, the corrective protocol was preregistered as H009-R, and the 30-seed NCI1 experiment was rerun from the corrected commit. H39-H41 verdicts now come from Claim 12b; the historical artifact remains provenance only. |
| Artifact | `notebooks/results/h009_nci1_sheaf_30seeds.{json,md}` retained for provenance only |
| Preregistered | `docs/hypotheses/HYPOTHESIS-009-sheaf-laplacian.md` |
| Reproduce | Do not treat the historical command as a current scientific reproduction. See the H009 invalidation and rerun requirements in the hypothesis document. |

---

## Claim 12b: H009-R resolves H39-H41 with the repaired scalar sheaf operator

| Field | Value |
|---|---|
| Status | **H39 positive difference; H40 no significant difference detected; H41 inconclusive (falsification condition not met)** |
| Domain | H009-R, corrective replication of H009 |
| Setup | NCI1, 4110 graphs, 30 seeds, 10 epochs, hidden_dim=32, matched-capacity protocol; `sheaf-residual` 2.0.0 (repaired `delta.T @ delta` construction, invariant-tested at merge `b89d196`), 2,403 parameters versus 2,338 for the controls (+2.78%, inside the 5% tolerance) |
| Per-arm medians | sheaf 0.604 [0.580, 0.624]; Hodge 0.605 [0.572, 0.612]; gin-residual 0.630 [0.605, 0.648]; MLP 0.523 [0.513, 0.566] |
| H39 (sheaf vs MLP) | median Delta = +0.0809; p_BH = 4.73 x 10^-3; r = +0.333; sheaf above MLP on 20/30 seeds. **Positive difference under the preregistered 0.05 rule.** |
| H40 (sheaf vs Hodge) | median Delta = -0.0006; p_BH = 0.428; r = +0.133. **No significant difference detected. Not an equivalence claim.** |
| H41 (sheaf vs gin-residual) | median Delta = -0.0262; p_BH = 0.0342; r = -0.467; sheaf below gin-residual on 22/30 seeds. The preregistered falsification condition requires p_BH < 0.01, so **H41 is not falsified and remains inconclusive**: conventionally significant at 0.05 but not crossing the preregistered 0.01 falsification threshold, and not an equivalence result. |
| Replication note | The invalidated historical run's qualitative pattern is reproduced by the valid operator (sheaf median 0.604 in both; sheaf-gin deficit in the 0.01-0.05 band both times). The historical numbers remain excluded from evidence; only H009-R carries inferential status. |
| Interpretation | Under this fixed short-budget protocol: the repaired construction is above the matched MLP (H39); no significant sheaf-Hodge difference is detected at this power (not an equivalence finding); and it is directionally below the matched normalized-adjacency arm without crossing the strict falsification threshold (H41 inconclusive). No learned-sheaf advantage over either fixed operator was detected in this regime. |
| Execution | GitHub Actions `Run Experiment` #8 (run ID 31419047248) at commit `79329df`; SHA-256 of the uploaded workflow artifact archive: `3d595aa27a1c95ab258051ac1a152d4e4fc4a4d8b04012e3dcf605a6f93dc920` (GitHub's recorded digest); the archive's contents are committed unmodified, with per-file digests recorded in the H009-R hypothesis document section 9.1 |
| Artifact | `notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-009R-sheaf-corrective-replication.md` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models sheaf-residual hodge-mp-residual gin-residual mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |

---

## Claim 13: Cross-dataset operator controls show no unique Hodge advantage, but do not establish the preregistered dataset-property mechanism

| Field | Value |
|---|---|
| Status | **H42 confirmed; H43 no significant difference; H44 inconclusive; H45/H46 refuted** |
| Domain | H010 |
| MUTAG | gin-residual 0.789 vs Hodge 0.750; p_BH = 7.44 x 10^-3. Significant gin-residual advantage. |
| PROTEINS | Hodge 0.686 vs gin-residual 0.675; p_BH = 0.292. No significant operator difference detected. |
| NCI1 | Hodge 0.609 vs gin-residual 0.629; p_BH = 1.01 x 10^-2. Significant gin-residual advantage at the H010 alpha=0.05 family threshold. |
| H44 decision | **Inconclusive.** Gap magnitude varies, but H44 predicted association with a dataset-level property and no such association was established. Its stated falsification condition is also not met because magnitudes differ. |
| MLP controls | On MUTAG, Hodge-residual is significantly below MLP (p_BH = 8.61 x 10^-3) and gin-residual has no significant difference from MLP (p_BH = 0.438). On PROTEINS, neither arm has a significant positive difference from MLP. |
| Interpretation | No tested dataset provides evidence of Hodge superiority over the matched normalized-adjacency arm. This does not justify the stronger claim that one spectral filter type is universally superior or that H44's dataset-level mechanism was found. |
| Artifact | `notebooks/results/h010_{mutag,proteins}_operator_30seeds.{json,md}`, with NCI1 reused from H008c |
| Preregistered | `docs/hypotheses/HYPOTHESIS-010-cross-dataset-operator.md` |
| Reproduce | See the H010 hypothesis document |

---

## Claim 14: The NCI1 `L_1` experiment does not establish a higher-order advantage

| Field | Value |
|---|---|
| Status | **H47/H48/H49 refuted; higher-order mechanism inconclusive on NCI1** |
| Domain | H011 |
| Per-arm medians | L_1 0.590; L_0 Hodge 0.609; gin-residual 0.629; MLP 0.523 |
| L_1 vs MLP | median Delta = +0.0669; p_BH = 0.0957. No significant positive difference detected; H47 refuted. |
| L_1 vs L_0 Hodge | median Delta = -0.0195; p_BH = 0.0787. No significant L_1 advantage; H48 refuted under its stated rule. |
| L_1 vs gin-residual | median Delta = -0.0389; p_BH = 0.00676; r = -0.533. Crosses H49's preregistered 0.01 falsification threshold. |
| Structural limitation | 3961/4110 NCI1 graphs (96%) contain no triangles, so the triangle-based `B_2 B_2^T` up-Laplacian term is absent for almost all graphs. The experiment is therefore a poor test of the intended triangle-rich higher-order mechanism. |
| Artifact | `notebooks/results/h011_nci1_l1_30seeds.{json,md}` |
| Preregistered | `docs/hypotheses/HYPOTHESIS-011-l1-edge-propagation.md` |
| Reproduce | See H011 |

---

## Pending: H011b `L_1` on triangle-rich COLLAB

| Field | Value |
|---|---|
| Status | **Pending, no statistical claim licensed** |
| Dataset | COLLAB, 5000 graphs, triangle-rich follow-up target |
| Smoke result | One seed, one epoch: L_1 0.668 vs MLP 0.520. Directional only. |
| Confirmatory design | 30 seeds, 10 epochs. A separate 18-seed compute attempt exceeded the GitHub Actions time limit and is not a completed substitute for the preregistered design. |
| Required final audit | The completed artifact must record the triangle census under the exact dataset loader and preprocessing. |
| Preregistered | `docs/hypotheses/HYPOTHESIS-011b-l1-collab.md` |

---

## Deferred: DRIVE segmentation with `CubicalTopologyLoss`

| Field | Value |
|---|---|
| Status | **Deferred; infrastructure exists, downstream benefit not yet tested at statistical rigor** |
| Domain | Image segmentation |
| Current evidence | `CubicalTopologyLoss` is implemented and gradient-tested. No preregistered powered DRIVE result is currently part of the evidence record. |
| Script | `notebooks/drive_unet_topology_loss.py` |

---

## Quality-floor metrics

| Metric | Current contract |
|---|---|
| Test suite | Required CI matrix; exact verified full-dependency snapshot is recorded in `docs/CLAIMS_TO_EVIDENCE.md` Claim 1 |
| Package coverage | 100% line and 100% branch on `topogeoml/` under the required full-dependency gate; no 100% coverage claim is made for `benchmarks/` |
| Lint | ruff enforced across the declared source/research paths |
| Type checking | mypy strict on `topogeoml/` |
| Every-PR validation | Python 3.11/3.12 Linux/macOS CI, full-dependency package coverage/dependency audit, and the MUTAG/PROTEINS/NCI1 Hodge smoke matrix |
| Path-conditional validation | The diff-PH benchmark runs on PRs that change `topogeoml/nn/diff_ph.py`, `benchmarks/**`, or `.github/workflows/benchmark.yml`; it is not an unconditional check on unrelated PRs |
| Registered benchmark arms | 11, including the repaired `sheaf-residual` 2.0.0 arm validated by H009-R; registry presence is not evidence validity |
| DOI | [10.5281/zenodo.20365816](https://doi.org/10.5281/zenodo.20365816) |

## Adding new evidence

1. Write the preregistered hypothesis and explicit decision rule before running a confirmatory experiment.
2. Run the declared design without changing thresholds after observing results.
3. Save machine-readable and human-readable result artifacts.
4. Report the result using the status vocabulary above. Do not convert non-significance into equivalence and do not relax a preregistered threshold after the fact.
5. Update the statistical summary if the comparison family or investigation-wide multiplicity changes.
6. Merge only after the result artifact, hypothesis interpretation, public summary, and required CI agree.

Negative, null, inconclusive, invalidated, and positive outcomes all remain part of the record.
