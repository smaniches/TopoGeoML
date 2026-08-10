---
title: Claims to evidence
nav_order: 5
---

# Claims to Evidence

This document maps the current public claims in `README.md` and `STATUS.md` to code, CI, preregistration records, and result artifacts. It distinguishes three kinds of statements:

1. **Software capability:** the implementation exists and is exercised by tests.
2. **Software correctness invariant:** CI enforces a stated property such as package coverage.
3. **Empirical claim:** a seeded or deterministic study supports a stated scientific conclusion within its declared scope.

Implementation and test coverage do not by themselves establish downstream task benefit. Conversely, a negative or invalidated result for one research architecture does not invalidate unrelated library components.

## Methodology

- Empirical values are taken from committed result artifacts in `notebooks/results/` or from the CI command that produces the invariant.
- Pairwise graph experiments use their preregistered decision rules and within-family Benjamini-Hochberg correction.
- The former investigation-wide 59-comparison and 76-entry sensitivity analysis is withdrawn as a current claim because it included invalidated H009 comparisons. With H009-R complete the validated comparison set exists, but the retrospective table has not been regenerated and remains withdrawn until it is, with documented inclusion rules. See [STATISTICAL_SUMMARY.md](STATISTICAL_SUMMARY.md).
- A non-significant comparison is reported as no significant difference detected, not as proof of equivalence.
- Where a preregistered threshold differs from a conventional alpha=0.05 interpretation, the preregistered threshold controls the hypothesis verdict.
- Invalidated experiments remain in the public record for provenance but are excluded from current scientific conclusions until corrected and rerun.

---

## Claim 1: The importable `topogeoml` package is gated at 100% line and 100% branch coverage under full dependencies

| Field | Evidence |
|---|---|
| CI command | `pytest -m "not gpu" --cov=topogeoml --cov-branch --cov-fail-under=100` after `pip install -e ".[all]"` |
| Workflow | `.github/workflows/ci.yml`, full-dependency coverage gate |
| Verified snapshot | At merge commit `9ddfa0d8156637bfd1d42ec9609f299ce337bf00`: **507 passed, 35 skipped; 1,297 statements, 0 missed; 402 branches, 0 partial; 100.00% package coverage** |
| Maintained invariant | The exact test-item count can change as regression tests are added. The maintained invariant is 100% line and 100% branch coverage on the importable `topogeoml` package in the required full-dependency gate. |
| Scope | `benchmarks/` is research infrastructure outside the package-coverage claim. The default dev-only test matrix can skip torch-dependent paths; the full-dependency gate is authoritative for the package coverage statement. |

---

## Claim 2: Hodge-residual has a narrow positive difference from MLP on NCI1

| Field | Evidence |
|---|---|
| Artifact | `notebooks/results/nci1_hodge_ablation_30seeds.{json,md}` |
| Comparison | `hodge-mp-residual` vs `mlp-baseline` |
| Result | median Delta = +0.086; within-experiment p_BH = 4.83 x 10^-3; Hodge median 0.609, MLP median 0.523 |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models hodge-mp-classifier hodge-mp-normalised hodge-mp-residual hodge-mp-deep-residual mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |
| Investigation-wide sensitivity | The previous 59-comparison retrospective BH statement is withdrawn because that comparison family included invalidated H009 comparisons. |
| Scope | NCI1, 30 seeds, 10 epochs, hidden_dim=32, matched-capacity protocol. Later operator controls show that the positive difference is not unique to the Hodge `L_0` operator. |

---

## Claim 3: No unique `L_0` Hodge advantage is supported by the matched operator controls

| Field | Evidence |
|---|---|
| H008c NCI1 | `gin-residual` 0.629 vs Hodge 0.609; median Delta = +0.0195; p_BH = 1.01 x 10^-2 |
| H010 MUTAG | `gin-residual` 0.789 vs Hodge 0.750; p_BH = 7.44 x 10^-3 |
| H010 PROTEINS | Hodge 0.686 vs `gin-residual` 0.675; p_BH = 0.292, so no significant operator difference is detected |
| Artifacts | `notebooks/results/h008c_nci1_gin_residual_30seeds.{json,md}` and `notebooks/results/h010_{mutag,proteins}_operator_30seeds.{json,md}` |
| Preregistrations | `docs/hypotheses/HYPOTHESIS-008c-gin-residual.md`, `docs/hypotheses/HYPOTHESIS-010-cross-dataset-operator.md` |
| Supported conclusion | No tested dataset provides evidence that the matched Hodge `L_0` arm is superior to the normalized-adjacency arm. MUTAG and NCI1 favor the adjacency arm at the H010 alpha=0.05 family threshold; PROTEINS detects no significant difference. |
| Not supported | A universal claim that adjacency is always superior, or that the H010 H44 dataset-level mechanism was identified. H44 remains inconclusive. |

---

## Claim 4: The tested external-residual adjacency formulation recovers NCI1 performance after normalization with an internal self path does not

| Field | Evidence |
|---|---|
| H008b | `gin-normalised` median 0.500 on NCI1 |
| H008c | `gin-residual` median 0.629; `gin-residual` vs `gin-normalised` p_BH = 5.20 x 10^-6 |
| Implementation | `benchmarks/hodge/models.py` shows `gin-normalised` uses `MLP((1+eps)h + A_norm h)`, while `gin-residual` uses `act(A_norm h W + b) + h` |
| Supported conclusion | The external-residual formulation is a successful tested architectural change and removes any need to invoke a unique Hodge operator effect in this NCI1 comparison. |
| Causal limitation | The two GIN formulations differ in placement and parameterization of the self path. The experiment does not prove that residual connections, abstracted from their exact computation form, are the sole causal mechanism in arbitrary models or datasets. |

---

## Claim 5: Graph-structural classification signal is detected on the three tested datasets under constant features

| Field | Evidence |
|---|---|
| Artifact | `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.{json,md}` |
| MUTAG | gap from class prior +0.098; p_BH = 4.53 x 10^-6 |
| PROTEINS | +0.088; p_BH = 1.41 x 10^-4 |
| NCI1 | +0.071; p_BH = 1.93 x 10^-5 |
| Reproduce | See `REPRODUCING.md` section H006, including the `benchmarks.hodge.h006_analysis` resolver step that performs the class-prior tests and family correction |
| Scope | These are Hodge-architecture comparisons against theoretical class-prior controls under constant node features. They demonstrate exploitable graph structure in the three tested datasets, not a unique Hodge mechanism or a universal claim about graph datasets. |

---

## Claim 6: H009 is invalidated as evidence for a cellular-sheaf Laplacian

| Field | Evidence |
|---|---|
| Historical artifact | `notebooks/results/h009_nci1_sheaf_30seeds.{json,md}` retained for provenance only |
| Implementation defect | The historical `sheaf-residual` model processed both orientations of each undirected edge independently. The resulting learned matrix was not guaranteed symmetric or positive semidefinite and was not, in general, the claimed `delta.T @ delta` scalar cellular-sheaf Laplacian. |
| Capacity defect | On NCI1, `Linear(64, 2)` contributes 130 learner parameters, producing 2,468 total trainable parameters versus 2,338 for the fixed-operator controls. The difference is about 5.56%, slightly outside the stated 5% tolerance. |
| Current status | The historical H009 pairwise statistics have no inferential status for a sheaf-Laplacian claim. H39, H40, and H41 are now resolved by the corrective replication H009-R (Claim 6b), not by this artifact. |
| Repair completed | One undirected edge representation, symmetric PSD construction, identity reduction to `L_0`, gradient tests, and exact parameter accounting were established at merge `b89d196`; the fresh 30-seed NCI1 corrective replication was preregistered as H009-R and executed from commit `79329df`. |
| Source document | `docs/hypotheses/HYPOTHESIS-009-sheaf-laplacian.md` contains the full invalidation and rerun requirements. |

---

## Claim 6b: The H009-R corrective replication resolves H39-H41 for the repaired scalar sheaf operator

| Field | Evidence |
|---|---|
| Artifact | `notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}`, committed unmodified from GitHub Actions `Run Experiment` #8 (run ID 31419047248), artifact SHA-256 `3d595aa27a1c95ab258051ac1a152d4e4fc4a4d8b04012e3dcf605a6f93dc920` |
| Preregistration | `docs/hypotheses/HYPOTHESIS-009R-sheaf-corrective-replication.md`, committed and merged before execution; decision rules for H39-H41 fixed in advance |
| Model | `sheaf-residual` 2.0.0: one coboundary row per undirected edge, `L_F = delta.T @ delta` symmetric PSD by construction, invariant-tested; 2,403 parameters versus 2,338 for the controls (+2.78%, inside the 5% tolerance) |
| H39 (sheaf vs MLP) | Supported. Median 0.604 versus 0.523; median Delta = +0.0809; p_BH = 4.74 x 10^-3 < 0.05; sheaf above MLP on 20/30 seeds |
| H40 (sheaf vs Hodge) | Not supported. Median 0.604 versus 0.605; p_BH = 0.428. No significant difference detected; not an equivalence claim |
| H41 (sheaf vs gin-residual) | Falsification condition not met. Median 0.604 versus 0.630; median Delta = -0.0262; p_BH = 0.0342, which does not cross the preregistered 0.01 threshold. Conventionally significant at 0.05, so the comparison is inconclusive under the stricter preregistered rule; the observed direction (sheaf below gin on 22/30 seeds) is reported without an equivalence or non-inferiority interpretation |
| Supported conclusion | Under the fixed matched-capacity, one-layer, 10-epoch NCI1 protocol, the repaired learned scalar sheaf construction performs like the fixed Hodge operator: above the matched MLP, statistically indistinguishable from Hodge at this power, and directionally below the matched normalized-adjacency arm. |
| Not supported | Any learned-sheaf advantage over fixed operators; sheaf-Hodge equivalence; generalization beyond this dataset, capacity, depth, and training budget; validation of the invalidated historical H009 numbers. |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models sheaf-residual hodge-mp-residual gin-residual mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` (bit-identical reproduction additionally requires the recorded dependency versions; the artifact records `torch` 2.13.0+cpu, `torch_geometric` 2.8.0.post1) |

---

## Claim 7: The NCI1 `L_1` experiment does not establish a higher-order advantage

| Field | Evidence |
|---|---|
| Artifact | `notebooks/results/h011_nci1_l1_30seeds.{json,md}` |
| L_1 vs MLP | median Delta = +0.0669; p_BH = 0.0957. H47 refuted. |
| L_1 vs L_0 Hodge | median Delta = -0.0195; p_BH = 0.0787. H48 refuted under its stated rule. |
| L_1 vs gin-residual | median Delta = -0.0389; p_BH = 0.00676. H49 refuted under its preregistered 0.01 rule. |
| Structural audit | 3961/4110 NCI1 graphs contain no triangles, so the `B_2 B_2^T` up-Laplacian term is absent for 96% of graphs. |
| Supported conclusion | The tested NCI1 `L_1` arm does not outperform the node-level controls under the preregistered rules. NCI1 is also a poor test of the intended triangle-rich higher-order mechanism. |
| Frontier | H011b on COLLAB is the correct triangle-rich follow-up. Its one-seed smoke result is directional only and the preregistered 30-seed statistical run remains incomplete. |

---

## Claim 8: The public library surface is implemented and tested, but downstream benefit is component-specific and often unproven

| Capability | Implementation | Test evidence | Claim boundary |
|---|---|---|---|
| Persistent-homology feature pipeline | `topogeoml/pipelines/feature_pipeline.py` | `tests/test_feature_pipeline.py`, full package gate | Implemented scikit-learn transformer; task-specific predictive benefit depends on data and model |
| Differentiable Vietoris-Rips primitives and `TopologyRegularizer` | `topogeoml/nn/diff_ph.py` | `tests/test_diff_ph.py`, full package gate, path-conditional diff-PH benchmark workflow | Gradient path is tested in exercised regimes; not a claim of globally smooth persistence or universal training improvement |
| Differentiable cubical persistence and `CubicalTopologyLoss` | `topogeoml/nn/cubical_diff_ph.py` | `tests/test_cubical_diff_ph.py`, full package gate | Implemented and gradient-tested; no powered end-to-end segmentation improvement claim yet |
| Simplicial/Hodge algebra and fixed-complex neural primitive | `topogeoml/core/complexes.py`, `topogeoml/nn/hodge.py` | package tests and full coverage gate | Building blocks, not a full variable-topology simplicial training framework |
| Signal topology | `topogeoml/signal/` | `tests/test_signal.py`, full package gate | Feature extraction implemented; downstream predictive value is domain-dependent |
| Embedding audit | `topogeoml/audits/embedding_audit.py` | `tests/test_embedding_audit.py`, full package gate | Prototype diagnostic with heuristic persistence threshold, not an exact topological certification tool |
| Experiment configuration/provenance | `topogeoml/experiments/configs.py` and related package modules | package tests and full coverage gate | Reproducibility utility, not cryptographic provenance or transactional storage guarantee |

---

## Claim 9: The preregistered series contains 14 hypothesis documents and 53 falsifiable sub-predictions

| Field | Evidence |
|---|---|
| Hypothesis documents | H001, H002, H003, H004, H005, H006, H007, H008, H008b, H008c, H009, H010, H011, H011b |
| Count | 14 documents |
| Sub-predictions | H1-H3 (3) + H4-H7 (4) + H8-H12 (5) + H13-H17 (5) + H18-H21 (4) + H22-H25 (4) + H26-H27 (2) + H28-H32 (5) + H33-H35 (3) + H36-H38 (3) + H39-H41 (3) + H42-H46 (5) + H47-H50 (4) + H51-H53 (3) = 53 |
| Verification | Git history for each hypothesis file supplies the preregistration timestamp; experiment artifacts were added later |
| Sequential-design limitation | Hypothesis selection was sequential and informed by earlier results. The former investigation-wide 59/76 sensitivity analysis is withdrawn until a validated comparison set is rebuilt after H009 repair. |

---

## Current unresolved evidence

- Investigation-wide multiplicity sensitivity table: the withdrawn 59/76 retrospective analysis has not been regenerated from the validated comparison set now that H009-R is complete; no current conclusion depends on it.
- H011b COLLAB `L_1`: one-seed, one-epoch directional smoke result only; no statistical claim licensed.
- `CubicalTopologyLoss` downstream segmentation benefit: implementation exists, but the powered end-to-end study is not complete.
- Topology-divergence callback: exploratory, floor-limited, no non-overfitting negative control.
- Independent external reproduction: the validated empirical record has not yet been reproduced by an independent team.
