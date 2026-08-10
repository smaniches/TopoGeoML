---
title: H009-R · Corrective sheaf replication
parent: Hypotheses (H001-H011b)
nav_order: 11.5
---

# H009-R: Corrective replication of the NCI1 scalar-sheaf experiment

**Status: PREREGISTERED. NOT RUN.**

This document fixes the design of the corrective H009 replication before any result from the repaired `sheaf-residual` 2.0.0 implementation is executed or observed.

The original H009 run is permanently invalidated as evidence about a cellular-sheaf Laplacian because its implementation did not guarantee the stated `delta.T @ delta` operator and its parameter accounting was incorrect. The historical artifact remains public for provenance. H009-R does not reinterpret or reuse those numerical results.

The repaired implementation was merged before this preregistration at commit:

`b89d196fdaec87a545e048df3106807b1b5fd45b`

The repaired model is `sheaf-residual` version `2.0.0`.

## 1. Purpose

H009-R is a corrective replication intended to resolve the original H39-H41 questions using the mathematical operator that H009 meant to test. It is not a new hypothesis family and does not add new sub-prediction IDs; the project still contains 14 original hypothesis documents and 53 original falsifiable sub-predictions, with H009-R serving as a preregistered corrective protocol for H39-H41.

It is not an independent replication because the invalidated historical H009 numbers are already known. Its value is narrower: the decision rules are fixed before observing any result from the repaired implementation, the repaired operator is tested against explicit mathematical invariants, and the new result will be kept separate from the invalid historical artifact.

No model selection, hyperparameter search, seed selection, or threshold change will be performed after seeing the H009-R result.

## 2. Repaired operator fixed before execution

For each undirected edge `e = {i, j}`, the repaired model processes the edge exactly once. A shared scalar restriction learner is evaluated in both endpoint orders to obtain one restriction scalar for each endpoint. With an arbitrary orientation, one coboundary row is assembled and the learned operator is

`L_F = delta.T @ delta`.

The implementation therefore enforces the intended scalar cellular-sheaf construction rather than symmetrizing an arbitrary learned matrix after the fact.

Before this preregistration, the repair PR established the following invariants in automated tests:

1. one undirected edge creates one coboundary row;
2. identity restrictions recover the ordinary combinatorial graph Laplacian exactly to numerical tolerance;
3. isolated vertices retain zero rows under the degree-support normalization convention;
4. randomized learned operators are symmetric positive semidefinite to numerical tolerance;
5. the learned operator is consistent under node relabeling;
6. gradients reach both the restriction learner and downstream message-passing parameters through a real loss;
7. exact parameter accounting is inside the original matched-capacity tolerance.

The repaired NCI1 parameter counts are fixed before the run:

| Arm | Trainable parameters |
|---|---:|
| `sheaf-residual` 2.0.0 | 2,403 |
| `hodge-mp-residual` | 2,338 |
| `gin-residual` | 2,338 |
| `mlp-baseline` | 2,338 |

The repaired sheaf arm is 65 parameters, approximately 2.78%, larger than the controls and remains inside the original 5% matched-capacity tolerance.

## 3. Dataset and training protocol

The corrective run preserves the historical H009 experimental family except for the repaired sheaf implementation and the new artifact path.

| Setting | Fixed value |
|---|---|
| Dataset | NCI1 via the existing TopoGeoML Hodge benchmark loader |
| Dataset version recorded by runner | current loader-reported version in the output artifact |
| Models | `sheaf-residual`, `hodge-mp-residual`, `gin-residual`, `mlp-baseline` |
| Sheaf model version | `2.0.0` |
| Hidden dimension | 32 |
| Seeds | integers 0 through 29, all included |
| Epochs | 10 |
| Learning rate | 0.01 |
| Optimizer | existing benchmark Adam training path |
| Split | deterministic stratified 80/20 split generated separately for each seed; all arms share the same split within a seed |
| Primary metric | held-out graph classification accuracy |
| Confidence interval | existing BCa bootstrap 95% interval reported by the benchmark runner |
| Pairwise test | paired Wilcoxon signed-rank test matched by seed |
| Multiplicity | Benjamini-Hochberg correction over all six pairwise comparisons produced by this four-arm run |

No early stopping, post-result epoch extension, alternative seed set, hidden-dimension change, learning-rate change, or model-family change is permitted for the confirmatory run described here.

## 4. Original H39-H41 decision rules

H009-R is designed to resolve the original H39-H41 questions rather than invent new favorable thresholds after implementation repair.

| ID | Original question | Preregistered decision rule retained for H009-R |
|---|---|---|
| **H39** | Does `sheaf-residual` strictly beat `mlp-baseline` on NCI1? | Positive-difference support requires sheaf accuracy above MLP with `p_BH < 0.05`. If `p_BH >= 0.05`, H39 is not supported. A significant result in the opposite direction is reported as a regression, not as support. |
| **H40** | Does `sheaf-residual` strictly beat `hodge-mp-residual` on NCI1? | Support requires sheaf accuracy above Hodge with `p_BH < 0.05`. Otherwise H40 is not supported. |
| **H41** | Does `sheaf-residual` avoid strict underperformance relative to `gin-residual`? | The original falsification condition is sheaf strictly below gin-residual with `p_BH < 0.01`. Failure to cross that condition is not described as statistical equivalence because no equivalence or non-inferiority margin was preregistered. |

The three scientific decisions above use the BH-adjusted p-values from the fixed six-comparison family. The other three pairwise comparisons are retained in the artifact because they are part of that multiplicity family, but they do not create new H009-R hypotheses.

## 5. Execution command

The confirmatory run must use this model family, seed set, and training budget:

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models sheaf-residual hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h009r_nci1_sheaf_v2_30seeds.json \
  --markdown notebooks/results/h009r_nci1_sheaf_v2_30seeds.md
```

The result files above do not exist at preregistration time and must not be populated until this preregistration is committed and merged.

## 6. Result-handling rules fixed in advance

- Every seed from 0 through 29 is retained. No seed may be removed because its result is unfavorable or unusual.
- The complete four-arm run is treated as one paired comparison family. Arms are not rerun separately and combined afterward.
- The first valid completed artifact from the repaired implementation is the confirmatory H009-R result.
- If execution fails because of infrastructure without changing model behavior, the same fixed protocol may be rerun.
- If a software defect affecting model mathematics, training, data splits, or statistics is discovered after execution begins, partial results are not interpreted. The defect must be corrected and documented before a complete fresh run.
- Any material protocol change requires a new preregistration amendment committed before observing results under the changed protocol.
- Negative, null, inconclusive, and positive outcomes are all retained and reported.
- The invalidated H009 artifact remains unchanged and separate from H009-R.

## 7. Interpretation boundaries

Even a favorable H009-R result would establish only the behavior of this scalar, feature-conditioned sheaf construction on NCI1 under the fixed matched-capacity, one-layer, short-budget protocol.

It would not establish that learned sheaf methods are generally superior to Hodge, GIN, graph neural networks, or other topology-aware architectures. It would not validate the invalid historical H009 run. It would not constitute independent external replication.

A null result would not establish statistical equivalence. A result failing H41's strict-underperformance screen would only mean that the original H41 falsification condition was not met.

## 8. Evidence update after completion

Only after the complete H009-R artifact exists may the repository:

1. resolve H39-H41 in the current evidence index;
2. add H009-R to `STATUS.md` and `docs/CLAIMS_TO_EVIDENCE.md`;
3. update `docs/STATISTICAL_SUMMARY.md`;
4. decide whether a new retrospective investigation-wide multiplicity sensitivity analysis is useful and, if so, regenerate it from the validated comparison set with the exact inclusion and de-duplication rules documented.

The historical H009 artifact and invalidation notice remain in the audit trail regardless of the H009-R outcome.
