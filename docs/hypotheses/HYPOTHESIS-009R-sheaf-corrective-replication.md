---
title: H009-R · Corrective sheaf replication
parent: Hypotheses (H001-H011b)
nav_order: 11.5
---

# H009-R: Corrective replication of the NCI1 scalar-sheaf experiment

**Status: RESOLVED 2026-08-10.** The confirmatory run completed as `Run Experiment` workflow run #8 (run ID 31419047248) on `main` at commit `79329df1b49b15867daf6a7959acb4911d994d6a`, producing `notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}`. Under the preregistered decision rules in section 4: **H39 is supported** (sheaf above MLP, p_BH = 4.74 x 10^-3), **H40 is not supported** (sheaf versus Hodge p_BH = 0.428), and **H41's falsification condition is not met** (sheaf below gin-residual at p_BH = 0.0342, which does not cross the preregistered 0.01 threshold; this is not a statistical-equivalence claim). See section 9 for the full resolved outcome.

Sections 1 through 8 below are the preregistration text, unchanged. Section 9 was appended after the artifact existed.

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

## 9. Resolved outcome (appended 2026-08-10, after the artifact existed)

### 9.1 Execution and protocol compliance

The confirmatory run executed on GitHub Actions (`Run Experiment` #8, run ID 31419047248) from `main` at commit `79329df1b49b15867daf6a7959acb4911d994d6a`, which contains the repaired implementation merged at `b89d196`. The uploaded artifact (`h009r_nci1_sheaf_v2_30seeds-results`, SHA-256 `3d595aa27a1c95ab258051ac1a152d4e4fc4a4d8b04012e3dcf605a6f93dc920`) is committed unmodified at `notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}`.

Protocol fields recorded in the artifact match section 3 exactly: dataset `nci1` (loader version 1.0.0, 4,110 graphs, per-seed stratified 80/20 split with n_train = 3,288 and n_test = 822), models `sheaf-residual` 2.0.0 / `hodge-mp-residual` / `gin-residual` / `mlp-baseline`, seeds 0 through 29 all present in every arm, 10 epochs, learning rate 0.01, paired Wilcoxon with Benjamini-Hochberg correction over the six-comparison family. An earlier dispatch attempt (run #7) failed for infrastructure reasons before producing any result artifact; under the section 6 rules the same fixed protocol was rerun without change, and run #8 produced the first valid completed artifact, which is therefore the confirmatory H009-R result.

Environment note: this run used `torch` 2.13.0+cpu and `torch_geometric` 2.8.0.post1 (the invalidated historical run used `torch` 2.12.0+cu130 and `torch_geometric` 2.7.0). Each (model, seed) cell is seeded independently and all four arms within a seed share one split and one environment, so every paired comparison below is within-run and unaffected by cross-run environment drift.

### 9.2 Per-arm results

| Arm | Median accuracy | BCa 95% CI | n_seeds |
|---|---:|---|---:|
| `sheaf-residual` 2.0.0 | 0.604 | [0.580, 0.624] | 30 |
| `hodge-mp-residual` | 0.605 | [0.572, 0.612] | 30 |
| `gin-residual` | 0.630 | [0.605, 0.648] | 30 |
| `mlp-baseline` | 0.523 | [0.513, 0.566] | 30 |

### 9.3 Preregistered decisions

The six-comparison family with BH correction, as fixed in section 3. `median delta` is the difference of arm medians as reported by the benchmark runner; `r` is the runner's matched-pairs effect statistic `(n_pos - n_neg) / n_nonzero`; wins/losses count per-seed paired differences.

| Comparison | median delta | p_raw | p_BH | r | seeds A>B / A<B / tie |
|---|---:|---:|---:|---:|---|
| sheaf vs hodge | -0.0006 | 0.428 | 0.428 | +0.133 | 17 / 13 / 0 |
| sheaf vs gin | -0.0262 | 0.0285 | 0.0342 | -0.467 | 8 / 22 / 0 |
| sheaf vs mlp | +0.0809 | 0.00237 | 0.00474 | +0.333 | 20 / 10 / 0 |
| hodge vs gin | -0.0255 | 0.000181 | 0.000543 | -0.733 | 4 / 26 / 0 |
| hodge vs mlp | +0.0815 | 0.0174 | 0.0260 | +0.379 | 20 / 9 / 1 |
| gin vs mlp | +0.1071 | 0.0000886 | 0.000531 | +0.600 | 24 / 6 / 0 |

Applying the section 4 rules:

- **H39 — supported.** `sheaf-residual` is above `mlp-baseline` (median 0.604 versus 0.523) with p_BH = 4.74 x 10^-3 < 0.05. The positive-difference condition is met.
- **H40 — not supported.** `sheaf-residual` versus `hodge-mp-residual` has p_BH = 0.428. The arm medians differ by less than one test graph (0.604 versus 0.605). No significant difference is detected; per section 7 this is not an equivalence claim.
- **H41 — falsification condition not met.** `sheaf-residual` is below `gin-residual` (median delta -0.0262) with p_BH = 0.0342. The preregistered falsification condition requires p_BH < 0.01, so H41 is not falsified. Because 0.0342 sits between 0.01 and 0.05, the comparison is conventionally significant at 0.05 while remaining inconclusive under the stricter preregistered rule. It is reported as inconclusive with an observed deficit direction (sheaf below gin on 22 of 30 seeds), not as equivalence and not as a refutation.

### 9.4 Post-hoc descriptive analysis (not preregistered; no decision weight)

These statistics are recomputed from the committed per-seed artifact for interpretation only.

- Paired per-seed differences: sheaf minus mlp has mean +0.0457 and median +0.0450 (paired Cohen's d_z = +0.65); sheaf minus hodge has mean +0.0090 and median +0.0036 (d_z = +0.20); sheaf minus gin has mean -0.0199 and median -0.0176 (d_z = -0.36).
- Seed-level variability is large under the fixed 10-epoch budget: per-arm standard deviations are 0.047 to 0.056. `sheaf-residual` finishes at or below 0.52 accuracy (near the 0.5005 class prior) on 3 of 30 seeds, `hodge-mp-residual` on 4, `gin-residual` on 1, and `mlp-baseline` on 14. The MLP distribution is strongly bimodal (either near chance or 0.56+), which is why its median (0.523) sits well below its mean (0.551).
- Per-seed accuracies correlate moderately between the message-passing arms (Pearson r = 0.56 sheaf-hodge, 0.64 hodge-gin) and essentially not at all with the MLP, consistent with shared split difficulty among the graph arms.
- Final training losses after 10 epochs are 0.77 to 0.78 for all three message-passing arms and 0.67 for the MLP, so no arm is near convergence; the protocol measures a fixed short-budget regime, as section 7 states.

### 9.5 Relationship to the invalidated H009 artifact

The invalidated historical numbers are not evidence and are not combined with H009-R. It is nevertheless part of the audit trail that the repaired operator reproduces the historical pattern rather than overturning it: sheaf median 0.604 (both runs), no significant sheaf-Hodge difference, and a sheaf-gin deficit in the 0.01-to-0.05 band (historical p_BH = 0.0137; H009-R p_BH = 0.0342). The implementation defect, once corrected, did not change any qualitative conclusion.

### 9.6 Scope

Section 7's interpretation boundaries apply unchanged. H009-R establishes the behavior of this scalar, feature-conditioned sheaf construction on NCI1 under the fixed matched-capacity, one-layer, 10-epoch protocol, and nothing beyond it.
