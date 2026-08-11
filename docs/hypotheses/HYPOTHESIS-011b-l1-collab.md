---
title: H011b · L₁ on COLLAB
parent: Hypotheses (H001–H011b)
nav_order: 14
---

# Hypothesis 011-b: L_1 edge-level message passing on COLLAB

**Status.** Preregistered 2026-05-25 and unresolved at statistical rigor. A one-seed, one-epoch smoke run completed with L_1 accuracy 0.668 and MLP accuracy 0.520. That result is directional only and licenses no statistical claim. A separate 18-seed compute attempt exceeded the GitHub Actions six-hour limit. The preregistered confirmatory design remains 30 seeds and has not been completed. A pre-execution execution-contract amendment (section 6) was committed 2026-08-11, before any confirmatory result exists; it changes no scientific design parameter and no decision threshold.

**Falsification target.** Whether the tested L_1 edge-level propagation architecture provides a classification advantage on a graph dataset with non-trivial triangle structure, where the up-Laplacian term B_2 B_2^T is present rather than nearly always zero.

**Prior result.** H011 on NCI1 is a poor test of that mechanism because 96% of NCI1 graphs have no triangles. The NCI1 L_1 arm does not significantly outperform MLP and is lower than gin-residual under the preregistered H49 rule. A triangle-rich follow-up is therefore required before making a higher-order claim.

**Why COLLAB.** COLLAB is a TUDataset graph-classification benchmark of scientific collaboration ego networks. It has no intrinsic node attributes in the project loader, so degree is used as a one-dimensional structural input. It was selected as a denser social-network benchmark in which triangle-based 2-simplices are expected to be materially more common than in NCI1.

The final confirmatory artifact must report the triangle census actually observed under the exact dataset loader and preprocessing used for the experiment. This avoids relying on an undocumented structural statistic when interpreting the up-Laplacian mechanism.

---

## 1. Design

Use the same broad matched-capacity comparison as H011, applied to COLLAB:

| Arm | Operator | Representation level | Self path |
|---|---|---|---|
| `l1-hodge-residual` | L_1 edge Laplacian | Edges | External identity residual |
| `hodge-mp-residual` | L_0 node Laplacian | Nodes | External identity residual |
| `gin-residual` | normalized adjacency | Nodes | External identity residual |
| `mlp-baseline` | no message-passing operator | Nodes | N/A |

All arms use degree as the one-dimensional node input supplied by the COLLAB dataset adapter.

## 2. Preregistered sub-hypotheses

The table below preserves the preregistered thresholds.

| ID | Sub-hypothesis | Prediction | Falsified if |
|---|---|---|---|
| **H51** | l1-hodge-residual outperforms mlp-baseline on COLLAB | p_BH < 0.05 | p_BH >= 0.05 |
| **H52** | l1-hodge-residual outperforms hodge-mp-residual on COLLAB | p_BH < 0.05 | p_BH >= 0.05 |
| **H53** | l1-hodge-residual outperforms gin-residual on COLLAB | p_BH < 0.05 | p_BH >= 0.05 |

The scientific motivation is that L_1 explicitly contains edge-space incidence structure and, when triangles are present, the up-Laplacian term B_2 B_2^T. Node-level models can still learn features correlated with triangle structure, so a positive H52/H53 result would be evidence for this tested L_1 architecture relative to the matched controls, not proof that triangle information is inaccessible to every L_0-based model.

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H51 + H52 + H53 confirmed | The tested L_1 architecture has a positive difference from MLP and both matched node-level controls on COLLAB under the preregistered design. This would justify a new, dataset-scoped higher-order result. |
| H51 confirmed; H52/H53 refuted | L_1 separates from the no-message-passing MLP control but does not separate from the matched node-level graph operators. No unique higher-order advantage is established. |
| H51 refuted | The tested L_1 architecture does not produce the preregistered positive difference from MLP on COLLAB. This would not falsify all possible higher-order Hodge models. |

## 4. Confirmatory design

- **Dataset:** COLLAB, 5000 graph instances as provided by TUDataset.
- **Node input:** one-dimensional degree feature from the project dataset adapter.
- **Models:** `l1-hodge-residual`, `hodge-mp-residual`, `gin-residual`, `mlp-baseline`.
- **Seeds:** 30.
- **Epochs:** 10.
- **Optimiser:** Adam(lr=1e-2).
- **Hidden dim:** 32.
- **Statistical procedure:** paired Wilcoxon comparisons with BH-FDR at alpha=0.05.
- **Required structural audit in final artifact:** number/fraction of graphs with triangles and a summary of triangle counts under the exact loaded graphs.

The 18-seed timed-out compute attempt is not a completed substitute for this design and is not included as confirmatory evidence.

## 5. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets collab \
  --models l1-hodge-residual hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h011b_collab_l1_30seeds.json \
  --markdown notebooks/results/h011b_collab_l1_30seeds.md
```

Until that run is completed, H011b remains pending and the one-seed smoke result must not be cited as evidence of a higher-order classification advantage.

---

## 6. Pre-execution amendment (2026-08-11): execution contract and result handling

This section was committed before any confirmatory H011b result exists. Sections 1 through 5 above are the original preregistration text, unchanged.

**This amendment changes no scientific design parameter.** The arms, dataset, node input, seeds (0 through 29), epochs (10), optimiser (Adam, lr = 1e-2), hidden dimension (32), statistical procedure (paired Wilcoxon with Benjamini-Hochberg correction at alpha = 0.05 over the six pairwise comparisons produced by the four-arm run), the H51-H53 decision rules, and the outcome decision tree are exactly as preregistered on 2026-05-25. No threshold is changed here and none may be changed after results are observed.

### 6.1 Why an amendment is required before execution

Three gaps blocked a valid confirmatory run and are resolved by the commit that introduces this section:

1. **The preregistered design was not executable on the sanctioned infrastructure.** The `l1-hodge-residual` implementation rebuilt the max_dim=2 clique complex and L_1 operator inside every forward pass. Under the confirmatory design each COLLAB graph is revisited about 246 times (0.8 x 10 training epochs + 0.2 x 1 evaluation, times 30 seeds), so identical operator construction was repeated ~250x. Measured on the exact loader-produced COLLAB graphs, one full construction pass over all 5,000 graphs costs on the order of an hour of CPU time, and the densest ~1% of graphs (more than 32,000 edges) cost ~25 seconds each; without memoization a single seed of the L_1 arm alone exceeds the GitHub Actions 360-minute job limit, and the 30-seed design requires hundreds of CPU-hours of redundant construction. This is the direct cause of the documented 18-seed timeout.
2. **The required triangle census had no implementation.** Sections "Falsification target" and 4 require the final artifact to report the triangle census under the exact dataset loader, but no benchmark code produced it.
3. **No result-handling rules were preregistered for H011b.** H009-R fixed its result-handling rules in advance (its section 6); H011b had no equivalent, which matters here because the known execution risk (timeouts) makes rules for failed and partial runs load-bearing.

### 6.2 Implementation repair fixed before execution

`l1-hodge-residual` version 1.0.1 memoizes the per-graph propagation operator (the normalized L_1 and the edge index), keyed by the identity of the stored per-graph L_0 tensor. Construction and normalization are deterministic, parameter-free functions of the graph — no learned state and no RNG participates — so the cached operator is the exact object the uncached composition produces, and a cached forward pass is numerically identical to an uncached one. `tests/test_h011b_contract.py` asserts this identity end-to-end (identical logits with a cold cache, a warm cache, and a cleared cache), asserts L_1 correctness on known graphs (filled triangle: L_1 = 3I, which requires the B_2 B_2^T up-term; triangle-free path: down-term only), and asserts the exact four-arm capacity match at COLLAB's input dimensionality (1,219 trainable parameters in every arm at input_dim = 1, num_classes = 3).

Model mathematics, parameter counts, initialization, training behaviour, and the split/statistics pipeline are unchanged. The version bump exists so the confirmatory artifact records which implementation executed.

### 6.3 Triangle census implementation

The census required by section 4 is produced by:

```bash
python -m benchmarks.hodge.triangle_census \
  --dataset collab \
  --output notebooks/results/h011b_collab_l1_30seeds.census.json
```

It reconstructs each graph from the stored per-graph L_0 exactly as the `l1-hodge-residual` model does (a test asserts the two reconstructions agree), counts triangles once each, and writes per-graph counts plus the summary fields section 4 requires (number and fraction of graphs with triangles, and summary statistics of the counts), together with the loader version and git commit. The census must be generated at the same commit and against the same dataset cache as the confirmatory run and committed alongside the confirmatory artifact.

### 6.4 Execution contract

- The confirmatory run uses exactly the section 5 command: all four arms in one invocation, seeds 0 through 29, 10 epochs, writing `notebooks/results/h011b_collab_l1_30seeds.{json,md}`.
- The four arms are one paired comparison family producing one six-comparison BH correction. Arms must not be run separately and combined afterward.
- The run must execute on hardware able to complete the single invocation without a wall-clock cap. GitHub Actions hosted runners are documented as unsuitable: the 360-minute job limit already terminated an 18-seed attempt, and the sanctioned `Run Experiment` workflow forbids shrinking the design to fit. A dedicated machine with at least 32 GB of memory is required (the memoized operator working set for the full dataset is several gigabytes; the exact measured figure is recorded in the pull request introducing this amendment).
- The environment is recorded by the runner artifact itself (dependency versions, platform, Python, git commit). Following the H009-R precedent, the resolved section must publish integrity digests (SHA-256) for the committed artifact files.
- The artifact set for resolution is: the runner JSON, the runner Markdown, and the census JSON of section 6.3.

### 6.5 Result-handling rules fixed in advance

- Every seed from 0 through 29 is retained. No seed may be removed because its result is unfavorable or unusual.
- The first valid completed artifact from the section 5 command under the repaired implementation is the confirmatory H011b result.
- If execution fails for infrastructure reasons (timeout, out-of-memory, hardware interruption) without changing model behaviour, the same fixed protocol may be rerun. Partial output from a failed run is not interpreted and is not evidence.
- If a software defect affecting model mathematics, training, data splits, or statistics is discovered after execution begins, partial results are not interpreted. The defect must be corrected and documented before a complete fresh run.
- Any further material protocol change requires a new amendment committed before observing results under the changed protocol.
- Negative, null, inconclusive, and positive outcomes are all retained and reported using the repository's status vocabulary.
- The one-seed smoke result and the timed-out 18-seed attempt remain non-evidence regardless of outcome.

### 6.6 What this amendment does not do

It reports no experimental result, interprets no accuracy number, and does not alter the H51-H53 rules or their alpha. Engineering measurements used to size the compute contract (operator construction times, operator sizes) involved no model training and no accuracy evaluation.
