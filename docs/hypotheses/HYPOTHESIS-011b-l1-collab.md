---
title: H011b · L₁ on COLLAB
parent: Hypotheses (H001–H011b)
nav_order: 14
---

# Hypothesis 011-b: L_1 edge-level message passing on COLLAB

**Status.** Preregistered 2026-05-25 and unresolved at statistical rigor. A one-seed, one-epoch smoke run completed with L_1 accuracy 0.668 and MLP accuracy 0.520. That result is directional only and licenses no statistical claim. A separate 18-seed compute attempt exceeded the GitHub Actions six-hour limit. The preregistered confirmatory design remains 30 seeds and has not been completed.

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
