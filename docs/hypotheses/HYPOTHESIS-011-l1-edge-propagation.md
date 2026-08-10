---
title: H011 · L₁ edge propagation
parent: Hypotheses (H001–H011b)
nav_order: 13
---

# Hypothesis 011: Does L_1 edge-level message passing capture structural signal that L_0 node-level propagation cannot?

**Status.** Partially resolved 2026-05-25. The NCI1 arm is complete. `l1-hodge-residual` does not have a significant positive difference from MLP (p_BH = 0.0957), has no significant difference from `hodge-mp-residual` (p_BH = 0.0787), and is lower than `gin-residual` (p_BH = 0.00676). H47, H48, and H49 are therefore refuted under their preregistered rules. H50 was not completed as a meaningful higher-order comparison because MUTAG has zero triangles. The triangle-rich follow-up is tracked as [H011b](HYPOTHESIS-011b-l1-collab.md); its directional smoke run is complete, while the full statistical run exceeded the GitHub Actions time limit and remains pending on higher-compute hardware.

**Falsification target.** Whether the 1-Hodge Laplacian L_1, operating on edge features, provides classification-relevant information beyond the tested L_0-based methods under this matched-capacity architecture.

**Why test L_1.** H001-H010 use node-level operators. The Hodge hierarchy becomes genuinely higher-order at k >= 1 because L_k acts on k-simplices and combines lower-incidence and upper-coincidence structure. H011 therefore moves the experiment from node space to edge space rather than treating L_0 as representative of all Hodge methods.

For an oriented 2-dimensional simplicial complex,

    L_1 = B_1^T B_1 + B_2 B_2^T

and the edge space decomposes orthogonally into gradient, harmonic, and curl subspaces under the usual real-coefficient Hodge decomposition. The decomposition itself is specific to L_1. This experiment, however, tests one particular neural use of L_1 as a propagation operator. It is not a generic test of every model that could use higher-order Hodge information.

A further distinction matters for interpretation. The up-Laplacian term B_2 B_2^T couples edges through shared 2-simplices (triangles in a clique complex). Ordinary 5- and 6-cycles do not become 2-simplices merely because they are rings. The original molecular motivation therefore overstated the connection between aromatic rings and shared-triangle adjacency. The post-preregistration triangle census below is the correct structural check.

---

## 1. Architecture

Edge-level message passing on L_1 with external residual:

1. Project node features: proj = proj_in(x)  (n_nodes, d)
2. Initialize edge features from endpoints: e_{ij} = proj[i] + proj[j]  (n_edges, d)
3. Propagate on normalised L_1: e' = act(L_1_tilde @ e @ W + b) + e  (external residual)
4. Pool edges to graph: graph_emb = sum(e')
5. Classify: head(graph_emb)

The clique complex is constructed with max_dim=2 so that L_1 includes both the edge-incidence down term and, when triangles exist, the up-Laplacian term.

For exact harmonic edge modes, the L_1 propagation term has zero response; the external residual preserves the input edge representation. The architecture should therefore be interpreted as one fixed L_1 filtering design, not as direct readout of every component of the Hodge decomposition.

## 2. Capacity matching

| Arm | Params (NCI1, input_dim=37, hidden_dim=32) |
|---|---|
| `l1-hodge-residual` | ~2338 (proj_in 1216 + mp_weight 1056 + head 66) |
| `hodge-mp-residual` | 2338 (L_0, node-level) |
| `gin-residual` | 2338 (adjacency, node-level) |
| `mlp-baseline` | 2338 |

The L_1 arm has the same reported parameter count as the L_0 Hodge arm. The representation level differs: the L_1 arm propagates and pools edge features, whereas the L_0 arms operate on node features.

## 3. Preregistered sub-hypotheses

The table below preserves the preregistered decision rules.

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H47** | l1-hodge-residual strictly outperforms mlp-baseline on NCI1 | p_BH < 0.05 | L_1 propagation may access edge-space structural information unavailable to an MLP on node features | p_BH >= 0.05 |
| **H48** | l1-hodge-residual outperforms hodge-mp-residual (L_0) on NCI1 | Uncertain | L_1 and L_0 operate on different representation spaces; either could dominate | p_BH >= 0.05 (no significant difference) |
| **H49** | l1-hodge-residual outperforms gin-residual on NCI1 | Uncertain; gin-residual is the current best arm (0.629) | L_1 would need to exceed the already-strong adjacency-based result | gin-residual strictly beats l1-hodge at p_BH < 0.01 |
| **H50** | l1-hodge-residual shows larger advantage on MUTAG than NCI1 (relative to MLP) | MUTAG advantage > NCI1 advantage | Original preregistration expected ring structure to favor L_1 | MUTAG advantage <= NCI1 advantage |

## 4. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H47 + H48 confirmed | The tested L_1 edge-propagation architecture provides a positive difference from MLP and L_0 Hodge under the NCI1 protocol. |
| H47 confirmed, H48 refuted | The tested L_1 architecture separates from MLP but not from the L_0 Hodge arm. |
| H47 refuted | The tested L_1 architecture does not produce the preregistered positive difference from MLP at this configuration. This does not falsify higher-order Hodge methods in general. |

## 5. Experimental design

- **Datasets:** NCI1 (10 epochs) and originally MUTAG (20 epochs), matched to prior experiments.
- **Models:** `l1-hodge-residual`, `hodge-mp-residual`, `gin-residual`, `mlp-baseline`.
- **Seeds:** 30, matched.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon, BH-FDR at alpha=0.05, with the stricter H49 falsification threshold preserved as preregistered.

## 6. Structural check after preregistration

The triangle census was performed after preregistration and before interpreting the result:

- MUTAG: 0 triangles in all 188 graphs.
- NCI1: 3961 of 4110 graphs (96%) have 0 triangles; only 149 graphs contain any triangle, with a maximum of 3.

Therefore the up-Laplacian B_2 B_2^T is zero on MUTAG and zero for almost all NCI1 graphs. In those cases L_1 contains only its down-Laplacian edge-incidence term B_1^T B_1. That operator is not the same matrix as L_0 because it acts on edge space, but the intended shared-triangle interaction is absent.

This structural fact makes NCI1 a poor test of the specific higher-order triangle mechanism and makes the preregistered MUTAG rationale invalid for the clique-complex construction. H011b was created to test L_1 on triangle-rich COLLAB instead.

## 7. NCI1 resolved outcome

Artifact: `notebooks/results/h011_nci1_l1_30seeds.{json,md}`. The Markdown artifact records git commit `7076c1097c28d08eb99351abec693d3c7d8086f3` and 30 seeds.

### Per-arm accuracy

| Arm | Median accuracy (95% bootstrap CI) |
|---|---|
| `l1-hodge-residual` | 0.590 [0.525, 0.615] |
| `hodge-mp-residual` | 0.609 [0.581, 0.625] |
| `gin-residual` | 0.629 [0.607, 0.641] |
| `mlp-baseline` | 0.523 [0.513, 0.566] |

### Pairwise results involving L_1

| Comparison | median Delta | p_BH | r | Interpretation |
|---|---|---|---|---|
| L_1 vs L_0 Hodge | -0.0195 | 0.0787 | -0.308 | No significant difference detected |
| L_1 vs gin-residual | -0.0389 | 0.00676 | -0.533 | gin-residual significantly higher; crosses H49's preregistered 0.01 falsification threshold |
| L_1 vs MLP | +0.0669 | 0.0957 | +0.267 | No significant positive difference detected |

### Sub-hypotheses

- **H47:** **REFUTED.** p_BH = 0.0957 for L_1 versus MLP, above the preregistered 0.05 threshold.
- **H48:** **REFUTED UNDER ITS STATED RULE.** L_1 does not significantly outperform L_0 Hodge; p_BH = 0.0787.
- **H49:** **REFUTED.** gin-residual is higher than L_1 by 3.89 pp with p_BH = 0.00676, crossing H49's preregistered 0.01 falsification threshold.
- **H50:** **NOT RESOLVED.** The planned MUTAG comparison does not test the intended triangle-based higher-order mechanism because MUTAG contains no triangles. The triangle-rich follow-up is H011b.

### Interpretation

The NCI1 result does not support the tested L_1 architecture as an improvement over the node-level baselines. More importantly, NCI1 is structurally unsuitable for the intended up-Laplacian question because almost all graphs have no 2-simplices in the clique complex. H011 therefore narrows the question but does not answer whether L_1 is useful on triangle-rich data.

The correct frontier is H011b on COLLAB. Its one-seed smoke result is directional only and is not a statistical claim.

## 8. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models l1-hodge-residual hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h011_nci1_l1_30seeds.json \
  --markdown notebooks/results/h011_nci1_l1_30seeds.md
```

## References

- Barbarossa, S. & Sardellitti, S. (2020). Topological signal processing over simplicial complexes. *IEEE TSP*, 68, 2992-3007.
- Bunch, E., You, Q., Fung, G., & Singh, V. (2020). Simplicial 2-complex convolutional neural networks. *NeurIPS Workshop on TDA and Beyond*.
- Ebli, S., Defferrard, M., & Spreemann, G. (2020). Simplicial neural networks. *NeurIPS Workshop on TDA and Beyond*.
- Schaub, M. T., Benson, A. R., Horn, P., Lippner, G., & Jadbabaie, A. (2020). Random walks on simplicial complexes and the normalized Hodge 1-Laplacian. *SIAM Review*, 62(2), 353-391.
