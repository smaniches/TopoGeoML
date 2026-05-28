---
title: H011 · L₁ edge propagation
parent: Hypotheses (H001–H011b)
nav_order: 13
---

# Hypothesis 011: Does L_1 edge-level message passing capture structural signal that L_0 node-level propagation cannot?

**Status.** Partially resolved 2026-05-25. NCI1 result: L_1 does not significantly outperform MLP (p_BH = 0.096) and underperforms gin-residual (p_BH = 0.007). Expected given that 96% of NCI1 graphs have 0 triangles — L_1's up-Laplacian component is effectively zero. COLLAB (triangle-rich) test running on GitHub Actions. See §8.

**Falsification target.** Whether the 1-Hodge Laplacian L_1 (operating on edge features, encoding shared-triangle adjacency) provides classification-relevant structural information beyond what L_0-based methods capture. This is the first experiment in this series that uses genuinely higher-order topological information.

**Why this is the mathematically motivated next step.** The entire H001-H010 investigation used L_0, the 0-th Hodge Laplacian on nodes. L_0 = D - A is algebraically equivalent to the graph Laplacian, which is the same structural information that GCN, GIN, and GAT access. H008-c proved that the choice of L_0-based operator is secondary to the residual architecture. But the Hodge theory's central contribution is not L_0 — it is the existence of HIGHER-ORDER Laplacians L_k that encode qualitatively different structural information.

**The L_1 Hodge decomposition.** The edge space of a graph decomposes as:

    C_1 = im(∂_1^T) ⊕ ker(L_1) ⊕ im(∂_2)

- **im(∂_1^T):** gradient component — edge flows derivable from a node-level potential (conservative flows)
- **ker(L_1):** harmonic component — edge flows that are cycles not bounding any triangle (topological holes, β_1)
- **im(∂_2):** curl component — edge flows derivable from triangle orientations

This decomposition is unique to the Hodge Laplacian. No L_0-based method can access it. L_1 propagation mixes edge features based on co-boundary adjacency (edges sharing a triangle), which encodes cycle and ring structure directly.

**Relevance to molecular classification.** Aromatic rings, ring systems, and functional group topology are primary determinants of mutagenicity (MUTAG) and anti-cancer activity (NCI1). L_1 message passing detects these structures through shared-triangle adjacency — edges within the same ring or ring system are L_1-adjacent and exchange features during propagation.

---

## 1. Architecture

Edge-level message passing on L_1 with external residual:

1. Project node features: proj = proj_in(x)  (n_nodes, d)
2. Initialize edge features from endpoints: e_{ij} = proj[i] + proj[j]  (n_edges, d)
3. Propagate on normalised L_1: e' = act(L_1_tilde @ e @ W + b) + e  (external residual)
4. Pool edges to graph: graph_emb = sum(e')
5. Classify: head(graph_emb)

The clique complex is constructed with max_dim=2 (nodes, edges, triangles) so that L_1 includes the up-Laplacian component ∂_2 ∂_2^T, which encodes shared-triangle structure.

## 2. Capacity matching

| Arm | Params (NCI1, input_dim=37, hidden_dim=32) |
|---|---|
| `l1-hodge-residual` | ~2338 (proj_in 1216 + mp_weight 1056 + head 66) |
| `hodge-mp-residual` | 2338 (L_0, node-level) |
| `gin-residual` | 2338 (adjacency, node-level) |
| `mlp-baseline` | 2338 |

The L_1 arm has identical parameter count to the L_0 Hodge arm — the only difference is the Laplacian (L_1 vs L_0) and the feature level (edges vs nodes).

## 3. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H47** | l1-hodge-residual strictly outperforms mlp-baseline on NCI1 | p_BH < 0.05 | L_1 propagation accesses structural information (ring/cycle topology) that MLP cannot read from node features | p_BH >= 0.05 |
| **H48** | l1-hodge-residual outperforms hodge-mp-residual (L_0) on NCI1 | Uncertain — L_1 captures different structure but edge-level pooling may lose node-level discrimination | The L_1 and L_0 propagations access orthogonal structural information; either could dominate | p_BH >= 0.05 (no significant difference) |
| **H49** | l1-hodge-residual outperforms gin-residual on NCI1 | Uncertain — gin-residual is the current best arm (0.629) | L_1 would need to exceed the already-strong adjacency-based result | gin-residual strictly beats l1-hodge at p_BH < 0.01 |
| **H50** | l1-hodge-residual shows larger advantage on MUTAG than NCI1 (relative to MLP) | MUTAG advantage > NCI1 advantage | MUTAG mutagenicity is determined by aromatic ring topology, which L_1 directly encodes via shared-triangle adjacency | MUTAG advantage <= NCI1 advantage |

## 4. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H47 + H48 confirmed (L_1 beats both MLP and L_0) | **Higher-order Hodge structure provides unique classification signal.** L_1 captures ring/cycle topology that L_0-based methods miss. This is the vindication of the Hodge theory applied to GNN classification. |
| H47 confirmed, H48 refuted (L_1 beats MLP but not L_0) | L_1 captures structural signal but L_0 already captures it equivalently or better. The higher-order decomposition adds computational cost without classification benefit. |
| H47 refuted (L_1 does not beat MLP) | Edge-level message passing with sum-of-endpoint initialisation fails to learn at this capacity/epoch budget. Possible causes: edge feature initialisation is too information-lossy, or the edge-to-graph pooling (sum over edges) discards per-node discrimination that sum-over-nodes preserves. |

## 5. Experimental design

- **Datasets:** NCI1 (10 epochs) and MUTAG (20 epochs), matched to prior experiments.
- **Models:** `l1-hodge-residual`, `hodge-mp-residual`, `gin-residual`, `mlp-baseline`.
- **Seeds:** 30, matched.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon, BH-FDR at alpha=0.05.

## 6. Implementation notes

L_1 is computed inside `forward_one` from the L_0 Laplacian: the edge set is extracted from L_0's off-diagonal entries, the clique complex is constructed with max_dim=2, and `hodge_laplacian(sc, k=1)` returns L_1. This avoids interface changes to the GraphSample dataclass or the training loop. The per-graph overhead is negligible at the tested graph sizes (18-30 nodes).

**Critical structural observation (discovered after preregistration, before results).** Triangle counts in the tested datasets:
- MUTAG: **0 triangles in all 188 graphs.** Molecular graphs are sparse; aromatic rings are 5-6 cycles, not 3-cliques.
- NCI1: **96% of graphs (3961/4110) have 0 triangles.** Only 149 graphs have any triangle; maximum is 3.

This means L_1's up-Laplacian component ∂_2 ∂_2^T (shared-triangle adjacency) is effectively zero for nearly all graphs. L_1 degenerates to the down-Laplacian ∂_1^T ∂_1 (edges sharing a vertex), which encodes the same neighbourhood structure as L_0. A negative H011 result should be interpreted as "these datasets lack the higher-order simplicial structure that L_1 is designed to exploit" rather than "L_1 message passing is uninformative in general."

Testing L_1 on datasets with rich triangle structure (social networks, collaboration graphs, protein contact maps) is the appropriate follow-up if H011 produces a null result on these molecular benchmarks.

## 7. Reproduction

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
