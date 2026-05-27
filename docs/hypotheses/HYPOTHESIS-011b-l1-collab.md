# Hypothesis 011-b: L_1 edge-level message passing on COLLAB (triangle-rich graphs)

**Status.** Preregistered 2026-05-25. Smoke test (1 seed, 1 epoch) completed on container: L_1 0.668 vs MLP 0.520 (directional only, not a claim). Full 18-seed run timed out on GitHub Actions (6h limit exceeded). Awaiting local execution on higher-compute hardware.

**Falsification target.** Whether L_1 edge-level message passing provides a classification advantage on a dataset with rich triangle structure, where the up-Laplacian component ∂_2 ∂_2^T is non-trivial.

**Prior result.** H011 on NCI1 was uninformative: 96% of NCI1 graphs have 0 triangles, so L_1 degenerates to the down-Laplacian. A proper test requires a dataset where L_1's shared-triangle adjacency has signal to propagate.

**Why COLLAB.** 5000 scientific-collaboration ego-network graphs, 3 classes (High Energy Physics, Condensed Matter, Astrophysics). Mean 9,290 triangles per graph. 100% of graphs have triangles. No node features (degree used as 1-dim input). Classification depends entirely on graph structure — exactly the setting where higher-order topology should matter if it matters anywhere.

---

## 1. Design

Same architecture as H011, applied to COLLAB:

| Arm | Operator | Level | Residual |
|---|---|---|---|
| `l1-hodge-residual` | L_1 (edge Laplacian with up-component) | Edges | External |
| `hodge-mp-residual` | L_0 (node Laplacian) | Nodes | External |
| `gin-residual` | I - L_tilde (normalised adjacency) | Nodes | External |
| `mlp-baseline` | None | Nodes | N/A |

All arms use degree as the 1-dim node feature (input_dim=1).

## 2. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H51** | l1-hodge-residual outperforms mlp-baseline on COLLAB | p_BH < 0.05 | COLLAB has no node features — structure IS the signal. L_1 accesses triangle-level structure that MLP (operating on degree alone) cannot. | p_BH >= 0.05 |
| **H52** | l1-hodge-residual outperforms hodge-mp-residual (L_0) on COLLAB | p_BH < 0.05 | COLLAB is triangle-rich; L_1's up-Laplacian component provides structural signal beyond node-level adjacency. This is the core test of higher-order Hodge theory. | p_BH >= 0.05 |
| **H53** | l1-hodge-residual outperforms gin-residual on COLLAB | p_BH < 0.05 | Same reasoning as H52 — L_1 encodes triangle co-boundary structure inaccessible to any L_0-based method. | p_BH >= 0.05 |

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H51 + H52 + H53 confirmed | **Higher-order Hodge structure provides unique classification signal on triangle-rich graphs.** L_1 captures structural information that L_0-based methods (Hodge, GIN) cannot access. This is the vindication of the Hodge approach — the value is in L_k for k >= 1, not in L_0. |
| H51 confirmed, H52/H53 refuted (L_1 beats MLP but not L_0) | L_1 captures structural signal, but L_0-based methods already capture it. The triangle-level information is redundant with node-level neighbourhood information on COLLAB. |
| H51 refuted (L_1 does not beat MLP on COLLAB) | Edge-level message passing with degree features fails on COLLAB. Possible causes: 1-dim degree input is insufficient, edge-to-graph pooling loses discrimination, or the L_1 computation on dense graphs is numerically unstable. |

## 4. Experimental design

- **Dataset:** COLLAB (5000 graphs, 3 classes), 1-dim degree features.
- **Models:** `l1-hodge-residual`, `hodge-mp-residual`, `gin-residual`, `mlp-baseline`.
- **Seeds:** 30.
- **Epochs:** 10.
- **Optimiser:** Adam(lr=1e-2).
- **Hidden dim:** 32.
- **Note:** COLLAB graphs are denser than NCI1 (mean 873 edges vs 32). Per-graph L_1 computation takes 0.02-0.13s. Estimated total wall time: ~8-12 hours on CPU.

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
