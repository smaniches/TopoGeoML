# Hypothesis 008: Comparative evaluation of Hodge-MP-residual against GIN and GAT baselines on NCI1

**Status.** Preregistered 2026-05-24, before execution.

**Falsification target.** Whether the NCI1 positive claim (H003: Hodge-MP-residual +8.6 pp over MLP at p_BH = 4.83e-3) generalises to comparisons against topology-aware GNN baselines — specifically GIN (Xu et al. 2019, ICLR) and GAT (Velickovic et al. 2018, ICLR) — or whether it is specific to the Hodge-vs-no-topology contrast.

**Prior result motivating this hypothesis.** H001-H007 established that Hodge-MP-residual strictly outperforms a matched-capacity no-topology MLP baseline on NCI1. The mechanism investigation (H004-H007) narrowed the effect to an architecture-data interaction consistent with a complementarity interpretation. The open question is whether this advantage reflects a property of the Hodge Laplacian specifically, or whether any topology-aware message-passing architecture captures the same structural signal. Answering this question is necessary before any claim about the Hodge Laplacian's unique contribution can be made.

---

## 1. Baseline selection and theoretical context

### GIN (Graph Isomorphism Network, Xu et al. 2019)

Update rule:

    h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)

GIN with sum aggregation is provably as powerful as the Weisfeiler-Lehman 1-dimensional test (WL-1) in distinguishing non-isomorphic graphs (Xu et al. 2019, Theorem 3). The WL-1 test is strictly more expressive than spectral methods based on graph Laplacian eigenvalues alone — co-spectral non-isomorphic graphs exist that WL-1 distinguishes. Since the Hodge-MP arm operates on L_0 (the graph Laplacian, a spectral operator on 0-simplices), GIN's theoretical expressiveness is at least as high as the Hodge arm's at the tested configuration. This makes GIN a principled upper-bound baseline.

### GAT (Graph Attention Network, Velickovic et al. 2018)

Update rule:

    h_v' = sigma(sum_{u in N(v)} alpha_{vu} W h_u)

where alpha_{vu} are learned attention coefficients. GAT learns adaptive neighbour weighting, representing a different inductive bias from GIN's uniform sum aggregation. The theoretical expressiveness of GAT relative to WL-1 is architecture-dependent (Brody et al. 2022 show that standard GAT is bounded by WL-1; GATv2 can be strictly more expressive in some configurations).

### Rationale for the comparison

The H003 positive claim compares Hodge-MP-residual against a no-topology MLP. Without testing against standard topology-aware baselines, it is impossible to determine whether the observed advantage reflects (a) topology vs. no topology, or (b) a Hodge-specific structural signal. This experiment discriminates between these interpretations.

## 2. Capacity matching

All arms use the parameter-matching discipline established in H001-H003:

| Arm | Architecture | Params (NCI1, input_dim=37, hidden_dim=32) |
|---|---|---|
| `hodge-mp-residual` | L_tilde @ proj(x) @ W + b + proj(x), sum-pool, head | 2338 |
| `mlp-baseline` | Linear -> ReLU -> Linear -> ReLU, sum-pool, head | 2338 |
| `gin-baseline` | proj(x) -> GIN((1+eps)*h + A@h, MLP) -> sum-pool, head | 2339 |
| `gat-baseline` | proj(x) -> GAT(W, attn-weighted neighbours) -> sum-pool, head | 2340 |

Parameters are matched to within 0.1%. The comparison isolates the aggregation mechanism, not model capacity.

## 3. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H28** | Hodge-MP-residual vs GIN on NCI1 | GIN at least matches Hodge (p_BH >= 0.05 or GIN > Hodge) | GIN's WL-1 expressiveness is theoretically at least as high as spectral methods on L_0. The Hodge arm's symmetric normalisation and external residual may partially offset this, but the theory favours GIN. | Hodge strictly beats GIN at p_BH < 0.01 |
| **H29** | Hodge-MP-residual vs GAT on NCI1 | Uncertain; GAT may match or underperform Hodge depending on whether NCI1's structural signal benefits from adaptive weighting | GAT's expressiveness is bounded by WL-1 (Brody et al. 2022) but its learned attention may help or hurt depending on the data | Hodge strictly beats GAT at p_BH < 0.01 with r > 0.3 |
| **H30** | GIN vs MLP on NCI1 | GIN strictly beats MLP (p_BH < 0.05) | GIN incorporates graph structure that MLP cannot access; the H003-H006 results confirm that NCI1 carries exploitable structural signal | p_BH >= 0.05 or GIN <= MLP |
| **H31** | GAT vs MLP on NCI1 | GAT strictly beats MLP (p_BH < 0.05) | Same reasoning as H30 | p_BH >= 0.05 or GAT <= MLP |
| **H32** | GIN vs GAT on NCI1 | Not significantly different (p_BH >= 0.05) | Both are topology-aware message-passing methods with comparable expressiveness on standard benchmarks | p_BH < 0.05 in either direction |

## 4. Outcome decision tree

| Pattern | Interpretation | Implication |
|---|---|---|
| H28 falsified (Hodge strictly beats GIN) | The Hodge Laplacian's spectral propagation captures classification-relevant structure on NCI1 that WL-1 aggregation misses at this capacity. This would be a surprising and significant finding, as it contradicts the theoretical expressiveness hierarchy. Requires careful examination of whether the advantage is attributable to the Laplacian, the normalisation scheme, or the residual architecture. | Investigate the source of the advantage (normalisation, residual, spectral properties). Test on additional datasets to assess generalisability. |
| H28 confirmed, H30 confirmed (GIN matches Hodge, both beat MLP) | The NCI1 positive claim reflects a generic "topology vs. no topology" advantage, not a Hodge-specific contribution. Any topology-aware message-passing architecture captures the same structural signal. The complementarity pattern from H006-H007 is architecture-independent. | The Hodge Laplacian does not confer a unique advantage on NCI1 at this configuration. Future work should focus on datasets or configurations where L_0 and WL-1 provably diverge. |
| H28 confirmed, H30 refuted (GIN matches Hodge, neither beats MLP) | The matched-capacity constraint may be too restrictive for GIN's architecture to exploit NCI1's structure. Re-examination of the capacity-matching protocol is warranted. | Investigate whether relaxing the capacity constraint (larger hidden_dim for GIN) changes the result. |
| All topology-aware arms match MLP | The H003 positive claim does not reproduce under the expanded comparison family. BH correction across a larger family may render the original result non-significant. | Document as a reproducibility finding. Re-examine whether the original H003 result is robust to comparison-family expansion. |

## 5. Experimental design

- **Dataset:** NCI1 (4110 graphs, 2 classes, Wale et al. 2008), identical to H003.
- **Models:** `hodge-mp-residual`, `gin-baseline`, `gat-baseline`, `mlp-baseline`.
- **Seeds:** 30, matched to H003 for direct comparison.
- **Epochs:** 10, matched to H003.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon signed-rank across all 6 arm pairs, BH-FDR correction at alpha=0.05. Note: the BH correction family is now 6 comparisons (vs 10 in H003), which affects the adjusted p-values.
- **Effect size:** Rank-biserial r (Kerby 2014) per comparison.
- **CIs:** BCa 95% bootstrap on per-arm accuracy median (10,000 replicates).

## 6. Implementation notes

GIN and GAT are implemented using the graph Laplacian L = D - A directly, without requiring an edge_index representation:

- **GIN aggregation:** The neighbourhood sum A @ H is computed as D @ H - L @ H, where D is the degree diagonal read from L's diagonal entries. This is algebraically equivalent to standard GIN sum aggregation and avoids modifying the dataset or training-loop interfaces.
- **GAT aggregation:** Attention coefficients are computed over L's off-diagonal sparsity pattern, which corresponds exactly to the edge set of the graph. Softmax normalisation is applied per destination node.

Both implementations use the existing `forward_one(x, laplacian)` interface. No modifications to the dataset loader, training loop, or runner infrastructure are required.

## 7. Wall-clock budget

| Arm | Estimated (30 seeds x 10 epochs, NCI1, CPU) |
|---|---|
| hodge-mp-residual | ~60 min |
| gin-baseline | ~60 min |
| gat-baseline | ~90 min (attention coefficient computation) |
| mlp-baseline | ~40 min |
| **Total** | **~4 hours** |

## 8. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-baseline gat-baseline mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h008_nci1_gin_gat_30seeds.json \
  --markdown notebooks/results/h008_nci1_gin_gat_30seeds.md
```

## References

- Brody, S., Alon, U., & Yahav, E. (2022). How attentive are Graph Attention Networks? *ICLR 2022*.
- Velickovic, P., Cucurull, G., Casanova, A., et al. (2018). Graph Attention Networks. *ICLR 2018*.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are Graph Neural Networks? *ICLR 2019*.
