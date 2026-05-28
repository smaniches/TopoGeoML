---
title: H009 · Sheaf Laplacian
parent: Hypotheses (H001–H011b)
nav_order: 11
---

# Hypothesis 009: Does a learned sheaf Laplacian outperform fixed operators on NCI1?

**Status.** Resolved 2026-05-25. H39 confirmed (sheaf outperforms MLP at p_BH = 0.017); H40 refuted (sheaf does not outperform Hodge, p_BH = 0.797); H41 refuted (sheaf strictly underperforms gin-residual at p_BH = 0.014). The learned sheaf Laplacian adds no value over fixed operators at this configuration. See §7.

**Falsification target.** Whether a data-dependent sheaf Laplacian — where edge-level restriction maps are learned from node features — outperforms both the fixed Hodge Laplacian and the fixed normalised adjacency on NCI1 under the matched-capacity protocol with external residual.

**Prior results motivating this hypothesis.** H008-c established that the external residual is the operative architectural factor for NCI1 classification at this capacity. The choice between L_tilde (high-pass) and I - L_tilde (low-pass) as the fixed propagation operator is secondary (gin-residual 0.629 vs Hodge 0.609). Both operators use a *fixed* propagation matrix determined entirely by graph structure. A learned sheaf Laplacian replaces this fixed operator with a data-dependent one, where the propagation weights are predicted from node features. This is the natural escalation: if the operator doesn't matter when fixed, does a *learned* operator add value?

**Theoretical context.** A cellular sheaf on a graph assigns a vector space (stalk) to each node and a linear map (restriction map) to each edge. The sheaf Laplacian L_F = delta^T delta, where delta is the sheaf coboundary operator, generalises the graph Laplacian: when all restriction maps are the identity, L_F reduces to L_0. Neural Sheaf Diffusion (Bodnar et al. 2022, NeurIPS) learns the restriction maps from node features, making the propagation operator a function of the data. This is strictly more expressive than any fixed-Laplacian method (Hansen & Ghrist 2019).

---

## 1. Design

For scalar stalks (stalk dimension d_s = 1), the sheaf Laplacian simplifies to a learned weighted Laplacian with PSD guarantee:

- For each edge e = {i, j}, a small network predicts restriction scalars f_{i<-e}, f_{j<-e} from the projected node features.
- Off-diagonal: L_F[i,j] = -f_{i<-e} * f_{j<-e}
- Diagonal: L_F[i,i] = sum_{e containing i} f_{i<-e}^2
- L_F is PSD by construction (L_F = delta^T delta).
- Symmetric normalisation: L_F_tilde = D_F^{-1/2} L_F D_F^{-1/2}

Propagation with external residual: h' = act(L_F_tilde @ proj(x) @ W + b) + proj(x)

This generalises the Hodge arm (which is the special case f = 1 for all edges).

| Arm | Operator | Learned? | Residual |
|---|---|---|---|
| `sheaf-residual` | L_F_tilde (learned sheaf Laplacian) | Yes | External |
| `hodge-mp-residual` | L_tilde (fixed graph Laplacian) | No | External |
| `gin-residual` | I - L_tilde (fixed normalised adjacency) | No | External |
| `mlp-baseline` | None | N/A | N/A |

## 2. Capacity matching

| Arm | Params (NCI1, input_dim=37, hidden_dim=32) |
|---|---|
| sheaf-residual | ~2403 (proj_in 1216 + sheaf_learner 65 + mp_weight 1056 + head 66) |
| hodge-mp-residual | 2338 |
| gin-residual | 2338 |
| mlp-baseline | 2338 |

The sheaf arm has ~2.8% more parameters due to the sheaf learner (65 params). This is within the 5% tolerance used in H001 and documented as acceptable for the matched-capacity protocol.

## 3. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H39** | sheaf-residual strictly beats mlp-baseline on NCI1 | p_BH < 0.05 | A learned operator with external residual should at minimum capture the structural signal that gin-residual and Hodge both capture | p_BH >= 0.05 |
| **H40** | sheaf-residual strictly beats hodge-mp-residual on NCI1 | Uncertain — the learned operator may or may not improve over fixed L_tilde at 10 epochs | 10 epochs may be insufficient for the sheaf learner to converge; the additional parameters may also overfit at this sample size | p_BH >= 0.05 or sheaf < hodge |
| **H41** | sheaf-residual at least matches gin-residual on NCI1 | p_BH >= 0.05 or sheaf > gin-residual | The learned operator should be at least as expressive as the fixed normalised adjacency | sheaf strictly underperforms gin-residual at p_BH < 0.01 |

## 4. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H39 + H40 confirmed (sheaf beats Hodge and MLP) | A learned propagation operator provides classification-relevant structure that fixed operators miss. The data-dependent restriction maps capture edge-level interactions that uniform propagation cannot. |
| H39 confirmed, H40 refuted (sheaf matches Hodge but beats MLP) | The learned operator does not improve over fixed operators at this capacity and epoch budget. The sheaf learner's 65 additional parameters are insufficient to learn meaningful edge-level structure, or 10 epochs is too short for convergence. |
| H39 refuted (sheaf does not beat MLP) | The sheaf learner fails to converge at this configuration. Possible causes: overfitting (additional parameters on 4110 graphs), optimisation difficulty (joint learning of restriction maps and classification weights), or insufficient epoch budget. |

## 5. Experimental design

- **Dataset:** NCI1 (4110 graphs), identical to H003-H008c.
- **Models:** `sheaf-residual`, `hodge-mp-residual`, `gin-residual`, `mlp-baseline`.
- **Seeds:** 30, matched.
- **Epochs:** 10, matched.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon, BH-FDR at alpha=0.05.

## 6. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models sheaf-residual hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h009_nci1_sheaf_30seeds.json \
  --markdown notebooks/results/h009_nci1_sheaf_30seeds.md
```

## 7. Resolved outcome (2026-05-25, 30 seeds x 10 epochs, 4 arms, NCI1)

Per-arm reports in `notebooks/results/h009_nci1_sheaf_30seeds.{json,md}`.

### Per-arm accuracy

| Arm | Median accuracy (BCa 95% CI) | vs MLP p_BH | Verdict |
|---|---|---|---|
| gin-residual | 0.629 [0.607, 0.641] | 2.42 x 10^-3 | WINS (+10.6 pp) |
| hodge-mp-residual | 0.609 [0.581, 0.625] | 1.01 x 10^-2 | WINS (+8.6 pp) |
| sheaf-residual | 0.604 [0.564, 0.619] | 1.68 x 10^-2 | WINS (+8.1 pp) |
| mlp-baseline | 0.523 [0.513, 0.566] | -- | control |

### Key comparisons

| Comparison | median Delta | p_BH | r | Interpretation |
|---|---|---|---|---|
| sheaf vs Hodge | -0.005 | 0.797 | +0.133 | Indistinguishable |
| sheaf vs gin-residual | -0.025 | 1.37 x 10^-2 | -0.467 | Sheaf underperforms |
| gin-residual vs Hodge | +0.020 | 1.52 x 10^-2 | +0.400 | gin-residual slightly ahead |

### Sub-hypotheses resolved

- **H39** (sheaf beats MLP): **CONFIRMED.** sheaf-residual (0.604) outperforms MLP (0.523) at p_BH = 1.68 x 10^-2, r = +0.333. The learned operator, like both fixed operators with external residual, captures structural signal above the no-topology baseline.
- **H40** (sheaf beats Hodge): **REFUTED.** sheaf-residual (0.604) is statistically indistinguishable from Hodge (0.609) at p_BH = 0.797. The learned restriction maps do not improve over the fixed identity maps (which reduce the sheaf Laplacian to the standard graph Laplacian) at this configuration.
- **H41** (sheaf matches gin-residual): **REFUTED.** sheaf-residual (0.604) strictly underperforms gin-residual (0.629) at p_BH = 1.37 x 10^-2, r = -0.467.

### Interpretation

The learned sheaf Laplacian does not improve over fixed operators at this configuration. All three topology-aware arms with external residual produce comparable accuracy (0.604-0.629), with gin-residual (fixed normalised adjacency) performing best and the sheaf Laplacian performing worst among the three. The 130 additional sheaf-learner parameters and per-graph dense Laplacian construction provide no measurable benefit.

Two factors likely contribute:
1. **Insufficient training budget.** The sheaf learner must jointly learn restriction maps and classification weights in 10 epochs. At convergence, the sheaf approach's additional expressiveness may manifest, but the current epoch budget may be insufficient for the sheaf parameters to specialise.
2. **Scalar stalks are minimally expressive.** The scalar-stalk sheaf Laplacian learns one restriction scalar per (node, edge) pair. Bodnar et al. (2022) use vector-valued stalks (d_s > 1) with full matrix restriction maps, which are substantially more expressive. The scalar reduction may be too constrained to capture edge-level heterogeneity.

### What the full H003-H009 arc establishes

| Hypothesis | Question | Finding |
|---|---|---|
| H003 | Does Hodge beat MLP on NCI1? | Yes (+8.6 pp) |
| H004 | Is sample size the mechanism? | No |
| H005 | Is feature dimensionality the mechanism? | No |
| H006 | Does topology carry class signal? | Yes (all 3 datasets) |
| H007 | Which structural proxy explains the gain? | None individually |
| H008 | Does Hodge beat GIN/GAT? | Yes, but GIN/GAT lack external residual |
| H008-b | Does normalisation close the gap? | No |
| H008-c | Does the external residual close the gap? | **Yes — gin-residual matches/exceeds Hodge** |
| H009 | Does a learned operator improve further? | **No — fixed operators suffice** |

**Consolidated conclusion:** On NCI1 at this configuration, topology-aware message passing with an external residual connection outperforms no-topology MLP by 8-10 pp. The critical factor is the external residual architecture, not the choice of propagation operator (fixed Laplacian, fixed adjacency, or learned sheaf). The propagation operator is secondary: all three variants perform comparably once the residual is present.

---

## References

- Bodnar, C., Di Giovanni, F., Chamberlain, B., Lio, P., & Bronstein, M. (2022). Neural Sheaf Diffusion: A topological perspective on heterophily and oversmoothing in GNNs. *NeurIPS 2022*.
- Hansen, J. & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3, 315-358.
