# Hypothesis 009: Does a learned sheaf Laplacian outperform fixed operators on NCI1?

**Status.** Preregistered 2026-05-24, before execution.

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

## References

- Bodnar, C., Di Giovanni, F., Chamberlain, B., Lio, P., & Bronstein, M. (2022). Neural Sheaf Diffusion: A topological perspective on heterophily and oversmoothing in GNNs. *NeurIPS 2022*.
- Hansen, J. & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3, 315-358.
