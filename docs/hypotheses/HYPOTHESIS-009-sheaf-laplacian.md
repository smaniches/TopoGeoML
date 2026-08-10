---
title: H009 · Sheaf Laplacian
parent: Hypotheses (H001–H011b)
nav_order: 11
---

# Hypothesis 009: Does a learned sheaf Laplacian outperform fixed operators on NCI1?

**Status.** Resolved 2026-05-25. H39 confirmed under its preregistered threshold (sheaf-residual outperforms MLP, p_BH = 0.0168). H40 refuted (no significant improvement over Hodge, p_BH = 0.797). H41 is inconclusive under its preregistered decision rule: the sheaf point estimate is below gin-residual and the pairwise p_BH is 0.0137, but H41 required p_BH < 0.01 to declare strict underperformance. The learned sheaf operator therefore shows no supported improvement over the fixed operators at this configuration. See §7.

**Falsification target.** Whether a data-dependent sheaf Laplacian, where edge-level restriction maps are learned from node features, outperforms both the fixed Hodge Laplacian and the fixed normalised adjacency on NCI1 under the matched-capacity protocol with external residual.

**Prior results motivating this hypothesis.** H008-c established that the external residual is the operative architectural factor for NCI1 classification at this capacity. The choice between L_tilde (high-pass) and I - L_tilde (low-pass) as the fixed propagation operator is secondary (gin-residual 0.629 vs Hodge 0.609). Both operators use a fixed propagation matrix determined entirely by graph structure. A learned sheaf Laplacian replaces this fixed operator with a data-dependent one, where propagation weights are predicted from node features. The experiment asks whether that additional operator flexibility produces a measurable benefit.

**Theoretical context.** A cellular sheaf on a graph assigns a vector space (stalk) to each node and a linear map (restriction map) to each edge. The sheaf Laplacian L_F = delta^T delta, where delta is the sheaf coboundary operator, generalises the graph Laplacian: when all restriction maps are the identity, L_F reduces to L_0. Neural Sheaf Diffusion (Bodnar et al. 2022, NeurIPS) learns restriction maps from node features, allowing a data-dependent propagation operator rather than a fixed graph operator.

---

## 1. Design

For scalar stalks (stalk dimension d_s = 1), the sheaf Laplacian simplifies to a learned weighted Laplacian with PSD guarantee:

- For each edge e = {i, j}, a small network predicts restriction scalars f_{i<-e}, f_{j<-e} from the projected node features.
- Off-diagonal: L_F[i,j] = -f_{i<-e} * f_{j<-e}
- Diagonal: L_F[i,i] = sum_{e containing i} f_{i<-e}^2
- L_F is PSD by construction (L_F = delta^T delta).
- Symmetric normalisation: L_F_tilde = D_F^{-1/2} L_F D_F^{-1/2}

Propagation with external residual: h' = act(L_F_tilde @ proj(x) @ W + b) + proj(x)

This generalises the Hodge arm, which is recovered when the scalar restriction maps are fixed to the corresponding identity case.

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
| **H40** | sheaf-residual strictly beats hodge-mp-residual on NCI1 | Uncertain; the learned operator may or may not improve over fixed L_tilde at 10 epochs | 10 epochs may be insufficient for the sheaf learner to converge; the additional parameters may also overfit at this sample size | p_BH >= 0.05 or sheaf < hodge |
| **H41** | sheaf-residual at least matches gin-residual on NCI1 | p_BH >= 0.05 or sheaf > gin-residual | The learned operator should be at least as expressive as the fixed normalised adjacency | sheaf strictly underperforms gin-residual at p_BH < 0.01 |

## 4. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H39 + H40 confirmed (sheaf beats Hodge and MLP) | A learned propagation operator provides classification-relevant structure that fixed operators miss. The data-dependent restriction maps capture edge-level interactions that uniform propagation cannot. |
| H39 confirmed, H40 refuted (sheaf does not improve over Hodge but beats MLP) | The learned operator does not improve over the fixed Hodge operator at this capacity and epoch budget. |
| H39 refuted (sheaf does not beat MLP) | The sheaf learner fails to produce the preregistered positive difference from MLP at this configuration. |

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

| Arm | Median accuracy (BCa 95% CI) | vs MLP p_BH | Result at alpha=0.05 |
|---|---|---|---|
| gin-residual | 0.629 [0.607, 0.641] | 2.42 x 10^-3 | Positive difference from MLP |
| hodge-mp-residual | 0.609 [0.581, 0.625] | 1.01 x 10^-2 | Positive difference from MLP |
| sheaf-residual | 0.604 [0.564, 0.619] | 1.68 x 10^-2 | Positive difference from MLP |
| mlp-baseline | 0.523 [0.513, 0.566] | -- | Control |

### Key comparisons

| Comparison | median Delta | p_BH | r | Interpretation |
|---|---|---|---|---|
| sheaf vs Hodge | -0.005 | 0.797 | +0.133 | No significant difference detected |
| sheaf vs gin-residual | -0.025 | 1.37 x 10^-2 | -0.467 | Lower sheaf point estimate; significant at 0.05 but not at H41's preregistered 0.01 falsification threshold |
| gin-residual vs Hodge | +0.020 | 1.52 x 10^-2 | +0.400 | Positive difference at alpha=0.05 in this H009 family |

### Sub-hypotheses resolved

- **H39** (sheaf beats MLP): **CONFIRMED.** sheaf-residual (0.604) exceeds MLP (0.523) at p_BH = 1.68 x 10^-2, satisfying H39's preregistered p_BH < 0.05 threshold.
- **H40** (sheaf beats Hodge): **REFUTED.** The sheaf point estimate is lower (0.604 vs 0.609) and the pairwise test detects no significant difference (p_BH = 0.797), satisfying H40's preregistered falsification rule.
- **H41** (sheaf at least matches gin-residual): **INCONCLUSIVE UNDER THE PREREGISTERED RULE.** sheaf-residual is lower by 2.5 pp and the pairwise p_BH is 1.37 x 10^-2. That is below 0.05, but H41 explicitly required p_BH < 0.01 to declare strict underperformance. The observed result therefore does not satisfy H41's stated confirmation condition or its stated falsification threshold.

### Interpretation

The learned sheaf Laplacian does not provide a supported improvement over the fixed Hodge operator at this configuration. Its median accuracy is 0.604, compared with 0.609 for Hodge and 0.629 for gin-residual. The comparison with Hodge detects no significant difference, while the comparison with gin-residual produces p_BH = 0.0137, which is suggestive under a conventional 0.05 threshold but does not cross H41's preregistered 0.01 falsification threshold.

The experiment therefore supports a narrower conclusion than the original post-result summary: adding the scalar learned sheaf operator did not improve the tested NCI1 classifier over the fixed operators. It does not establish that learned sheaf operators are generally inferior.

Possible explanations such as insufficient training budget or limited scalar-stalk expressivity remain post hoc hypotheses and require separate tests.

### What the H003-H009 arc establishes

| Hypothesis | Question | Finding |
|---|---|---|
| H003 | Does Hodge beat MLP on NCI1? | Positive difference at the tested configuration (+8.6 pp) |
| H004 | Is sample size the mechanism? | Sample size alone does not explain the cross-dataset sign change |
| H005 | Is feature dimensionality the mechanism? | Feature dimensionality alone does not explain it |
| H006 | Does graph structure carry class signal under constant features? | Yes on the three tested datasets |
| H007 | Which tested structural proxy explains the full-feature pattern? | None of the five tested proxies tracks it |
| H008 | Does Hodge separate from GIN/GAT in the matched-capacity NCI1 regime? | Yes; mechanism not isolated by H008 alone |
| H008-b | Does degree normalisation close the gap? | No |
| H008-c | Does the external residual close the gap? | The residual is the operative factor in the tested ablation; no unique Hodge advantage remains |
| H009 | Does the scalar learned sheaf operator improve further? | No supported improvement over the fixed Hodge operator; H41 remains inconclusive at its preregistered threshold |

---

## References

- Bodnar, C., Di Giovanni, F., Chamberlain, B., Lio, P., & Bronstein, M. (2022). Neural Sheaf Diffusion: A topological perspective on heterophily and oversmoothing in GNNs. *NeurIPS 2022*.
- Hansen, J. & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3, 315-358.
