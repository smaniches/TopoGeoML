---
title: H008c · External residual (primary finding)
parent: Hypotheses (H001–H011b)
nav_order: 10
---

# Hypothesis 008-c: Is the external residual formulation the operative factor in the Hodge-GIN gap?

**Status.** Resolved 2026-05-24. H36 confirmed: `gin-residual` exceeds `gin-normalised` at p_BH = 5.20 x 10^-6. H37 supports the substantive preregistered conclusion that there is no unique Hodge advantage: `gin-residual` has the higher median than `hodge-mp-residual` (+1.95 pp, p_BH = 0.0101), so the Hodge-superiority falsification condition is not met. H38 confirmed: `gin-residual` exceeds MLP at p_BH = 6.05 x 10^-4. The experiment shows that the tested external-residual formulation is sufficient to recover performance after normalization alone fails. It does not establish that an abstract residual connection, independent of its placement and self-path parameterization, is the sole causal mechanism.

**Falsification target.** Whether replacing the internal self contribution of the normalized GIN formulation with the same external identity skip used by the Hodge residual arm recovers NCI1 performance to at least the Hodge level. H008-b ruled out normalization alone.

**Prior results.**
- H008: GIN with raw adjacency and an internal self contribution is at the class prior (0.500) on NCI1 under the matched-capacity protocol.
- H008-b: GIN with normalized adjacency and the same internal self formulation is also at the class prior (0.500).
- Hodge-MP-residual: 0.609 on NCI1. Its self path is external: `act(L_tilde @ h @ W + b) + h`.

The tested difference is therefore not merely “has a self connection” versus “does not.” Both GIN formulations contain a self contribution. The distinction is where that contribution enters the computation. `gin-normalised` applies the learned transformation and nonlinearity jointly to `(1 + eps)h + A_norm h`; `gin-residual` applies the propagated affine/nonlinear path and then adds the identity representation outside the activation.

---

## 1. Design

The experiment gives normalized adjacency the same external residual architecture used by the Hodge residual arm.

| Arm | Update | Self path |
|---|---|---|
| `gin-normalised` (H008-b) | `MLP((1+eps)*h + A_norm @ h)` | Trainable internal self term before the affine/nonlinear update |
| `gin-residual` (this experiment) | `act(A_norm @ h @ W + b) + h` | External identity skip after activation |
| `hodge-mp-residual` (control) | `act(L_tilde @ h @ W + b) + h` | External identity skip after activation |

The `gin-residual` and Hodge residual arms match in architecture and differ only in propagation operator, which makes their comparison a clean operator ablation. The `gin-normalised` versus `gin-residual` comparison changes the self-path formulation, including the location of the self contribution relative to the learned affine map and nonlinearity; `gin-normalised` also has the trainable `eps` scalar. Therefore this comparison identifies the tested external-residual formulation as the successful design change, but it should not be interpreted as an operator-independent proof about every possible residual architecture.

## 2. Preregistered sub-hypotheses

The table below preserves the preregistered decision rules.

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H36** | `gin-residual` strictly beats `gin-normalised` on NCI1 | p_BH < 0.01 | Moving the self path outside the propagated nonlinear update should preserve the projected representation | p_BH >= 0.05 |
| **H37** | `gin-residual` at least matches `hodge-mp-residual` on NCI1 | p_BH >= 0.05 | If Hodge has no unique operator advantage once architecture is matched, gin-residual should not be worse | Hodge strictly beats gin-residual at p_BH < 0.01 |
| **H38** | `gin-residual` strictly beats `mlp-baseline` on NCI1 | p_BH < 0.05 | The matched external-residual adjacency arm should exploit graph structure beyond the MLP control | p_BH >= 0.05 |

H37's tabulated prediction used non-significance as the literal “match” condition. The observed result is instead significant in the favorable direction for gin-residual. That does not constitute statistical equivalence, but it more directly rules out the Hodge-superiority falsification condition and supports the substantive “no unique Hodge advantage” question.

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H36 supported and Hodge does not outperform gin-residual | The external-residual formulation recovers the normalized-adjacency arm and no unique Hodge operator advantage remains in the matched architecture. |
| H36 supported but Hodge still beats gin-residual at the preregistered falsification threshold | The external residual formulation helps, but the Hodge operator contributes independently. |
| H36 refuted | The tested external-residual reformulation is not sufficient to recover the normalized-adjacency arm. |

## 4. Experimental design

- **Dataset:** NCI1 (4110 graphs), identical to H003/H008/H008-b.
- **Models:** `hodge-mp-residual`, `gin-residual`, `gin-normalised`, `mlp-baseline`.
- **Seeds:** 30, matched.
- **Epochs:** 10, matched.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon, BH-FDR at alpha=0.05.

## 5. Resolved outcome (2026-05-24, 30 seeds x 10 epochs, 4 arms, NCI1)

Per-arm reports in `notebooks/results/h008c_nci1_gin_residual_30seeds.{json,md}`.

### Per-arm accuracy

| Arm | Median accuracy (BCa 95% CI) | vs MLP p_BH | Result |
|---|---|---|---|
| `gin-residual` | 0.629 [0.607, 0.641] | 6.05 x 10^-4 | Positive difference from MLP |
| `hodge-mp-residual` | 0.609 [0.581, 0.625] | 4.05 x 10^-3 | Positive difference from MLP |
| `gin-normalised` | 0.500 [0.500, 0.500] | 5.33 x 10^-5 | Lower than MLP |
| `mlp-baseline` | 0.523 [0.513, 0.566] | -- | Control |

### Pairwise comparisons

| Comparison | median Delta | p_BH | r |
|---|---|---|---|
| gin-residual vs hodge-mp-residual | +0.0195 | 1.01 x 10^-2 | +0.400 |
| gin-residual vs gin-normalised | +0.1290 | 5.20 x 10^-6 | +1.000 |
| gin-residual vs mlp-baseline | +0.1058 | 6.05 x 10^-4 | +0.600 |
| hodge-mp-residual vs gin-normalised | +0.1095 | 5.20 x 10^-6 | +1.000 |
| hodge-mp-residual vs mlp-baseline | +0.0864 | 4.05 x 10^-3 | +0.533 |

### Sub-hypotheses resolved

- **H36:** **CONFIRMED.** gin-residual 0.629 exceeds gin-normalised 0.500 at p_BH = 5.20 x 10^-6, crossing the preregistered 0.01 threshold.
- **H37:** **SUPPORTED IN THE FAVORABLE DIRECTION, NOT AS AN EQUIVALENCE CLAIM.** gin-residual is 1.95 pp higher than Hodge with p_BH = 0.0101. Hodge does not satisfy H37's preregistered falsification condition. The result rules out a unique Hodge advantage in this matched architecture, but it is not statistical evidence of equality.
- **H38:** **CONFIRMED.** gin-residual 0.629 exceeds MLP 0.523 at p_BH = 6.05 x 10^-4.

### Interpretation

Three conclusions are supported by the observed ablation:

1. Normalizing the internal-self GIN formulation is not enough to recover performance on NCI1 under this protocol (H008-b).
2. The tested external-residual adjacency formulation recovers performance strongly (H008-c).
3. Once the Hodge and adjacency arms use the same external-residual computation, the adjacency arm is not worse and in fact has the higher observed median. Therefore the Hodge `L_0` operator has no unique advantage in this experiment.

The causal statement must remain scoped. H008-c changes the placement and parameterization of the self path between `gin-normalised` and `gin-residual`. It therefore supports the external-residual formulation as the operative tested architectural change, not a universal theorem that residual connections alone explain message-passing performance.

### Scoped claim

> Under the matched-capacity NCI1 protocol (30 seeds, 10 epochs, hidden_dim=32), changing normalized adjacency from the tested internal-self formulation to the matched external-residual formulation raises median accuracy from 0.500 to 0.629. In the matched external-residual architecture, normalized adjacency reaches 0.629 and Hodge `L_0` reaches 0.609, so no unique Hodge operator advantage is supported. This result is specific to the tested architectures and does not establish that residual placement is the sole mechanism in other models or datasets.

## 6. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-residual gin-normalised mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h008c_nci1_gin_residual_30seeds.json \
  --markdown notebooks/results/h008c_nci1_gin_residual_30seeds.md
```
