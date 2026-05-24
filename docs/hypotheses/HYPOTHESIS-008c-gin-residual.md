# Hypothesis 008-c: Is the external residual the operative factor in the Hodge-GIN gap?

**Status.** Preregistered 2026-05-24, before execution.

**Falsification target.** Whether adding an external residual connection to GIN's normalised aggregation recovers performance comparable to Hodge-MP-residual on NCI1. H008-b ruled out normalisation alone. Three candidate factors remain: (1) spectral vs spatial operator, (2) weight-propagation order, (3) external vs internal residual. This experiment isolates factor (3).

**Prior results.**
- H008: GIN (raw A, internal self-loop) collapses to class prior (0.500) on NCI1.
- H008-b: GIN with normalised aggregation (D^{-1/2}AD^{-1/2}, internal self-loop) also collapses to class prior (0.500).
- Hodge-MP-residual: 0.609 on NCI1. Its residual is *external*: `act(L_tilde @ h @ W + b) + h`.
- H001: The residual variant's verdict inverts across datasets (hurts on MUTAG, helps on NCI1).

The external residual preserves the projected features through the propagation step, allowing the model to learn what to *add* to the per-node representation rather than what to *replace* it with. GIN's (1+eps)*h self-loop is functionally different: it is inside the MLP, so the nonlinearity is applied to the sum of the self-loop and the aggregation jointly.

---

## 1. Design

A single architectural modification: give normalised GIN the same external residual that the Hodge arm uses.

| Arm | Aggregation | Residual |
|---|---|---|
| `gin-normalised` (H008-b) | MLP((1+eps)*h + A_norm@h) | Internal (self-loop inside MLP) |
| `gin-residual` (this experiment) | act(A_norm @ h @ W + b) + h | External (skip outside activation) |
| `hodge-mp-residual` (control) | act(L_tilde @ h @ W + b) + h | External (skip outside activation) |

The `gin-residual` arm uses the normalised adjacency (I - L_tilde) for aggregation and an external residual, matching the Hodge arm's residual architecture exactly. The only remaining difference is the operator: normalised adjacency (low-pass, averages with neighbours) vs normalised Laplacian (high-pass, emphasises differences from neighbours).

## 2. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H36** | `gin-residual` strictly beats `gin-normalised` on NCI1 | p_BH < 0.01 | The external residual is the architectural element that enables learning at this capacity; without it, the aggregated signal overwrites the projected features | p_BH >= 0.05 |
| **H37** | `gin-residual` at least matches `hodge-mp-residual` on NCI1 | p_BH >= 0.05 | If the residual is the sole operative factor, then both arms — differing only in spectral (L_tilde) vs spatial (I-L_tilde) operator — should converge | Hodge strictly beats gin-residual at p_BH < 0.01 |
| **H38** | `gin-residual` strictly beats `mlp-baseline` on NCI1 | p_BH < 0.05 | With the external residual enabling learning, the normalised adjacency should provide exploitable structural signal | p_BH >= 0.05 |

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H36 + H37 + H38 confirmed | **The external residual is the mechanism.** Once the residual preserves the projected features, both spectral and spatial operators achieve comparable performance. The Hodge Laplacian does not confer a unique advantage — the architecture (residual placement) is what matters. |
| H36 confirmed, H37 falsified (Hodge still beats gin-residual) | **The residual is necessary but the spectral operator contributes independently.** The Laplacian's high-pass filtering provides classification-relevant signal that normalised adjacency averaging does not. This would be evidence for a Hodge-specific structural signal. |
| H36 refuted (gin-residual does not recover) | **The external residual alone is not sufficient.** The Hodge advantage involves the interaction between the spectral operator and the weight matrix (factor 2: L_tilde @ h @ W vs separate aggregation + MLP), not just the residual placement. |

## 4. Experimental design

- **Dataset:** NCI1 (4110 graphs), identical to H003/H008/H008-b.
- **Models:** `hodge-mp-residual`, `gin-residual`, `gin-normalised`, `mlp-baseline`.
- **Seeds:** 30, matched.
- **Epochs:** 10, matched.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon, BH-FDR at alpha=0.05.

## 5. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-residual gin-normalised mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h008c_nci1_gin_residual_30seeds.json \
  --markdown notebooks/results/h008c_nci1_gin_residual_30seeds.md
```
