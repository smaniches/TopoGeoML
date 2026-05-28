---
title: H008c · External residual (primary finding)
parent: Hypotheses (H001–H011b)
nav_order: 10
---

# Hypothesis 008-c: Is the external residual the operative factor in the Hodge-GIN gap?

**Status.** Resolved 2026-05-24. H36 confirmed (gin-residual strictly outperforms gin-normalised); H37 confirmed in the first decision-tree row (gin-residual matches or exceeds Hodge — the external residual is the mechanism); H38 confirmed (gin-residual strictly outperforms MLP). See §6.

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

## 5. Resolved outcome (2026-05-24, 30 seeds x 10 epochs, 4 arms, NCI1)

Per-arm reports in `notebooks/results/h008c_nci1_gin_residual_30seeds.{json,md}`.

### Per-arm accuracy

| Arm | Median accuracy (BCa 95% CI) | vs MLP p_BH | Verdict |
|---|---|---|---|
| **gin-residual** | **0.629 [0.607, 0.641]** | **6.05 x 10^-4** | **WINS (+10.6 pp)** |
| hodge-mp-residual | 0.609 [0.581, 0.625] | 4.05 x 10^-3 | WINS (+8.6 pp) |
| gin-normalised | 0.500 [0.500, 0.500] | 5.33 x 10^-5 | LOSES (-2.3 pp) |
| mlp-baseline | 0.523 [0.513, 0.566] | -- | control |

### Pairwise comparisons

| Comparison | median Delta | p_BH | r |
|---|---|---|---|
| gin-residual vs hodge-mp-residual | +0.0195 | 1.01 x 10^-2 | +0.400 |
| gin-residual vs gin-normalised | +0.1290 | 5.20 x 10^-6 | +1.000 |
| gin-residual vs mlp-baseline | +0.1058 | 6.05 x 10^-4 | +0.600 |
| hodge-mp-residual vs gin-normalised | +0.1095 | 5.20 x 10^-6 | +1.000 |
| hodge-mp-residual vs mlp-baseline | +0.0864 | 4.05 x 10^-3 | +0.533 |

### Sub-hypotheses resolved

- **H36** (gin-residual beats gin-normalised): **CONFIRMED.** gin-residual (0.629) strictly outperforms gin-normalised (0.500) at p_BH = 5.20 x 10^-6, r = +1.000. The external residual recovers learning from class-prior collapse.
- **H37** (gin-residual matches Hodge): **CONFIRMED in the first decision-tree row.** gin-residual (0.629) not only matches but slightly exceeds Hodge (0.609). The difference is significant at p_BH = 0.010, r = +0.400, favouring gin-residual. The external residual is sufficient — the choice of spectral vs spatial operator is secondary.
- **H38** (gin-residual beats MLP): **CONFIRMED.** gin-residual (0.629) strictly outperforms MLP (0.523) at p_BH = 6.05 x 10^-4, r = +0.600.

### Interpretation

The result corresponds to the first row of the preregistered decision tree: **the external residual is the mechanism.**

The ablation series H008 → H008-b → H008-c systematically isolated three candidate factors:

| Factor | Tested by | Outcome |
|---|---|---|
| Degree normalisation | H008-b (gin-normalised) | Does not recover (still at class prior) |
| Spectral vs spatial operator | H008-c (gin-residual uses I-L_tilde, Hodge uses L_tilde) | gin-residual matches or exceeds Hodge — operator choice is secondary |
| **External residual placement** | **H008-c (gin-residual adds external skip)** | **Recovers from class-prior collapse to 0.629 — the operative factor** |

The Hodge Laplacian (L_tilde, high-pass spectral operator) does not confer a unique classification advantage on NCI1 at this configuration. The critical architectural element is the external residual connection (`act(propagation @ W + b) + h`), which preserves the projected features through the propagation step. Without it, both GIN variants (normalised and unnormalised) fail to learn. With it, even the normalised adjacency operator (low-pass, I - L_tilde) slightly outperforms the Hodge Laplacian.

### Scoped claim

> Under the matched-capacity protocol on NCI1 (30 seeds, 10 epochs, hidden_dim=32), the external residual connection is the operative architectural factor that enables topology-aware classification above MLP baseline. With external residual, both Hodge (L_tilde, 0.609) and normalised-adjacency (I - L_tilde, 0.629) arms outperform MLP (0.523). Without external residual, both normalised and unnormalised GIN collapse to class prior (0.500). The choice of spectral operator (high-pass vs low-pass) is secondary to the residual architecture at this configuration.

### What the H003-H008c arc establishes

The complete investigation arc, from the initial NCI1 positive claim (H003) through mechanism elimination (H004-H007) and architecture comparison (H008-H008c), converges on a specific finding:

1. Topology-aware message passing outperforms no-topology MLP on NCI1 (+8-10 pp).
2. The advantage requires an external residual connection; without it, message-passing architectures fail to learn.
3. The advantage is not specific to the Hodge Laplacian — normalised adjacency aggregation with external residual performs comparably or better.
4. The mechanism is architectural (residual placement), not operator-specific (Laplacian vs adjacency).

---

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
