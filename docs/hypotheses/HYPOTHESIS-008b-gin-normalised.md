---
title: H008b · Normalised GIN
parent: Hypotheses (H001–H011b)
nav_order: 9
---

# Hypothesis 008-b: Does degree normalisation close the GIN-Hodge gap on NCI1?

**Status.** Resolved 2026-05-24. H33 refuted (normalised GIN does not recover); H34 falsified (Hodge strictly beats normalised GIN, p_BH = 6.36e-6, r = +1.000); H35 refuted (normalised GIN underperforms MLP). The candidate explanation from H008 is incorrect — normalisation alone does not account for the GIN-Hodge gap. See §6.

**Falsification target.** Whether the GIN collapse observed in H008 (GIN at class prior, 0.500, under matched-capacity protocol) is attributable to the absence of degree normalisation. If so, adding symmetric normalisation to GIN's aggregation (GCN-style: D^{-1/2} A D^{-1/2} instead of raw A) should recover performance comparable to Hodge-MP-residual.

**Prior result motivating this hypothesis.** H008 found that GIN and GAT collapse to class prior on NCI1 while Hodge-MP-residual achieves 0.609. The candidate explanation identified in H008 §8 is degree-dependent feature scaling: the unnormalised adjacency sum A@h scales linearly with node degree, burying the per-node feature signal. This is the same mechanism that explains the combinatorial Hodge arm's 9 pp deficit on MUTAG (H001). If normalisation is the operative factor, a GIN variant with normalised aggregation should recover.

---

## 1. Design

A single architectural modification to the H008 GIN baseline:

| Arm | Aggregation | Normalisation |
|---|---|---|
| `gin-baseline` (H008) | (1+eps)*h + A@h | None (raw adjacency sum) |
| `gin-normalised` (this experiment) | (1+eps)*h + D^{-1/2} A D^{-1/2} @ h | Symmetric degree normalisation |
| `hodge-mp-residual` (control) | activation(L_tilde @ proj(x) @ W + b) + proj(x) | Symmetric Laplacian normalisation |

The normalised GIN aggregation D^{-1/2} A D^{-1/2} @ h is computed from the Laplacian as D^{-1/2}(D - L)D^{-1/2} @ h = (I - L_tilde) @ h, where L_tilde is the symmetrically normalised Laplacian. This is algebraically equivalent to the GCN propagation rule (Kipf & Welling 2017) applied within the GIN update framework.

All other experimental parameters are held fixed from H008: 30 seeds, 10 epochs, Adam(lr=1e-2), hidden_dim=32, matched capacity.

## 2. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Prediction | Rationale | Falsified if |
|---|---|---|---|---|
| **H33** | `gin-normalised` strictly beats `gin-baseline` on NCI1 | p_BH < 0.01 | Normalisation prevents degree-dependent feature scaling, the candidate explanation for GIN's collapse in H008 | p_BH >= 0.05 |
| **H34** | `gin-normalised` at least matches `hodge-mp-residual` on NCI1 | p_BH >= 0.05 (no significant difference) | If normalisation is the sole operative factor, then normalised GIN and normalised Hodge should converge to similar performance | Hodge strictly beats normalised GIN at p_BH < 0.01 |
| **H35** | `gin-normalised` strictly beats `mlp-baseline` on NCI1 | p_BH < 0.05 | With normalisation, GIN should be able to exploit NCI1's structural signal (confirmed present by H006) | p_BH >= 0.05 or GIN-norm <= MLP |

## 3. Outcome decision tree

| Pattern | Interpretation |
|---|---|
| H33 + H34 + H35 confirmed | **Normalisation is the mechanism.** Normalised GIN matches Hodge and both beat MLP. The H003 positive claim is a normalisation effect, not a Hodge-specific effect. The Hodge Laplacian provides no unique advantage over standard normalised message passing at this configuration. |
| H33 confirmed, H34 falsified (Hodge still beats normalised GIN) | **Normalisation is necessary but not sufficient.** The Hodge Laplacian's propagation mechanism provides an advantage beyond what normalisation alone confers. This would be the strongest evidence for a Hodge-specific structural signal. |
| H33 refuted (normalised GIN does not recover) | **Normalisation alone does not explain the GIN collapse.** Other factors — the GIN MLP architecture, the (1+eps) self-loop parameterisation, or training dynamics — contribute to the failure. The candidate explanation from H008 is incomplete. |

## 4. Experimental design

- **Dataset:** NCI1 (4110 graphs), identical to H003 and H008.
- **Models:** `hodge-mp-residual`, `gin-normalised`, `gin-baseline`, `mlp-baseline`.
- **Seeds:** 30, matched.
- **Epochs:** 10, matched.
- **Optimiser:** Adam(lr=1e-2), matched.
- **Hidden dim:** 32, matched.
- **Statistical procedure:** Pairwise paired Wilcoxon signed-rank across all 6 arm pairs, BH-FDR correction at alpha=0.05.

## 5. Reproduction

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-normalised gin-baseline mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output notebooks/results/h008b_nci1_gin_normalised_30seeds.json \
  --markdown notebooks/results/h008b_nci1_gin_normalised_30seeds.md
```

## 6. Resolved outcome (2026-05-24, 30 seeds x 10 epochs, 4 arms, NCI1)

Per-arm reports in `notebooks/results/h008b_nci1_gin_normalised_30seeds.{json,md}`.

### Per-arm accuracy

| Arm | Median accuracy (BCa 95% CI) | vs MLP p_BH | Verdict |
|---|---|---|---|
| **hodge-mp-residual** | **0.609 [0.581, 0.625]** | **4.05 x 10^-3** | **WINS (+8.6 pp)** |
| gin-normalised | 0.500 [0.500, 0.500] | 5.33 x 10^-5 | LOSES (-2.3 pp) |
| gin-baseline | 0.500 [0.500, 0.505] | 2.96 x 10^-3 | LOSES (-2.3 pp) |
| mlp-baseline | 0.523 [0.513, 0.566] | -- | control |

### Headline: Hodge vs normalised GIN

| Comparison | median Delta | p_BH | r |
|---|---|---|---|
| Hodge vs gin-normalised | +0.1095 | 6.36 x 10^-6 | +1.000 |
| Hodge vs gin-baseline | +0.1095 | 6.36 x 10^-6 | +0.933 |
| gin-normalised vs gin-baseline | +0.0000 | 2.06 x 10^-2 | -0.263 |

### Sub-hypotheses resolved

- **H33** (normalised GIN beats raw GIN): **REFUTED.** Normalised GIN does not recover from class prior. Median is still 0.500, indistinguishable from raw GIN in practical terms (p_BH = 0.021, but both at class prior). Normalisation does not rescue GIN's learning failure at this configuration.
- **H34** (normalised GIN matches Hodge): **FALSIFIED.** Hodge strictly outperforms normalised GIN at p_BH = 6.36 x 10^-6 with r = +1.000 (perfect rank separation across all 30 seeds). The gap is +10.95 pp — larger than the Hodge-vs-MLP gap (+8.64 pp).
- **H35** (normalised GIN beats MLP): **REFUTED.** Normalised GIN strictly underperforms MLP at p_BH = 5.33 x 10^-5, r = -0.833.

### Interpretation

The preregistered candidate explanation — that degree normalisation alone accounts for the GIN-Hodge gap — is refuted. Adding symmetric degree normalisation to GIN's aggregation does not recover learning at this configuration. Both normalised and unnormalised GIN collapse to class prior on NCI1, while MLP achieves 0.523 and Hodge achieves 0.609.

**What this rules out.** The Hodge advantage on NCI1 is not attributable to normalisation alone. If it were, normalised GIN (which uses the same D^{-1/2} scaling) would recover. It does not.

**What distinguishes Hodge from normalised GIN architecturally.** The Hodge-MP-residual arm applies L_tilde @ proj(x) @ W + proj(x), where L_tilde is the normalised Laplacian. This is a *spectral* propagation: it computes a weighted combination of each node's features with the Laplacian-smoothed (high-pass filtered) features of its neighbourhood. The normalised GIN arm applies MLP((1+eps)*h + (I - L_tilde)@h), which is a *spatial* aggregation: it sums normalised neighbour features and passes the result through an MLP. The key architectural difference is:

1. **Operator.** Hodge uses L_tilde (Laplacian, high-pass: emphasises how nodes *differ* from neighbours). GIN uses I - L_tilde (normalised adjacency, low-pass: averages nodes *with* neighbours).
2. **Weight application.** Hodge applies a learned weight matrix W *after* spectral propagation: L_tilde @ h @ W. GIN applies the MLP *after* spatial aggregation: MLP(aggregated). The Hodge formulation allows the weight matrix to interact with the spectral-domain signal; GIN's MLP sees only the spatially-aggregated result.
3. **Residual.** Hodge adds a skip connection *outside* the activation: act(L_tilde @ h @ W + b) + h. GIN's self-loop (1+eps)*h is *inside* the MLP. The external residual preserves the projected features through the propagation step.

These three differences — spectral vs spatial operator, weight-propagation interaction, external vs internal residual — are individually testable. Each could be the operative factor. The current experiment establishes that the combination produces a measurable advantage; identifying which factor is responsible requires further ablation.

### Scoped claim

> Under the matched-capacity protocol on NCI1 (30 seeds, 10 epochs, hidden_dim=32), the Hodge Laplacian's spectral propagation with external residual produces classification accuracy (0.609) that neither unnormalised GIN (0.500), normalised GIN (0.500), nor no-topology MLP (0.523) achieves. The advantage is not attributable to degree normalisation alone. The operative architectural difference remains to be isolated via targeted ablation (spectral vs spatial operator, weight-propagation order, residual placement).

---

## References

- Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are Graph Neural Networks? *ICLR 2019*.
