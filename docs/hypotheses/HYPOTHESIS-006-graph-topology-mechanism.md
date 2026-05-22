# Hypothesis 006: Does graph-topology signal predict the Hodge advantage? A Weisfeiler-Lehman + feature-shuffle test

**Status.** Preregistered 2026-05-22, before any execution. Conditional on H004 + H005 having both refuted (which they did).
**Falsification target.** Whether, under constant-feature ablation, the Hodge MP arm retains above-prior predictive signal on the three TUDataset benchmarks (MUTAG, PROTEINS, NCI1). The preregistered sub-hypotheses (§2) frame this as an NCI1-vs-others contrast; the data may confirm that contrast, refute it, or reveal a more nuanced pattern. The verdict is recorded in §6 after the run completes.
**Prior results that motivate this hypothesis.** H004 refuted sample size as the mechanism (subsampling NCI1 to MUTAG-size leaves Hodge-residual *winning*); H005 refuted feature dimensionality (projecting NCI1 to 7-dim noise leaves Hodge-residual *winning* while MLP collapses to chance). One remaining candidate to test is whether graph-structural signal — measurable by removing node features entirely — accounts for the dataset-by-dataset difference. The preregistered prediction below is that the signal lives mostly in NCI1; the experiment is designed so the data can refute that prediction if it is wrong.

---

## 1. The "topology-only" diagnostic

The H005-A subfinding (NCI1-7d: Hodge 0.58 vs MLP 0.50 at chance) suggests a direct diagnostic: **how much classification signal can we extract using ONLY the graph topology**, with node features removed entirely?

**Experiment setup**: replace each graph's node features with a *constant* vector (e.g. all-ones of shape `(n_nodes, 1)`), keeping the Laplacian intact. Train `hodge-mp-residual` and `mlp-baseline` for 30 seeds × 10 epochs on each dataset {MUTAG, PROTEINS, NCI1}. The MLP cannot use topology at all; its accuracy floor on constant-feature graphs is the class-prior baseline. The Hodge model can still use topology via the Laplacian.

The gap `Hodge_acc - class_prior` measures **how much classification signal is encoded in pure topology**, per dataset. The hypothesis is that this gap *correlates* with the residual-vs-MLP win in the full-feature condition.

## 2. Preregistered sub-hypotheses

| ID | Sub-hypothesis | Predicted | Falsified if |
|---|---|---|---|
| **H22** | NCI1 constant-feature Hodge accuracy is significantly above class prior | p_BH < 0.05 | not significant |
| **H23** | MUTAG constant-feature Hodge accuracy is NOT significantly above class prior | p_BH ≥ 0.05 | significant |
| **H24** | PROTEINS constant-feature Hodge accuracy is between H23 and H22 levels | between MUTAG and NCI1 | falls outside |
| **H25** | The constant-feature Hodge-vs-class-prior gap correlates positively with the full-feature Hodge-vs-MLP gap across the three datasets (Spearman ρ > 0) | yes | ρ ≤ 0 |

## 3. Outcome decision tree (preregistered)

| Pattern | Mechanism verdict | Framework implication |
|---|---|---|
| H22+H23+H24+H25 confirmed | **Graph-topology signal IS the mechanism.** The Hodge architecture's value on a dataset is predicted by how much classification signal lives in pure graph structure. | Framework can claim a *predictive criterion* for when Hodge-MP helps: pre-train a constant-feature Hodge classifier; if it beats class prior significantly, Hodge will help on full features too. |
| H22 confirmed, H23 refuted | MUTAG's topology DOES carry signal but the Hodge architecture fails to exploit it under full features. Architecture/data interaction is more subtle. | Hypothesis 007 examines why the same topology signal is exploited on NCI1 but not on MUTAG (e.g. graph size, training dynamics). |
| H22 refuted | Even NCI1's topology doesn't carry signal under constant features — the H005-A finding was actually about projected-feature-noise *acting as a topology signal* via the residual connection. | Re-examine the mechanism story; possibly the residual + L̃ interaction is doing something subtler than "topology signal extraction". |
| H25 refuted (correlation null or negative) | Constant-feature accuracy is not predictive of full-feature gain | Reject the simple topology-signal hypothesis; mechanism is something else. |

## 4. Implementation plan

Two new infrastructure pieces, both small:

1. **`--constant-features` flag** on the bench CLI, threaded through `run_classification` like `max_graphs` and `feature_projection_dim`. When set, replace every graph's node features with `torch.ones((n_nodes, 1), dtype=torch.float64)`.
2. **Class-prior computation** added to the `ClassificationReport` so the "is Hodge above class prior?" test runs alongside the existing arm comparisons.

## 5. Wall-clock budget

Three datasets × 30 seeds × 10 epochs × 2 arms (`hodge-mp-residual`, `mlp-baseline`):

| Dataset | Wall time at constant features (lighter than full features) |
|---|---|
| MUTAG | ~3 min |
| PROTEINS | ~20 min |
| NCI1 | ~50 min |
| **Total** | **~75 min** |

Background-runnable.

### Reproduction commands

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
for DS in mutag proteins nci1; do
  python -m benchmarks.hodge \
    --datasets $DS \
    --models hodge-mp-residual mlp-baseline \
    --seeds $SEEDS \
    --n-epochs 10 \
    --constant-features \
    --output notebooks/results/h006_${DS}_constant_30seeds.json \
    --markdown notebooks/results/h006_${DS}_constant_30seeds.md
done

# Resolver: combines the three constant-feature JSONs above with the
# three H001/H002/H003 full-feature 30-seed JSONs to emit the verdicts.
python -m benchmarks.hodge.h006_analysis \
  --constant-results-dir notebooks/results \
  --full-results-dir notebooks/results
```

(The resolver fails loud if any of the six expected JSONs is missing.)

## 6. Resolved outcome (2026-05-22, 30 seeds × 10 epochs × constant-feature ablation, all three datasets)

The constant-feature run completed on MUTAG, PROTEINS, and NCI1. The resolver
(`benchmarks/hodge/h006_analysis.py`) consumes the six artifact JSONs (three
constant-feature outputs from this PR + three full-feature outputs from the
prior H001/H002/H003 30-seed runs) and emits the verdicts below. **All gaps,
p-values, and verdicts in this section are read directly from the per-seed
JSON artifacts now committed to `notebooks/results/h006_{ds}_constant_30seeds.json`;
nothing is hand-derived.**

### Artifact-backed dataset-by-dataset summary

| Dataset | Feature mode | Hodge score | Prior score | Gap | p_BH | Source artifact | Verdict (preregistered tag) |
|---|---|---|---|---|---|---|---|
| mutag | constant | 0.763 | 0.6649 | +0.0983 | 4.53e-06 | `notebooks/results/h006_mutag_constant_30seeds.json` | H23: **rejects** preregistered prediction (signal is present) |
| mutag | full | 0.750 | 0.789 (MLP, not prior) | -0.0395 | — | `notebooks/results/mutag_hodge_ablation_30seeds.json` | anchor for H25 correlation |
| proteins | constant | 0.684 | 0.5957 | +0.0882 | 1.41e-04 | `notebooks/results/h006_proteins_constant_30seeds.json` | H24: significant Hodge>prior; in-between predicate not satisfied |
| proteins | full | 0.686 | 0.675 (MLP, not prior) | +0.0112 | — | `notebooks/results/proteins_hodge_ablation_30seeds.json` | anchor for H25 correlation |
| nci1 | constant | 0.571 | 0.5005 | +0.0707 | 1.93e-05 | `notebooks/results/h006_nci1_constant_30seeds.json` | H22: **supports** (Hodge significantly above prior) |
| nci1 | full | 0.609 | 0.523 (MLP, not prior) | +0.0864 | — | `notebooks/results/nci1_hodge_ablation_30seeds.json` | anchor for H25 correlation |

### Preregistered sub-hypotheses, resolved

- **H22** (NCI1 constant-feature Hodge significantly above class prior): **SUPPORTED.** Observed gap +0.071 at p_BH = 1.93e-05.
- **H23** (MUTAG constant-feature Hodge NOT significantly above class prior): **REFUTED.** Observed gap +0.098 at p_BH = 4.53e-06. MUTAG does carry feature-independent graph-structural signal at this configuration, contrary to the preregistered prediction.
- **H24** (PROTEINS constant-feature Hodge between H23 and H22 levels): **REFUTED.** PROTEINS shows signal of similar magnitude to MUTAG and NCI1; the "in-between" predicate fails because the lower bound (no MUTAG signal) doesn't hold. PROTEINS observed gap +0.088 at p_BH = 1.41e-04.
- **H25** (Spearman ρ on constant-feature `Hodge − prior` gap vs full-feature `Hodge − MLP` gap, predicted > 0): **REFUTED.** Observed ρ = -1.0000 across the three datasets. The constant-feature gap ordering is MUTAG (+0.098) > PROTEINS (+0.088) > NCI1 (+0.071), while the full-feature Hodge−MLP gap ordering is NCI1 (+0.086) > PROTEINS (+0.011) > MUTAG (-0.040). With n=3 the Spearman significance test is uninformative, but the rank-ordering is exactly *inverted* — the constant-feature signal does **not** predict the full-feature gain on this sample. (Spearman ρ is reported descriptively, not as a p-value claim.)

### Scoped final claim (artifact-backed)

> *Under constant-feature control at the tested configuration (matched-capacity 1378-param arms, 30 seeds × 10 epochs × stratified 80/20 split, `hodge-mp-residual` vs `mlp-baseline`), the Hodge MP arm retains feature-independent graph-structural predictive signal on MUTAG (+0.098 over class prior, p_BH = 4.53e-06), PROTEINS (+0.088, p_BH = 1.41e-04), and NCI1 (+0.071, p_BH = 1.93e-05), suggesting that the model is exploiting graph-structural information under this configuration. The constant-feature gap is **not** monotone with the full-feature Hodge-vs-MLP gap across these three datasets (observed Spearman ρ = -1.0000), so the preregistered "topology-signal predicts full-feature gain" hypothesis (H25) is refuted on this sample. Evidence is consistent with an architecture × data-topology interaction under the tested configuration; no causal mechanism claim is asserted.*

The diagnostic isolates **feature-independent graph-structural signal**, not homology specifically — degree distribution, clustering coefficient, connectivity, and Laplacian spectrum are all conflated under this test. Whether homology specifically (as distinct from other graph-structural properties) drives the effect is a separate question not answered by H006.

### What this resolves and what it leaves open

H006 resolves: the Hodge MP arm reads feature-independent graph-structural signal at this configuration on three TUDataset benchmarks. The mechanism is not specific to NCI1.

H006 leaves open:
1. **The mechanism question is harder than predicted.** The constant-feature gap is anti-correlated with the full-feature gap (H25 refuted), so the simple "topology-signal predicts gain" story doesn't hold on this sample. Future work would need to identify what *interaction* between graph structure and node features explains why MUTAG (which has the strongest constant-feature gap) shows no Hodge advantage under full features, while NCI1 (with the weakest constant-feature gap) shows the largest.
2. **Whether homology specifically drives the signal.** A Weisfeiler-Lehman kernel + linear SVM would partially disentangle homology from degree/clustering, but is out of scope here.
3. **Generality beyond three datasets.** With n=3, the rank-order observation has limited statistical force; H25's refutation is a sample observation, not a population claim.

---

## 7. What hypothesis 006 deliberately does NOT do

- Does not implement Weisfeiler-Lehman graph kernels (a more sophisticated topology-signal diagnostic). The constant-feature diagnostic is cheaper and answers the headline question directly: "is the signal in topology, in features, or both?"
- Does not vary architecture beyond `hodge-mp-residual` + `mlp-baseline`. The mechanism test holds those constant.
- Does not run on additional datasets (DD, COLLAB, etc.). One mechanism question at a time per the preregistration discipline.

## 8. If H22 + H23 + H24 + H25 all confirm — hypothesis 007

The natural next step is to *validate the predictive criterion* on a new dataset that wasn't part of mechanism identification. Take a fresh TUDataset (e.g. DD, COLLAB, or NCI109), measure its constant-feature Hodge-vs-class-prior gap, and predict the full-feature Hodge-vs-MLP outcome. Then run the full ablation. If the prediction holds, the framework has a *useful predictive criterion* for when to recommend topology-aware methods.

## 9. Future work (deliberately out of scope here)

Future PRs may explore a broader Algebra → Topology → Geometry architecture, in which structural-class-aware metrics, regime-conditioned distances, and metric-blindness diagnostics could refine the mechanism story. This is **not** implemented in the current PR — H006 is purely the constant-feature ablation described in §1-5. The framing here makes no commitment to any future abstraction beyond what the resolver actually computes from the per-seed JSON outputs.