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

## 6. Resolved outcome

*Pending — preregistration written before any execution.*

---

## 7. What hypothesis 006 deliberately does NOT do

- Does not implement Weisfeiler-Lehman graph kernels (a more sophisticated topology-signal diagnostic). The constant-feature diagnostic is cheaper and answers the headline question directly: "is the signal in topology, in features, or both?"
- Does not vary architecture beyond `hodge-mp-residual` + `mlp-baseline`. The mechanism test holds those constant.
- Does not run on additional datasets (DD, COLLAB, etc.). One mechanism question at a time per the preregistration discipline.

## 8. If H22 + H23 + H24 + H25 all confirm — hypothesis 007

The natural next step is to *validate the predictive criterion* on a new dataset that wasn't part of mechanism identification. Take a fresh TUDataset (e.g. DD, COLLAB, or NCI109), measure its constant-feature Hodge-vs-class-prior gap, and predict the full-feature Hodge-vs-MLP outcome. Then run the full ablation. If the prediction holds, the framework has a *useful predictive criterion* for when to recommend topology-aware methods.

## 9. Future work (deliberately out of scope here)

Future PRs may explore a broader Algebra → Topology → Geometry architecture, in which structural-class-aware metrics, regime-conditioned distances, and metric-blindness diagnostics could refine the mechanism story. This is **not** implemented in the current PR — H006 is purely the constant-feature ablation described in §1-5. The framing here makes no commitment to any future abstraction beyond what the resolver actually computes from the per-seed JSON outputs.