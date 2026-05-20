# Diagnostics and Audits

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §9, §10, §11

---

## What it is

A family of layer-4 components that probe a trained or partially-trained model for structural failures: representation collapse, cohort-specific topology degradation, regression against expected invariants.

Three subsystems:

1. **Representation collapse diagnostics** — detect when the latent space has lost effective dimensionality.
2. **Cohort topology audits** — re-run `audit_embedding` per subgroup and compare against a reference cohort or each other.
3. **Latent topology regression tests** — pytest-style assertions that specific Betti numbers, persistence-entropy values, or descriptor statistics hold for named cohorts.

## Why it matters

Aggregated metrics hide subgroup failures. A recommender with 0.78 AUC overall might have 0.40 AUC for a minority cohort; the topology of the minority cohort's embedding might also be degenerate. These three subsystems are how you find that out before deployment.

Collapse detection is particularly important during training-time monitoring: it provides an early signal that something has gone wrong with the regularization, the architecture, or the data, *before* the loss curve makes it obvious.

## Core constructs

### Representation collapse

- `CollapseSignal` — named indicator (rank deficiency; persistence entropy floor; NN-distance variance floor)
- `CollapseDetector` — composes signals; returns severity
- thresholds calibrated per architecture family

### Cohort audits

- `Cohort` — labeled subset of the dataset
- `CohortAudit` — per-cohort `EmbeddingTopologyAudit` plus pairwise comparison
- `DivergenceReport` — which cohorts diverge most in topology

### Regression tests

- `TopologyAssertion` — declarative invariant requirement; consumed by pytest
- `RegressionSuite` — a YAML/JSON list of assertions tied to a model identifier

## Open questions

- [ ] **Collapse thresholds**: what's a sensible default for "the H₀ persistence entropy has collapsed"? Calibration on a reference dataset?
- [ ] **Cohort definition**: where do cohorts come from — config, dataset metadata, or user-provided indicator functions?
- [ ] **Regression-test ergonomics**: pytest collection of TopologyAssertion suites from YAML; or first-class pytest plugin?
- [ ] **Causal cohort analysis**: when cohort divergence appears, is it caused by data imbalance or model behavior? (overlaps with [10-applications.md](10-applications.md) causal-geometry track.)

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | Seed: `audits.audit_embedding` is the single-cohort, single-shot prototype |
| v0.1 | `CohortAudit`, `CollapseDetector` (rank + persistence-entropy signals), `TopologyAssertion` with pytest plugin |
| v0.2 | NN-distance variance signal; auto-cohort discovery |
| v1.0 | Causal cohort analysis integrated with [18-causal-geometry] direction |
