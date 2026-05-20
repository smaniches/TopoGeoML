# Scope Lock — v0.0.1

**STATUS:** AUTHORITATIVE — defines what v0.0.1 ships; modifications require a new scope-lock document for v0.0.2 or v0.1
**Date locked:** 2026-05-20
**Spec source:** TopoGeomML Specification Part I (v0.0.1 scope section)

---

This document records exactly what v0.0.1 implements, what it deliberately does *not* implement, and the reasoning behind every exclusion. It exists so that the next session, the next contributor, or the next year's Santiago can see the scope decisions without rediscovering them.

## What ships in v0.0.1

Eleven items. All implemented. All tested. All documented.

| # | Item | Module | Tests |
|---|---|---|---|
| 1 | Topology feature pipeline for point clouds | `topogeoml.pipelines.TopologyFeaturePipeline` | 11 |
| 2 | Diagram statistics and Betti curve vectorizers | `topogeoml.core.PersistenceImageVectorizer`, `BettiCurveVectorizer` | 7 |
| 3 | Synthetic point cloud classification benchmark | `examples/circles_vs_lines.py` + `examples/run_experiment.py` | (covered by 1+2) |
| 4 | Cubical mask topology diagnostic | `topogeoml.core.cubical_mask_diagnostic` | 10 |
| 5 | Graph to clique complex lift | `topogeoml.data.graph_to_clique_complex` | 11 |
| 6 | Boundary operator validation | `topogeoml.core.is_chain_complex` | 3 |
| 7 | Hodge Laplacian utility | `topogeoml.core.hodge_laplacian`, `betti_numbers` | 9 |
| 8 | Minimal Hodge message passing layer | `topogeoml.nn.hodge.HodgeMessagePassing` | 9 |
| 9 | Embedding topology audit prototype | `topogeoml.audits.audit_embedding` | 9 |
| 10 | YAML experiment configs and JSON outputs | `topogeoml.experiments.load_experiment_config` / `write_results` | 9 |
| 11 | Documentation with explicit limitations | `LIMITATIONS.md` + `docs/architecture/` | — |

Plus carry-over from the initial scaffold: 10 diagram-contract tests, 9 filtration tests, 7 vectorizer tests. **Total: 104 tests passing.**

## Mathematical contracts validated

- ∂² = 0 on triangle, tetrahedron, two-triangles-glued, K₅ clique complex
- Hodge β recovery: D² (1,0,0); S¹ (1,1); S² (1,0,1); disjoint vertices; disjoint triangles
- L₀ on path graph = standard combinatorial graph Laplacian (exact)
- Cubical β: disk (1,0); annulus (1,1); two disks (2,0); disk-with-two-holes (1,2)
- Clique complex topology: C₄ → (1,1); K₄ at max_dim=2 → S² → (1,0,1)
- Hodge MP forward/backward, gradient flow through `torch.sparse.mm`, layer stacking
- Embedding audit: single circle → β₁=1; two circles → β₁=2 with β₀=2

## What v0.0.1 does NOT ship — and why

The following are deliberately excluded from v0.0.1. Each exclusion has a specific reason; do not "just add" any of these in v0.0.x patch releases.

### Layer 3 — additional operators

| Excluded | Reason |
|---|---|
| Bottleneck distance | Requires careful diagram normalization; not load-bearing for v0.0.1 use cases |
| Wasserstein distance | Same; OT solver dependency adds install weight |
| Persistence landscapes (Bubenik) | Vectorization is sufficient via PI + Betti; landscapes are v0.1 luxury |
| Persistence entropy | Trivial to add; deferred only to keep v0.0.1 scope honest |
| Alpha complex filtration | Requires GUDHI; adds heavyweight install for one filtration |
| Cubical filtration on real-valued images | The diagnostic is binary-only; full real-valued PH is v0.1 |
| 3D β₁ in `cubical_mask_diagnostic` | Requires cubical persistence backend (cripser / GUDHI); not just scipy.ndimage |
| Distance matrix input to `TopologyFeaturePipeline` | Trivial extension but adds a code path that needs testing |
| Multi-rank simplicial neural network | The Hodge MP layer is the building block; the full architecture is v0.2 |

### Layer 4 — epistemic training and evaluation

All of layer 4 is excluded from v0.0.1. The `audit_embedding` prototype is layer-3-with-a-veneer; it does not yet do cohort splits, training-time monitoring, regression assertions, or autotopology search.

**Reason:** layer 4 depends on a stable layer 3. Building layer 4 before layer 3 has been used in anger guarantees that layer 3 will need refactoring within months, breaking the layer-4 surface. The first user of `audit_embedding` is the strongest forcing function for layer-3 stability; only after that has happened (v0.1) does layer 4 get built.

### Layer 5 — claims and governance

All of layer 5 is excluded from v0.0.1. The JSON output from `write_results` is the **seed** of an evidence bundle but is not an evidence bundle yet (no signing, no hashing, no formal schema, no claim graph composition).

**Reason:** see above. Plus: the governance layer needs a stable mental model from at least one real consumer before its surface is designed. The candidate first consumer is a CI gate that blocks a model deployment if its β₁ regressed by more than 2σ relative to the previous release. v0.1 picks one consumer and builds the minimum claim graph + evidence bundle to serve it.

### EMLIR

Excluded from v0.0.1. The Epistemic ML Intermediate Representation is an attractive abstraction but unproven. Strong prior that EMLIR is over-engineering for v0.1 and that layers 4 and 5 can be built directly on Pydantic + JSON schemas. EMLIR's earliest possible introduction is v0.2, and only if its absence becomes a real pain point in v0.1 work.

### TF / JAX adapters

Excluded. Only PyTorch is exercised (in the Hodge MP layer). TF and JAX adapters appear in `pyproject.toml` only as optional extras; no code paths use them in v0.0.1.

**Reason:** PyTorch is the right first backend for research flexibility. Adding TF/JAX before PyTorch integration has been used in real workflows is speculative API design.

### Real-world dataset adapters

Excluded. v0.0.1 ships exactly one dataset: synthetic circles vs lines. No MUTAG, no PROTEINS, no ZINC, no Open Graph Benchmark loaders.

**Reason:** dataset loaders are tedious to write well and easy to write badly. Postpone until v0.1 has a clear set of three or four benchmark targets selected.

### GPU acceleration

Excluded. All compute is CPU.

**Reason:** the v0.0.1 surface is fast enough on CPU for the synthetic benchmark; GPU acceleration is a v0.2 concern, possibly via `ripserplusplus` integration or PyTorch sparse on CUDA.

### MLflow / W&B / external trackers

Excluded. The JSON output is the only tracker.

**Reason:** the JSON output IS the canonical truth. External trackers can be added as one-way adapters (`json → mlflow`) once the schema is stable.

### CLI

Excluded. `examples/run_experiment.py` is the closest thing to a CLI, but it's a Python script, not a `topogeoml` command.

**Reason:** v0.1 will land a `topogeoml` console-script entry point that wraps `run_experiment` plus the audit and benchmark workflows.

### FastAPI service (`topogeoml/services/`)

Excluded. The subpackage exists as a stub.

**Reason:** premature. The shape of the right API only becomes clear after layer 5 has shipped.

## How to add scope

If a future contributor wants to add capability beyond this lock:

1. **In a v0.0.x patch**: only if it is a strict bug fix or documentation-only change. No new features.
2. **In v0.1**: must be motivated by a real use case (Kaggle competition, paper experiment, deployment pipeline). Speculative additions are rejected.
3. **In v0.2 and beyond**: must be tied to a section of an architecture document. Drive-by additions without an architecture-doc home are rejected.

## Versioning policy reminder

`v0.x` is **pre-stable**. APIs may change. Pin exact versions downstream. See `LIMITATIONS.md` for the version-semantics policy.

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) — TOPOLOGICA LLC
