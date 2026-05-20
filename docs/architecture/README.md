# Architecture Documents

This directory contains the system architecture for TopoGeomML. Each file is a separate concern; together they describe the project as a complete artifact rather than a TDA utility package.

## Reading order

1. **[00-positioning.md](00-positioning.md)** — Why TopoGeomML is not another TDA package, and what it is instead.
2. **[01-five-layer-architecture.md](01-five-layer-architecture.md)** — The five-layer system model.
3. **[SCOPE_LOCK_v0.0.1.md](SCOPE_LOCK_v0.0.1.md)** — What v0.0.1 actually ships and the rationale for every cut.
4. The rest in any order — they are independent skeletons.

## Status overview

| Doc | Concept | Status |
|---|---|---|
| 00-positioning | Strategic positioning above tensor frameworks | DRAFT |
| 01-five-layer-architecture | Backend / data / operators / training / governance | DRAFT |
| 02-emlir | Epistemic ML Intermediate Representation | PROPOSED |
| 03-claim-graphs | Claims, edges, attestations | PROPOSED |
| 04-evidence-bundles | Bundle format, signing, verification | PROPOSED |
| 05-invariant-ledgers | Topological provenance | PROPOSED |
| 06-shape-of-learning | Training-time topology reports | PROPOSED |
| 07-diagnostics-and-audits | Collapse, cohort, regression | PROPOSED |
| 08-autotopology | Search + complexity penalty | PROPOSED |
| 09-benchmark-foundry | Cross-competition harness | PROPOSED |
| 10-applications | Recommender, marketplace, biogeom, sheaf, causal | PROPOSED |
| 11-adapters | TF / PyTorch / JAX strategy | DRAFT |
| 12-governance | Model cards, deployment gates | PROPOSED |
| 13-moat | Strategic differentiation | DRAFT |

In v0.0.1, **all** architecture documents are at DRAFT or PROPOSED status. Implementation lands progressively from v0.1 onward according to the per-document `## Implementation plan` section.

## What's IMPLEMENTED today (v0.0.1)

Only the operator-layer primitives:

- `topogeoml.core.RipsFiltration` — VR persistent homology
- `topogeoml.core.PersistenceImageVectorizer`, `BettiCurveVectorizer`
- `topogeoml.core.SimplicialComplex` + `boundary_matrix` + `is_chain_complex` + `hodge_laplacian` + `betti_numbers`
- `topogeoml.core.cubical_mask_diagnostic`
- `topogeoml.data.graph_to_clique_complex`
- `topogeoml.pipelines.TopologyFeaturePipeline`
- `topogeoml.audits.audit_embedding` (prototype)
- `topogeoml.nn.hodge.HodgeMessagePassing` (requires torch)
- `topogeoml.experiments.load_experiment_config` / `write_results`

Everything else in these documents is forward-looking design intent.

## Architecture diagram (high level)

```
                            ┌──────────────────────────────────────┐
                            │  LAYER 5 — CLAIMS & GOVERNANCE        │
                            │  claim graphs, evidence bundles,      │
                            │  invariant ledgers, model cards,      │
                            │  deployment gates                     │
                            └──────────────────┬────────────────────┘
                                               │
                            ┌──────────────────▼────────────────────┐
                            │  LAYER 4 — EPISTEMIC TRAINING/EVAL    │
                            │  shape-of-learning reports, audits,   │
                            │  autotopology search, benchmark       │
                            │  foundry                              │
                            └──────────────────┬────────────────────┘
                                               │
                            ┌──────────────────▼────────────────────┐
                            │  LAYER 3 — TOPOLOGICAL OPERATORS      │
                            │  filtrations, vectorizers, complexes, │
                            │  Hodge Laplacians, descriptor calculus│
                            │  ← v0.0.1 lives mostly here           │
                            └──────────────────┬────────────────────┘
                                               │
                            ┌──────────────────▼────────────────────┐
                            │  LAYER 2 — GEOMETRIC DATA              │
                            │  point clouds, graphs, complexes,      │
                            │  masks, embeddings; canonical schemas  │
                            │  and ingestion                         │
                            └──────────────────┬────────────────────┘
                                               │
                            ┌──────────────────▼────────────────────┐
                            │  LAYER 1 — BACKEND RUNTIME             │
                            │  numpy / scipy / PyTorch / TF / JAX /  │
                            │  GUDHI / ripser / TopoNetX / FAISS     │
                            │  — composed, never duplicated         │
                            └────────────────────────────────────────┘
```

The crucial design choice: layers 1 and 2 are **borrowed**, not built. Layer 3 is where TopoGeomML adds value at the operator level (v0.0.1). Layers 4 and 5 are the differentiated product.

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) — TOPOLOGICA LLC
