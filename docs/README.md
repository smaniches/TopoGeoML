# TopoGeomML Documentation

This directory holds long-form documentation that lives in-repo and ships with each release.

## Structure

```
docs/
├── README.md                    ← you are here
└── architecture/                ← architecture & specification documents
    ├── README.md                  Architecture index and reading order
    ├── SCOPE_LOCK_v0.0.1.md       What v0.0.1 ships and why; rationale for cuts
    ├── 00-positioning.md          Strategic positioning: epistemic ML operating layer
    ├── 01-five-layer-architecture.md  The five-layer model
    ├── 02-emlir.md                EMLIR — Epistemic ML Intermediate Representation
    ├── 03-claim-graphs.md         Claim graphs and attestations
    ├── 04-evidence-bundles.md     Evidence bundle format and verification
    ├── 05-invariant-ledgers.md    Invariant ledgers and topological provenance
    ├── 06-shape-of-learning.md    ShapeOfLearning reports
    ├── 07-diagnostics-and-audits.md  Collapse, cohort, regression diagnostics
    ├── 08-autotopology.md         AutoTopology search + complexity penalty
    ├── 09-benchmark-foundry.md    Benchmark Foundry
    ├── 10-applications.md         Recommender, marketplace, exposure, biogeom, sheaf, causal
    ├── 11-adapters.md             TF, PyTorch, JAX adapter strategy
    ├── 12-governance.md           Model cards from evidence; deployment gates
    └── 13-moat.md                 Strategic moat — mathematical, technical, commercial
```

## Reading order

If you are new to the project:

1. Top-level `README.md` for the v0.0.1 surface and quick-start.
2. `architecture/00-positioning.md` for the strategic frame.
3. `architecture/01-five-layer-architecture.md` for the system shape.
4. `architecture/SCOPE_LOCK_v0.0.1.md` to understand what is and isn't built yet.
5. The other architecture docs in any order — they are independent skeletons today.

## Status conventions

Every architecture document declares a status header:

- `STATUS: IMPLEMENTED` — code exists in `topogeoml/`, tests cover it
- `STATUS: PARTIAL` — some pieces in code, full design still drafted
- `STATUS: DRAFT` — design only, no implementation
- `STATUS: PROPOSED` — concept stage, open to redesign

A document with `STATUS: DRAFT` or `STATUS: PROPOSED` is a contract for future work, not a description of present capability. Read accordingly.

## How these documents relate to code

Each architecture document names the modules it specifies. When implementation lands, the doc's status is upgraded and a `## Implementation` section is added linking to the module + tests. Documents that describe shipped features serve as the authoritative spec; documents that describe DRAFT features serve as design intent.

## Citation and provenance

Architecture documents originate from the **TopoGeomML Specification (Part I + Part II)** canvas authored by Santiago Maniches. Section numbers in the canvas correspond to the `## Spec source` line at the top of each architecture doc.

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) — TOPOLOGICA LLC
