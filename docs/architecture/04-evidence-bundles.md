# Evidence Bundles

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §6
**Depends on:** [03-claim-graphs.md](03-claim-graphs.md)

---

## What it is

An evidence bundle is a versioned, content-addressed, signed container of artifacts attached to a specific model version. It is the unit of exchange between training systems (producers) and governance systems (consumers).

Concretely: a zip-or-tar archive with a manifest JSON listing every artifact (persistence diagrams, audit reports, training-time topology trajectories, configuration echoes, environment snapshots) by content hash. The manifest itself is signed.

## Why it matters

A claim graph (see [03-claim-graphs.md](03-claim-graphs.md)) names evidence by hash. The evidence bundle is **what those hashes resolve to**. Without bundles, a claim graph is a graph of dangling pointers; with bundles, claims are independently verifiable.

Bundles also make evidence portable. A model trained on team A's infrastructure can ship its evidence to team B's deployment system without requiring shared databases, secret keys, or live API calls.

## Bundle format (draft)

```
my_model_v3.7.evidence/
├── manifest.json              ← signed; lists all artifacts by sha256
├── manifest.json.sig          ← signature (sigstore / GPG / in-tree)
├── claims.json                ← claim graph
├── artifacts/
│   ├── {sha256}.diagram.json  ← persistence diagram + provenance
│   ├── {sha256}.audit.json    ← embedding audit report
│   ├── {sha256}.shape.json    ← shape-of-learning trajectory
│   ├── {sha256}.config.yaml   ← experiment config
│   └── {sha256}.env.json      ← environment snapshot
└── README.md                  ← human-readable summary auto-generated
```

## Core constructs

- `Bundle` — the in-memory representation
- `Manifest` — the signed top-level listing
- `Artifact` — a content-addressed payload with a known schema
- `BundleBuilder` — accumulator used by training pipelines
- `BundleVerifier` — verifies signature and per-artifact hashes

## Open questions

- [ ] **Signing infrastructure**: sigstore (modern, transparent log) vs minisign (simple) vs GPG (mature). Lean toward sigstore.
- [ ] **Bundle size**: a per-epoch shape-of-learning trajectory could be megabytes. Compression? Subsampling at write time?
- [ ] **Differential bundles**: can release v3.8's bundle reference v3.7's bundle and only ship what changed?
- [ ] **Bundle schema versioning**: how do older verifiers handle newer bundles?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | Seed: `topogeoml.experiments.write_results` produces a single JSON that is a degenerate one-artifact bundle |
| v0.1 | Multi-artifact bundle (zip), manifest, unsigned; `BundleBuilder` API |
| v0.2 | Signed bundles (sigstore); bundle verification CLI |
| v1.0 | Differential bundles; bundle schema versioning |
