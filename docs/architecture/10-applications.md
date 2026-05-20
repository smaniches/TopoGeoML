# Application Tracks

**STATUS:** PROPOSED (all tracks)
**Phase:** v0.1+ depending on track
**Spec source:** TopoGeomML Specification Part II §15-20

---

## What it is

Application tracks are *vertical* applications of the TopoGeomML stack to specific problem domains. Each track is a self-contained subproject with its own datasets, baselines, claims, and product surface — built on the horizontal stack defined by the other architecture documents.

## Why have them

Horizontal infrastructure without vertical demonstrations is impossible to sell, fund, or publish. Each application track is the answer to "what would I actually do with this in domain X?" The horizontal stack is the substrate; the tracks are the proof.

## Tracks

### Recommender and personalization audit layer

**Phase:** v0.2

Apply layer-4 cohort audits and layer-5 invariant ledgers to recommender embeddings: track user-embedding latent topology, detect collapse, audit cohort-specific recommendation quality. Existing recommender stacks (Pinterest's Pixie, TikTok's two-tower, Amazon's collaborative filtering) emit user embeddings; the track ingests them and produces a deployment-gate-ready report.

### Marketplace geometry health

**Phase:** v0.2

For two-sided marketplaces, audit the geometric structure of buyer/seller embeddings to detect imbalances. β₁ of seller-side embeddings detects degenerate seller clusters; cohort comparison detects systematic disadvantage for new entrants.

### Topological exposure diversity

**Phase:** v0.2

Quantify diversity of recommendations using descriptor entropy on the manifold of recommended items, not just on item categories. Two systems with identical category-level diversity can have different topology-level diversity. Useful for "is my recommender narrowing the user's exposure even if the categories look balanced?"

### BioGeom

**Phase:** v0.2 (depends on which subtrack)

Domain-specific applications to biology:

- Protein structure / function inference (drift tensor + Hodge MP on residue graphs)
- Single-cell RNA-seq cell-type topology (β₁ of UMAP embeddings; existing TOPOLOGICA scRNA paper)
- Drug-target binding via topology-aware embeddings

### Sheaf consistency track

**Phase:** v1.0+

Multi-source data fusion via sheaf cohomology. Each data source is a section of a sheaf over a problem graph; sheaf cohomology measures inconsistency. Applies to: federated learning consistency, multi-omics integration, multi-rater image labels.

### Causal geometry

**Phase:** v1.0+

Use topology to detect interventional changes: does an intervention (a new feature, a new policy, a treatment) preserve the topology of the affected subspace? If β₁ changes only in the treated cohort, that's structural evidence for a causal effect beyond mean shift.

## Open questions

- [ ] **Track prioritization**: which track is the v0.1 first-application demo? Strong prior: synthetic shape classification + one BioGeom subtrack (likely scRNA or protein-function).
- [ ] **External datasets**: do we package dataset loaders inside `topogeoml.data`, or as a separate `topogeoml-datasets` package?
- [ ] **Track ownership**: does each track get its own subpackage (`topogeoml.recommender`), or its own repo? Leaning toward separate repos to keep the core small.

## Implementation plan

Each track is its own roadmap. The shared substrate (layers 1-5) must be solid before any track is meaningful.
