# Strategic Moat

**STATUS:** DRAFT
**Phase:** binding from v0.0.1; reviewed each minor release
**Spec source:** TopoGeomML Specification Part II §24

---

## What it is

The strategic moat is the set of advantages that make TopoGeomML defensible against competitors over a multi-year horizon. The moat is *not* features. Features can be copied. The moat is the combination of mathematical, technical, research, and commercial assets that make TopoGeomML hard to displace once adopted.

## The four moats

### 1. Mathematical moat

TopoGeomML carries proprietary mathematical contributions that are not yet in the public competitive landscape:

- **Drift tensor `D: M → TM`** — Riemannian correction layer applied to embeddings. (TOPOLOGICA-proprietary; v0.2.)
- **PH metric cascade** (Euclidean → Spectral → Fermat, selected by intrinsic-dimension estimates) — the right-metric-first discipline that prevents the most common PH failure mode (using Euclidean PH on data where d_int ≈ d_amb). (Validated; published as paper.)
- **K-FAC curvature filtration on transformer MLPs** — TDA applied to learned curvature structure. (Paper in submission.)
- **Topographic prominence as a parameter-free shell detector** — applied to nuclear physics; transferable to ML latent-space cluster discovery. (Paper in submission.)
- **P00 invariant classification framework** — a unifying classification system for topology-aware models. (Long-term research arc.)

Each of these is a publication-track contribution from a years-long research arc. They are not easily replicated by a competitor reading the docs.

### 2. Technical moat

- **Provenance discipline everywhere.** Every fitted object carries a `FitProvenance`; every diagram carries a `DiagramProvenance`; every JSON output carries an environment snapshot. Retrofitting this discipline into an existing TDA package is harder than writing one from scratch with it baked in.
- **Verification gate** (interpolator check, correction audit, derivative inheritance, validation provenance) applied uniformly. Most ML libraries do not have such a gate; adding one requires reviewing every operator.
- **The five-layer architecture** with strict adapter discipline. Cleanly separating math from frameworks (see [11-adapters.md](11-adapters.md)) avoids the tar-pit that catches projects that try to be "PyTorch-but-with-topology".

### 3. Research moat

- 14+ unpublished papers across TDA, nuclear physics, information geometry, chirality/entropy, P00 invariant classification. These are publication-track contributions tied to the platform.
- Ongoing relationships with BlueDot Impact (Biosecurity, AI Safety, Future of AI cohorts), the Kairos group, and the AI safety research community.
- A persistent research voice and ORCID identity (0009-0005-6480-1987) tied to the platform.

A competitor would need to either reproduce this research independently or partner with someone who already has — both are slow.

### 4. Commercial moat

- The **regulated-industry market** for model evidence (healthcare, finance, defense, public-sector) is large, growing, and currently unserved at the framework level.
- **Switching costs once deployed**: once a customer's deployment pipeline gates on TopoGeomML evidence bundles, replacing TopoGeomML means re-validating every model against a new evidence schema. This is non-trivial.
- **Network effects via the Benchmark Foundry**: as the Foundry's leaderboard grows, every new published comparison cites TopoGeomML as the harness, increasing visibility and adoption.
- **First-mover at the epistemic-layer positioning**. No incumbent is currently building this. Six months of head-start in positioning is worth more than six months of head-start in features.

## How to defend the moat

- **Publish the math.** Drift tensor, metric cascade, K-FAC filtration — get them to arXiv and Zenodo. Published math is harder to claim than unpublished math.
- **Patent the system, not the math.** The system-level innovations (provenance-disciplined topological audit pipelines, evidence-bundle composition for ML governance) are patentable. The underlying math is not, and shouldn't be.
- **Be the reference implementation.** When somebody publishes a topology-aware ML paper in 2027, TopoGeomML should be the package they cite for filtration + audit + evidence handling.
- **Cultivate the regulated-industry channel early.** A single deployment at a regulated-industry customer (healthcare, finance) is worth more than ten academic citations.

## What erodes the moat

- **Letting feature creep blur the positioning** (see [00-positioning.md](00-positioning.md)). Every "let's add an autograd engine" PR weakens the moat.
- **Failing to ship.** The 14+ unpublished papers are a moat *only if published*. Unpublished work is unverifiable and uncited.
- **Allowing a competitor to position first.** If a major tensor framework adds a `tf.audit` or `torch.governance` submodule that does even a poor version of layer 4 + 5, the positioning advantage evaporates.

## The single most important defensive move

Ship the v0.0.1 to PyPI, claim the namespace, tag the release. The moat exists only if the package exists publicly. An internal-only TopoGeomML, however brilliant, is not a moat.
