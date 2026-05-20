# Five-Layer Architecture

**STATUS:** DRAFT
**Phase:** binding from v0.0.1; layers 1-3 partially implemented, 4-5 specified
**Spec source:** TopoGeomML Specification Part II §3
**Depends on:** [00-positioning.md](00-positioning.md)

---

## Summary

TopoGeomML is a five-layer system. Each layer has a single responsibility, a narrow interface to the layer below, and a clear ownership boundary: which parts are *borrowed* from existing libraries, which are *adapted*, and which are *owned* by TopoGeomML.

The discipline of the architecture: **never re-implement what a competent neighbouring library already does**. The product is the upper layers, not the lower ones.

## Layer 1 — Backend runtime

**Responsibility:** numerical compute, sparse linear algebra, autograd, tensor manipulation, GPU acceleration.

**Owners (borrowed, not built):**

- `numpy`, `scipy` — dense and sparse linear algebra
- `torch`, `tensorflow`, `jax` — autograd-enabled tensors and training
- `ripser`, `gudhi`, `cripser` — persistent homology backends
- `networkx`, `scipy.sparse.csgraph` — graph algorithms
- `faiss` — approximate nearest-neighbour search
- `toponetx` — combinatorial complex types

**TopoGeomML's contribution at this layer:** none. We *compose*. The only thing TopoGeomML writes here is **adapters**: thin wrapper modules that normalize the interface of these libraries so the upper layers don't depend on backend identity. See [11-adapters.md](11-adapters.md).

**Failure mode to avoid:** "We need our own faster Rips backend." If true, contribute the optimization upstream to ripser. If ripser is fundamentally unsuited, write a new backend in a *separate package*, not inside TopoGeomML.

## Layer 2 — Geometric data

**Responsibility:** canonical representations of geometric data and the ingestion paths into them.

**Canonical types:**

- **Point cloud**: `NDArray[float64]` of shape `(n_points, ambient_dim)`
- **Distance matrix**: `NDArray[float64]` of shape `(n_points, n_points)`, symmetric, zero diagonal
- **Graph**: `networkx.Graph` or scipy sparse adjacency
- **Simplicial complex**: `topogeoml.core.SimplicialComplex`
- **Cell complex / combinatorial complex**: deferred to TopoNetX
- **Binary mask**: `NDArray[bool]` of shape `(H, W)` or `(D, H, W)`
- **Real-valued image / volume**: `NDArray[float]` of shape `(H, W)` or `(D, H, W)` (v0.1)
- **Embedding matrix**: `NDArray[float64]` of shape `(n_points, embedding_dim)`

**Adapters provided in v0.0.1:** `data.graph_to_clique_complex` lifts a graph into a SimplicialComplex.

**Adapters planned for v0.1:**

- `data.pointcloud_to_filtration` — calibrated `max_edge_length` selection via intrinsic-dimension estimation
- `data.image_to_cubical_filtration` — real-valued cubical sub-level set filtration
- `data.embeddings_from_torch` — adapter from `torch.Tensor` to embedding ndarray (zero-copy where possible)
- `data.embeddings_from_tf` — adapter from `tf.Tensor`
- `data.dataset_loaders` — adapters for standard benchmark datasets (MUTAG, PROTEINS, etc.)

**Contracts:**

- Every canonical type carries a known dtype and shape.
- Every adapter is *pure*: input in canonical form → output in canonical form, no side effects.
- Adapters are stateless when possible; when state is required (e.g. a calibration), they expose a fit/transform interface so the verification gate applies.

## Layer 3 — Topological operators

**Responsibility:** the topological mathematics. Filtrations, complexes, boundary operators, Hodge structure, descriptors, and the verification machinery that makes their output trustworthy.

**Owned by TopoGeomML:**

- `core.diagrams.PersistenceDiagram` + `DiagramProvenance`
- `core.filtrations.RipsFiltration`
- `core.cubical.cubical_mask_diagnostic` (v0.0.1: binary masks only; real-valued v0.1)
- `core.complexes.SimplicialComplex` + `boundary_matrix` + `is_chain_complex` + `hodge_laplacian` + `betti_numbers`
- `core.vectorizers.PersistenceImageVectorizer` + `BettiCurveVectorizer`
- `core.distances` (v0.1) — bottleneck, Wasserstein, sliced approximations
- `core.descriptors` (v0.1) — landscapes, persistence entropy, kernel-based features

**Why owned, not borrowed:** the layer-5 governance contract requires that every topological quantity carry verifiable provenance and pass an interpolator / leakage check. Existing libraries (GUDHI, giotto-tda) implement the math, but do not enforce the provenance discipline. TopoGeomML wraps them where it can (ripser via `RipsFiltration`) and writes its own where the discipline is load-bearing (the boundary-operator construction; `is_chain_complex`).

**Verification gate (applies to every operator):**

1. **Interpolator check** — no operator is an exact interpolator over its input.
2. **Correction audit** — no operator leaks training-fold information into transform.
3. **Derivative inheritance** — operators that depend on multiple sub-results require every sub-result to be independently valid.
4. **Provenance** — every operator records its inputs, parameters, and backend identity into a `DiagramProvenance` or `FitProvenance` record.

See [03-claim-graphs.md](03-claim-graphs.md) for how these provenance records compose into claims.

## Layer 4 — Epistemic training and evaluation

**Responsibility:** turn raw operator outputs into structured judgements about training and model behaviour.

**Components (all PROPOSED / DRAFT, v0.1+):**

- **ShapeOfLearning reports** — per-epoch β₀ / β₁ trajectories, drift summaries; see [06-shape-of-learning.md](06-shape-of-learning.md)
- **Representation collapse diagnostics** — singular-value floor, persistence-entropy floor, NN-distance variance; see [07-diagnostics-and-audits.md](07-diagnostics-and-audits.md)
- **Cohort topology audits** — re-run audit per subgroup, compare against reference
- **Latent topology regression tests** — pytest-style assertions over expected Betti numbers
- **AutoTopology search** — grid / Bayesian search over filtration parameters with complexity penalty; see [08-autotopology.md](08-autotopology.md)
- **Benchmark Foundry** — reproducible cross-task harness; see [09-benchmark-foundry.md](09-benchmark-foundry.md)

**v0.0.1 seed:** `topogeoml.audits.audit_embedding` is the prototype of cohort topology audits. The full subgroup-aware machinery lands in v0.1.

**Output contract:** every layer-4 component produces a structured artifact (JSON or Python dataclass with a JSON schema) that is *consumed by layer 5*. No layer-4 output is human-only; everything must be machine-parseable for evidence-bundle composition.

## Layer 5 — Claims and governance

**Responsibility:** make decisions about deployment based on evidence.

**Components (all PROPOSED, v0.1+):**

- **Claim graphs** — directed graphs of claims, attestations, and supporting evidence; see [03-claim-graphs.md](03-claim-graphs.md)
- **Evidence bundles** — signed, versioned collections of layer-4 artifacts attached to a specific model version; see [04-evidence-bundles.md](04-evidence-bundles.md)
- **Invariant ledgers** — append-only logs of topological invariants observed across training and deployment, with regression-detection; see [05-invariant-ledgers.md](05-invariant-ledgers.md)
- **Model card generation** — produce a model card from an evidence bundle, not from a notebook; see [12-governance.md](12-governance.md)
- **Deployment gates** — CI hooks that block model promotion when a claim graph's invariants regress

**v0.0.1 seed:** the JSON output schema in `topogeoml.experiments.write_results` is the seed of an evidence bundle. It carries config echo, environment snapshot, timestamp, and results — the minimum viable evidence record.

**Output contract:** every layer-5 artifact is signed (v0.2+: cryptographically; v0.1: hashed content). Every layer-5 decision is reproducible from the artifacts alone — no external state, no notebook context.

## Cross-layer rules

1. **Higher layers never reach below layer 3** without going through the canonical type interfaces. A trainer at layer 4 does not import `numpy` directly to compute distances; it uses the operator at layer 3 that does, which then uses numpy.
2. **Provenance flows upward by composition.** A layer-5 evidence bundle contains layer-4 reports which contain layer-3 provenance records which name layer-1 / layer-2 versions. Nothing is lost on the way up.
3. **Backend identity is layer-3 hidden state.** A layer-4 component that wants Hodge homology asks `hodge_laplacian(complex, k)`; it does not know or care whether the boundary matrices came from our internal construction or from a TopoNetX bridge.
4. **Adapters are bidirectional.** Every adapter that brings external data *in* must have a dual that exports back out. A model card produced from an evidence bundle must be re-ingestible as input for a downstream verifier.

## Open questions

- [ ] Should layer 4 be split into "evaluation" and "training-time monitoring"? Probably yes for v0.2 — they have different latency contracts.
- [ ] Does layer 5 own the **signing infrastructure** (cryptographic primitives, key management), or is that another borrowed library (`cryptography` / `sigstore`)? Almost certainly borrowed.
- [ ] What is the **canonical wire format** for crossing layer boundaries? Candidate: a Pydantic-validated JSON object with a discriminator for type. Decision deferred to v0.1.

## Implementation map

| Layer | v0.0.1 | v0.1 | v0.2 | v1.0 |
|---|---|---|---|---|
| 1: Backend | borrowed | borrowed | borrowed | borrowed |
| 2: Data | `graph_to_clique_complex` | `pointcloud_to_filtration`, `image_to_cubical`, framework adapters | dataset loaders | benchmark dataset registry |
| 3: Operators | Rips, complexes, Hodge, vectorizers | metric cascade, distances, more descriptors | differentiable PH | GPU-batched |
| 4: Training/eval | `audit_embedding` (proto) | ShapeOfLearning, cohort audits, regression tests, autotopology | benchmark foundry | full epistemic training loop |
| 5: Governance | JSON output (seed) | claim graphs, evidence bundles | invariant ledgers, model cards | signed deployment gates |
