# Limitations of TopoGeomML v0.0.1

This document lists what TopoGeomML v0.0.1 **does not do**, what it does **only partially**, and the **failure modes** the user should expect. Honest accounting is part of the contract: every claim in the codebase must name its validation, and every absence must be named here.

If you hit a behavior that is not listed here and is not what the docstring promises, that is a bug — open an issue.

---

## 1. Scope limits (deliberate v0.0.1 cuts)

### 1.1 Only Vietoris–Rips persistent homology

`RipsFiltration` is the only filtration shipped. **Cubical persistence on real-valued images, alpha complexes, witness complexes, lower-star filtrations on functions, and the PH metric cascade are not implemented in v0.0.1.** The cubical *diagnostic* (`cubical_mask_diagnostic`) handles binary masks but does not produce a persistence diagram.

### 1.2 Only two vectorizers

Persistence images (Adams et al. 2017) and Betti curves. **Persistence landscapes (Bubenik 2015), persistence entropy, kernel-based diagram features, and silhouettes are deferred.**

### 1.3 Single point cloud per fit/transform call

`TopologyFeaturePipeline` operates on batches of point clouds, but each cloud is processed serially. **No GPU acceleration. No batched persistence computation.** For batches of >100 clouds with >500 points each, expect minutes of compute.

### 1.4 No persistence-diagram distances

`bottleneck_distance` and `wasserstein_distance` are not in v0.0.1. The `core/distances.py` module is reserved for v0.1. Until then, diagrams can only be vectorized and compared in feature space.

### 1.5 Hodge Laplacian: real coefficients only

Hodge Laplacian construction uses real coefficients. **Z/2, Z/p, and field-coefficient generalizations are not provided.** This is the correct choice for spectral methods (Z/2 makes ∂ᵀ∂ useless as a Laplacian), but it means Hodge homology may disagree with persistent homology computed over Z/2 in degenerate cases.

### 1.6 Cubical mask diagnostic: 2D only for β₁

`cubical_mask_diagnostic` reports β₀ in 2D and 3D but **β₁ only in 2D**. The 3D case requires a full cubical persistence backend (GUDHI's `cubical_complex` or `cripser`); 3D inputs return `betti_1=-1` as a sentinel. Coming in v0.1.

### 1.7 Embedding audit: prototype only

`audit_embedding` is explicitly a prototype. **The persistence-significance threshold defaults to `2 × median nearest-neighbor distance`, which is a heuristic, not a calibrated noise floor.** The PH metric cascade (Euclidean → Spectral → Fermat with d_int/d_amb selection) that would replace this heuristic lands in v0.1. Audit results are diagnostic, not metric-correct.

### 1.8 Hodge message passing: minimal, not state-of-the-art

`HodgeMessagePassing` implements one round of `x' = σ(L̃ₖ @ x @ W + b)`. **No multi-rank simplicial neural network (SCN), no equivariance machinery, no batching across complexes, no temporal extension, no attention.** This is a building block, not a competitive architecture. Real simplicial NNs (Ebli et al. 2020; Bunch et al. 2020) require composing this with up/down separation, residual connections, and dimension-mixing layers.

### 1.9 Experiment configs support exactly one dataset and one pipeline kind

`load_experiment_config` and `run_experiment.py` recognize only `dataset.name = "synthetic_shapes"` and `pipeline.kind = "topology_feature"`. Real-world datasets, alternative pipelines, and multi-stage benchmarks land in v0.1.

---

## 2. Failure modes you will hit

### 2.1 Rips explodes for large point clouds without `max_edge_length`

ripser's complex size scales combinatorially. For point clouds with `n > 2000` and unset `max_edge_length`, expect seconds to minutes of compute and gigabytes of RAM. **Always set a finite `max_edge_length` calibrated to your data scale.** v0.1 will provide automatic calibration via intrinsic-dimension estimation.

### 2.2 PersistenceImageVectorizer outputs are sensitive to `fallback_max` and `sigma`

`fallback_max` substitutes for infinite deaths and bounds the pixel grid. **The pipeline auto-estimates this from the training-batch scale, but for novel test inputs at different scales, you may need to fit-then-transform within each CV fold (which the pipeline does by default if used inside `sklearn.model_selection`).** `sigma` controls the Gaussian smoothing bandwidth — too small produces sparse spiky images, too large washes out structure. Defaults (`sigma=0.1`, `resolution=20`) are reasonable for unit-scale point clouds; rescale your data or tune both for other regimes.

### 2.3 Betti curve discriminability degrades with noise

Above noise levels of ~10% of the data diameter, Betti curves count spurious short-lived loops at the noise scale. The `tests/test_vectorizers.py::test_betti_curve_h1_line_much_smaller_than_circle` test documents this: σ=0.05 noise on a 50-point line produces ~6 grid-cells of spurious β₁ signal. Use comparison ratios, not absolute thresholds, when this matters.

### 2.4 Clique-complex enumeration is exponential

`graph_to_clique_complex` calls `networkx.find_cliques` (Bron–Kerbosch). For dense graphs (edge density > 0.5) with > 50 nodes, expect noticeable compute. Cap `max_dim` conservatively (default 2). A complete graph K_n at `max_dim=3` produces C(n,4) tetrahedra, C(n,3) triangles, C(n,2) edges, n vertices — quartic in n.

### 2.5 Hodge Laplacian Betti numbers are dense

`betti_numbers(complex_)` uses `np.linalg.eigvalsh` on the dense form of each Lₖ. **For complexes with > 1000 simplices in any dimension, this is slow and memory-heavy.** Use the persistent-homology pipeline (`RipsFiltration` + vectorization) instead.

### 2.6 Hodge MP layer assumes a fixed complex

`HodgeMessagePassing` stores its normalized Laplacian as a buffer. **The complex is fixed for the lifetime of the layer.** If your inputs are batched graphs of varying topology, you need to instantiate one layer per graph or use a per-batch construction pattern. The standard PyG batching trick (block-diagonal Laplacian across a batch) is not implemented in v0.0.1.

### 2.7 Embedding audit's β estimates are conservative

`audit_embedding` declares a bar significant if its lifetime exceeds the default threshold. **For embeddings with multimodal density (some clusters tight, others loose), a single global threshold under-counts loops in the loose regions.** This is one of the motivations for the metric cascade in v0.1.

### 2.8 JSON output writes are not strictly atomic

`write_results` uses a `tmp` + `os.replace` pattern. **`os.replace` is atomic on POSIX same-filesystem and on Windows for non-existing targets, but a Windows-side replace of an open file may fail.** Don't rely on atomicity for production pipelines until v0.1, which adds explicit cross-platform atomic writes.

---

## 3. What is **not validated**

The following claims appear in docstrings or roadmap text but **have no executable test** in v0.0.1:

- **Discriminability on real-world datasets.** All discrimination tests are on synthetic shapes (circles vs lines, two-circle complexes). No real benchmark dataset (MNIST, Cora, MUTAG, ZINC, PROTEINS, ROGII, BirdCLEF embeddings, etc.) is exercised. Real-data benchmarks land in v0.1 (item 3 will expand from "synthetic" to "Kaggle topology-shaped competition").
- **Performance characteristics.** No `pytest-benchmark` micro-benchmark suite exists. Runtime/memory claims in this document are estimates from manual inspection, not measurements.
- **Numerical stability at scale.** All tests run on small complexes (n_simplices < 20) or small point clouds (n_points ≤ 100). Behavior at production scale is not characterized.
- **Cross-platform behavior.** CI is configured for Ubuntu and macOS on Python 3.11/3.12 but has not been run on Windows. Path handling and file I/O (`os.replace` semantics, BOM handling) are likely fine but unverified.
- **Hodge MP layer learning convergence.** The forward/backward tests verify shape and gradient flow; **no test verifies that the layer learns anything useful on a real task.** Convergence behavior, training stability, and depth limits are uncharacterized.

---

## 4. What is **explicitly out of scope** for v0.0.1

The following are deliberately deferred and will not be addressed by minor patches:

- Drift-tensor correction layer (`D: M → TM`) — v0.1 / proprietary TOPOLOGICA contribution
- PH metric cascade (Euclidean → Spectral → Fermat with d_int/d_amb selection) — v0.1
- Topology losses for segmentation (Clough et al. style cubical persistence losses) — v0.1
- TopoNetX integration for non-simplicial complexes (cell complexes, combinatorial complexes) — v0.1
- Differentiable persistent homology (`autograd` through the bar matching) — v0.2
- GPU batching of Rips persistence — v0.2 (depends on upstream `ripserplusplus` integration)
- MLflow / Weights & Biases tracking adapters — v0.1
- Real-data dataset adapters (MUTAG, ZINC, PROTEINS, etc.) — v0.1
- FastAPI inference service (`services/`) — v0.1
- Multi-rank simplicial neural network (full SCN architecture) — v0.2

---

## 5. How to read the version

`v0.0.1` is the deliberate signal: pre-stable. **APIs may change without notice.** Pin exact versions in any downstream project that depends on TopoGeomML, and re-test before upgrading.

The semantics of the major versions will be:

- `v0.x` — pre-stable, breaking changes expected
- `v1.0` — stable API, full TopoNetX integration, peer-reviewed publication, GPU-batched differentiable persistence
- `v1.x` — backward-compatible additions
- `v2.x` — new domains (TDA on transformers, sheaf-theoretic data fusion) if needed

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) — TOPOLOGICA LLC
