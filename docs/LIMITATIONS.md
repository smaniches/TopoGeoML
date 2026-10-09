# Limitations and Failure Modes

This document describes the current `topogeoml` code and the separate research record. It is an operational boundary, **not** a promise of future capabilities. For the mathematical implementation see [Architecture](ARCHITECTURE.md); for executable workflows see [Usage](USAGE.md).

## What can fail immediately?

| Input or operation | Current behavior | Maintainer action | Code |
|---|---|---|---|
| Fit a feature pipeline on an empty batch | `ValueError` | Supply at least one two-dimensional point cloud | `pipelines/feature_pipeline.py` |
| Transform before `fit` | scikit-learn fitted-state check fails | Call `fit` on training data first | `pipelines/feature_pipeline.py` |
| Pass a one-dimensional sample or malformed batch | Shape validation raises | Provide `(n_points, dimension)` arrays, a three-dimensional batch, or a sequence of two-dimensional arrays | `pipelines/feature_pipeline.py` |
| Set `metric="precomputed"` on a non-square matrix | `ValueError` | Supply a square distance matrix, and declare that metric explicitly | `core/filtrations.py`, `pipelines/feature_pipeline.py` |
| Use an invalid filtration threshold or homology dimension | Filtration constructor rejects negative homology dimension or a non-positive finite cutoff | Check configuration before computing persistence | `core/filtrations.py` |
| Use a persistence image with non-positive bandwidth or too-small resolution | Vectorizer validation fails | Choose a positive `sigma` and an accepted resolution | `core/vectorizers.py` |
| Import `topogeoml.nn.diff_ph` without PyTorch | Import error with installation guidance | Install `.[torch]` from a checkout or `topogeoml[torch]` from a package index | `nn/diff_ph.py` |
| Select `h1_tie_policy="gudhi"` without GUDHI when an ambiguous H₁ bar needs pairing | Import error; no silent fallback to the old heuristic | Install `.[tda]` as well, or use `"reject"` | `nn/diff_ph.py` |
| Write a result to an existing path with `overwrite: false` | `FileExistsError` | Choose another output path or opt in to overwrite | `experiments/configs.py` |

Not all malformed finite values are explicitly rejected before entering NumPy, SciPy, ripser, or GUDHI. Validate NaNs, infinities, symmetry, and intended metric meaning in upstream data preparation. Passing the documented shape checks is not proof that an arbitrary distance matrix defines the geometry intended by the caller.

## Where does computation stop scaling usefully?

**Vietoris–Rips:** `RipsFiltration` calls CPU ripser. Dense complexes can grow combinatorially as point count, homology dimension, or allowed edge length grows. A universal maximum safe number of points is **not benchmarked** in this repository. Use a defensible finite `max_edge_length` or a documented subsample; both change which features are observable.

**Feature transformation:** `TopologyFeaturePipeline.transform` loops over samples and calls ripser for each cloud. The `n_jobs` field is currently stored but is not used to parallelize that loop. An end-to-end parallel scaling benefit is **not benchmarked**.

**Vectorization:** Persistence-image output size equals the number of selected homology dimensions times `resolution²`; Betti-curve output size is proportional to `resolution`. Gaussian persistence images also allocate intermediate grids whose work depends on the number of bars. No general memory ceiling is established.

**Simplicial complexes:** `SimplicialComplex` enumerates all faces of facets; `graph_to_clique_complex` enumerates cliques subject to `max_dim`. Dense or high-dimensional graph complexes may be impractical. The Laplacians are sparse, but `betti_numbers` explicitly converts them to dense arrays for eigenvalues. Sparse storage does not make the Betti computation scalable.

**Differentiable persistence:** Both Vietoris–Rips and cubical differentiable paths use CPU combinatorial backends, with data copied or detached from autograd. GUDHI H₁ generator matching may construct a two-dimensional Rips skeleton and perform interval matching; its cutoff is restricted to the relevant finite death range, but a universal runtime or memory bound is **not benchmarked**. GPU tensors do not imply GPU-native persistence reduction.

## Which numerical or semantic assumptions matter?

**Training scale:** `TopologyFeaturePipeline.fit` estimates the fixed vectorization scale from up to the first eight training samples. Samples elsewhere in the training set, or future test samples at different scale, can exceed that grid. Infinite deaths are replaced by the estimated scale. Feature vectors are therefore scale-dependent and lossy. Keep fitting inside each training fold, and do not use test data to recalibrate it.

**Floating point:** The public diagram wrapper stores float64 arrays, while ripser's reported filtration values are subject to its internal precision. Tiny perturbations can change combinatorial pairings. The differentiable distance path avoids the former squared-norm cancellation at large translations, but that correction is not a guarantee against every conditioning problem.

**H₀ / zero distances:** `nn/diff_ph.py` applies a small positive distance floor because the SciPy sparse minimum-spanning-tree representation interprets stored zeros differently from positive edges. This changes how exact zero distances are represented internally; treat duplicate-point behavior as a separately tested regime.

**H₁ / gradient identifiability:** The original nearest-death-edge rule assigned a nonzero gradient to an unrelated edge of equal length. The Issue #114 regression falsified that rule. Current code rejects ambiguous H₁ autograd routes at ripser's numerical filtration precision by default. Explicit `"gudhi"` selects generator edges, but a tie may still be nondifferentiable. **No unique classical gradient, Clarke-subgradient membership, or guaranteed descent at a tie is established.** The corrected behavior may cause a previously executing training loop to raise; passing the policy through `TopologyRegularizer` lets the caller opt into GUDHI deliberately.

**Cubical images:** `cubical_diagram_torch` uses GUDHI critical pixels and requires float64 input. Pairings can change discontinuously under ties and threshold changes. The separate `cubical_mask_diagnostic` computes two-dimensional holes but returns sentinel values for three-dimensional H₁ and Euler characteristic. Do not interpret those sentinels as measured topology.

**Benchmark scope:** Finite persistence diagrams are compared with a bottleneck metric that permits matching short bars to the diagonal. A successful forward diagram check does not validate all gradients or infinite-bar behavior. The `--quick` run reduces repetitions and is a smoke/configuration check, not a confirmatory scientific result. The scientific correctness gate must be requested by name with `--require-correctness-backend`.

## What does the repository deliberately not supply?

- A GPU-native general persistent-homology solver or a custom differentiable simplicial reduction.
- A public all-purpose diagram-distance API or general alpha, Čech, witness, and lower-star point-cloud filtration interfaces.
- A production-scale batched higher-order graph-learning framework or an automatic guarantee that Hodge propagation improves classification.
- Automatic cross-validation leakage prevention when the caller fits a transformer manually before splitting; the caller must use a fold-local pipeline.
- A demonstrated end-to-end segmentation improvement from `CubicalTopologyLoss`. Its implemented loss and gradients are not equivalent to downstream benefit.
- A validated forecasting guarantee from `ShapeOfLearningCallback`. The archived divergence study is exploratory, fires at its earliest allowed probe in all reported seeds, and lacks a non-overfitting negative control.
- Complete cross-machine historical bitwise reproduction: some archived runs do not record an exact dependency environment.
- A production service, long-running application server, or deployment system; this is a library plus experiments.

These absences are observable from the exported package and source-tree entry points; they are not claims that such systems could never be built.

## Which research conclusions remain bounded or unresolved?

The graph-classification investigation is a collection of controlled experiments, not a general benchmark leaderboard. A matched normalized-adjacency external-residual control removes a supported **unique** zero-dimensional Hodge advantage in the tested configurations. A positive Hodge-versus-multilayer-perceptron comparison does not establish Hodge-specific causality.

The earlier H009 cellular-sheaf result was invalidated by implementation and parameter-count problems. Its files remain only as historical provenance. H009-R is the repaired, separately preregistered replication. H011b on COLLAB has a directional smoke result but not the intended confirmatory multi-seed result. The earlier investigation-wide multiplicity analysis was withdrawn because it included invalidated H009 comparisons; it has not been regenerated from the validated family. Non-significant comparisons are **not** equivalence claims.

For the precise designs and artifacts, consult [STATUS](../STATUS.md), [Statistical Summary](STATISTICAL_SUMMARY.md), [Claims to Evidence](CLAIMS_TO_EVIDENCE.md), and [Reproducing](../REPRODUCING.md).

## What is tested, and what is not?

The required GitHub Actions coverage gate measures line and branch execution on `topogeoml` with full optional dependencies; `benchmarks/` is excluded from that coverage denominator. The test matrix and research workflows report exercised scenarios, not a mathematical proof for unseen distributions, model families, or hardware. Published version v0.0.7 predates the latest `main` correctness fixes. Consult the current git revision when comparing API behavior to the published package.
