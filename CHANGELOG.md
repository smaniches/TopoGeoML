# Changelog

All notable changes to TopoGeomML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.1
- Differentiable persistence (PyTorch autograd through diagram → loss)
- PH metric cascade (Euclidean → Spectral → Fermat) with d_int/d_amb auto-selection
- Cubical filtration for real-valued image / segmentation inputs
- Drift-tensor correction layer (TOPOLOGICA proprietary)
- Benchmark harness on one topology-shaped Kaggle competition
- Bottleneck and Wasserstein diagram distances
- Real-data dataset adapters (MUTAG, PROTEINS, ZINC, etc.)
- Cross-platform atomic JSON write
- MLflow / W&B tracking adapters

## [0.0.1] — 2026-05-20

Initial pre-stable release. The eleven-item v0.0.1 scope lock is fully implemented.

### Added

**Core mathematical objects**
- `PersistenceDiagram` frozen dataclass with mandatory `DiagramProvenance`
- `RipsFiltration` via ripser with float64 enforcement and provenance recording
- `PersistenceImageVectorizer` (Adams et al. 2017) via persim
- `BettiCurveVectorizer` (vectorized sampling on uniform grid)
- `SimplicialComplex` with lexicographic simplex ordering and automatic face closure
- `boundary_matrix(k)` — signed sparse boundary operator ∂_k over R
- `is_chain_complex` — verifies ∂_{k-1} ∂_k = 0 within numerical tolerance
- `hodge_laplacian(k)` — symmetric PSD sparse Laplacian L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T
- `betti_numbers` via dense eigendecomposition (discrete Hodge theorem)
- `cubical_mask_diagnostic` — β_0/β_1 + Euler characteristic for binary 2D masks (3D β_0 only)

**Data adapters**
- `graph_to_clique_complex` — NetworkX graph or adjacency matrix → SimplicialComplex with bounded max_dim

**Pipelines**
- `TopologyFeaturePipeline` — sklearn `BaseEstimator + TransformerMixin`, supports list-of-arrays and 3D ndarray inputs, captures `FitProvenance`

**Audits**
- `audit_embedding` — Rips-based topology audit of embedding matrices with NN-distance threshold heuristic
- `EmbeddingTopologyAudit` dataclass with β_0/β_1 estimates, total persistence, longest H_1 lifetime

**Neural-network layers (requires torch)**
- `HodgeMessagePassing` — minimal one-round propagation x' = σ(L̃_k @ x @ W + b)
- `normalize_hodge_laplacian` — symmetric normalization D^{-1/2} L D^{-1/2}
- `sparse_scipy_to_torch` — sparse format converter
- `build_hodge_layer_from_complex` — convenience constructor

**Experiments**
- `load_experiment_config` — YAML loader with dataclass validation
- `write_results` — JSON writer with config echo, environment snapshot, UTC timestamp, numpy-aware serialization
- `ExperimentConfig`, `DatasetConfig`, `PipelineConfig`, `ValidationConfig`, `OutputConfig` dataclasses
- `examples/run_experiment.py` — end-to-end YAML → JSON runner
- `examples/configs/synthetic_shapes.yaml` — first benchmark config

**Documentation**
- `LIMITATIONS.md` — explicit scope cuts, failure modes, and unvalidated claims
- `README.md` — quick-start for each of the 11 items, architecture diagram, contracts, citation block

**Infrastructure**
- `pyproject.toml` (hatchling backend, MIT license, optional extras: `torch`, `tda`, `higher-order`, `api`, `dev`, `all`)
- GitHub Actions CI matrix (Python 3.11/3.12 × Ubuntu/macOS)
- PEP 561 `py.typed` marker

### Verified
- Boundary identity ∂² = 0 on triangle, tetrahedron, two-triangle complexes
- Hodge β recovery: D² (1,0,0); S¹ (1,1); S² (1,0,1); disjoint vertices; disjoint triangles
- L_0 on path graph reduces to standard combinatorial graph Laplacian
- Cubical β on disk (1,0), annulus (1,1), two disks (2,0), disk-with-two-holes (1,2)
- Clique complex topology on K_3, K_4, C_4, K_4-boundary
- Hodge MP forward/backward, gradient flow, shape contracts, layer stacking
- Embedding audit on single-circle (β_1=1), two-circle (β_1=2) layouts
- YAML round-trip, JSON output schema with mandatory fields, numpy serialization
- End-to-end synthetic-shapes benchmark: 5-fold CV accuracy 1.0000 ± 0.0000

### Test suite

118 tests passing, 3 skipped (torch-gated: Hodge MP layer, differentiable PH, ShapeOfLearning callback — collected and run only when the `[torch]` extra is installed). Verified on Python 3.11 and 3.12, Ubuntu and macOS, via `.github/workflows/ci.yml`.
