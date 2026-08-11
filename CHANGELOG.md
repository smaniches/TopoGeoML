# Changelog

All notable changes to TopoGeoML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **H011b execution contract preregistered (no results).** `docs/hypotheses/HYPOTHESIS-011b-l1-collab.md` gains a pre-execution amendment (section 6) fixing the execution venue requirements, artifact set, and H009-R-style result-handling rules for the pending 30-seed COLLAB run, before any confirmatory result exists. No scientific design parameter, decision rule, or threshold changes. A new `benchmarks/hodge/triangle_census.py` CLI implements the triangle census the preregistration requires in the final artifact (reconstructing each graph from the stored per-graph `L_0` exactly as the `l1-hodge-residual` model does), and `tests/test_h011b_contract.py` adds the previously missing coverage for the H011b surface: `L_1` correctness on known graphs (filled triangle `L_1 = 3I`, proving the `B_2 B_2^T` up-term; triangle-free path, down-term only), exact four-arm capacity match at COLLAB's `input_dim=1` (1,219 parameters per arm), census values and reconstruction consistency, and the degree-feature loader path.

### Fixed

- **`l1-hodge-residual` per-forward operator reconstruction (version 1.0.1).** The model rebuilt the max_dim=2 clique complex and `L_1` inside every forward pass; under the preregistered 30-seed H011b design each COLLAB graph would be reconstructed ~250 times, which is the measured cause of the documented 18-seed GitHub Actions timeout (the densest ~1% of COLLAB graphs cost ~25 s per construction). The normalized propagation operator and edge index — deterministic, parameter-free functions of the graph — are now memoized per stored Laplacian with weakref eviction. Cached and fresh forwards are numerically identical (asserted end-to-end by `tests/test_h011b_contract.py`); model mathematics, parameters, and training behaviour are unchanged.

- **H009-R corrective sheaf replication resolved.** The preregistered 30-seed NCI1 rerun of the repaired `sheaf-residual` 2.0.0 operator completed on GitHub Actions (`Run Experiment` #8) and its artifact is committed unmodified at `notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}` (SHA-256 verified against the uploaded artifact digest). Under the preregistered decision rules: H39 supported (sheaf median 0.604 above MLP 0.523, p_BH = 4.73e-3), H40 not supported (sheaf versus Hodge p_BH = 0.428, no equivalence claim), H41 falsification condition not met (sheaf below gin-residual at p_BH = 0.0342, above the preregistered 0.01 threshold; inconclusive). The evidence index (`LEADERBOARD.md` Claim 12b), `STATUS.md`, `docs/CLAIMS_TO_EVIDENCE.md`, `docs/STATISTICAL_SUMMARY.md`, `docs/limitations.md`, `docs/hypotheses/`, `README.md`, and `REPRODUCING.md` are updated accordingly. The withdrawn investigation-wide 59/76 sensitivity table stays withdrawn until regenerated from the validated comparison set.

### Security

- Hardened manual experiment dispatch by removing workflow-input shell interpolation, disabling long-option abbreviation in the Hodge CLI, keeping workflow-managed protocol/output arguments authoritative, and adding regression tests for abbreviated-option bypasses found during review.
- Read-only CI, benchmark, experiment, and release checkout steps no longer persist GitHub credentials when subsequent authenticated Git operations are unnecessary.
- The full-dependency CI gate now runs `pip-audit --local`. The audit surfaced an unused `giotto-tda` dependency that pinned vulnerable scikit-learn 1.3.2; `giotto-tda` was removed, the scikit-learn minimum was raised to 1.5, and setuptools is upgraded before auditing.

### Changed

- Dependabot now uses its default labels instead of repository labels that did not exist.

## [0.0.6] — 2026-07-06

Documentation-accuracy and packaging-hygiene release. No code-behaviour changes,
no new empirical results, and no change to any reported claim.

### Fixed

- **Silenced a torch sparse-tensor `UserWarning`** in `topogeoml.nn.hodge.sparse_scipy_to_torch`. Converting a scipy sparse matrix to a torch COO tensor emitted "Sparse invariant checks are implicitly disabled" on every call. The COO comes from a valid scipy sparse matrix, so the construction is wrapped in torch's documented `torch.sparse.check_sparse_tensor_invariants(False)` context manager, making the already-default behaviour explicit and removing the warning. Behaviour is unchanged; the scoped context manager leaks no global state.
- **Test-count claims reconciled to 504.** `CITATION.cff`, `CONTRIBUTING.md`, `docs/index.md`, `docs/CLAIMS_TO_EVIDENCE.md`, and `STATUS.md` still reported "500 tests" while `README.md` and `.zenodo.json` had moved to 504; the full suite reports 504 passing tests under `.[all]`. `docs/CLAIMS_TO_EVIDENCE.md` Claim 1 also asserted "500 test functions as counted by `grep -c \"def test_\"`", but that command returns 509 (the suite defines 509 test functions that resolve to 504 passing items after parametrization and optional-dependency skips); the verification note now states the pytest-based measurement.
- **Project description in `.zenodo.json` brought up to date.** The Zenodo deposit description still described "Seven hypotheses (H001-H007) with 27+ falsifiable sub-predictions"; the investigation now spans 14 hypotheses (H001–H011b) with 53 falsifiable sub-predictions, matching `CITATION.cff`, `README.md`, and `STATUS.md`.
- **`rips_diagram_torch` docstring corrected** (`topogeoml.nn.diff_ph`): the `max_edge_length` parameter was documented as raising `NotImplementedError` when it truncates the H_0 filtration. The v0.0.5 reconstruction handles a truncating threshold correctly — keeping the MST edges that die at or below it as finite bars and emitting one essential bar per surviving component, matching ripser exactly (covered by `tests/test_library_coverage.py`) — so the docstring now describes the implemented behaviour.

### Removed

- **Dropped FastAPI service scaffolding.** The empty `topogeoml.services` package and the `api` optional-dependency group (`fastapi`, `uvicorn`, `pydantic`) were remnants of an inference service that `LIMITATIONS.md` records as dropped. Neither was imported, exercised, or documented; removing them makes the package tree consistent with the stated decision.

## [0.0.5] — 2026-06-21

### Fixed

- **Hodge Laplacian normalization for isolated simplices** (`topogeoml.nn.hodge.normalize_hodge_laplacian`): a degree-0 (isolated) simplex now follows the standard symmetric-normalization convention `D^{-1/2}_ii = 0` instead of `1/sqrt(epsilon)` (~1e3). The discriminator is `degree > 0` (scale-invariant), so a small but genuinely positive degree is normalized rather than zeroed; only an exactly-zero degree is treated as isolated. The former `epsilon` floor also biased *every* positive-degree diagonal by ~5e-7; normalized values now equal the exact `epsilon -> 0` limit, so output shifts by ~5e-7 on connected complexes (a correctness improvement, not a regression). The `epsilon` keyword is retained for API compatibility.
- **Differentiable Rips H0 under `max_edge_length`** (`topogeoml.nn.diff_ph.rips_diagram_torch`): the H0 reconstruction assumed `n-1` finite bars from the full-matrix MST, producing a silently-incorrect diagram when `max_edge_length` truncated the filtration. It now keeps only the MST edges that die at or below the threshold as finite bars and emits one essential bar per surviving component, matching ripser's H0 diagram exactly; the default (unbounded) path is unchanged.
- **Betti regularization loss** (`topogeoml.nn.diff_ph.betti_regularization_loss`): counts every essential (infinite-death) bar as a permanent component instead of assuming exactly one, and penalizes the *least* prominent excess finite bars (preserving the most prominent within the target budget) rather than the most prominent. An all-essential diagram returns zero explicitly, since infinite lifetimes are not differentiable and there is no finite bar to shrink.
- **Block-bootstrap label** (`benchmarks/stats.py`): corrected "non-overlapping" to "overlapping" — the implementation draws a block start at every valid position (the overlapping moving-block bootstrap, Kunsch 1989), as the docstring body and the computed estimator already reflect.

### Added

- A real `torch.autograd.gradcheck` test for `cubical_diagram_torch` (analytical gradients vs. finite differences), replacing a docstring claim of gradcheck coverage that did not previously exist in the test suite.

### Changed

- Scoped the README "float64 dtype on every numerical array" claim to NumPy arrays; torch layers follow torch's float32 default and preserve float64 when the caller requests it.

## [0.0.4] — 2026-06-16

Hardening and quality release on the v0.0.3 library. No new empirical results and no
change to any reported claim; the published artifact now carries the post-0.0.3
correctness and quality fixes that previously lived only on `main`. The citable DOI is
now the **concept DOI** (`10.5281/zenodo.20365816`), which always resolves to the latest version.

### Fixed

- **Float64-safe sparse conversion** in `topogeoml.nn.hodge.sparse_scipy_to_torch`: standard float32/float64 data now passes through without a float32 intermediate (which truncated precision for float64 callers); any other input dtype is normalised to float64 before the caller's requested precision is applied (#44).

### Changed

- **Coverage gate tightened** to 100% line *and* 100% branch coverage on the `topogeoml` package under full dependencies, enforced in CI (#50).
- Runtime-introspectable type hints; `mypy --strict` and stale-test cleanups (#49, #52).
- Divergence claim reconciled to exploratory wording and the reproduction-script verdict gated; discoverability metadata (project URLs) added (#52).
- `is_chain_complex` guard simplified — the loop bounds already guarantee the checked condition; behaviour unchanged (#44).

### Added

- `PROBLEMS.md` — persistent self-audit log of known issues and their resolutions (#54).

### Quality gates for v0.0.4
- `ruff` / `black`: clean; `mypy --strict topogeoml`: 0 errors.
- `pytest` (full dependencies): 100% line and 100% branch coverage on `topogeoml`, gated in CI.
- All CI workflows on `main` green.

## [0.0.3] — 2026-06-05

Packaging/distribution release — no library or result changes.

### Added

- **PyPI distribution.** `topogeoml` is now published to PyPI via GitHub Actions Trusted Publishing (OIDC, no stored token), with a CycloneDX SBOM, build-provenance + SBOM attestations, and Sigstore signing — the same release pipeline used across the author's other packages. Install with `pip install topogeoml` (add the `[torch]` extra for the differentiable layers in `topogeoml.nn`).

## [0.0.2] — 2026-05-24

Headline: **the framework has its first strict positive-difference real-data claim** (hypothesis 003, NCI1). The v0.0.2 release gate set in `docs/hypotheses/HYPOTHESIS-002-hodge-proteins.md` §5 ("strictly beats MLP at p_BH < 0.01") is met by the `hodge-mp-residual` arm on NCI1 (p_BH = 4.83 × 10⁻³, +8.6 pp). A preregistered hypothesis series (H001–H007, 27 falsifiable sub-predictions) investigates the mechanism through systematic elimination.

### Added — empirical results (all 30-seed, BCa CIs, paired Wilcoxon + BH-FDR; per-seed reports in `notebooks/results/`)

- **PR #11 — Topology-divergence callback validated.** `ShapeOfLearningCallback.divergence_score` fires no later than a textbook val-loss-ratio watchdog on a controlled overfitting regime (200-sample `sklearn.load_digits`, p_raw = 5.77 × 10⁻⁴, r = +1.0).
- **PR #15 (hypothesis 001) — MUTAG ablation, five-arm matched-capacity.** Symmetric Laplacian normalisation is sufficient to make a one-layer Hodge MP match an MLP baseline on MUTAG (p_BH = 0.714). The combinatorial Laplacian underperforms by 9 pp (p_BH = 5.66 × 10⁻⁴). The residual variant *underperforms* MLP at this scale (p_BH = 0.019).
- **PR #16 (hypothesis 002) — PROTEINS replication.** Two-dataset equality holds for the symm-normalised arm (p_BH = 0.548). The MUTAG combinatorial-L harm does not replicate (Δ shrinks by ~10×). Strong "topology beats MLP" claim refuted on PROTEINS.
- **PR #19 (hypothesis 003) — NCI1, the headline.** On 4110 chemical-compound graphs, the symm-normalised + residual variant **strictly beats** MLP at p_BH = 4.83 × 10⁻³ (median Δ = +0.086, BCa 95% CI: [0.581, 0.625] vs MLP's [0.513, 0.566]). The residual variant *inverts* its verdict from MUTAG to NCI1 — residuals scale with dataset size at this architectural class.

### Added — public API surface

**Neural-network layers** (requires torch)
- `topogeoml.nn.diff_ph` — differentiable Vietoris-Rips persistent homology via critical-edge indexing (Hofer 2017 / Carrière 2021). Public surface: `rips_diagram_torch`, `finite_lifetimes`, `total_persistence_loss`, `persistence_entropy_loss`, `betti_matching_loss`.
- `topogeoml.nn.cubical_diff_ph` — differentiable lower-star cubical persistent homology on 2-D/3-D images, with `CubicalTopologyLoss(nn.Module)` for image-segmentation training in the Clough et al. 2020 style.
- `topogeoml.training.ShapeOfLearningCallback` — empirically validated topology-divergence watchdog for PyTorch training loops (see PR #11 row above).

**Benchmark framework** (`benchmarks/`)
- 4 backends × 4 measurement axes with statistically defensible reporting; `python -m benchmarks` CLI with `--quick` smoke tier.
- `benchmarks.stats` — bootstrap CI (percentile, BCa, block), Mann-Whitney U + Cliff's δ, Wilcoxon signed-rank + rank-biserial, Benjamini-Hochberg FDR. All citations in module docstring.
- `benchmarks.hodge` — graph-classification subsystem with five matched-capacity classifier arms (combinatorial L, symm L̃, +residual, +2-stacked+residual, MLP control) and three dataset adapters (MUTAG, PROTEINS, NCI1).

**Documentation + discipline**
- `LEADERBOARD.md` — single navigable record of every empirical claim with reproducibility instructions and discipline rules.
- `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md` through `HYPOTHESIS-003-hodge-nci1.md` — preregistered hypothesis docs with falsifiable sub-predictions and post-hoc resolved outcomes.

### Changed

- The README's roadmap is narrower and honest: no "drift-tensor", no "TOPOLOGICA proprietary", no "GPU-batched", no peer-review or DOI promises. v0.0.2 is gated on the NCI1 positive claim; v1.0 is conditional on a deeper empirical record and a methods paper.
- The "Hodge MP layer: minimal, not state-of-the-art" caveat in `LIMITATIONS.md` §1.8 is preserved; the framework now ships *five* Hodge arms in the benchmark, and the architectural element that produces the NCI1 win (residual + normalisation) is named explicitly.

### Fixed

- **Critical bug in `benchmarks/hodge/models.py` (PR #12).** The original `HodgeMessagePassing` layer was instantiated inside `forward_one()` per graph, so its weights were never registered with the optimizer and were re-randomised on every forward call. The pre-PR-#12 MUTAG numbers measured a 2-layer MLP through a random topology filter, not the Hodge architecture. Two regression tests prevent recurrence.
- **`benchmarks/cli.py` exit-code logic** — `SkippedNonDifferentiable` and `UnavailableBackend` cells no longer count as failures (PR #16, #17). Real cell failures now surface on stderr.
- **Bench workflow torch/torchvision ABI mismatch** — install both from the CPU wheel index so `torchvision::nms` resolves (PR #17).

### Deferred indefinitely (no implementation timeline)
- PH metric cascade (Euclidean → Spectral → Fermat)
- TopoNetX integration for non-simplicial complexes
- GPU-batched Rips
- MLflow / W&B tracking
- Multi-rank simplicial neural network (full SCN)
- Real DRIVE numbers — pipeline shipped (PR #9), user-side GPU run pending

### Added — academic infrastructure

- `docs/RESEARCH_REPORT.md` — structured technical report documenting the full preregistered hypothesis series (H001–H007) with results, discussion, and bounded claims
- `CITATION.cff` — CFF 1.2.0 machine-readable citation (GitHub renders "Cite this repository")
- `.zenodo.json` — Zenodo deposit metadata for DOI minting
- `CONTRIBUTING.md` — academic collaboration guidelines (preregistration pattern, code standards, statistical discipline)
- `REPRODUCING.md` — per-hypothesis reproduction guide with commands, wall-clock estimates, and expected outputs

### Changed

- `pyproject.toml`: version bumped to 0.0.2, classifier updated to "Development Status :: 4 - Beta", URLs fixed to `smaniches/TopoGeoML`
- `README.md`: badges updated, status section rewritten to reflect positive results, mechanism-investigation section added, roadmap updated, citation section updated with CITATION.cff reference

### Quality gates for v0.0.2
- `ruff check topogeoml tests benchmarks scripts notebooks`: clean
- `pytest`: 497 passed (up from 118 at v0.0.1)
- `coverage(topogeoml/ + benchmarks/)`: 100%
- `mypy topogeoml`: 0 errors (CI enforcement deferred to a separate PR pending constrained-env reproduction)
- 6 CI workflows on main, all green

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