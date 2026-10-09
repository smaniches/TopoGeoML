---
title: "Design decisions"
nav_order: 3
description: "Observed engineering choices, rejected alternatives, and their costs."
---

# Design Decisions

This document distinguishes **observed implementation choices** from **historical decisions**. The code and in-repository comments establish what was implemented and, in some cases, why. An alternative listed below is an engineering comparison, **not evidence that an earlier maintainer formally evaluated and rejected it**. When deliberation or benchmarks are absent, the historical reason is **not recorded** and performance is **not benchmarked**.

Each entry gives a constraint, identifiable options, the implemented option, why another option is not used, its consequence, and the source.

## 1. Keep the core install separate from training dependencies

**Constraint.** Point-cloud feature extraction uses NumPy, SciPy, scikit-learn, ripser, NetworkX, and PyYAML. The neural components use PyTorch and sometimes GUDHI or PyTorch Geometric.

**Options.** One mandatory dependency set; a small core with optional extras.

**Chosen, because.** `pyproject.toml` puts training and higher-order dependencies in `[torch]`, `[tda]`, and `[higher-order]` extras, and `topogeoml/__init__.py` avoids importing PyTorch modules. This lets the core feature code import without the neural stack.

**Not chosen / cost.** A single mandatory environment is not implemented. Optional extras cost explicit installation and allow optional modules to be unavailable. The repository does not benchmark installation size or dependency-resolution time.

## 2. Delegate Vietoris–Rips reduction rather than implement it again

**Constraint.** The library needs persistence diagrams from point clouds and precomputed distances.

**Options.** Reimplement persistent-homology reduction; delegate to an existing implementation.

**Chosen, because.** `RipsFiltration.compute` passes arguments to `ripser` and wraps its `dgms` output in typed arrays and provenance, keeping the library's additional responsibility at the interface level.

**Not chosen / cost.** No homegrown Vietoris–Rips reduction is present in this path. Depending on ripser costs a third-party CPU computation and its supported metric and numerical conventions. No in-repository performance comparison with a hypothetical independent reduction is recorded.

## 3. Store provenance with diagrams and fits

**Constraint.** A barcode or feature row has little audit value without knowing the filtration, metric, threshold, and fitting context.

**Options.** Raw arrays only; arrays plus structured metadata.

**Chosen, because.** `PersistenceDiagram` pairs `bars` with `DiagramProvenance`, and `TopologyFeaturePipeline.fit` creates `FitProvenance`. Those fields identify which computation produced the data and which training-scale calibration was used.

**Not chosen / cost.** Unannotated diagram arrays are not the public filtration return type. Metadata adds fields but does not store input bytes, lock all dependencies, or make nested arrays immutable. The historical reason for choosing frozen dataclasses is not recorded; the current contract is in `topogeoml/core/diagrams.py`.

## 4. Vectorize diagrams for conventional estimators

**Constraint.** Barcodes can have different numbers of intervals per input, but scikit-learn estimators expect fixed-width rows.

**Options.** Leave diagrams ragged for a specialized model; compute persistence images; compute Betti curves.

**Chosen, because.** `PersistenceImageVectorizer` sums lifetime-weighted Gaussian contributions on a grid, and `BettiCurveVectorizer` counts live bars at sampled thresholds. `TopologyFeaturePipeline` concatenates dimensions to make one numeric row per cloud.

**Not chosen / cost.** A general ragged-diagram model is not provided by the feature pipeline. Both vectorizers discard information; image output width grows with the square of resolution, while Betti-curve width grows linearly. No broad superiority claim for either representation is supported by the repository.

## 5. Learn vectorization scale from training inputs

**Constraint.** Persistence images, Betti curves, and infinite-bar substitutions require a finite, consistent scale.

**Options.** Refit grids for each test sample; fix an arbitrary global scale; estimate once in `fit`.

**Chosen, because.** `TopologyFeaturePipeline.fit` probes the first at most eight training clouds to estimate `fallback_max` using the declared metric. The resulting grid remains fixed during `transform`. This prevents held-out samples from recalibrating the feature representation.

**Not chosen / cost.** Per-test adaptation is not implemented. The bounded probe saves fitting work but can miss a later training cloud's larger scale. Out-of-range births or persistence values are not automatically regridded. The estimator must be fitted **inside each training fold** to avoid leakage.

## 6. Select distance interpretation explicitly

**Constraint.** A square coordinate array with a zero diagonal can also resemble a distance matrix.

**Options.** Infer the input type from its shape; use the configured `metric`.

**Chosen, because.** `_estimate_filtration_scale` branches on `metric="precomputed"` versus a coordinate metric. Euclidean coordinates use a bounding-box diagonal; other supported metrics use `scipy.spatial.distance.pdist`; precomputed matrices use off-diagonal entries. This fixes the previously demonstrated ambiguity.

**Not chosen / cost.** A shape-based heuristic is not used. Callers must set `metric="precomputed"` correctly and supply a square matrix. Unsupported or malformed metric input can still fail at SciPy or ripser.

## 7. Treat combinatorial persistence and numeric differentiation as separate steps

**Constraint.** PyTorch needs a computation graph, but ripser's discrete simplex reduction is not a PyTorch differentiable operator.

**Options.** Custom differentiable reduction; numerical finite differences at every training step; identify critical inputs outside autograd and index back into differentiable distances.

**Chosen, because.** `topogeoml/nn/diff_ph.py` detaches distances to compute ripser bar and cocycle identities, then reconstructs values by indexing the live PyTorch distance tensor. `cubical_diff_ph.py` applies the same pattern with GUDHI critical pixels. This retains standard PyTorch gradients where the selected critical pairing remains valid.

**Not chosen / cost.** End-to-end differentiation through the persistence reduction is not implemented. Pairing changes are discrete, and the construction does not generally provide a derivative at ties. The detached CPU step also carries a device-transfer cost. No GPU-native speed advantage is claimed.

## 8. Refuse ambiguous H₁ gradients by default

**Constraint.** Multiple edges can have the same H₁ birth or death filtration value. A nearest-equal-distance heuristic once selected a remote, unrelated edge and returned a demonstrably wrong derivative.

**Options.** Silently choose any equal-length edge; always require GUDHI; refuse the derivative unless the caller selects a generator convention.

**Chosen, because.** `rips_diagram_torch` and `TopologyRegularizer` default to `h1_tie_policy="reject"` for autograd-connected ambiguous H₁ inputs. `"gudhi"` is an explicit alternative using GUDHI flag-persistence generator edges matched to ripser finite intervals. The original six-point counterexample and regression tests are in `tests/test_h1_gradient_issue114.py`.

**Not chosen / cost.** Silent nearest-edge routing is no longer accepted for tied autograd inputs because it failed the independent finite-difference test. Automatically adding GUDHI to every core install is not implemented. Default rejection can interrupt existing training on grid-like or duplicate-point inputs; opt-in GUDHI adds a CPU combinatorial path and **does not** guarantee a unique derivative or descent at an exact tie. This change arrived on `main` after the v0.0.7 release.

## 9. Preserve the exact simplicial boundary conventions

**Constraint.** Hodge operators require consistently ordered simplices and signed boundary maps satisfying `∂² = 0`.

**Options.** Ad hoc adjacency propagation; explicit simplicial faces and boundary matrices.

**Chosen, because.** `SimplicialComplex` sorts the faces of each facet, constructs signed sparse boundaries, and supports `is_chain_complex`. `hodge_laplacian` composes lower and upper boundary terms. This exposes inspectable linear-algebra operators instead of hiding topology inside a graph layer.

**Not chosen / cost.** A generic dynamic higher-order network framework is not implemented. Enumerating every face grows rapidly for large facets, clique enumeration can be expensive, and `betti_numbers` uses dense eigenvalue computation despite sparse boundary storage.

## 10. Keep research comparators distinct from the library default

**Constraint.** A positive Hodge-versus-multilayer-perceptron result does not identify whether the Hodge operator caused the improvement.

**Options.** Advertise Hodge as an unconditional graph-classification upgrade; compare matched capacities and adjacency-based controls.

**Chosen, because.** `benchmarks/hodge/` registers separate model arms, while `docs/hypotheses/` and `notebooks/results/` retain the sequence of comparisons. H008c and H010 show no supported unique zero-dimensional Hodge advantage after the matched adjacency controls.

**Not chosen / cost.** The README does not recommend Hodge propagation as a universal replacement for adjacency-based graph neural networks. The evidence is constrained by the datasets, architecture, seed structure, and training budgets actually tested. See `STATUS.md`.

## 11. Separate software execution from scientific correctness

**Constraint.** A benchmark can run without throwing an exception and still produce a wrong persistence diagram.

**Options.** Treat exit success as validity; record a separate scientific verdict and make it enforceable.

**Chosen, because.** `benchmarks/axes/correctness.py` uses finite-diagram bottleneck distance with diagonal matching. `benchmarks/cli.py` accepts `--require-correctness-backend` and returns a failure when a required backend has missing or failed correctness evidence. Historical lexicographically sorted bar differences remain diagnostics rather than the deciding metric.

**Not chosen / cost.** Row-by-row barcode alignment is not a valid multiset comparison and is not used as the verdict. Bottleneck distance requires an additional comparison implementation and a declared tolerance. A passing forward-diagram comparison does not establish gradient correctness.

## 12. Keep per-experiment provenance and preserve negative results

**Constraint.** A changing registry or revised implementation can alter a comparison family after a result was reported.

**Options.** Keep only summary statistics and overwrite old output; store configurations, seeds, software context, and original artifacts separately.

**Chosen, because.** `topogeoml/experiments/configs.py` serializes YAML configuration and an environment snapshot, writing JSON via a temporary file followed by `os.replace`. `benchmarks/hodge/runner.py` records the requested model family and uses paired comparisons and within-family Benjamini–Hochberg adjustments. `REPRODUCING.md` requires naming historical families explicitly; `docs/hypotheses/` keeps preregistrations and `notebooks/results/` keeps artifacts.

**Not chosen / cost.** Archived findings are not silently revised. Invalidated H009 and incomplete H011b stay visible, making the documentation less concise but preventing selective reporting. A complete historical dependency lockfile is **not** present for every experiment; bit-identical replication is not promised.

## 13. Enforce tests and static checks without equating them with proof

**Constraint.** Numerical changes can preserve shape while altering values or derivatives.

**Options.** Rely on import and smoke tests alone; require regression tests, lint, strict typing, multi-platform CI, and a full-dependency coverage gate.

**Chosen, because.** `.github/workflows/ci.yml` runs ruff and mypy, tests supported Python/platform combinations, and requires 100% measured line and branch coverage on `topogeoml` under its full optional dependency stack. Benchmark workflows add algorithm-specific checks.

**Not chosen / cost.** A green unit test or coverage percentage is not treated as universal correctness. Strict coverage means new branches require tests and can delay integration. The benchmark and historical-experiment trees are not inside the library coverage claim.

For operational instructions see [Usage](USAGE.md); for failure boundaries and open scientific questions see [Limitations](LIMITATIONS.md).
