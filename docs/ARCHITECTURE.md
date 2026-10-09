---
title: "Architecture"
nav_order: 2
description: "Implemented modules, data flow, provenance, and validation boundaries."
---

# Architecture

**Scope:** the code on `main` as inspected for this guide, including the H₁ tie-handling correction merged in pull request (PR) #115. Paths below are the evidence for the implementation descriptions. This document describes code behavior, not a speculative future platform.

## What is installed?

`pyproject.toml` builds the `topogeoml` Python package with Hatchling. The default installation depends on NumPy, SciPy, scikit-learn, ripser, NetworkX, and PyYAML. PyTorch, PyTorch Geometric, GUDHI (Geometry Understanding in Higher Dimensions), and TopoNetX are optional dependency groups, because the package's point-cloud feature path does not need the training stack. The root `topogeoml/__init__.py` exports the core classes but does not import the PyTorch-dependent modules.

`benchmarks/`, `notebooks/`, `examples/`, and `docs/hypotheses/` are source-tree research and demonstration material, not additional modules in the installed wheel. The `benchmarks` command therefore requires a source checkout, appropriate extras, and (for real datasets) any external dataset downloads.

## How does a point cloud become a feature row?

The primary path, implemented in `topogeoml/pipelines/feature_pipeline.py`, is:

```text
Array (n_points, ambient_dim) or precomputed distance matrix
    → TopologyFeaturePipeline.fit() [training samples only]
        → calibration scale + RipsFiltration + vectorizer
    → TopologyFeaturePipeline.transform() [each input sample]
        → RipsFiltration.compute() → PersistenceDiagram
        → vectorizer.transform_one() → fixed-length NumPy row
    → feature matrix (n_samples, n_features)
```

`TopologyFeaturePipeline` is a scikit-learn `BaseEstimator` and `TransformerMixin`. It accepts a single two-dimensional array, a three-dimensional batch, or a sequence of two-dimensional arrays with varying point counts. Its `fit` method constructs the filtration and vectorizer, estimates a fallback scale from the first at most eight training clouds, and records a `FitProvenance`. Its `transform` method preallocates a float64 matrix and processes each sample serially. It does not fit a classifier.

The filtration uses `RipsFiltration` (`topogeoml/core/filtrations.py`), which delegates persistence to ripser and returns a `PersistenceDiagram` (`topogeoml/core/diagrams.py`). The latter contains `bars: dict[int, ndarray]`, each value shaped `(n_bars, 2)` with birth and death columns, plus a `DiagramProvenance` recording metric, filtration, input dimensions, threshold, coefficient field, and backend. The dataclasses are frozen, **but their nested dictionaries and arrays are not made deeply immutable**.

The vectorizers live in `topogeoml/core/vectorizers.py`. A persistence image maps birth–death intervals to weighted Gaussian contributions on a fixed birth–persistence grid. A Betti curve counts the bars alive at each sampled filtration level. Both replace infinite deaths with the fitted fallback scale, because a bounded feature grid requires finite coordinates. That choice loses information and makes train-time scale calibration consequential. Fit the transformer inside each cross-validation fold, not on the full dataset before splitting.

## Where do differentiable topology losses get their derivatives?

`topogeoml/nn/diff_ph.py` takes a PyTorch point-cloud tensor and computes Euclidean pairwise distances with a direct-coordinate path. It detaches a central processing unit (CPU) distance matrix for ripser to identify persistence bars and cocycles, then selects entries from the original differentiable distance tensor for the reconstructed births and deaths. PyTorch propagates gradients through those selected tensor entries; it does **not** differentiate through ripser's combinatorial reduction.

The differentiated Vietoris–Rips reconstruction currently implements only zero- and one-dimensional homology; passing `max_dim > 1` to `rips_diagram_torch` does not return higher-dimensional bars, despite ripser being asked to compute them. For zero-dimensional homology (H₀), minimum-spanning-tree edges give finite death values, and a filtration cutoff changes how many components remain essential. For first homology (H₁), a numeric death value alone may correspond to multiple edges. A previous selector attached a loop death to an unrelated same-length edge. The merged Issue #114 correction detects ambiguities at ripser's filtration precision, rejects the associated autograd gradient by default, and offers an explicit `h1_tie_policy="gudhi"`. GUDHI supplies a selected pair of birth/death generator edges, matched to ripser's finite intervals with a numerical tolerance. This is a selected combinatorial branch, not a proof of a unique gradient at ties.

`TopologyRegularizer` exposes the same policy when used in a training loop and optionally subsamples input points. Subsampling changes the analyzed topology; it is not a transparent optimization.

`topogeoml/nn/cubical_diff_ph.py` handles two- or three-dimensional image-like tensors differently: GUDHI computes cubical persistence vertex pairings on detached CPU data, and PyTorch indexing reconstructs differentiable birth and death values. `CubicalTopologyLoss` selects and penalizes excess bars relative to target Betti counts. Its result is an auxiliary loss term, not a full image segmentation model. A separate `topogeoml/core/cubical.py` function estimates topology of binary masks through SciPy connected-component labeling; it is **not** the differentiable cubical backend.

## How are graph and signal structures represented?

`topogeoml/core/complexes.py` represents a simplicial complex by enumerating all faces of supplied facets, sorting simplices by dimension, and constructing signed sparse boundary matrices. `graph_to_clique_complex` (`topogeoml/data/graph_to_complex.py`) builds a capped flag complex from a NetworkX graph or adjacency matrix. The sparse Hodge Laplacian is computed from the boundary operators. `betti_numbers` converts these Laplacians to dense matrices before calculating their eigenvalues, so it is meant for small complexes.

`topogeoml/nn/hodge.py` normalizes a precomputed Laplacian and registers it as a non-trainable PyTorch buffer. `HodgeMessagePassing` applies a sparse Laplacian operation followed by a learned linear transformation and activation. It does not create a dynamically changing graph, batch arbitrary simplicial complexes, or establish that Hodge propagation beats a matched adjacency operator.

`topogeoml/signal/delay_embedding.py` constructs delay-coordinate point clouds from univariate signals. `topogeoml/signal/sliding_window.py` calls ripser on successive windows, computes finite-bar statistics, and pools them into a fixed feature vector. `topogeoml/audits/embedding_audit.py` separately samples embedding rows and combines ripser summaries with nearest-neighbor distances; the default significance cutoff is a heuristic, not a calibrated hypothesis test.

`topogeoml/training/callbacks.py` hooks a named PyTorch layer, evaluates fixed probe inputs periodically, stores `ShapeSnapshot` records, and compares current topology summaries against a rolling baseline. The associated experiment is exploratory and does not establish prospective overfitting prediction.

## How does evidence move through the repository?

There are two independent research paths:

- `examples/run_experiment.py` loads a YAML (YAML Ain't Markup Language) `ExperimentConfig`, generates circles and lines, fits a scikit-learn pipeline *inside* stratified cross-validation, then writes JavaScript Object Notation (JSON) with scores, timing, configuration, environment, and a timestamp through `topogeoml/experiments/configs.py`.
- `python -m benchmarks` runs available persistence backends across named datasets and correctness, stability, speed, or optimization axes (`benchmarks/runner.py` and `benchmarks/axes/`). It captures failures per cell. Its command-line interface (CLI) has `--require-correctness-backend` because execution without an exception is **not** proof that a persistence diagram is correct. The Hodge graph-classification harness is separate: `python -m benchmarks.hodge` uses a named model/dataset family, paired comparisons, and within-family Benjamini–Hochberg adjustment. The registered model family matters to the statistical result.

Archived hypotheses and seed-level artifacts remain in `docs/hypotheses/` and `notebooks/results/`. `STATUS.md`, `LEADERBOARD.md`, `docs/STATISTICAL_SUMMARY.md`, and `docs/CLAIMS_TO_EVIDENCE.md` distinguish current evidence from invalidated H009 data, its H009-R correction, and incomplete H011b. Run new experiments into new output paths instead of replacing archived artifacts.

## Where are the validation boundaries?

`.github/workflows/ci.yml` runs ruff, mypy, a Python/platform test matrix, and a strict full-dependency line-and-branch-coverage gate on the importable `topogeoml` package. The benchmark workflows perform additional checks. Coverage measures exercised code paths, not mathematical validity or downstream model generalization. The release workflow is tag-triggered; editing documentation or merging a pull request is not itself a package release.

Read [Design decisions](DESIGN_DECISIONS.md) for costs and alternatives, [Usage](USAGE.md) for invocation contracts, and [Limitations](LIMITATIONS.md) before extending or deploying an algorithm.
