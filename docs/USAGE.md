# Usage and API Contracts

This guide describes the checked-out `main` implementation. The published v0.0.7 package predates several fixes, including `h1_tie_policy`. If you require the code documented here, install from the current repository checkout, not an older published wheel. The short first-run commands are in the [README](../README.md).

## Installation: which environment is required?

The declared versions in `pyproject.toml` include Python 3.11 and 3.12. On Linux or macOS, use an isolated source checkout:

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

For Windows PowerShell, avoid dependence on activation policy:

```powershell
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import topogeoml; print(topogeoml.__version__)"
```

The base install is for CPU persistent-homology features and non-neural operators. Install **only** the extras you actually need, because PyTorch Geometric and GUDHI enlarge the environment:

| From a source checkout | Adds |
|---|---|
| `python -m pip install -e ".[torch]"` | PyTorch and PyTorch Geometric for neural modules |
| `python -m pip install -e ".[torch,tda]"` | Neural modules and GUDHI for cubical persistence or the explicit H₁ generator convention |
| `python -m pip install -e ".[dev]"` | Test, lint, type-check, and development tools |
| `python -m pip install -e ".[all]"` | All declared optional groups, used by the full-dependency coverage gate |
| `python -m pip install -e ".[bench]"` | Comparator backends and MNIST/graph benchmarking dependencies |

An extra does not download every research dataset. `[bench]` adds the software dependencies, while benchmark dataset loaders may fetch or read their own data. For authoritative tests see [Reviewer Guide](../REVIEWER.md).

## Core API: how do point clouds become features?

`TopologyFeaturePipeline` is imported from `topogeoml` (or `topogeoml.pipelines.feature_pipeline`). Input is either one float-convertible array shaped `(points, coordinates)`, a three-dimensional batch, or a sequence of differently sized two-dimensional clouds. `fit` records its scale and fitted transformation; `transform` returns float64 `(samples, features)`.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from topogeoml import TopologyFeaturePipeline

rng = np.random.default_rng(42)
angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
circle = np.column_stack([np.cos(angles), np.sin(angles)])
line = np.column_stack([np.linspace(-1, 1, 24), np.zeros(24)])
clouds = [
    base + rng.normal(0.0, 0.02, size=base.shape)
    for _ in range(6)
    for base in (circle, line)
]
labels = np.tile(np.array([1, 0]), 6)

estimator = Pipeline([
    ("topology", TopologyFeaturePipeline(max_homology_dim=1, resolution=8)),
    ("classifier", LogisticRegression(max_iter=500)),
])
scores = cross_val_score(estimator, clouds, labels, cv=3)
print(scores)
```

These are smoke-example scores, **not benchmark claims**. `cross_val_score` fits each transformer only on its training fold. To inspect the representation without a classifier, call `features = TopologyFeaturePipeline(resolution=8).fit_transform(clouds)` and inspect `features.shape` and `fit_provenance_`.

| `TopologyFeaturePipeline` option | Current contract |
|---|---|
| `max_homology_dim` | Highest homology dimension passed to ripser; default H₀ and H₁ |
| `max_edge_length` | Positive finite Rips cutoff; `None` leaves the backend unbounded |
| `vectorizer` | `"persistence_image"` or `"betti_curve"` |
| `resolution` | Image grid width per dimension, or Betti-curve sample count |
| `sigma` | Positive Gaussian width for persistence images; not used by Betti curves |
| `metric` | `"euclidean"`, `"precomputed"`, or a ripser/SciPy-compatible metric |
| `n_jobs` | Reserved field; it does **not** parallelize `transform` |

With selected homology dimensions `0..max_homology_dim`, persistence-image width is `(max_homology_dim + 1) * resolution**2` and Betti-curve width is `(max_homology_dim + 1) * resolution`. These are dimensional contracts derived from `core/vectorizers.py`, not empirical throughput numbers.

For a precomputed distance matrix, set `metric="precomputed"` explicitly and supply a square matrix. Do not rely on array shape to identify its meaning.

The lower-level `RipsFiltration(...).compute(cloud)` returns `PersistenceDiagram`, whose `bars[k]` is float64 with shape `(n_bars, 2)`. Death may be infinite; `diagram.lifetimes(k)` excludes infinite bars by default. The diagram carries `DiagramProvenance`, while a fitted pipeline carries `FitProvenance`. See `core/diagrams.py` for exact fields.

## Differentiable topology: what is the contract?

Install `.[torch]`, and additionally `.[tda]` if selecting GUDHI or using the cubical backend.

```python
import torch
from topogeoml.nn.diff_ph import rips_diagram_torch, TopologyRegularizer

cloud = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [0.3, 0.9], [1.1, 1.2]],
    dtype=torch.float64,
    requires_grad=True,
)
diagrams = rips_diagram_torch(cloud, max_dim=0)
print(diagrams[0].shape)  # one H0 row per input point, including essential
regularizer = TopologyRegularizer(max_dim=0, loss_type="total_persistence")
loss = regularizer(cloud)
loss.backward()
assert cloud.grad is not None
```

`rips_diagram_torch` takes a two-dimensional PyTorch point cloud and returns a list of tensors, one per homology dimension, each with two birth/death columns; infinite deaths remain infinite. It builds a distance matrix, delegates bar identities to ripser on detached CPU data, then indexes those distances to make selected values differentiable. It does not differentiate through ripser.

For an autograd-connected H₁ input with ambiguous birth or death edges, the default `h1_tie_policy="reject"` raises rather than inventing a critical edge. If you explicitly accept a GUDHI-selected combinatorial branch, use:

```python
from topogeoml.nn.diff_ph import TopologyRegularizer
regularizer = TopologyRegularizer(
    max_dim=1, loss_type="total_persistence", h1_tie_policy="gudhi"
)
```

This requires the `[tda]` extra and changes both dependency and runtime behavior. The choice is **not** a mathematical guarantee of a unique gradient at tied distances. If the application cannot tolerate exceptions during training, decide how to handle ambiguous inputs before deployment. Do not silently replace them with gradients from arbitrary equal-length edges.

`TopologyRegularizer` accepts `loss_type="total_persistence"`, `"entropy"`, or `"betti_regularization"`, with `p`, `target_betti`, `max_points`, `seed`, and `h1_tie_policy` parameters. Input is `(n_points, embedding_dim)`, not a batch of independent point clouds. It may subsample when `max_points` is exceeded.

For image-like tensors, `CubicalTopologyLoss` in `topogeoml.nn.cubical_diff_ph` accepts `(H,W)`, `(B,H,W)`, or `(B,1,H,W)`, computes cubical persistence, and returns a scalar auxiliary loss. `target_betti` is a mapping from dimension to desired count; `invert=True` maps high-valued foreground to the low-valued part of the filtration. `cubical_diagram_torch` itself requires float64, while the loss module casts predictions. A loss primitive is not proof of improved segmentation accuracy.

## Higher-order graphs, signal features, and audits

**Graph operators.** `SimplicialComplex` builds face-closed simplices from facets. `graph_to_clique_complex(graph, max_dim)` lifts a NetworkX graph or adjacency array into a capped flag complex. `hodge_laplacian(complex_, k)` returns a sparse matrix. For a PyTorch layer:

```python
import networkx as nx
import torch
from topogeoml import graph_to_clique_complex
from topogeoml.nn.hodge import build_hodge_layer_from_complex

complex_ = graph_to_clique_complex(nx.cycle_graph(4), max_dim=1)
layer = build_hodge_layer_from_complex(
    complex_, k=0, in_features=3, out_features=2
)
output = layer(torch.ones(complex_.n_simplices(0), 3))
assert output.shape == (4, 2)
```

This requires `[torch]`. The layer expects a fixed complex and features for the selected simplex dimension; it does not automatically handle arbitrary graph batches.

**Signals.** `takens_embedding(signal, embedding_dim, delay)` maps a univariate sample series into a delay-coordinate cloud. `sliding_window_topology_features(cloud, TopologyFeatureConfig(...))` pools finite-bar counts, persistence sums, entropy, and longest lifetimes over successive windows. `topology_feature_names(config)` returns names in the same order. Modules: `signal/delay_embedding.py` and `signal/sliding_window.py`. Delay estimation via `estimate_delay_autocorrelation` is a heuristic.

**Embedding audits.** `audit_embedding(embeddings, max_points, max_homology_dim, persistence_threshold, seed)` returns an `EmbeddingTopologyAudit` with persistence and nearest-neighbor summaries. Without an explicit persistence threshold it uses a multiple of the median nearest-neighbor distance. Treat `beta_0_estimate` and `beta_1_estimate` as heuristic diagnostics, not inferential estimates.

**Training callback.** `ShapeOfLearningCallback(model, probe_inputs, layer_name, ...)` hooks a named layer and emits `ShapeSnapshot` records from `on_step(step, loss)` at configured intervals. Call `detach()` after training to remove its hook. Its alert threshold is a rolling statistic, not a validated predictor of future generalization failure.

## YAML experiment: how do I avoid overwriting archived evidence?

The tracked `examples/configs/synthetic_shapes.yaml` uses `output.overwrite: true` and targets a file already in the repository. **Do not run it unchanged when you need to preserve the archived result.** Write a separate config and output:

```bash
mkdir -p reproductions
python - <<'PY'
from pathlib import Path

original = Path("examples/configs/synthetic_shapes.yaml").read_text()
edited = original.replace(
    "path: examples/outputs/synthetic_shapes_v001.json",
    "path: reproductions/synthetic_shapes_demo.json",
).replace("overwrite: true", "overwrite: false")
Path("reproductions/synthetic_shapes_demo.yaml").write_text(edited)
PY
python examples/run_experiment.py reproductions/synthetic_shapes_demo.yaml
```

The driver only implements `dataset.name="synthetic_shapes"` and `pipeline.kind="topology_feature"`. It uses `StratifiedKFold`, a fold-local topology transformer, `StandardScaler`, and `LogisticRegression`. The resulting JSON contains configuration, scores, timing, environment information, and a timestamp. `OutputConfig.overwrite` is enforced by `write_results`, which writes a temporary file and replaces the target path.

## Benchmarks and reproduction: which command proves what?

For a lightweight differentiable-persistence comparator run from the checkout (requires `.[bench]`):

```bash
mkdir -p reproductions
python -m benchmarks \
  --backends topogeoml-diff-ph \
  --datasets mnist_mock_digit_0 \
  --axes correctness \
  --quick \
  --require-correctness-backend topogeoml-diff-ph \
  --output reproductions/diff_ph_smoke.json \
  --markdown reproductions/diff_ph_smoke.md
```

The `--quick` flag reduces repetitions; this is not evidence for general performance. The explicit correctness gate fails the command when the named backend's finite-diagram comparison is missing or fails. Other backends can be unavailable because their optional dependencies are not installed; this is reported separately from a wrong scientific result.

For a *non-confirmatory* graph-classification smoke run (requires `.[torch]` and dataset access):

```bash
python -m benchmarks.hodge \
  --datasets mutag \
  --models hodge-mp-residual mlp-baseline \
  --seeds 0 1 2 \
  --n-epochs 5 \
  --output reproductions/hodge_smoke.json \
  --markdown reproductions/hodge_smoke.md
```

The graph harness uses paired seed-level comparisons. Its registered model list can grow, and the adjusted p-values depend on the requested comparison family. To reproduce an archived hypothesis, use the **exact models, dataset, seeds, modifiers, and preregistered threshold** recorded in [Reproducing](../REPRODUCING.md) and the corresponding [hypothesis](hypotheses/index.md). Do not substitute a smoke result for the unfinished H011b 30-seed protocol.

## Maintenance and troubleshooting

From a clean checkout, `python -m pip install -e ".[dev]"` provides the basic tests. The full package gate used in CI installs `.[all]` and runs:

```bash
python -m pip install -e ".[all]"
ruff check topogeoml tests benchmarks notebooks scripts
mypy topogeoml
pytest -m "not gpu" --cov=topogeoml --cov-branch --cov-fail-under=100
```

The README example is separately exercised from a new Python 3.11 virtual environment by `.github/workflows/docs-quickstart.yml`. Package coverage excludes the `benchmarks/` code.

| Symptom | Likely cause and check |
|---|---|
| `ModuleNotFoundError: topogeoml` | The shell is using a different Python than the installation. Verify `python -m pip --version` inside the active virtual environment. |
| `ModuleNotFoundError: torch` or `gudhi` | An optional module was imported without its declared extra. Install `.[torch]` or `.[torch,tda]` as required. |
| `unexpected keyword argument: h1_tie_policy` | An older published release may be installed. Verify `topogeoml.__version__` and the current checkout; v0.0.7 predates this argument. |
| `Non-identifiable H1 gradient` | Tied or near-tied filtration edges. See [Limitations](LIMITATIONS.md); either preserve default rejection, or intentionally opt into GUDHI with its caveats. |
| Precomputed distance errors | Set `metric="precomputed"` and validate matrix shape, symmetry, and distance meaning upstream. |
| `NotFittedError` or fitted-state check | Call `fit` using the training partition before calling `transform`. |
| Slow or memory-intensive persistence | Reduce cloud size or limit `max_edge_length`, then record that the chosen geometry or filtration changed. `n_jobs` does not enable pipeline parallelism. |
| `FileExistsError` from a YAML experiment | Point output to a new path or explicitly set `overwrite: true`. Archived artifacts should not be overwritten. |
| Benchmark exits without a required comparator | `--require-correctness-backend` intentionally fails closed if that backend cannot produce a valid correctness result. Install the required benchmark extra. |
| Results differ from archived graph studies | Confirm historical commit, exact model family, seed split, and dependencies; not all historical runs have complete lockfile provenance. |

For the code-level map read [Architecture](ARCHITECTURE.md), for costs and alternatives [Design Decisions](DESIGN_DECISIONS.md), and for hard boundaries [Limitations](LIMITATIONS.md).
