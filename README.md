# TopoGeomML

**Topology-aware geometric machine learning.**

A Python-first research and engineering stack connecting persistent homology, higher-order topological domains, and geometric deep learning into reproducible ML pipelines.

```text
                            ┌─────────────────────────┐
  point cloud / graph ─────►│  filtration / lift      │
                            └────────────┬────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                ┌─────────────▼──────┐    ┌─────────▼──────────┐
                │ persistence diagram│    │ simplicial complex │
                └─────────┬──────────┘    └──────────┬─────────┘
                          │                          │
                ┌─────────▼─────────┐      ┌─────────▼─────────┐
                │   vectorization   │      │  Hodge Laplacian  │
                └─────────┬─────────┘      └─────────┬─────────┘
                          │                          │
                          ▼                          ▼
                 features for sklearn         message passing in PyTorch
```

[![Tests](https://img.shields.io/badge/tests-104%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.0.1--alpha-orange)](#status)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Status

**v0.0.1 — pre-stable.** The eleven items of the v0.0.1 scope lock are all implemented and tested. APIs may change without notice; pin exact versions in downstream projects.

See [`LIMITATIONS.md`](LIMITATIONS.md) for an honest accounting of what this version **does not** do.

For architecture, strategic positioning, and the design intent behind v0.0.1 + the v0.1+ roadmap, see [`docs/architecture/`](docs/architecture/README.md) — in particular [`00-positioning.md`](docs/architecture/00-positioning.md) (why this is not a TDA utility package) and [`SCOPE_LOCK_v0.0.1.md`](docs/architecture/SCOPE_LOCK_v0.0.1.md) (what ships now and why).

## What's in v0.0.1

| # | Item | Status | Where |
| --- | --- | --- | --- |
| 1 | Topology feature pipeline for point clouds | ✅ | `topogeoml.pipelines.TopologyFeaturePipeline` |
| 2 | Diagram statistics and Betti curve vectorizers | ✅ | `topogeoml.core.PersistenceImageVectorizer`, `BettiCurveVectorizer` |
| 3 | Synthetic point cloud classification benchmark | ✅ | `examples/circles_vs_lines.py` + `examples/run_experiment.py` |
| 4 | Cubical mask topology diagnostic | ✅ | `topogeoml.core.cubical_mask_diagnostic` |
| 5 | Graph to clique complex lift | ✅ | `topogeoml.data.graph_to_clique_complex` |
| 6 | Boundary operator validation | ✅ | `topogeoml.core.is_chain_complex` |
| 7 | Hodge Laplacian utility | ✅ | `topogeoml.core.hodge_laplacian`, `betti_numbers` |
| 8 | Minimal Hodge message passing layer | ✅ | `topogeoml.nn.hodge.HodgeMessagePassing` (requires torch) |
| 9 | Embedding topology audit prototype | ✅ | `topogeoml.audit_embedding` |
| 10 | YAML experiment configs and JSON outputs | ✅ | `topogeoml.experiments.load_experiment_config` / `write_results` |
| 11 | Documentation with explicit limitations | ✅ | [`LIMITATIONS.md`](LIMITATIONS.md) |

## Installation

```bash
# Core: feature pipeline, vectorizers, complexes, audits, configs.
pip install topogeoml

# With PyTorch (enables the Hodge MP layer)
pip install "topogeoml[torch]"

# With GUDHI + giotto-tda (additional TDA backends)
pip install "topogeoml[tda]"

# With TopoNetX (higher-order domains — v0.1)
pip install "topogeoml[higher-order]"

# Everything
pip install "topogeoml[all]"
```

From source:

```bash
git clone https://github.com/topologica-llc/topogeoml.git
cd topogeoml
pip install -e ".[dev]"
pytest
```

## Quick start

**Feature pipeline (item 1):**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topogeoml import TopologyFeaturePipeline

rng = np.random.default_rng(42)
theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
t = np.linspace(-1, 1, 50)

X, y = [], []
for _ in range(10):
    X.append(np.stack([np.cos(theta), np.sin(theta)], axis=1) + 0.05 * rng.standard_normal((50, 2)))
    X.append(np.stack([t, np.zeros(50)], axis=1) + 0.05 * rng.standard_normal((50, 2)))
    y.extend([1, 0])

clf = Pipeline([
    ("topology", TopologyFeaturePipeline(max_homology_dim=1, resolution=10)),
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(random_state=42)),
])
clf.fit(X, np.array(y))
print(clf.score(X, y))  # 1.0
```

**Cubical mask diagnostic (item 4):**

```python
import numpy as np
from topogeoml import cubical_mask_diagnostic

mask = np.zeros((20, 20), dtype=bool)
mask[3:17, 3:17] = True
mask[7:13, 7:13] = False  # one hole
d = cubical_mask_diagnostic(mask)
print(d.betti_0, d.betti_1, d.euler_characteristic)  # 1 1 0
```

**Graph → clique complex → Hodge Laplacian (items 5, 6, 7):**

```python
import networkx as nx
from topogeoml import (
    graph_to_clique_complex, is_chain_complex, hodge_laplacian, betti_numbers,
)

G = nx.cycle_graph(4)               # C_4 — one connected loop
sc = graph_to_clique_complex(G, max_dim=2)
assert is_chain_complex(sc)         # boundary identity ∂² = 0
L0 = hodge_laplacian(sc, 0)         # standard graph Laplacian
b = betti_numbers(sc, max_dim=1)
print(b)  # {0: 1, 1: 1}            # one component, one loop
```

**Hodge MP layer (item 8):**

```python
import torch
from topogeoml import graph_to_clique_complex
from topogeoml.nn.hodge import build_hodge_layer_from_complex

sc = graph_to_clique_complex(nx.complete_graph(5), max_dim=2)
layer = build_hodge_layer_from_complex(sc, k=0, in_features=16, out_features=8)
x = torch.randn(sc.n_simplices(0), 16)
out = layer(x)
print(out.shape)  # torch.Size([5, 8])
```

**Embedding audit (item 9):**

```python
import numpy as np
from topogeoml import audit_embedding

emb = np.random.default_rng(42).standard_normal((500, 16))
audit = audit_embedding(emb, max_points=200)
print(audit.summary())
```

**YAML experiment (item 10):**

```bash
python examples/run_experiment.py examples/configs/synthetic_shapes.yaml
# Writes examples/outputs/synthetic_shapes_v001.json with config echo,
# CV scores, timings, environment snapshot, and UTC timestamp.
```

## Architecture

```text
topogeoml/
├── core/                       # Mathematical objects (no torch)
│   ├── diagrams.py             ✅ PersistenceDiagram + DiagramProvenance
│   ├── filtrations.py          ✅ RipsFiltration (via ripser)
│   ├── vectorizers.py          ✅ PersistenceImage, BettiCurve
│   ├── complexes.py            ✅ SimplicialComplex, ∂_k, L_k, betti_numbers
│   ├── cubical.py              ✅ Binary-mask topology diagnostic
│   └── distances.py            🚧 v0.1: bottleneck, Wasserstein
├── data/
│   └── graph_to_complex.py     ✅ Graph → clique complex
├── pipelines/
│   └── feature_pipeline.py     ✅ TopologyFeaturePipeline (sklearn-compatible)
├── audits/
│   └── embedding_audit.py      ✅ Embedding topology audit (prototype)
├── experiments/
│   └── configs.py              ✅ YAML loader + JSON writer
├── nn/                         # Requires torch
│   └── hodge.py                ✅ HodgeMessagePassing layer
├── services/                   🚧 v0.1: FastAPI descriptor service
└── tests/                      ✅ 104 tests
```

## Mathematical object contracts

Every public type carries a strict contract:

- **`PersistenceDiagram`** — frozen dataclass; `bars[k]` is a contiguous `float64` `(n,2)` array; `DiagramProvenance` is mandatory.
- **`RipsFiltration`** — stateless; `compute(X)` is pure, coerces to `float64`.
- **Vectorizers** — deterministic fixed-length output (`output_dim` known before transform).
- **`SimplicialComplex`** — auto-closes under faces; lexicographic ordering of k-simplices guarantees deterministic boundary-matrix column indices.
- **`is_chain_complex`** — verifies ∂_{k-1} ∘ ∂_k = 0 within numerical tolerance, in all dimensions.
- **`hodge_laplacian`** — returns symmetric PSD sparse matrix; `dim ker L_k = β_k` (discrete Hodge theorem).
- **`TopologyFeaturePipeline`** — sklearn-compatible `BaseEstimator + TransformerMixin`; captures `fit_provenance_` per verification gate.
- **`HodgeMessagePassing`** — `nn.Module`; sparse Laplacian as buffer; gradient flows back through `torch.sparse.mm`.
- **`audit_embedding`** — returns `EmbeddingTopologyAudit` with full provenance dict.
- **`load_experiment_config` / `write_results`** — YAML in, JSON out; environment snapshot, UTC timestamp, and full config echo are mandatory output fields.

## Standards

The package complies with the `elite-code-standards` failure-prevention rules:

- Explicit `float64` dtype on every numerical array
- No Python sample loops for numerical computation (construction loops permitted)
- `random_state=42` / `np.random.default_rng(42)` for reproducible RNG
- Provenance dict on every fit
- Verification gate before any quantitative claim: interpolator check, correction audit, derivative inheritance, validation provenance

## Testing

```bash
pytest                          # all tests
pytest -m "not slow"            # skip slow tests
pytest -m "not torch"           # skip layer tests if torch is unavailable
pytest --cov=topogeoml          # with coverage
```

Current coverage: 104 tests across diagrams, filtrations, vectorizers, complexes, cubical, graph lift, Hodge MP layer, embedding audit, configs, and the end-to-end pipeline. Topology recovery is verified on shapes with known Betti numbers (S¹, D², S², two circles, tetrahedron boundary, K_4 boundary).

## Limitations

This is v0.0.1 — pre-stable. The deliberate cuts in scope, known failure modes, and unvalidated claims are listed in [`LIMITATIONS.md`](LIMITATIONS.md). Read it before relying on this package for anything load-bearing.

## Roadmap

**v0.0.1 (current)** — 11-item scope lock, all delivered.

**v0.1** — Differentiable persistence (PyTorch autograd), PH metric cascade (Euclidean → Spectral → Fermat with d_int/d_amb selection), cubical filtration on real-valued images, drift-tensor correction layer (TOPOLOGICA proprietary), benchmark harness on one topology-shaped Kaggle competition, bottleneck/Wasserstein diagram distances.

**v0.2** — TopoNetX integration for cell and combinatorial complexes, GPU-batched Rips, topology losses for segmentation, full simplicial neural network architecture, MLflow/W&B adapters.

**v1.0** — Stable API, peer-reviewed publication, GPU-batched differentiable persistence.

## Citation

```bibtex
@software{maniches_topogeoml_2026,
  author       = {Maniches, Santiago},
  title        = {TopoGeomML: topology-aware geometric machine learning},
  year         = {2026},
  version      = {0.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/topologica-llc/topogeoml},
  orcid        = {0009-0005-6480-1987}
}
```

## License

MIT. See [LICENSE](LICENSE).

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) — [TOPOLOGICA LLC](https://topologica.ai)
