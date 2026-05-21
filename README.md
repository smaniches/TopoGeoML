# TopoGeoML

**A rigorous research toolkit for topology-aware machine learning.**

Differentiable persistent homology layers, Hodge message passing, and a benchmark framework with statistically defensible reporting — positioned as *complementary* to PyTorch / TensorFlow, not a replacement.

```text
                            ┌─────────────────────────┐
  point cloud / image ─────►│  filtration / lift      │
                            └────────────┬────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                ┌─────────────▼──────┐    ┌─────────▼──────────┐
                │ persistence diagram│    │ simplicial complex │
                │ (Rips, cubical)    │    │ (clique complex)   │
                └─────────┬──────────┘    └──────────┬─────────┘
                          │ autograd                  │
                ┌─────────▼─────────┐      ┌─────────▼─────────┐
                │   topology loss   │      │  Hodge Laplacian  │
                │   (nn.Module)     │      │  message passing  │
                └─────────┬─────────┘      └─────────┬─────────┘
                          │                          │
                          ▼                          ▼
                  PyTorch training              PyTorch training
                       loop                          loop
```

[![CI](https://github.com/smaniches/TopoGeoML/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/smaniches/TopoGeoML/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.0.1--alpha-orange)](#status)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Status (honest)

**Pre-stable research artefact.** The library is internally consistent (476 tests, 100% coverage on the library and benchmark framework), the mathematical layers are correctly implemented, and the statistical machinery is rigorous. **What it does NOT yet have**: a "topology helps on a real benchmark" claim that survives BH-corrected significance testing — the only such claim attempted so far on a real dataset is a *negative* result (see [`Empirical evidence`](#empirical-evidence) below).

This is a research toolkit, sized at ~7K LOC, positioned for researchers who need correct + citable topology-aware layers. It is **not** a production training framework. APIs will change without notice until v1.0.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list of what does *not* work yet.

---

## Empirical evidence

Every claim in the rest of this README is backed by an in-repo experiment or a literature citation. Two empirical experiments have been run so far; both are reproducible from the scripts in `notebooks/`.

### 1. Topology divergence score detects overfitting no later than a val-loss watchdog (positive)

A controlled overfitting regime on 200 examples of `sklearn.load_digits` (8×8 handwritten digits), 64-hidden MLP, Adam(lr=1e-2), 600 steps, 30 independent seeds. Two watchdogs run at the same 10-step probe cadence:
- **loss watchdog** — fires when val_loss > 1.10 × running_min
- **topology watchdog** — `ShapeOfLearningCallback.divergence_score` ≥ 2.0

Result (full report in `notebooks/results/topology_predicts_divergence_30seeds.md`):

| Statistic | Value |
|---|---|
| Direction count (topology earlier / tie / loss earlier) | **14 / 16 / 0** |
| Rank-biserial r | **+1.000** |
| Paired Wilcoxon p_raw | **5.77 × 10⁻⁴** |
| BCa 95% CI on median advantage | [+0.0, +10.0] steps |

The directional verdict is unambiguous — topology never fires *later* than loss. The magnitude is lower-bounded by the topology watchdog's baseline-window floor (every topology firing landed at step 30, the earliest possible step).

Reproduce: `python notebooks/topology_predicts_divergence.py --n-seeds 30`.

### 2. Minimal one-layer HodgeMP on MUTAG does NOT beat an MLP baseline (negative)

MUTAG mutagenicity benchmark (188 molecular graphs, 2 classes, Debnath 1991 via PyG TUDataset), 30 independent seeds × 20 epochs of Adam(lr=1e-2), 80/20 stratified split per seed. Models with matched hidden dimension (32):
- **hodge-mp-classifier**: per-node linear projection → 1 round of inline Hodge propagation (`activation(L @ X @ W + b)`) → sum-pool → linear head
- **mlp-baseline**: same shape, but the middle step ignores the Laplacian (matched-capacity control)

Result (full report in `notebooks/results/mutag_hodge_vs_mlp_30seeds.md`):

| Model | Median accuracy (95% bootstrap CI) |
|---|---|
| `hodge-mp-classifier` | 0.697 [0.658, 0.750] |
| `mlp-baseline` | **0.789 [0.763, 0.816]** |

Paired Wilcoxon (BH-corrected): median Δ = **−0.092** (Hodge is *worse* by ~9 pp), **p = 5.66 × 10⁻⁵**, rank-biserial **r = −0.760**.

**What this means.** The minimal Hodge MP architecture currently shipped — one layer, combinatorial L_0, ReLU, sum-pool — **reliably underperforms** an MLP that ignores topology, on this dataset, with statistical significance. This is a negative finding about the *architecture*, not about Hodge methods in general; the natural follow-up is to test deeper architectures, normalised Laplacians, and richer features. The result also reveals that the original PR #6 numbers (Hodge ≈ MLP at ~70%) were measuring a buggy model whose HodgeMP weights were not registered with the optimizer (fixed in PR #12).

Reproduce: `python -m benchmarks.hodge --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 20`.

---

## What's actually in the box

| Subsystem | Module | Status | Notes |
|---|---|---|---|
| Persistent-homology core | `topogeoml.core.{diagrams,filtrations,vectorizers,complexes,cubical}` | done | Rips diagrams, persistence images, Betti curves, simplicial complexes |
| Graph → clique complex | `topogeoml.data.graph_to_clique_complex` | done | Bron-Kerbosch via networkx |
| Topology feature pipeline | `topogeoml.pipelines.TopologyFeaturePipeline` | done | sklearn-compatible |
| Hodge Laplacian + MP layer | `topogeoml.nn.hodge` | done | One round of `activation(L @ X @ W + b)`; minimal SCN building block |
| **Differentiable PH (Rips)** | `topogeoml.nn.diff_ph` | done | autograd through critical-edge indexing (Hofer 2017, Carrière 2021) |
| **Differentiable PH (cubical)** | `topogeoml.nn.cubical_diff_ph` | done | autograd through critical-pixel indexing; `CubicalTopologyLoss(nn.Module)` for image-segmentation training (Clough 2020-style) |
| Topology-divergence callback | `topogeoml.training.ShapeOfLearningCallback` | done | empirically validated — see evidence section above |
| Signal analysis | `topogeoml.signal.{delay_embedding,sliding_window}` | done | Takens embedding + windowed topology features |
| Embedding audit | `topogeoml.audits.audit_embedding` | prototype | heuristic significance threshold; calibrated noise floor pending |
| **Benchmark framework** | `benchmarks/` | done | 4 backends × 4 axes (correctness/stability/speed/optimization), 100% coverage |
| **Hodge subsystem benchmark** | `benchmarks/hodge/` | done | MUTAG classification with paired Wilcoxon + BH |
| **Statistical machinery** | `benchmarks.stats` | done | BCa + block + percentile bootstrap; Wilcoxon, Mann-Whitney, BH-FDR; 100% coverage |

---

## Installation

```bash
# Core: complexes, persistence, vectorizers, audits, configs (no torch).
pip install topogeoml

# With PyTorch (enables nn.diff_ph, nn.cubical_diff_ph, nn.hodge).
pip install "topogeoml[torch]"

# Plus GUDHI for the cubical PH backend.
pip install "topogeoml[tda]"

# Plus torch-geometric for the Hodge benchmark on TUDataset.
pip install "topogeoml[bench]"
```

From source:

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
pip install -e ".[dev]"
pytest
```

---

## Quick start

### Topology feature pipeline (sklearn-compatible)

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

### Differentiable cubical topology loss (for image segmentation)

```python
import torch
from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

# Penalise predictions whose foreground has more than one connected component.
topo_loss = CubicalTopologyLoss(target_betti={0: 1}, invert=True)

pred = torch.rand(4, 1, 64, 64, dtype=torch.float64, requires_grad=True)  # (B, 1, H, W)
loss = topo_loss(pred)
loss.backward()  # gradients flow through the persistent-homology computation
```

See `notebooks/drive_unet_topology_loss.py` for the DRIVE retinal-vessel segmentation pipeline (Dice + BCE + λ·CubicalTopologyLoss vs Dice + BCE baseline).

### Hodge message passing layer

```python
import networkx as nx
import torch
from topogeoml import graph_to_clique_complex
from topogeoml.nn.hodge import build_hodge_layer_from_complex

sc = graph_to_clique_complex(nx.complete_graph(5), max_dim=2)
layer = build_hodge_layer_from_complex(sc, k=0, in_features=16, out_features=8)
x = torch.randn(sc.n_simplices(0), 16)
out = layer(x)
print(out.shape)  # torch.Size([5, 8])
```

### Benchmark CLI

```bash
# Full-rigor run (~hours on CPU; preferred on GPU / Modal):
python -m benchmarks

# CI smoke tier (thinned seeds/repeats; ~10-15 min on CPU):
python -m benchmarks --quick
```

The benchmark writes a JSON leaderboard + Markdown report with bootstrap CIs and BH-corrected paired Wilcoxon for every cross-backend comparison.

### Statistical machinery (usable standalone)

```python
import numpy as np
from benchmarks.stats import bootstrap_ci, BootstrapMethod, compare_paired

x = np.random.lognormal(size=120)
ci = bootstrap_ci(x, statistic="median", method=BootstrapMethod.BCA)
print(f"BCa 95% CI: [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")
```

Three interval methods are supported: percentile (Efron 1979), BCa (Efron 1987), and block (Künsch 1989). See `benchmarks/stats.py` for the citations behind every procedure.

---

## Standards

The package enforces the following floor:

- Explicit `float64` dtype on every numerical array
- No Python sample loops for numerical computation (construction loops permitted)
- `random_state=42` / `np.random.default_rng(42)` for reproducible RNG
- Provenance dict on every fit + every benchmark cell
- 100% coverage on the library (`topogeoml/`) and the benchmark framework (`benchmarks/`)
- ruff clean across all source directories
- Every empirical claim in any docstring or README must point to either a literature citation or an in-repo experiment (negative results count and are shipped)

---

## Testing

```bash
pytest                          # 476 tests
pytest -m "not slow"            # skip slow tests
pytest --cov=topogeoml --cov=benchmarks  # with coverage
```

Coverage is 100% on `topogeoml/` and `benchmarks/`. Torch-gated tests skip cleanly when torch is not installed.

---

## Roadmap (only what's planned, not what's hoped)

**v0.0.1 (current).** Library subsystems above, benchmark framework with BCa/block bootstrap, two empirical experiments (one positive, one negative — see [Empirical evidence](#empirical-evidence)).

**v0.0.2 (next).** A real-data benchmark with a positive empirical claim (target: DRIVE retinal-vessel segmentation using `CubicalTopologyLoss`, or a deeper Hodge architecture on a TUDataset where the minimal one-layer model failed). The bar is paired Wilcoxon p < 0.01 after BH correction, with BCa CIs reported. If no positive result is found at v0.0.2, the README will say so and the work continues.

**v0.1 and later.** Not planned in detail yet — it depends on whether v0.0.2 produces a positive empirical claim and which direction is most promising from that signal. No promises about GPU-batched persistence, distributed training, simplicial neural network architectures, or replacing PyTorch / TensorFlow.

---

## Citation

```bibtex
@software{maniches_topogeoml_2026,
  author       = {Maniches, Santiago},
  title        = {TopoGeoML: a research toolkit for topology-aware machine learning},
  year         = {2026},
  version      = {0.0.1-alpha},
  url          = {https://github.com/smaniches/TopoGeoML},
  orcid        = {0009-0005-6480-1987}
}
```

No DOI is minted at this version. The empirical record is too thin to lock in permanently.

---

## License

MIT. See [LICENSE](LICENSE).

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)).
