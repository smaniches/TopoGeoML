# TopoGeoML

**Topology-aware machine learning tools for persistent-homology features, differentiable topology losses, simplicial and Hodge operators, signal analysis, and reproducible evaluation.**

[![CI](https://github.com/smaniches/TopoGeoML/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/smaniches/TopoGeoML/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.0.7--beta-green)](#status)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20365816.svg)](https://doi.org/10.5281/zenodo.20365816)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

TopoGeoML is a Python scientific-software library for cases where connected components, loops, persistence, simplicial structure, or topology through training are part of the signal you want to measure or constrain. It provides a scikit-learn feature pipeline, PyTorch topology losses, simplicial and Hodge primitives, topology features for signals, embedding diagnostics, and a reproducible research harness.

The repository also contains a preregistered graph-classification investigation. That study is evidence about where one family of topological methods did and did not help. It is not the definition of the library.

## Where TopoGeoML fits

| Problem | Use | Current status |
|---|---|---|
| Point clouds, shapes, geometric samples | `TopologyFeaturePipeline` converts Vietoris-Rips persistence into persistence-image or Betti-curve features for ordinary ML pipelines | Implemented, scikit-learn compatible |
| Differentiable topology on learned point sets | `topogeoml.nn.diff_ph` and `TopologyRegularizer` route persistence values back to PyTorch tensors | Research primitive with gradient and stability tests |
| Topology-constrained image segmentation | `CubicalTopologyLoss` penalizes excess connected components or loops in soft masks | Implemented and gradient-tested; no powered end-to-end segmentation benefit claim yet |
| Simplicial or higher-order experiments | `SimplicialComplex`, Hodge Laplacians, and `HodgeMessagePassing` expose inspectable algebraic building blocks | Implemented; fixed-complex layer, not a full simplicial training framework |
| Time series and sensor signals | Takens delay embedding plus sliding-window persistence features | Implemented |
| Learned embeddings | `audit_embedding` reports topology diagnostics using a heuristic persistence threshold | Prototype diagnostic |
| Reproducible topology-ML research | Seeded benchmarks, BCa bootstrap, paired tests, BH-FDR, provenance, and claim-to-evidence documents | Repository research infrastructure |

TopoGeoML is most useful when topology is a plausible source of signal or a desired structural constraint and you want a small, inspectable pipeline around it. It is not intended to replace broad computational-topology libraries, general graph-learning frameworks, or specialized high-throughput GPU persistence implementations.

## What the graph study established

The graph study asks whether Hodge-Laplacian propagation itself provides a unique classification advantage. Across the tested matched-capacity configurations, no unique `L_0` Hodge advantage is supported. In H008c on NCI1, a normalized-adjacency arm using the same external identity-skip architecture as Hodge reaches 0.629 versus 0.609 for Hodge. H010 then finds a significant adjacency advantage on MUTAG, no significant operator difference on PROTEINS, and the same favorable adjacency direction on NCI1.

H008c also shows that the tested external-residual adjacency formulation recovers performance after normalization with an internal self contribution does not. That causal statement is intentionally scoped. `gin-normalised` places a trainable self term inside the affine/nonlinear update, whereas `gin-residual` adds an identity skip outside the activation. The result identifies the successful tested self-path formulation; it does not prove that residual connections alone are the sole mechanism in arbitrary architectures.

There is a narrow positive result. On NCI1, `hodge-mp-residual` outperforms the matched-capacity MLP baseline by a median 8.6 percentage points within the H003 comparison family (`p_BH = 4.83e-3`). Later operator ablations show that the improvement is not unique to the Hodge Laplacian. The former investigation-wide 59-comparison and 76-entry sensitivity analysis remains withdrawn as a current claim because it included invalidated H009 comparisons and has not been regenerated from the validated comparison set.

The sheaf question is now resolved by a corrective replication. The historical H009 run was invalidated by implementation audit (its arm did not guarantee the stated cellular-sheaf Laplacian and missed the capacity tolerance) and remains provenance only. The repaired, invariant-tested `sheaf-residual` 2.0.0 operator was rerun under the preregistered H009-R protocol: the learned scalar sheaf construction beats the matched MLP (`p_BH = 4.73e-3`), shows no significant difference from the fixed Hodge operator (`p_BH = 0.428`, not an equivalence claim), and sits below the matched normalized-adjacency arm without crossing H41's strict 0.01 falsification threshold (`p_BH = 0.0342`, inconclusive). No learned-sheaf advantage over the fixed operator was detected in this regime.

The higher-order question also remains open. H011's NCI1 `L_1` arm does not significantly outperform MLP and is tested on a dataset where 96% of graphs have no triangles. The triangle-rich COLLAB follow-up H011b has only a one-seed directional smoke result so far.

This matters for users. TopoGeoML does not present `L_0` Hodge propagation as a universal GNN upgrade. The negative, inconclusive, and invalidated results remove unjustified use cases while leaving the reusable topology features, losses, diagnostics, signal tools, and higher-order research primitives intact.

See [`STATUS.md`](STATUS.md), [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md), [`LEADERBOARD.md`](LEADERBOARD.md), and [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) for the current evidence record. The older [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) is explicitly preserved as the Version 0.0.2 historical snapshot through H008c.

## Status

TopoGeoML 0.0.7 is beta scientific software. Public APIs can change before 1.0.

Required CI enforces 100% line and 100% branch coverage on the importable `topogeoml` package under full dependencies. Mypy strict mode and ruff are enforced in CI. The `benchmarks/` tree is separate research infrastructure and is not included in the package-coverage claim. The exact verified test and coverage snapshot is recorded in [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md).

The project reports negative, null, exploratory, invalidated, and positive findings separately. A non-significant comparison is not treated as proof of equivalence. An equivalence claim would require a preregistered equivalence margin and an appropriate equivalence procedure.

## Installation

Core topology features and data structures:

```bash
pip install topogeoml
```

Differentiable Vietoris-Rips persistence and Hodge layers:

```bash
pip install "topogeoml[torch]"
```

Differentiable cubical persistence and `CubicalTopologyLoss` require both PyTorch and GUDHI:

```bash
pip install "topogeoml[torch,tda]"
```

For development from source:

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
pip install -e ".[dev]"
pytest
```

## Quick start

### Persistent-homology features in scikit-learn

```python
import numpy as np

from topogeoml import TopologyFeaturePipeline

rng = np.random.default_rng(42)
theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
t = np.linspace(-1, 1, 50)

circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
circle += 0.05 * rng.standard_normal((50, 2))

line = np.stack([t, np.zeros(50)], axis=1)
line += 0.05 * rng.standard_normal((50, 2))

pipe = TopologyFeaturePipeline(max_homology_dim=1, resolution=10)
features = pipe.fit_transform([circle, line])

print(features.shape)  # (2, 200)
```

`TopologyFeaturePipeline` is a standard scikit-learn transformer, so it can be placed inside a `Pipeline` and fit within each cross-validation fold.

### Cubical topology loss for segmentation

```python
import torch

from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

# Example structural prior: one foreground connected component.
topo_loss = CubicalTopologyLoss(target_betti={0: 1}, invert=True)

pred = torch.rand(
    4, 1, 64, 64,
    dtype=torch.float64,
    requires_grad=True,
)

loss = topo_loss(pred)
loss.backward()
```

Use this as an auxiliary structural loss when the desired topology is known. The implementation is tested as a differentiable primitive; the repository does not claim that adding it improves every segmentation task.

### Hodge message passing

```python
import networkx as nx
import torch

from topogeoml import graph_to_clique_complex
from topogeoml.nn.hodge import build_hodge_layer_from_complex

complex_ = graph_to_clique_complex(nx.complete_graph(5), max_dim=2)
layer = build_hodge_layer_from_complex(
    complex_,
    k=0,
    in_features=16,
    out_features=8,
)

x = torch.randn(complex_.n_simplices(0), 16)
out = layer(x)
print(out.shape)  # torch.Size([5, 8])
```

At `k=0`, the normalized Hodge operator is the normalized graph Laplacian. On positive-degree vertices it has the familiar form `I - D^{-1/2} A D^{-1/2}`; isolated vertices follow the implementation's zero degree-support convention and keep zero normalized rows. It is not the standard Kipf-Welling GCN propagation operator. `HodgeMessagePassing` is a fixed-complex research building block, not a claim of state-of-the-art graph classification.

## Implemented library surface

| Subsystem | Public entry points | Notes |
|---|---|---|
| Persistence diagrams | `PersistenceDiagram`, `DiagramProvenance`, `RipsFiltration` | CPU Vietoris-Rips persistence through ripser |
| Vectorization | `PersistenceImageVectorizer`, `BettiCurveVectorizer` | Fixed-length ML features |
| Feature pipeline | `TopologyFeaturePipeline` | scikit-learn estimator/transformer |
| Simplicial algebra | `SimplicialComplex`, `hodge_laplacian`, `betti_numbers`, `is_chain_complex` | Inspectable chain and Hodge primitives |
| Graph conversion | `graph_to_clique_complex` | Clique-complex construction |
| Differentiable Rips PH | `topogeoml.nn.diff_ph` | PyTorch critical-value routing and topology losses |
| Differentiable cubical PH | `topogeoml.nn.cubical_diff_ph` | GUDHI pairings plus PyTorch critical-pixel routing |
| Hodge neural primitive | `topogeoml.nn.hodge.HodgeMessagePassing` | Fixed-complex sparse propagation |
| Signal topology | `takens_embedding`, `estimate_delay_autocorrelation`, `sliding_window_topology_features` | Time-series feature extraction |
| Embedding diagnostics | `audit_embedding`, `EmbeddingTopologyAudit` | Prototype heuristic audit |
| Experiment provenance | `ExperimentConfig`, `load_experiment_config`, `write_results` | Reproducible configuration and result metadata |

Torch-dependent neural symbols are imported from their subpackages so the core package remains importable without PyTorch.

## Research harness

The `benchmarks/`, `notebooks/`, and hypothesis files belong to the source repository and support the empirical record. They are not the same thing as the installed `topogeoml` wheel.

A short Hodge smoke run from a source checkout is:

```bash
pip install -e ".[all]"
python -m benchmarks.hodge \
  --datasets mutag \
  --seeds 0 1 2 \
  --n-epochs 5
```

This exercises the real benchmark path but is not a reproduction of the full seeded claims. Exact reproduction commands for the preregistered studies are in [`REPRODUCING.md`](REPRODUCING.md).

## Evidence discipline

The repository separates software correctness from empirical claims.

- Package correctness is gated by tests, strict type checking, linting, dependency audit, and 100% line and branch coverage under the full-dependency package gate.
- Empirical claims point to seeded result artifacts and reproduction commands.
- Pairwise graph experiments use paired Wilcoxon tests and Benjamini-Hochberg correction within their declared families.
- The former investigation-wide 59/76 sensitivity analysis remains withdrawn; H009-R completes the validated comparison set, but no regenerated table is claimed yet.
- Non-significance is reported as no detected difference at the tested power, not as statistical equivalence.
- Smoke runs and exploratory diagnostics are not promoted to confirmatory claims.

## What remains unproven

The public contract is intentionally narrower than the code surface.

- No claim says topology improves every ML task.
- No claim says Hodge propagation is generally better than a well-tuned GNN.
- The historical H009 result is not valid evidence about a cellular-sheaf Laplacian; the corrected H009-R replication is the valid record, and it shows no learned-sheaf advantage over the fixed Hodge operator at the tested configuration.
- No powered end-to-end result currently shows `CubicalTopologyLoss` improving medical-image segmentation.
- The H011b COLLAB `L_1` experiment has a one-seed directional smoke result, but the full statistical run remains incomplete.
- The topology-divergence callback result is exploratory because it is floor-limited and lacks a non-overfitting negative control.
- Large-scale GPU persistence, variable-topology batched Hodge networks, and a broad computational-topology algorithm catalog are outside the current package scope.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the detailed engineering and scientific limits.

## Citation

```bibtex
@software{maniches_topogeoml_2026,
  author  = {Maniches, Santiago},
  title   = {TopoGeoML: A Preregistered Investigation into Topology-Aware Graph Classification},
  year    = {2026},
  version = {0.0.7},
  doi     = {10.5281/zenodo.20365816},
  url     = {https://doi.org/10.5281/zenodo.20365816}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff). DOI: [10.5281/zenodo.20365816](https://doi.org/10.5281/zenodo.20365816).

## License

MIT. See [LICENSE](LICENSE).

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)).
