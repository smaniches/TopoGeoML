# TopoGeoML

## 1. What this is and what problem it solves

TopoGeoML is a Python library for using topology in machine-learning workflows. Its core path converts point clouds into fixed-width features through **Vietoris–Rips persistent homology** and a scikit-learn-compatible transformer. Optional modules provide PyTorch topology losses, simplicial and Hodge operators, time-series features, and embedding diagnostics. The repository also holds a separate, versioned graph-classification research record. Start with the feature pipeline; use the specialized modules only when your problem requires them.

## 2. Why existing approaches fall short

A persistence diagram contains a variable number of birth–death pairs, while conventional estimators expect a fixed feature width. A raw persistence computation also does not supply gradients back to PyTorch inputs. This codebase connects established persistence algorithms to feature vectorization or selected differentiable critical values. Those connections have costs: vectorization loses information, and persistence gradients are not uniquely defined at tied filtration values. The code does **not** replace ripser, GUDHI, or a general machine-learning framework.

## 3. What is genuinely new and what is inherited

The repository's own implementation includes its pipeline, provenance records, persistence-image and Betti-curve vectorizers, training adapters, benchmark correctness gates, and a publicly retained series of negative, corrected, and unresolved experiments. These are identifiable software contributions, **not** a claim that the underlying mathematics or approach was invented here. Persistent homology comes from existing theory; ripser computes Vietoris–Rips diagrams; GUDHI supplies selected persistence generators and cubical pairings; scikit-learn and PyTorch supply estimator and automatic-differentiation interfaces. See [architecture](docs/ARCHITECTURE.md), [design decisions](docs/DESIGN_DECISIONS.md), and the [evidence index](docs/CLAIMS_TO_EVIDENCE.md).

## 4. Quickstart

Requires Git, network access for installation, a POSIX shell (Linux or macOS), and Python 3.11. The package also declares support for Python 3.12. Run these commands in a clean shell:

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Then execute the complete first-use example:

```bash
python - <<'PY'
import numpy as np
from topogeoml import TopologyFeaturePipeline

rng = np.random.default_rng(7)
angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
circle = np.column_stack((np.cos(angles), np.sin(angles)))
line = np.column_stack((np.linspace(-1, 1, 24), np.zeros(24)))
clouds = [c + 0.02 * rng.normal(size=c.shape) for c in (circle, line)]

features = TopologyFeaturePipeline(
    max_homology_dim=1, vectorizer="persistence_image", resolution=8
).fit_transform(clouds)
assert features.shape == (2, 128)
assert np.isfinite(features).all()
print("TopoGeoML quickstart OK:", features.shape)
PY
```

The expected final line is `TopoGeoML quickstart OK: (2, 128)`. The example fits a feature transformer, **not** a classifier or a scientific performance study. Windows commands and the full evaluation workflows are in [Usage](docs/USAGE.md).

## 5. Where it breaks

Vietoris–Rips persistence can become expensive on large or dense clouds; the feature pipeline processes samples serially and calibrates its vectorization grid from training inputs. Differentiable one-dimensional homology (H₁) rejects ambiguous critical-edge gradients by default, because the former nearest-edge rule could return a demonstrably incorrect gradient. Selecting GUDHI generators is optional and does not make a derivative unique at a tie. Higher-order graph results are configuration-bound: the H011b confirmatory COLLAB experiment and the withdrawn investigation-wide multiplicity analysis are not completed claims. See [Limitations](docs/LIMITATIONS.md).

## 6. Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Implementation choices and their costs](docs/DESIGN_DECISIONS.md)
- [Failures, scaling, and scientific boundaries](docs/LIMITATIONS.md)
- [API, installation, workflows, and troubleshooting](docs/USAGE.md)
- [Current evidence and unresolved experiments](STATUS.md) · [Reproduce the graph study](REPRODUCING.md) · [License](LICENSE)
