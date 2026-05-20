# TopoGeoML benchmark framework

Rigorous comparison of differentiable persistent-homology implementations
in PyTorch, with statistical machinery, full provenance, and a versioned
leaderboard.

## Scope

This framework validates the **topology** subsystem of TopoGeoML —
specifically the differentiable Vietoris-Rips persistent-homology layer
in `topogeoml.nn.diff_ph`. The library's name decomposes into three
distinct subsystems and this benchmark covers only the first:

| Subsystem | Module(s) | This benchmark validates? |
|---|---|---|
| **Topo** — persistent homology, diagram vectorizers, diff-PH layer | `topogeoml.core`, `topogeoml.nn.diff_ph` | **Yes** (this directory) |
| **Geo** — Hodge Laplacian message passing, Takens embedding, simplicial complexes as geometric objects | `topogeoml.nn.hodge`, `topogeoml.signal`, `topogeoml.core.complexes` | Not yet — separate benchmark, separate PR |
| **ML** — training callbacks, pipeline integration, downstream task accuracy | `topogeoml.pipelines`, `topogeoml.training` | Not yet — Phase 3 (real-data datasets + downstream-task axis) |

We make no claim about the geometry or end-to-end ML subsystems from
this benchmark's output. Cross-subsystem claims require cross-subsystem
benchmarks.

## What it measures

Each cell in the benchmark is a tuple `(backend, dataset, axis)`. The
runner is a matrix product over the registered backends, datasets, and
axes. No cell collapses into a single "winner" number — the tradeoffs
across axes *are* the finding.

### Axes (Phase 1)

| Axis | Question it answers | Statistical reporting |
|---|---|---|
| `correctness` | Does the backend's persistence diagram match the ripser reference (Bauer 2021) to numerical tolerance? Does it preserve `float64`? | Pass/fail per seed; no ranking. |
| `stability` | Does the backend satisfy the Cohen-Steiner / Chazal-de Silva-Oudot stability theorem for Rips persistence? What is its empirical gradient Lipschitz constant? Does it pass `torch.autograd.gradcheck`? | Theorem violations counted exactly; Lipschitz reported with bootstrap 95% CI; gradcheck pass rate. |
| `speed` | Forward and forward+backward latency, with GC-disabled measurement windows. | Per-cell min-of-medians across 5 outer passes; paired Wilcoxon signed-rank tests across backends with Benjamini-Hochberg FDR correction. |
| `optimization` | When used as a loss (longest-H_1 inflation), does descent converge? | Diagnostic only — subgradient choices can legitimately differ across backends (Hofer 2017; Carrière 2021). |

### Backends (Phase 1)

| Backend | Library | Version (at framework v1.0.0) | Reference |
|---|---|---|---|
| `topogeoml-diff-ph` | `topogeoml.nn.diff_ph` (this repo) | 0.0.1 | Elder Lemma for H_0, cocycle representative for H_1, autograd via tensor indexing |
| `torch-topological` | `torch-topological` on PyPI | 0.1.9 | Bastian Rieck et al.; wraps gudhi |

Each backend is one file in `benchmarks/backends/` implementing the
`PHBackend` protocol. Adding a new backend (gudhi-python, gudhi.tensorflow,
the Hofer 2017 reference implementation, PersLay, dionysus) is a single
PR per backend; the orchestration does not change.

### Datasets (Phase 1)

Synthetic, deterministic, with known ground-truth Betti numbers:

- `noisy_circle` — points on the unit circle plus Gaussian noise; β₁ = 1
- `two_circles` — two disjoint unit circles; β₁ = 2
- `gaussian_blob` — isotropic Gaussian point cloud; β₁ = 0 (control)

Real-data datasets (MUTAG, MNIST topology, DRIVE retinal vessels) and the
end-to-end downstream-task axis land in Phase 3.

## Statistical reporting rules (strict)

1. **Never claim a directional difference without a significance label.**
   When two backends are present, paired comparisons use Wilcoxon
   signed-rank; the family p-values are corrected via the Benjamini-Hochberg
   step-up procedure at α = 0.05. The report writes "no significant
   difference" when the BH-adjusted p-value exceeds α.

2. **Distinguish *underpowered* from *not significant*.** Comparisons
   with n < 20 paired observations are labeled `underpowered`; we report
   the point estimate and effect size but refuse to compute a p-value
   (the asymptotic null distribution is unreliable below that floor —
   Conover 1999, p. 281).

3. **Bootstrap CIs report the percentile interval** with ≥ 10 000
   resamples (Davidson & MacKinnon 2000). The percentile interval is not
   transformation-invariant; the BCa interval (Efron 1987) is a planned
   refinement.

4. **Cohen-Steiner is a deterministic check, not a statistical one.**
   The theorem holds for every (X, X′) pair or it does not. The report
   counts violations exactly. A single violation in any cell is a hard
   failure, not a tail event.

## Provenance contract

Every run writes a JSON file with the full provenance needed to reproduce
the result on another machine: git SHA (with dirty flag), Python and
library versions, platform string, CPU count, system memory, and an
explicit `deterministic_algorithms_set` flag. Without this metadata, a
result is not citable. The schema is versioned at `SCHEMA_VERSION` in
`benchmarks/runner.py` and the leaderboard JSON.

## Running the bench

```bash
# Run everything that's available.
python -m benchmarks

# Restrict to specific backends, datasets, or axes.
python -m benchmarks --backends topogeoml-diff-ph --datasets noisy_circle --axes correctness stability

# Write the markdown report alongside the JSON.
python -m benchmarks --output /tmp/bench.json --markdown /tmp/bench.md
```

The bench requires the `[bench]` extra:

```bash
pip install -e ".[bench]"
```

which installs `torch` and `torch-topological`. The core `topogeoml`
package does not depend on either; the framework gracefully reports
`UnavailableBackend` for any backend whose dependencies are missing.

## In CI

`.github/workflows/benchmark.yml` runs the bench on every PR that touches
`topogeoml/nn/diff_ph.py` or `benchmarks/**`, posts the markdown report
to the GHA step summary, and uploads the JSON as a build artifact for
download.

The CI hardware is a hosted `ubuntu-latest` runner (CPU-only). **CPU
rankings can flip on GPU** — both backends target GPU and that's where
they're intended to run. The report's caveat block surfaces this on
every run.

## Adding a backend

Drop one file into `benchmarks/backends/`:

```python
from typing import ClassVar
import torch
from benchmarks.backends import register_backend

@register_backend
class MyBackend:
    name: ClassVar[str] = "my-backend"
    version: ClassVar[str] = ""

    @staticmethod
    def available() -> bool:
        try:
            import my_library
        except ImportError:
            return False
        MyBackend.version = my_library.__version__
        return True

    @staticmethod
    def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
        ...

    @staticmethod
    def loss_longest_h1(X: torch.Tensor) -> torch.Tensor:
        ...
```

Add the import in `benchmarks/backends/__init__.py` so registration
happens at package-import time. The runner will pick it up automatically.

## References

- Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. *Journal of Applied and Computational Topology* 5(3), 391–423.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57(1).
- Chazal, F., de Silva, V., & Oudot, S. (2014). Persistence stability for geometric complexes. *Geometriae Dedicata* 173(1), 193–214.
- Cliff, N. (1996). *Ordinal Methods for Behavioral Data Analysis*. Erlbaum.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry* 37(1), 103–120.
- Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
- Davidson, R., & MacKinnon, J. (2000). Bootstrap tests: how many bootstraps? *J. Econometrics & Dyn. Control* 24.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Hofer, C., Kwitt, R., Niethammer, M., & Uhl, A. (2017). Deep learning with topological signatures. *NeurIPS*.
- Stinner, V. (2017). *pyperf — Python performance benchmarking toolkit.*
