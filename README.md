# TopoGeoML

**A preregistered, self-falsifying investigation into topology-aware graph classification — plus a differentiable-TDA toolkit.**

Differentiable persistent-homology layers, Hodge message passing, and a benchmark framework with preregistered hypotheses and statistically defensible reporting. The headline scientific question — *does encoding topological structure via the Hodge Laplacian improve graph classification beyond node features?* — was tested across 14 preregistered hypotheses and **answered in the negative**: once an external residual connection is present, a plain normalised-adjacency operator matches or exceeds the Hodge Laplacian. The operative factor is the residual, not the topology. The library is positioned as *complementary* to PyTorch / TensorFlow, not a replacement.

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
[![Version](https://img.shields.io/badge/version-0.0.5--beta-green)](#status)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20365816.svg)](https://doi.org/10.5281/zenodo.20365816)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Status

**A research investigation with a primarily negative headline result, plus a working toolkit.** The library is internally consistent (504 tests; 100% line and 100% branch coverage on the `topogeoml` package when run with full dependencies, gated in CI; the `benchmarks/` research harness is below 100% — see [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) Claim 6), type-checked with mypy in strict mode, and statistically validated with investigation-wide FDR control (see [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md)).

**Primary finding (negative).** Across 14 preregistered hypotheses (H001–H011b, 53 falsifiable sub-predictions), encoding topological structure via the Hodge Laplacian does **not** confer a unique advantage for graph classification on any tested dataset. Once an external residual connection is present, a normalised *adjacency* operator matches or exceeds the Hodge Laplacian; on NCI1, without that residual both fall to near the class prior (~0.51 vs a 0.50 prior), while on MUTAG and PROTEINS the no-residual arms still sit above it. The operative architectural factor is the residual connection, not the topology (see H008c).

**Secondary finding (positive, narrow).** On NCI1 (4110 graphs), a one-layer message-passing classifier *with* an external residual outperforms a matched-capacity MLP by 8–10 pp (paired Wilcoxon p_BH < 0.01; survives investigation-wide BH but not Bonferroni).

> **Important — regime caveat, read before citing any accuracy number.** All results are obtained under a deliberately constrained *matched-capacity* protocol (1 layer, hidden_dim=32, 10–20 epochs, no batch normalisation, ~1.4–2.3k parameters per arm). Under this protocol the standard GNN baselines (GIN, GAT) **collapse to the class prior (0.500)** on NCI1, and the best arm reaches ~0.61–0.63 — roughly **20 percentage points below the ~0.80+ that properly-trained GNNs achieve** on this benchmark in the literature. These comparisons isolate *architectural mechanism at fixed capacity*; they are **not** statements about leaderboard performance, and phrases like "outperforms GIN/GAT" must be read in that light. See [`LIMITATIONS.md`](LIMITATIONS.md) §0 and [`LEADERBOARD.md`](LEADERBOARD.md).

This is a research toolkit, sized at ~7K LOC, positioned for researchers who need correct + citable topology-aware layers and a rigorous statistical harness. It is **not** a production training framework, and it does **not** claim competitive benchmark accuracy. APIs will change without notice until v1.0.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list of what does *not* work yet.

---

## Empirical evidence

Every claim in the rest of this README is backed by an in-repo experiment or a literature citation, and every experiment is reproducible from the scripts in `notebooks/`. The full empirical record — including pending experiments and the discipline rules — lives in [`LEADERBOARD.md`](LEADERBOARD.md). **All accuracy numbers below are obtained in the constrained matched-capacity regime described in the [regime caveat](#status) above**; they isolate architectural mechanism at fixed capacity and are not benchmark-performance claims.

### 1. Topology divergence watchdog never fires later than a val-loss watchdog (exploratory — floor-limited, no control yet)

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

The directional verdict is consistent across all 30 seeds — topology never fires *later* than loss (14 earlier, 16 tied at the same step, 0 later). **Why this is exploratory, not a positive finding:** the topology watchdog fired at step 30 — its *earliest possible* step (the baseline window must fill first) — in every one of the 30 seeds, and all 30 runs overfit (train loss → 0). Because it fires at its floor every time, the data establish only that topology is never *slower* than the loss watchdog; they do **not** establish that topology *anticipates* divergence. A no-overfitting control — a run where divergence should not be flagged at all — has not been performed, so a genuine falsification test of "topology predicts divergence" does not yet exist. We therefore report this as exploratory and inconclusive.

Reproduce: `python notebooks/topology_predicts_divergence.py --n-seeds 30`.

### 2. Symmetric-normalised one-layer Hodge MP on MUTAG matches an MLP baseline; the combinatorial variant loses by 9 pp (mixed)

MUTAG mutagenicity benchmark (188 molecular graphs, 2 classes, Debnath 1991 via PyG TUDataset), 30 independent seeds × 20 epochs of Adam(lr=1e-2), 80/20 stratified split per seed. Five matched-capacity arms (1378-1442 trainable params each) tested as a single literature-grounded ablation; see `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md` for the falsifiable hypotheses, the four citations behind each architectural choice (Kipf-Welling 2017, Bunch 2020, HL-HGAT 2024, Hodgelet GP 2024), and the resolved outcomes.

Per-arm result (full report in `notebooks/results/mutag_hodge_ablation_30seeds.md`):

| Arm | Median accuracy (95% BCa CI) | Wilcoxon p_BH vs MLP | Verdict |
|---|---|---|---|
| `hodge-mp-classifier` (combinatorial L) | 0.697 [0.658, 0.750] | **5.66 × 10⁻⁴** | loses by 9 pp |
| **`hodge-mp-normalised`** (symm L̃ = D⁻¹/² L D⁻¹/²) | **0.789 [0.763, 0.816]** | **0.714** | **matches MLP** |
| `hodge-mp-residual` (above + identity skip) | 0.750 [0.724, 0.789] | 0.019 | loses by 4 pp (surprise) |
| `hodge-mp-deep-residual` (above + 2 stacked layers) | 0.776 [0.737, 0.789] | 0.102 | matches (weak) |
| `mlp-baseline` | 0.789 [0.763, 0.816] | — | control |

**What this licenses the framework to claim.**

On MUTAG at 30 seeds × 20 epochs × hidden_dim=32, a one-layer Hodge message-passing classifier using a **symmetrically-normalised Laplacian** is statistically indistinguishable from a no-topology MLP of matched capacity (paired Wilcoxon p_BH = 0.714, median Δ = +0.000, BCa 95% CI on Hodge accuracy: [0.763, 0.816]). The unnormalised combinatorial variant underperforms by 9 pp (p_BH = 5.66 × 10⁻⁴). Symmetric normalisation is *the* architectural choice that closes the gap; **residual connections and stacked layers do not further improve performance** at this scale, and the residual variant actually slightly underperforms MLP (p_BH = 0.019).

This is a **positive equality claim** ("topology with proper normalisation is competitive"), not a "topology helps" claim ("topology beats MLP"). The literature consensus (Errica et al. 2020, arXiv 1810.09155; Yang et al. 2024 Hodgelet GPs at 88.06 ± 7.99) is that MUTAG cannot discriminate between simple architectures at its scale; both confirmation and refutation of the strong "topology helps" claim require a larger dataset.

Reproduce: `python -m benchmarks.hodge --datasets mutag --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 20`.

### 3. Cross-dataset replication on PROTEINS — equality holds; strict-positive refuted (mixed)

PROTEINS benchmark (1113 protein graphs, 2 classes, Borgwardt et al. 2005 / Dobson & Doig 2003 via PyG TUDataset; 5.9× MUTAG's sample size, 2.2× MUTAG's average graph size). Same 5-arm ablation, 30 seeds × 10 epochs, matched-capacity. Preregistered as hypothesis 002 (`docs/hypotheses/HYPOTHESIS-002-hodge-proteins.md`) before the result was known.

Per-arm result (full report in `notebooks/results/proteins_hodge_ablation_30seeds.md`):

| Arm | Median accuracy (95% BCa CI) | Wilcoxon p_BH vs MLP | Verdict |
|---|---|---|---|
| `hodge-mp-classifier` (combinatorial L) | 0.646 [0.605, 0.700] | 0.646 | matches MLP |
| `hodge-mp-normalised` (H1) | 0.688 [0.670, 0.704] | 0.548 | matches MLP |
| `hodge-mp-residual` (H2) | 0.686 [0.670, 0.717] | 0.339 | matches MLP |
| `hodge-mp-deep-residual` (H3) | 0.695 [0.659, 0.709] | 0.426 | matches MLP |
| `mlp-baseline` | 0.675 [0.596, 0.706] | — | control |

**What this means.** After BH correction across the 10 pairwise comparisons, **no arm produces a statistically significant difference from MLP**. The strong hypothesis (H1 *beats* MLP at p_BH < 0.01) is **refuted**. The cross-dataset equality (H1 = MLP) is **reconfirmed**: p_BH = 0.548 on PROTEINS replicates the p_BH = 0.714 on MUTAG.

**Surprising cross-dataset cancellation.** The MUTAG combinatorial-L harm (9 pp gap to MLP, p_BH = 5.66 × 10⁻⁴) **does not replicate** on PROTEINS (2.9 pp gap, p_BH = 0.65). Effect size drops by ~10× — meaning the combinatorial vs symm-normalised contrast that defines hypothesis 001 is a small-graph phenomenon (MUTAG avg 18 nodes/graph) that washes out at PROTEINS scale (39 nodes/graph). Two interpretations remain in play: a discrimination ceiling that PROTEINS also sits below, or genuine cancellation by larger-graph sum-pooling.

**Bottom line.** The Geo subsystem has a defensible *two-dataset equality* claim. A strict "topology helps" claim requires either a richer architecture (HL-HGAT attention, polynomial filters, SCConv up-down) or a substantially larger dataset (NCI1, DD, COLLAB). Hypothesis 003 picks the direction.

Reproduce: `python -m benchmarks.hodge --datasets proteins --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10`.

### 4. Scale-escalation on NCI1 — positive-difference result (matched-capacity regime)

NCI1 benchmark (4110 chemical-compound graphs, 2 classes, Wale et al. 2008 via PyG TUDataset; **22× MUTAG's sample size, 3.7× PROTEINS'**). Same 5-arm ablation, 30 seeds × 10 epochs, matched-capacity. Preregistered as hypothesis 003 (`docs/hypotheses/HYPOTHESIS-003-hodge-nci1.md`) BEFORE the result was known, with five sub-hypotheses (H8–H12) and an outcome decision tree.

Per-arm result (full report in `notebooks/results/nci1_hodge_ablation_30seeds.md`). The `p_BH` column is per-family Benjamini-Hochberg within the H003 ablation:

| Arm | Median accuracy (95% BCa CI) | Wilcoxon p_BH vs MLP | Verdict |
|---|---|---|---|
| `hodge-mp-classifier` (combinatorial L) | 0.506 [0.501, 0.511] | **2.6 × 10⁻⁴** | loses 1.7 pp |
| `hodge-mp-normalised` (H1) | 0.516 [0.511, 0.523] | 0.253 | matches MLP |
| **`hodge-mp-residual` (H2)** | **0.609 [0.581, 0.625]** | **4.83 × 10⁻³** | **BEATS MLP by 8.6 pp** |
| `hodge-mp-deep-residual` (H3) | 0.603 [0.594, 0.623] | 1.18 × 10⁻² | beats MLP by 8.0 pp |
| `mlp-baseline` | 0.523 [0.513, 0.566] | — | control |

Under investigation-wide BH across the 59 distinct comparisons (of 76 total computed; the investigation-wide FDR is computed over the distinct set), the `hodge-mp-residual` vs MLP result is rank 22/59 (threshold 1.86 × 10⁻²) — significant, but not Bonferroni-significant (see [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md) §2).

**Defensible claim (the framework's one narrow strict positive-difference real-data result):**

> On NCI1 at 30 seeds × 10 epochs × hidden_dim=32, a one-layer Hodge MP classifier with a symmetrically-normalised Laplacian AND an identity residual connection strictly outperforms a no-topology MLP baseline of matched capacity (median Δ = +0.086, paired Wilcoxon p_BH = 4.83 × 10⁻³, rank-biserial r = +0.533, BCa 95% CI on Hodge accuracy: [0.581, 0.625]).

> **Important — regime caveat.** This is a *matched-capacity, fixed-architecture* comparison, not a benchmark-performance claim. The MLP baseline sits at 0.523 and the Hodge arm at 0.609 — both ~20 pp below the ~0.80+ that properly-tuned GNNs reach on NCI1. The result says only "at ~1.4k parameters and one layer, the residual architecture extracts more signal than a same-capacity MLP." Crucially, H008c shows a normalised-*adjacency* operator with the same external residual does this *slightly better* than the Hodge Laplacian — so this is **not** evidence that topology per se helps.

**Surprising cross-dataset twist.** The residual variant — which *lost* to MLP on MUTAG (p_BH = 0.019) and *matched* on PROTEINS (p_BH = 0.339) — **wins** on NCI1. The residual's contribution scales positively with dataset size at this architectural class. The cross-dataset behaviour table:

| Architecture | MUTAG (188) | PROTEINS (1113) | NCI1 (4110) |
|---|---|---|---|
| combinatorial L | LOSES (-9pp) | matches | LOSES (-1.7pp) |
| symm L̃ | matches | matches | matches |
| symm L̃ + residual | LOSES (-4pp) | matches | **WINS (+8.6pp)** |
| symm L̃ + 2L + residual | matches | matches | **WINS (+8pp)** |

The same architecture's verdict inverts across dataset scale. Two candidate mechanisms were proposed: (a) NCI1's 37-dim dense features let the residual augment the propagated signal rather than displacing sparse one-hots; (b) NCI1's larger training set lets the optimiser actually learn to use the residual. Both were investigated and ruled out — see [Mechanism Investigation](#mechanism-investigation-h004h007) below.

Reproduce: `python -m benchmarks.hodge --datasets nci1 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10`.

---

## Mechanism Investigation (H004–H007)

The cross-dataset inversion above motivated a systematic mechanism-elimination program. Full details in [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md); summary below.

### Findings

| Hypothesis | Question | Outcome |
|---|---|---|
| **H004** (sample size) | Does subsampling NCI1 to MUTAG-size kill the Hodge advantage? | **No.** NCI1@188 graphs: Δ = +0.019, p_BH = 0.897. The advantage persists at all sample sizes tested. |
| **H005** (feature density) | Does projecting NCI1 to 7-dim features kill the Hodge advantage? | **No.** NCI1-7d: Δ = +0.081, p_BH = 4.93 × 10⁻⁴. MLP collapses to class prior; Hodge remains above chance. |
| **H006** (graph topology) | Can Hodge classify from graph structure alone (constant features)? | **Yes, on all datasets.** MUTAG +0.098, PROTEINS +0.088, NCI1 +0.071 (all p_BH < 5 × 10⁻⁴). |
| **H007** (structural proxies) | Which graph-structural property explains the full-feature gain? | **None individually.** All five proxies (size, degree, WL, cycle, spectral) are rank-inverted vs. the full-feature gain. |

### Positive results from mechanism investigation

1. **Feature-degradation robustness:** On NCI1-7d, MLP drops to 0.500 (class prior) while Hodge-residual achieves 0.581 — the Hodge architecture reads graph-structural signal the MLP cannot access.
2. **Universal graph-structural signal:** Under constant-feature control, the Hodge architecture extracts classification signal from topology alone on ALL three datasets (all p_BH < 5 × 10⁻⁴).
3. **Complementarity pattern:** The Hodge advantage under full features is largest where graph-structural separability is *lowest* (NCI1) — consistent with the Hodge Laplacian providing *complementary* information where the MLP fails to extract class signal from features alone.

### Current interpretation

The mechanism is narrowed to an architecture-data complementarity interaction. The Hodge architecture adds value where the no-topology baseline cannot extract class signal from node features — not where graph structure inherently carries the most information. Deeper architectures and additional datasets are the next experimental direction.

Reproduce all mechanism experiments: see [`REPRODUCING.md`](REPRODUCING.md).

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
| Topology-divergence callback | `topogeoml.training.ShapeOfLearningCallback` | done | implemented; the divergence-vs-loss comparison is exploratory (floor-limited, no negative control) — see the empirical evidence section above |
| Signal analysis | `topogeoml.signal.{delay_embedding,sliding_window}` | done | Takens embedding + windowed topology features |
| Embedding audit | `topogeoml.audits.audit_embedding` | prototype | heuristic significance threshold; calibrated noise floor pending |
| **Benchmark framework** | `benchmarks/` | done | 4 backends × 4 axes (correctness/stability/speed/optimization); cross-backend tests need the `bench` extra |
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

- Explicit `float64` dtype on every NumPy numerical array; torch layers follow torch's `float32` default and preserve `float64` when the caller requests it
- No Python sample loops for numerical computation (construction loops permitted)
- `random_state=42` / `np.random.default_rng(42)` for reproducible RNG
- Provenance metadata (model, seed, platform, dependency versions) on every benchmark cell
- 100% line and 100% branch coverage on the library (`topogeoml/`) with full dependencies, enforced by a dedicated full-deps CI gate (`--cov-branch --cov-fail-under=100`); the `benchmarks/` research harness is high but below 100% and is intentionally out of the gated scope (see [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) Claim 6)
- ruff clean across all source directories
- Every empirical claim in any docstring or README must point to either a literature citation or an in-repo experiment (negative results count and are shipped)

---

## Testing

```bash
pytest                          # 504 tests
pytest -m "not slow"            # skip slow tests
pytest --cov=topogeoml --cov=benchmarks  # with coverage
```

Coverage is 100% line **and** 100% branch on the `topogeoml/` package (with torch installed), proven by the full-deps `coverage-gate` CI job, which installs `.[all]` (torch CPU wheels) and fails below 100% under `--cov-branch`. The `benchmarks/` research harness is ~93% line: cross-backend tests skip without the `torch-topological` backend (the `bench` extra), and a few hodge-analysis paths are partially covered. The harness is deliberately kept out of the gated scope (the gate is `--cov=topogeoml`), so the 100% gate is not diluted by harness gaps. Torch-gated tests skip cleanly when torch is not installed. See [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) Claim 6.

---

## Roadmap

**v0.0.5 (current).** The current package and citable version — a correctness and reviewer-driven precision release over v0.0.4 (scale-invariant Hodge isolated-simplex normalization, a corrected differentiable Rips H0 reconstruction under `max_edge_length`, full essential-bar counting in the Betti regularization loss, a real cubical-complex `gradcheck` test, and statistical-claim doc-honesty fixes); no new empirical results. Primary finding negative: the Hodge Laplacian confers no unique advantage over a normalised-adjacency operator once an external residual is present (H008c). One narrow, regime-bound positive difference on NCI1 (+8.6 pp, p_BH = 4.83 × 10⁻³; survives investigation-wide BH but not Bonferroni; absolute accuracy ~20 pp below SOTA — see regime caveat). Preregistered hypothesis series H001–H011b (including GIN/GAT comparison, residual-placement ablation, sheaf Laplacian, and L_1 edge-level propagation). Full academic infrastructure (CITATION.cff, Zenodo DOI, reproduction guide, investigation-wide statistical summary). 504 tests; 100% line and 100% branch coverage on the `topogeoml` package with full dependencies, gated in CI (the `benchmarks/` harness is below 100%); type-checked with mypy strict in CI.

**Next.** Cross-domain validation (DD, COLLAB, social-network benchmarks). DRIVE retinal-vessel segmentation with `CubicalTopologyLoss` (Dice + BCE + λ·topo vs baseline). Continued mechanism ablation (spectral vs spatial operator isolation). The bar remains paired Wilcoxon p < 0.01 after BH correction.

**v0.1 and later.** Cross-domain validation (social networks, citation graphs). Cross-PLM experiments (ProtT5, SaProt embeddings as node features). Feature-interaction ablations with controlled dimensionality sweeps. Conditional on the empirical results above determining which direction has the most signal.

---

## Citation

```bibtex
@software{maniches_topogeoml_2026,
  author       = {Maniches, Santiago},
  title        = {TopoGeoML: A Preregistered Investigation into Topology-Aware Graph Classification},
  year         = {2026},
  version      = {0.0.5},
  doi          = {10.5281/zenodo.20365816},
  url          = {https://doi.org/10.5281/zenodo.20365816},
  orcid        = {0009-0005-6480-1987}
}
```

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button from it). DOI: [10.5281/zenodo.20365816](https://doi.org/10.5281/zenodo.20365816).

---

## License

MIT. See [LICENSE](LICENSE).

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)).
