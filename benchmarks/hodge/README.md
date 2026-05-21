# TopoGeoML Hodge subsystem benchmark

Rigorous validation of `topogeoml.nn.hodge.HodgeMessagePassing` as a graph-classification building block on a real-data TUDataset benchmark.

## Scope (Phase 1 — Geo subsystem)

This bench answers one question per registered dataset:

> *Does a HodgeMP-based classifier outperform a feature-MLP baseline on this graph-classification task, with the difference statistically significant under paired Wilcoxon + Benjamini-Hochberg FDR correction at α=0.05?*

Phase 1 ships:
- **Datasets**: MUTAG (188 molecular graphs, 2 classes; Debnath et al. 1991).
- **Models**: `hodge-mp-classifier` (one inline Hodge propagation step + sum-pool + linear head), `mlp-baseline` (no-topology control with matched capacity).
- **Axis**: graph-classification test accuracy on a stratified 80/20 split, averaged across N seeds.

The bench reuses `benchmarks/stats.py` (bootstrap CI, Mann–Whitney / Wilcoxon, BH correction) so the reporting discipline is identical to the diff-PH bench.

## Empirical result (MUTAG, 30 seeds, 20 epochs)

The current published result for the minimal one-layer Hodge architecture is **negative**: the Hodge classifier *underperforms* the MLP baseline.

| Model | Median accuracy (95% bootstrap CI) |
|---|---|
| `hodge-mp-classifier` | 0.697 [0.658, 0.750] |
| `mlp-baseline` | **0.789 [0.763, 0.816]** |

Paired Wilcoxon (BH-corrected): median Δ = **−0.092**, **p = 5.66 × 10⁻⁵**, rank-biserial **r = −0.760**.

This negative result is **only visible after** fixing the critical bug in `models.py` (PR #12) where the previous `HodgeMessagePassing` layer was instantiated inside `forward_one()` and its weights were neither registered with the optimizer nor preserved across calls. The pre-fix bench reported both models at ~70% — measuring an MLP through a random topology filter, not the Hodge architecture.

**Honest interpretation.** This rules out the minimal one-layer Hodge architecture as a useful inductive bias on MUTAG. It does *not* rule out deeper Hodge architectures, normalised Laplacians (the current bench uses combinatorial L_0), richer node features, or attention-weighted propagation. Those are the natural next steps for the Geo subsystem.

Full per-seed report: `notebooks/results/mutag_hodge_vs_mlp_30seeds.md`.

## Architecture

```
benchmarks/hodge/
├── models.py             # GraphClassifier protocol + 2 registered models
├── datasets.py           # TUDataset adapters; MUTAG in Phase 1
├── classification.py     # train + test_accuracy axis
├── runner.py             # orchestrates models × datasets + pairwise tests
├── __main__.py           # `python -m benchmarks.hodge` CLI
└── leaderboard/          # versioned JSON output (created at first run)
```

Adding a TUDataset (PROTEINS, NCI1, ENZYMES) is one entry in `benchmarks/hodge/datasets.py`. Adding a model (e.g. a Hodge multi-layer SCN) is one entry in `models.py`.

## Why per-graph training

The HodgeMP layer caches a Laplacian as a fixed buffer. Batching graphs with different topology would require either block-diagonal Laplacians (PyG's standard trick) or padding to the largest graph. For MUTAG (188 graphs, avg 18 nodes) per-graph training is fast enough that we avoid both engineering costs.

## Running

```bash
# Default: both models × MUTAG × 5 seeds × 20 epochs
python -m benchmarks.hodge

# Restrict
python -m benchmarks.hodge --models mlp-baseline --seeds 0 1 --n-epochs 5

# Output a markdown report alongside the JSON
python -m benchmarks.hodge --markdown /tmp/hodge_report.md
```

Requires the `[bench]` extra (`pip install -e ".[bench]"`).

## In CI

`.github/workflows/benchmark-hodge.yml` runs the bench on PRs touching:
- `topogeoml/nn/hodge.py`
- `topogeoml/core/complexes.py`
- `topogeoml/data/graph_to_complex.py`
- `benchmarks/hodge/**`

The TUDataset download is cached between runs via `actions/cache`.

## Methodology citations

- Lim, L.-H. (2020). "Hodge Laplacians on Graphs." *SIAM Review* 62(3).
- Debnath, A. K., Lopez de Compadre, R. L., et al. (1991). "Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds." *J. Med. Chem.* 34.
- Morris, C., Kriege, N. M., Bause, F., et al. (2020). "TUDataset." *ICML 2020 GRL+ workshop*.
- Wilcoxon, F. (1945). "Individual comparisons by ranking methods." *Biometrics Bulletin* 1.
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate." *JRSS-B* 57(1).
