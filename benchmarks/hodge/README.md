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

## Empirical result — full ablation (MUTAG, 30 seeds, 20 epochs)

Five matched-capacity (1378-1442 params) arms tested as a single literature-grounded ablation; full hypothesis + four citations in `docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md`.

| Arm | Median accuracy (95% BCa CI) | Wilcoxon p_BH vs MLP | Verdict |
|---|---|---|---|
| `hodge-mp-classifier` (combinatorial L) | 0.697 [0.658, 0.750] | **5.66 × 10⁻⁴** | loses by 9 pp |
| **`hodge-mp-normalised`** (symm L̃ = D⁻¹/² L D⁻¹/²) | **0.789 [0.763, 0.816]** | **0.714** | **matches MLP** |
| `hodge-mp-residual` (above + identity skip) | 0.750 [0.724, 0.789] | 0.019 | loses by 4 pp (surprise) |
| `hodge-mp-deep-residual` (above + 2 stacked layers) | 0.776 [0.737, 0.789] | 0.102 | matches (weak) |
| `mlp-baseline` | 0.789 [0.763, 0.816] | — | control |

**Defensible claim.** On MUTAG at 30 seeds × 20 epochs × hidden_dim=32, a one-layer Hodge MP classifier using a symmetrically-normalised Laplacian is statistically indistinguishable from a no-topology MLP of matched capacity (paired Wilcoxon p_BH = 0.714, median Δ = +0.000, BCa CI on Hodge accuracy: [0.763, 0.816]).

**Findings.**
- **H1 (normalisation alone fixes it)** — *confirmed*. Symmetric Laplacian normalisation closes the entire 9 pp gap.
- **H2 (residual helps on top)** — *refuted*. The residual variant actually underperforms MLP at p_BH = 0.019.
- **H3 (depth helps on top)** — *refuted*. The 2-layer variant is no better than the 1-layer normalised arm.

**Honest interpretation.** Symmetric normalisation is the architectural choice that makes a minimal Hodge MP competitive with no-topology baselines on MUTAG. Residual connections and stacked layers — contrary to the literature-inspired prediction — do not further improve performance at this dataset's scale, consistent with Errica et al. 2020's finding that MUTAG cannot discriminate between simple architectures.

Full per-seed report: `notebooks/results/mutag_hodge_ablation_30seeds.md`. The prior 2-arm-only result (combinatorial Hodge vs MLP) is preserved at `notebooks/results/mutag_hodge_vs_mlp_30seeds.md` for the audit trail.

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
