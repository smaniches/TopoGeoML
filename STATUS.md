# Project Status

## What this project is

TopoGeoML is a preregistered research investigation into whether Hodge decomposition improves graph neural network classification. It is a research prototype, not a production framework.

## Investigation summary

14 preregistered hypotheses (H001-H011b) with 53 falsifiable sub-predictions, tested across 4 datasets (MUTAG, PROTEINS, NCI1, COLLAB), 11 model variants, and 59 distinct pairwise statistical comparisons (76 computed in total, including 17 baseline re-reports across hypothesis families).

### Findings

1. **Topology-aware message passing with external residual outperforms MLP on NCI1 by 8-10 pp.** The Hodge-residual-vs-MLP comparison is significant at per-hypothesis BH (p_BH = 4.83e-3, within the H003 ablation family) and survives investigation-wide Benjamini-Hochberg correction over the 59 distinct comparisons (rank 22/59, threshold 1.86e-2) but not Bonferroni (threshold 0.05/59 = 8.47e-4). *Regime-bound:* this is a matched-capacity comparison (best arm 0.609 vs MLP 0.523, both ~20 pp below the ~0.80+ that properly-trained GNNs reach on NCI1; GIN/GAT collapse to class prior under the same protocol). It isolates architectural mechanism at fixed capacity and is not a benchmark-performance claim.

2. **The external residual connection is the operative architectural factor.** The choice of propagation operator (Hodge Laplacian, normalised adjacency, or learned sheaf) is secondary. All three operators perform comparably once the external residual is present; all collapse to class prior without it.

3. **The Hodge Laplacian (L_0) does not confer a unique advantage** on any tested dataset. Normalised adjacency with external residual matches or exceeds Hodge on all three chemistry/protein benchmarks.

4. **The NCI1 advantage does not transfer to MUTAG or PROTEINS** at the tested configuration (1-layer, hidden_dim=32, 10-20 epochs, no batch normalisation).

5. **L_1 (edge-level Hodge Laplacian) is untested on triangle-rich data at statistical rigor.** NCI1 is triangle-sparse (96% of graphs have 0 triangles), so L_1 effectively degenerates to the down-Laplacian. COLLAB (mean 9,290 triangles per graph) showed a directionally strong smoke result (+14.8 pp, 1 seed) but the full experiment has not completed due to compute constraints.

### What is not claimed

- "Topology helps graph classification" — not supported across datasets
- "Hodge is better than GNNs" — refuted (H008-c)
- "L_1 captures unique structural signal" — not yet tested at rigor
- Any result beyond the tested configuration

## Quality

| Metric | Value |
|---|---|
| Test suite | Required CI matrix; the exact verified full-dependency snapshot and reproduction command are recorded in [docs/CLAIMS_TO_EVIDENCE.md](docs/CLAIMS_TO_EVIDENCE.md) |
| Coverage | 100% line **and** 100% branch coverage on the `topogeoml` package with full dependencies (`.[all]`), enforced by the full-deps `coverage-gate` CI job (`--cov-branch --cov-fail-under=100`); the `benchmarks/` research harness is intentionally outside the gated package scope |
| Type checking | mypy strict enforced in CI |
| Lint | ruff, all checks passing |
| DOI | [10.5281/zenodo.20365816](https://doi.org/10.5281/zenodo.20365816) |
| Statistical analysis | Investigation-wide BH-FDR across 59 distinct comparisons (76 computed; [docs/STATISTICAL_SUMMARY.md](docs/STATISTICAL_SUMMARY.md)) |
| Preregistration | 14 hypothesis documents with git-timestamped commit history |
| Negative results | 28 of 59 distinct comparisons (47%), and 29 of 76 total (38%), are non-significant; all reported |

## Open items

| Item | Status | Next step |
|---|---|---|
| H011-b (L_1 on COLLAB) | Smoke test completed; full run pending | Run locally on higher-compute hardware |
| Historical research report | [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) is the Version 0.0.2 snapshot through H008c; H009-H011b are documented in the current hypothesis/evidence files | Preserve the historical snapshot; use this status file, [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md), and [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) for current project state |
| Cross-domain validation | Chemistry/protein experiments are statistically evaluated; COLLAB provides a social-network smoke result only | Complete H011-b on COLLAB before making any cross-domain L_1 claim |
| Multi-layer architectures | All results at 1-layer, hidden_dim=32 | Test with 2-5 layers and batch normalisation only as a new preregistered research extension |

## How to verify

See [REVIEWER.md](REVIEWER.md) for a 10-minute verification path. See [docs/CLAIMS_TO_EVIDENCE.md](docs/CLAIMS_TO_EVIDENCE.md) for every claim mapped to evidence.
