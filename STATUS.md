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
| Tests | 497 |
| Coverage | 100% line coverage with full dependencies (`.[all]`); reported but not gated in CI (torch-less environment) |
| Type checking | mypy strict enforced in CI |
| Lint | ruff, all checks passing |
| DOI | [10.5281/zenodo.20564298](https://doi.org/10.5281/zenodo.20564298) |
| Statistical analysis | Investigation-wide BH-FDR across 59 distinct comparisons (76 computed; [docs/STATISTICAL_SUMMARY.md](docs/STATISTICAL_SUMMARY.md)) |
| Preregistration | 14 hypothesis documents with git-timestamped commit history |
| Negative results | 28 of 59 distinct comparisons (47%), and 29 of 76 total (38%), are non-significant; all reported |

## Open items

| Item | Status | Next step |
|---|---|---|
| H011-b (L_1 on COLLAB) | Smoke test completed; full run pending | Run locally on higher-compute hardware |
| RESEARCH_REPORT.md | Covers H001-H007; does not reflect H008-H011 findings | Update after COLLAB result |
| Cross-domain validation | Only chemistry/protein datasets tested | Test on social-network, citation-graph benchmarks |
| Multi-layer architectures | All results at 1-layer, hidden_dim=32 | Test with 2-5 layers, batch normalisation |

## How to verify

See [REVIEWER.md](REVIEWER.md) for a 10-minute verification path. See [docs/CLAIMS_TO_EVIDENCE.md](docs/CLAIMS_TO_EVIDENCE.md) for every claim mapped to evidence.
