---
title: Claims → evidence
nav_order: 5
---

# Claims to Evidence

Principal numerical and empirical claims in the README, mapped to evidence artifacts, reproduction commands, expected tolerances, and limitations. Per-arm statistics in the Empirical Evidence tables are sourced directly from the JSON artifacts listed below; this document maps the summary-level claims, not every individual table cell.

## Methodology

- Claims are extracted from README.md as of the current commit.
- Each claim must have a JSON artifact in `notebooks/results/` or a CI command that produces it.
- Tolerances account for hardware-specific floating-point variation (see [REPRODUCING.md](https://github.com/smaniches/TopoGeoML/blob/main/REPRODUCING.md) §Expected Numerical Variation).
- "Survives global BH" indicates whether the claim's p-value survives investigation-wide Benjamini-Hochberg correction across the 59 distinct comparisons (of 76 total computed; see [STATISTICAL_SUMMARY.md](STATISTICAL_SUMMARY.md)).

---

## Claim 1: "100% line and 100% branch coverage on the `topogeoml` package under the required full-dependency gate"

| Field | Value |
|---|---|
| Evidence | `pytest -m "not gpu" --cov=topogeoml --cov-branch --cov-fail-under=100` after `pip install -e ".[all]"`. |
| Artifact | The dedicated full-deps `coverage-gate` job in `.github/workflows/ci.yml` installs `.[all]` with the CPU torch wheel set and fails the build below 100% package coverage. |
| Verified snapshot | At merge commit `9ddfa0d8156637bfd1d42ec9609f299ce337bf00`, the required full-deps gate reported **507 passed, 35 skipped; 1,297 statements, 0 missed; 402 branches, 0 partial; 100.00% package coverage**. |
| Invariant | The number of collected test items may increase as regression tests are added. The maintained invariant is 100% line and 100% branch coverage on the importable `topogeoml` package under the required full-dependency gate. |
| Limitation | The default `test` CI job installs `.[dev]` without torch, so `topogeoml/nn/` code paths may be import-skipped and package coverage is reported rather than gated there. The separate full-deps gate is the authoritative package-coverage check. The `benchmarks/` research harness is outside this package coverage claim; see Claim 6. |

---

## Claim 2: NCI1 positive difference (+8.6 pp, p_BH = 4.83 x 10^-3)

| Field | Value |
|---|---|
| Evidence | `notebooks/results/nci1_hodge_ablation_30seeds.json` |
| Artifact key | `pairwise_comparisons[hodge-mp-residual vs mlp-baseline]` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |
| Expected | median_diff: 0.086 +/- 0.005; p_BH: 4.83e-3 +/- factor of 2 |
| Survives global BH | Yes (rank 22/59, threshold 1.86e-2) |
| Survives Bonferroni | No (threshold 6.58e-4) |
| Limitation | One dataset (NCI1), one configuration (1-layer, hidden_dim=32, 10 epochs). Does not replicate on MUTAG or PROTEINS at this configuration. Subsequent ablation (H008-c) showed the operative factor is the external residual, not the Hodge Laplacian. |

---

## Claim 3: "topology-aware message passing with external residual outperforms MLP by 8-10 pp"

| Field | Value |
|---|---|
| Evidence | `notebooks/results/h008c_nci1_gin_residual_30seeds.json` |
| Artifact key | `pairwise_comparisons[gin-residual vs mlp-baseline]`: Delta +0.106, p_BH = 6.05e-4 |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models gin-residual mlp-baseline --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10` |
| Expected | median_diff: 0.106 +/- 0.005; p_BH < 0.001 |
| Survives global BH | Yes |
| Survives Bonferroni | Yes |
| Limitation | NCI1 only. Does not hold on MUTAG (gin-residual matches MLP) or PROTEINS (not significantly different). |

---

## Claim 4: "external residual connection — not the Hodge Laplacian specifically — as the operative architectural factor"

| Field | Value |
|---|---|
| Evidence (H008) | `notebooks/results/h008_nci1_gin_gat_30seeds.json` — GIN/GAT without external residual collapse to class prior |
| Evidence (H008-b) | `notebooks/results/h008b_nci1_gin_normalised_30seeds.json` — normalised GIN without external residual also collapses |
| Evidence (H008-c) | `notebooks/results/h008c_nci1_gin_residual_30seeds.json` — gin-residual (with external residual) achieves 0.629 vs Hodge 0.609 |
| Reproduce | See [REPRODUCING.md](https://github.com/smaniches/TopoGeoML/blob/main/REPRODUCING.md) §H008, §H008-b, §H008-c |
| Limitation | Tested at one capacity point (1-layer, 32 hidden). Standard GIN/GAT with batch normalisation and multiple layers were not tested. |

---

## Claim 5: "graph-structural signal on all 3 datasets (all p_BH < 5 x 10^-4)"

| Field | Value |
|---|---|
| Evidence | `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.json` |
| Artifact key | Hodge accuracy vs class prior per dataset |
| Reproduce | See [REPRODUCING.md](https://github.com/smaniches/TopoGeoML/blob/main/REPRODUCING.md) §H006 |
| Expected | MUTAG: gap +0.098, p = 4.53e-6; PROTEINS: gap +0.088, p = 1.41e-4; NCI1: gap +0.071, p = 1.93e-5 |
| Survives global BH | Yes (all three) |
| Limitation | These p-values are from the Hodge-vs-class-prior comparison within the H006 resolver, not the Hodge-vs-MLP comparison in the raw JSON. The class prior is the theoretical baseline (majority-class accuracy), not the MLP's constant-feature accuracy. |

---

## Claim 6: Coverage scope for the library and research harness

| Field | Value |
|---|---|
| Library (`topogeoml`) | **100% line and 100% branch coverage** under the required full-dependency gate. Every importable `topogeoml/*` file in that gate reports no missed lines and no partial branches. |
| Research harness (`benchmarks`) | Outside the package coverage invariant. The benchmark tree is research infrastructure and is exercised through its own tests and dedicated benchmark workflows, including diff-PH and the required Hodge smoke matrix. No 100% coverage claim is made for `benchmarks/`. |
| Limitation | The repository's 100% coverage statement is deliberately scoped to the distributable `topogeoml` package. It must not be interpreted as a 100% coverage claim for the experimental benchmark harness. |

---

## Claim 7: "preregistered hypothesis series (H001-H011, 50+ falsifiable sub-predictions)"

| Field | Value |
|---|---|
| Evidence | `docs/hypotheses/HYPOTHESIS-*.md` (14 files) |
| Sub-prediction count | H1-H3 (3) + H4-H7 (4) + H8-H12 (5) + H13-H17 (5) + H18-H21 (4) + H22-H25 (4) + H26-H27 (2) + H28-H32 (5) + H33-H35 (3) + H36-H38 (3) + H39-H41 (3) + H42-H46 (5) + H47-H50 (4) + H51-H53 (3) = 53 |
| Preregistration verification | `git log --format="%H %ai" -- docs/hypotheses/HYPOTHESIS-008-gin-gat-comparison.md | tail -1` — commit timestamp precedes experiment result timestamp. Replace the filename with any hypothesis document to verify. |
| Limitation | Hypothesis selection was sequential (each informed by the prior). This is acknowledged in [STATISTICAL_SUMMARY.md](STATISTICAL_SUMMARY.md) §4 as legitimate sequential testing, not p-hacking. |

---

## Claims not yet independently validated

The following claims have not been reproduced outside the original compute environment:

- All per-seed accuracies (hardware-dependent floating-point variation expected)
- The investigation-wide BH analysis (computed from the archived JSON artifacts; a third party should re-run the analysis script to verify)
- The full H011-b COLLAB L_1 statistical experiment remains pending; the repository currently contains only the smoke result
