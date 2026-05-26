# Claims to Evidence

Every numerical or empirical claim in the README, mapped to its evidence artifact, reproduction command, expected tolerance, and limitations.

## Methodology

- Claims are extracted from README.md as of the current commit.
- Each claim must have a JSON artifact in `notebooks/results/` or a CI command that produces it.
- Tolerances account for hardware-specific floating-point variation (see `REPRODUCING.md` §Expected Numerical Variation).
- "Survives global BH" indicates whether the claim's p-value survives investigation-wide Benjamini-Hochberg correction across all 76 comparisons (see `docs/STATISTICAL_SUMMARY.md`).

---

## Claim 1: "497 tests, coverage enforced at 100%"

| Field | Value |
|---|---|
| Evidence | `pytest --cov=topogeoml --cov-fail-under=100` |
| Artifact | CI output (reproduced on every push) |
| Tolerance | Exact: 497 test functions as counted by `grep -c "def test_" tests/*.py` |
| Limitation | Coverage gate (100%) applies to `topogeoml/` only. `benchmarks/` coverage is measured but not gated because torch-dependent test paths skip in CI environments without GPU/torch. `__init__.py` files are omitted per `pyproject.toml [tool.coverage.run]`. |

---

## Claim 2: NCI1 positive difference (+8.6 pp, p_BH = 4.83 x 10^-3)

| Field | Value |
|---|---|
| Evidence | `notebooks/results/nci1_hodge_ablation_30seeds.json` |
| Artifact key | `pairwise_comparisons[hodge-mp-residual vs mlp-baseline]` |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --seeds 0..29 --n-epochs 10` |
| Expected | median_diff: 0.086 +/- 0.005; p_BH: 4.83e-3 +/- factor of 2 |
| Survives global BH | Yes (rank 28/76, threshold 1.84e-2) |
| Survives Bonferroni | No (threshold 6.58e-4) |
| Limitation | One dataset (NCI1), one configuration (1-layer, hidden_dim=32, 10 epochs). Does not replicate on MUTAG or PROTEINS at this configuration. Subsequent ablation (H008-c) showed the operative factor is the external residual, not the Hodge Laplacian. |

---

## Claim 3: "topology-aware message passing with external residual outperforms MLP by 8-10 pp"

| Field | Value |
|---|---|
| Evidence | `notebooks/results/h008c_nci1_gin_residual_30seeds.json` |
| Artifact key | `pairwise_comparisons[gin-residual vs mlp-baseline]`: Delta +0.106, p_BH = 6.05e-4 |
| Reproduce | `python -m benchmarks.hodge --datasets nci1 --models gin-residual mlp-baseline --seeds 0..29 --n-epochs 10` |
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
| Reproduce | See `REPRODUCING.md` §H008, §H008-b, §H008-c |
| Limitation | Tested at one capacity point (1-layer, 32 hidden). Standard GIN/GAT with batch normalisation and multiple layers were not tested. |

---

## Claim 5: "graph-structural signal on all 3 datasets (all p_BH < 5 x 10^-4)"

| Field | Value |
|---|---|
| Evidence | `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.json` |
| Artifact key | Hodge accuracy vs class prior per dataset |
| Reproduce | See `REPRODUCING.md` §H006 |
| Expected | MUTAG: gap +0.098, p = 4.53e-6; PROTEINS: gap +0.088, p = 1.41e-4; NCI1: gap +0.071, p = 1.93e-5 |
| Survives global BH | Yes (all three) |
| Limitation | These p-values are from the Hodge-vs-class-prior comparison within the H006 resolver, not the Hodge-vs-MLP comparison in the raw JSON. The class prior is the theoretical baseline (majority-class accuracy), not the MLP's constant-feature accuracy. |

---

## Claim 6: "100% coverage on the library and benchmark framework"

| Field | Value |
|---|---|
| Evidence | CI enforces `--cov-fail-under=100` on `topogeoml/` and `benchmarks/` |
| Reproduce | `pytest --cov=topogeoml --cov=benchmarks --cov-fail-under=100` |
| Limitation | Coverage target applies to `topogeoml/` and `benchmarks/` only. `__init__.py` files and lines guarded by `TYPE_CHECKING` are excluded per `pyproject.toml [tool.coverage.report]`. |

---

## Claim 7: "preregistered hypothesis series (H001-H011, 50+ falsifiable sub-predictions)"

| Field | Value |
|---|---|
| Evidence | `docs/hypotheses/HYPOTHESIS-*.md` (14 files) |
| Sub-prediction count | H1-H3 (3) + H4-H7 (4) + H8-H12 (5) + H13-H17 (5) + H18-H21 (4) + H22-H25 (4) + H26-H27 (2) + H28-H32 (5) + H33-H35 (3) + H36-H38 (3) + H39-H41 (3) + H42-H46 (5) + H47-H50 (4) + H51-H53 (3) = 53 |
| Preregistration verification | `git log --format="%H %ai" -- docs/hypotheses/HYPOTHESIS-NNN-*.md | tail -1` — commit timestamp precedes experiment result timestamp |
| Limitation | Hypothesis selection was sequential (each informed by the prior). This is acknowledged in `docs/STATISTICAL_SUMMARY.md` §4 as legitimate sequential testing, not p-hacking. |

---

## Claims not yet independently validated

The following claims have not been reproduced outside the original compute environment:

- All per-seed accuracies (hardware-dependent floating-point variation expected)
- The investigation-wide BH analysis (computed from the archived JSON artifacts; a third party should re-run the analysis script to verify)
- COLLAB L_1 experiment (H011-b) — pending GitHub Actions completion
