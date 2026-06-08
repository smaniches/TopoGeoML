---
title: Overview
nav_order: 1
description: "What TopoGeoML is, what it found, and what it does not claim."
---

# TopoGeoML
{: .no_toc }

**A preregistered, self-falsifying investigation into topology-aware graph classification — plus a differentiable-TDA toolkit.**

TopoGeoML asks one question and tries hard to answer it honestly: *does encoding topological structure via the Hodge Laplacian improve graph classification beyond what node features already provide?* The question was tested across 14 preregistered hypotheses (53 falsifiable sub-predictions), with the hypothesis documents committed to git **before** each experiment ran.

This site is the navigable record of that investigation. It is written to inform, not to sell. Every accuracy figure on it is bounded by the regime caveat below.

---

## Primary finding — negative

Encoding topological structure via the Hodge Laplacian does **not** confer a unique advantage for graph classification on any tested dataset. Once an external residual connection is present, a plain normalised-*adjacency* operator matches or slightly exceeds the Hodge Laplacian; without that residual, both collapse to the class prior. **The operative architectural factor is the residual connection, not the topology** (hypothesis H008c).

This refutes the strong "topology helps graph classification" hypothesis at the tested configuration. It is reported with the same prominence as any positive result — that is the point of the project.

## Secondary finding — positive, narrow

On NCI1 (4110 chemical-compound graphs), a one-layer message-passing classifier *with* an external residual outperforms a matched-capacity MLP baseline by 8–10 percentage points (paired Wilcoxon p_BH = 4.83 × 10⁻³; survives investigation-wide Benjamini–Hochberg correction but not Bonferroni).

{: .warning }
> **Read before citing any accuracy number.** All results are obtained under a deliberately constrained *matched-capacity* protocol (1 layer, hidden_dim = 32, 10–20 epochs, no batch normalisation, ~1.4–2.3k parameters per arm). Under this protocol the standard GNN baselines (GIN, GAT) **collapse to the class prior (0.500)** on NCI1, and the best arm reaches only ~0.61–0.63 — roughly **20 percentage points below the ~0.80+ that properly-trained GNNs achieve** on this benchmark in the literature. These comparisons isolate *architectural mechanism at fixed capacity*; they are **not** statements about leaderboard performance, and a phrase like "outperforms GIN/GAT" means only "at equal, severely-limited capacity."

## What this is — and is not

- **It is** a rigorous, reproducible research investigation and a small (~7K LOC) toolkit of correct, citable, type-checked topology-aware layers and a statistical harness.
- **It is not** a production training framework, a competitive benchmark entry, or evidence that topological methods beat well-tuned GNNs. No claim of generality beyond the tested configuration is made.

---

## What is in the toolkit

| Component | Module | Notes |
|---|---|---|
| Differentiable persistent homology (Rips) | `topogeoml.nn.diff_ph` | autograd through critical-edge indexing (Hofer 2017 / Carrière 2021) |
| Differentiable cubical PH + topology loss | `topogeoml.nn.cubical_diff_ph` | `CubicalTopologyLoss` for image-segmentation training |
| Hodge Laplacian message-passing layer | `topogeoml.nn.hodge` | one round of `σ(L̃ₖ x W + b)`; a minimal building block, not a SOTA architecture |
| Persistence vectorizers + pipeline | `topogeoml.core`, `topogeoml.pipelines` | persistence images, Betti curves; sklearn-compatible |
| Statistical machinery | `benchmarks.stats` | BCa + block + percentile bootstrap; Wilcoxon, Mann–Whitney, BH-FDR |

Quality floor: 497 tests, 100% line coverage with full dependencies, mypy strict, ruff clean.

## How the investigation was run

- **Preregistration.** Each hypothesis was committed to git with falsifiable sub-predictions and a pre-specified outcome decision tree *before* the experiment ran.
- **Statistics.** 30 seeds per experiment, paired Wilcoxon signed-rank with Benjamini–Hochberg FDR control, BCa bootstrap confidence intervals; investigation-wide correction across the 59 distinct comparisons (76 computed in total).
- **Negative results shipped.** 28 of the 59 distinct comparisons (47%), and 29 of the 76 total (38%), are non-significant and are reported with identical formatting to the positive ones.

---

## Start here

- [Research report]({% link RESEARCH_REPORT.md %}) — the full structured technical write-up (H001–H007 arc).
- [Hypotheses (H001–H011b)]({% link hypotheses/index.md %}) — every preregistration and its resolved outcome.
- [Statistical summary]({% link STATISTICAL_SUMMARY.md %}) — multiple-testing burden, power, and FDR across the whole investigation.
- [Claims → evidence]({% link CLAIMS_TO_EVIDENCE.md %}) — each claim mapped to the artefact that backs it.
- [Limitations & scope]({% link limitations.md %}) — what does not work, and the regime bounds.
- [Mathematical foundations]({% link mathematics/foundations.md %}) — the underlying TDA / Hodge theory.

Code, reproduction commands, and the full leaderboard live in the [GitHub repository](https://github.com/smaniches/TopoGeoML).

---

## Citation

```bibtex
@software{maniches_topogeoml_2026,
  author  = {Maniches, Santiago},
  title   = {TopoGeoML: A Preregistered Investigation into Topology-Aware Graph Classification},
  year    = {2026},
  version = {0.0.3},
  doi     = {10.5281/zenodo.20564298},
  url     = {https://doi.org/10.5281/zenodo.20564298}
}
```
