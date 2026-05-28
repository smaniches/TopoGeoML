---
title: Limitations & scope
nav_order: 7
description: "What does not work, and the regime that bounds every result."
---

# Limitations & scope
{: .no_toc }

Honest accounting is part of the contract. This page summarises the bounds; the full, continuously-maintained list lives in the canonical [`LIMITATIONS.md`](https://github.com/smaniches/TopoGeoML/blob/main/LIMITATIONS.md) at the repository root.

{: .warning }
> **The regime that bounds every accuracy number.** All empirical results come from a deliberately constrained *matched-capacity* protocol: 1 layer, hidden_dim = 32, 10–20 epochs, Adam(lr = 1e-2), no batch normalisation, no learning-rate schedule, ~1.4–2.3k parameters per arm. On NCI1 the best arm reaches ~0.61–0.63 and the MLP baseline ~0.52, versus the ~0.80+ that properly-trained GNNs reach in the literature; the standard GNN baselines (GIN, GAT) collapse to the class prior (0.500) under this protocol. **Every "outperforms X" statement is therefore *at equal, deliberately-constrained capacity*** — it isolates architectural mechanism and is not a benchmark-performance or expressiveness claim.

## Empirical scope

- **Datasets:** MUTAG (188), PROTEINS (1113), NCI1 (4110), COLLAB (5000, pending). One domain family (chemistry / proteins / social).
- **Architecture:** one architectural class (one-layer Hodge MP / message passing). Deeper architectures, attention, polynomial filters, and full simplicial networks are untested.
- **Statistical power:** minimum detectable effect |r| = 0.289 at n = 30. Null results are failures to reject at the tested power, not positive equality claims.
- **Multiple testing:** the NCI1 Hodge-vs-MLP result survives investigation-wide Benjamini–Hochberg but **not** Bonferroni.

## Toolkit scope (selected)

- Only Vietoris–Rips and cubical persistent homology ship; no alpha/witness complexes, no diagram distances (`bottleneck`/`wasserstein`) in the core.
- The Hodge message-passing layer is a **minimal building block** (`σ(L̃ₖ x W + b)`), not a competitive simplicial neural network.
- The embedding audit is a prototype with a heuristic significance threshold.
- No GPU-batched persistence; large point clouds and dense clique complexes are compute-heavy.

## What is explicitly *not* claimed

- "Topology helps graph classification" — **not supported** across datasets.
- "Hodge is better than GNNs" — **refuted** (H008c; and the GNN baselines here are capacity-starved, not competitive).
- "L₁ captures unique structural signal" — **not yet tested at rigor** (triangle-rich COLLAB run is pending).
- Any result beyond the tested configuration.

For the complete, authoritative list of failure modes and deferred features, see [`LIMITATIONS.md`](https://github.com/smaniches/TopoGeoML/blob/main/LIMITATIONS.md).
