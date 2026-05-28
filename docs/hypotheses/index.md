---
title: Hypotheses (H001–H011b)
nav_order: 3
has_children: true
description: "The preregistered hypothesis series and its resolved outcomes."
---

# Preregistered hypotheses (H001–H011b)
{: .no_toc }

Each hypothesis below was written with falsifiable sub-predictions and a pre-specified outcome decision tree, then **committed to git before its experiment ran** — the commit history is the preregistration timestamp. Results were appended afterwards, with the original predictions preserved for audit.

{: .note }
> The honest headline of this series is a **negative** one: across all 14 hypotheses, the Hodge Laplacian never confers a unique advantage once an external residual is present (H008c). Outcomes below are reported exactly as they resolved — including the many nulls and refutations.

## Outcomes at a glance

| # | Question (abbreviated) | Resolved outcome |
|---|---|---|
| **H001** | Why does minimal Hodge MP lose on MUTAG; what closes the gap? | Symmetric normalisation closes it (matches MLP); residual *hurts* here |
| **H002** | Does the winning architecture beat MLP on PROTEINS? | Equality; strong "topology beats MLP" claim **refuted** |
| **H003** | Does scale (NCI1) lift the Hodge = MLP ceiling? | **Narrow positive** (+8.6 pp), regime-bound |
| **H004** | Is sample size the mechanism? | Mechanism **ruled out** |
| **H005** | Is feature density the mechanism? | Mechanism **ruled out** (+ a feature-degradation robustness result) |
| **H006** | Can Hodge classify from graph structure alone? | Signal present on all 3 datasets, but rank-inverted vs the full-feature gain |
| **H007** | Which structural proxy explains the gain? | **None** — no single proxy is predictive |
| **H008** | Does the NCI1 result hold vs GIN / GAT? | Regime-bound: GIN/GAT **collapse to class prior** at matched capacity |
| **H008b** | Does degree normalisation close the GIN–Hodge gap? | **Refuted** — normalisation alone does not recover |
| **H008c** | Is the *external residual* the operative factor? | **Confirmed — the primary finding.** Operator (Hodge vs adjacency) is secondary |
| **H009** | Does a learned sheaf Laplacian beat fixed operators? | **No** — adds no value |
| **H010** | Does high-pass (Hodge) beat low-pass cross-dataset? | **No** — high-pass is equal or worse |
| **H011** | Does L₁ edge propagation capture signal L₀ cannot? | Degenerate on NCI1 (≈ no triangles); inconclusive |
| **H011b** | L₁ on triangle-rich COLLAB | **Pending** — directional smoke result only; full run compute-constrained |

Individual preregistrations and their full resolved outcomes are listed in the sidebar.
