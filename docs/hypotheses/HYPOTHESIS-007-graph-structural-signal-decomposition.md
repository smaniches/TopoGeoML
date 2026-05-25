# Hypothesis 007: Which graph-structural proxies explain the H006 constant-feature signal?

**Status.** Resolved 2026-05-22. H26 refuted (all proxies correlate positively with constant-feature gap, rho = +1.0); H27 refuted (no proxy correlates positively with full-feature gain, rho = -1.0). No single structural proxy explains the full-feature Hodge-vs-MLP gain. See §8.

**Falsification target.** For each of five graph-structural proxies — graph size, degree distribution, Weisfeiler-Lehman (WL) subtree features, cycle statistics, and normalized Hodge Laplacian spectral summaries — measure the per-class separability on MUTAG, PROTEINS, and NCI1. Then ask: which proxy (if any) tracks the dataset-by-dataset *constant-feature* Hodge-vs-prior gap from H006? And which (if any) tracks the *full-feature* Hodge-vs-MLP gain?

**Prior result that motivates this hypothesis.** H006 (PR #22) showed that under constant-feature control, the Hodge MP arm retains above-prior predictive signal on all three datasets (MUTAG, PROTEINS, NCI1) at p_BH < 5e-4, but the constant-feature gap is rank-order *inverted* relative to the full-feature Hodge-vs-MLP gap (Spearman ρ = -1.0000 across the three datasets). So the simple "constant-feature signal predicts full-feature gain" mechanism is refuted. H007 decomposes what graph-structural *proxy* — beyond the Laplacian-based Hodge model itself — actually carries class information per dataset, and whether any proxy explains either of the two empirical curves.

---

## 1. The five graph-structural proxies

For each graph in the three TUDataset benchmarks (MUTAG: 188, PROTEINS: 1113, NCI1: 4110), compute:

1. **Size proxy** (1-D scalar): `n_nodes`. Tests whether class can be predicted by graph size alone — the sum-pool MLP baseline already gets graph-size signal for free.
2. **Degree proxy** (5-D vector): `[mean_degree, max_degree, std_degree, n_isolated_nodes, edge_density]`. Tests whether degree-distribution statistics separate the classes.
3. **WL subtree proxy** (32-D vector): 2-iteration Weisfeiler-Lehman subtree-label histogram, bucketed to 32 dimensions via a deterministic hash. WL is the classical "graph kernel" baseline for graph classification.
4. **Cycle proxy** (4-D vector): `[n_cycles_basis, mean_cycle_length, n_triangles, n_4cycles]`. Cycle basis size is closely related to the first Betti number β₁ (rank of H₁); triangle and 4-cycle counts are local topology.
5. **Spectral proxy** (5-D vector): top-5 eigenvalues of the symmetrically-normalised Laplacian L̃ = D^{-1/2} L D^{-1/2}. The Hodge-MP arm's representational basis.

## 2. Class separability metric (unified across proxies)

For each (proxy, dataset, component) triple, compute the Mann-Whitney U statistic between the two class-conditional samples and convert to the rank-biserial correlation `r = 2U/(n₁n₂) − 1` ∈ [-1, 1]. Take `|r|` so the result is in [0, 1] where 0 = chance and 1 = perfect class separation by that component.

For multi-dimensional proxies, report `max |r|` across components — an upper bound on the 1-D separability achievable by any single proxy feature. This is conservative (a multivariate classifier could exceed this), but it gives a comparable scalar across all five proxies.

## 3. Preregistered predictions (light)

For each proxy, the preregistered prediction asks: *"if this proxy explained the H006 result, what would the dataset-by-dataset separability look like?"* Predictions in [low, medium, high] qualitative ranks.

| Proxy | Predicted const-feature-correlation explanation | Predicted full-feature-gain explanation |
|---|---|---|
| Size | MUTAG and PROTEINS have larger size-class separation than NCI1 → could explain const-feature ordering (MUTAG > PROTEINS > NCI1) | unlikely; MLP sum-pool already reads size |
| Degree | uncertain; chemical compounds have constrained valence | uncertain |
| WL | classical graph-kernel separability → if WL ranks NCI1 > PROTEINS > MUTAG, then WL **could** explain full-feature gain (which has the same ordering) | possible |
| Cycle | NCI1 has more aromatic rings → if cycles drive class signal, cycle ranking would be NCI1 > PROTEINS > MUTAG, matching full-feature gain | possible |
| Spectral | hardest to predict; the Hodge-MP arm uses the spectrum directly | possible |

**Predicted ranking refutation (H26):** if no proxy's dataset-by-dataset separability Spearman-correlates with the H006 constant-feature gap at ρ > 0, then the constant-feature gap reflects an interaction between proxies that no single proxy explains.

**Predicted ranking confirmation (H27):** if at least one proxy's separability Spearman-correlates with the full-feature Hodge-vs-MLP gain at ρ > 0 across the three datasets, that proxy is a **candidate explanation** for which datasets see Hodge benefit under full features. (With n=3, ρ is descriptive, not significant.)

## 4. What this PR deliberately does NOT do

- No new model architecture, no Hodge-MP variant, no MLP variant. The model arms are held fixed at H006's `hodge-mp-residual` and `mlp-baseline`.
- No leaderboard update unless an analysis result passes a preregistered threshold; descriptive ρ across n=3 datasets does NOT pass.
- No vessel datasets, no DRIVE, no "Same Dice Different Topology" infrastructure.
- No `topogeoml/regimes/` abstraction.
- No `StructuralClass`, no `AlgebraicObject`, no `GeometrySelector`.
- No new test loss, no new optimizer, no architectural change at all.
- No causal claim. Each proxy is a *graph-structural proxy*, not a "topology mechanism." Only if a proxy specifically isolates a topological invariant (e.g. cycle-basis size for β₁) does the analysis text say so explicitly.

## 5. Implementation plan

Single analysis module `benchmarks/hodge/h007_analysis.py` (~250 LOC):

- `compute_size_features(graph) -> NDArray[float]` — per-graph 1-vector
- `compute_degree_features(graph) -> NDArray[float]` — per-graph 5-vector
- `compute_wl_features(graph, n_iter=2, n_buckets=32) -> NDArray[int]` — per-graph 32-vector
- `compute_cycle_features(graph) -> NDArray[float]` — per-graph 4-vector
- `compute_spectral_features(laplacian, k=5) -> NDArray[float]` — per-graph 5-vector
- `class_separability(features, labels) -> tuple[float, int]` — max |rank-biserial r|, best component
- `run_h007_analysis(...) -> dict` — top-level entry point; returns JSON-serialisable summary
- `render_markdown(result) -> str` — analysis table for the research note
- `main(argv) -> int` — CLI entry point

Output: `notebooks/results/h007_structural_decomposition.json` and `.md`.

## 6. Wall-clock budget

All three datasets, 5 proxies, computed once (no seed loop — these are deterministic graph properties):
- MUTAG (188 graphs, ~18 nodes): <10s
- PROTEINS (1113 graphs, ~39 nodes): ~1 min
- NCI1 (4110 graphs, ~30 nodes): ~2 min

Total: well under 5 min on CPU.

## 7. Reproduction commands

```bash
python -m benchmarks.hodge.h007_analysis \
  --output notebooks/results/h007_structural_decomposition.json \
  --markdown notebooks/results/h007_structural_decomposition.md
```

## 8. Resolved outcome (2026-05-22, deterministic analysis)

The analysis module (`benchmarks/hodge/h007_analysis.py`) ran on MUTAG (188 graphs), PROTEINS (1113 graphs), and NCI1 (4110 graphs). Outputs: `notebooks/results/h007_structural_decomposition.{json,md}`. All numbers below are read directly from the JSON artifact; nothing is hand-derived.

### Per-(dataset × proxy) class separability (`|rank-biserial r|` ∈ [0, 1])

| Dataset | size | degree | wl | cycle | spectral |
|---|---|---|---|---|---|
| mutag | 0.7634 | 0.7722 | 0.6721 | **0.8083** | 0.7431 |
| proteins | 0.5226 | 0.5001 | 0.2067 | **0.5485** | 0.4656 |
| nci1 | **0.3683** | 0.3658 | 0.1808 | 0.2977 | 0.3176 |

(Boldface marks the highest-separability proxy per dataset.)

### Cross-dataset correlation (n=3, descriptive only — Spearman p-values not reported)

| Proxy | ρ vs H006 const-feature gap | ρ vs H006 full-feature gain |
|---|---|---|
| size | +1.0000 | -1.0000 |
| degree | +1.0000 | -1.0000 |
| wl | +1.0000 | -1.0000 |
| cycle | +1.0000 | -1.0000 |
| spectral | +1.0000 | -1.0000 |

### What this resolves

Under this set of five graph-structural proxies on these three TUDataset benchmarks (deterministic analysis, no seeded sampling):

1. **Every proxy's per-dataset class separability follows the same rank order: MUTAG > PROTEINS > NCI1.** This holds for all five proxies — size, degree, WL subtree histogram, cycle (including the β₁ topological-invariant component), and normalised Laplacian spectrum.

2. **The H006 constant-feature gap follows the same rank order** (MUTAG +0.098 > PROTEINS +0.088 > NCI1 +0.071), so every proxy is rank-order consistent with the H006 const-feature gap (Spearman ρ = +1.0000 for all five). This is the unanimous finding: under constant-feature control, the Hodge MP arm's gap over the class prior tracks *whichever* graph-structural property carries the most class signal on the dataset — and on these three datasets, all five proxies agree on the ordering.

3. **No proxy explains the full-feature Hodge-vs-MLP gain.** The full-feature gain has the opposite rank order (NCI1 +0.086 > PROTEINS +0.011 > MUTAG -0.040), so Spearman ρ = -1.0000 for every proxy. The full-feature gain is **rank-order inverted** relative to graph-structural class separability across these three datasets.

### Preregistered sub-hypothesis verdicts

- **H26** (no proxy correlates positively with the H006 const-feature gap at ρ > 0): **REFUTED.** Every proxy gives ρ = +1.0000. The const-feature gap is consistent with all five graph-structural proxies under the tested configuration.
- **H27** (at least one proxy correlates positively with the H006 full-feature gain at ρ > 0): **REFUTED.** Every proxy gives ρ = -1.0000 — the opposite direction. None of the five proxies is a positive predictor of full-feature gain under this analysis.

### Scoped interpretation (descriptive, not causal)

Under the tested configuration, two distinct observations emerge:

1. *Under constant-feature control* (H006's regime), the Hodge MP arm's per-class advantage tracks the same dataset-by-dataset rank order as every measured graph-structural proxy — including the cycle-basis-size component, which is the only entry in the proxy set that specifically isolates a topological invariant (β₁). The Hodge MP arm is consistent with "reading whichever graph-structural class signal is present in the dataset."

2. *Under full-feature training* (H001-H003's regime), the Hodge-vs-MLP gain has the *opposite* dataset-by-dataset rank order from graph-structural separability. The dataset where graph structure is **least** class-informative (NCI1) is where the Hodge architecture beats the MLP baseline by the largest margin; the dataset where graph structure is **most** class-informative (MUTAG) is where Hodge has no advantage.

These two observations are descriptive across n=3 datasets and carry no causal claim. They are consistent with an **architecture × dataset interaction** in which the Hodge advantage is largest on datasets where the no-topology MLP baseline fails to extract the class signal from node features alone — not on datasets where graph structure inherently carries more information.

**The H007 analysis does not identify a single structural proxy that uniquely explains H006. All five proxies are rank-order indistinguishable on these three datasets, so H007 cannot disentangle their relative contributions.** Disentangling would require either (a) more datasets, or (b) ablations that vary one proxy while holding others fixed — neither is in scope for this PR.

---

## 9. Acceptance criteria (per the PR scope contract)

1. CI green.
2. Analysis outputs reproducible (deterministic — no seeded sampling at this layer).
3. H007 §8 states which proxies explain or fail to explain (a) the constant-feature gap and (b) the full-feature gain.
4. No overclaim: "graph-structural proxy" rather than "topology mechanism" unless the proxy specifically isolates a topological invariant.
5. Tests for the JSON output schema.
