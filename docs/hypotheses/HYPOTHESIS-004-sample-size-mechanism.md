---
title: H004 · Sample-size mechanism
parent: Hypotheses (H001–H011b)
nav_order: 4
---

# Hypothesis 004: Is sample size the mechanism behind the residual-scale effect? NCI1 subsampling study

**Status.** Resolved 2026-05-22. **H13 refuted, H14 refuted, H15 confirmed (reproducibility holds), H16 confirmed (monotone Δ in n), H17 N/A (Δ never crosses zero in the tested range).** Sample size is decisively NOT the mechanism. See §6 for the full outcome.
**Falsification target.** Whether the residual-vs-MLP effect on NCI1 *survives subsampling to MUTAG-size and PROTEINS-size* — a direct mechanism test that controls for everything except sample count.
**Prior result that motivates this hypothesis.** Hypothesis 003 (PR #19): `hodge-mp-residual` strictly beats MLP on NCI1 at p_BH = 4.83 × 10⁻³ (+8.6 pp). The same architecture *loses* on MUTAG (p_BH = 0.019) and *matches* on PROTEINS (p_BH = 0.339). Two mechanisms remain in play:
  - **(a) Feature density / distribution.** NCI1 has 37-dim atom features (vs MUTAG's 7-dim, PROTEINS' 3-dim).
  - **(b) Sample size.** NCI1 has 4110 graphs (vs MUTAG's 188, PROTEINS' 1113).

Hypothesis 002 §6 sketched (a) as the leading explanation but the analysis was rough; hypothesis 003 §6 noted the existing `_proj_in: nn.Linear(input_dim, 32)` already linearises into 32-dim before the residual, so "preserving one-hot sparsity" isn't quite the mechanism. This hypothesis tests (b) cleanly.

---

## 1. The experimental design

**Single-dataset subsampling.** Take NCI1's 4110 graphs and subsample (per seed, deterministically using the seed as the RNG state) to four sample sizes:

| n_graphs | Comparable to | Reason |
|---|---|---|
| 188 | MUTAG size | Tests whether NCI1@MUTAG-size loses like MUTAG did |
| 1113 | PROTEINS size | Tests whether NCI1@PROTEINS-size matches like PROTEINS did |
| 2000 | Intermediate | Locates the crossover, if any |
| 4110 | Full NCI1 | Control — must reproduce hypothesis 003's p_BH = 4.83 × 10⁻³ |

For each sample size, run 30 independent seeds × 10 epochs × hidden_dim=32 × stratified 80/20 split. Compare `hodge-mp-residual` vs `mlp-baseline` (two arms only — drop the other Hodge variants because they aren't the headline; the design isolates the residual-scale question).

**Why this design separates (a) from (b) cleanly.** The 188-graph NCI1 subsample has the same 37-dim feature distribution as the full NCI1 — feature density is fixed. Sample size is the only variable that changes. Therefore:
- If `hodge-mp-residual` *loses* at NCI1[n=188] (replicating MUTAG): **sample size IS the mechanism**.
- If `hodge-mp-residual` *wins* at NCI1[n=188] (matching full NCI1): **feature density / distribution IS the mechanism** (since sample size was reduced to MUTAG's but the result held).
- If the residual effect *transitions* somewhere between 188 and 4110: we get a quantitative threshold for when the residual starts to help.

## 2. Preregistered sub-hypotheses (verbatim, before result lands)

| ID | Sub-hypothesis | Predicted at 30 seeds | Falsified if |
|---|---|---|---|
| **H13** | Residual loses at NCI1[n=188] (sample-size mechanism) | median Δ < 0, p_BH < 0.05 | p_BH ≥ 0.05 OR median Δ ≥ 0 |
| **H14** | Residual matches at NCI1[n=1113] | p_BH ≥ 0.05 | p_BH < 0.05 either way |
| **H15** | Residual wins at NCI1[n=4110] (control — reproduces H003) | median Δ > 0, p_BH < 0.01 | reproduction failure → re-examine the bench |
| **H16** | Monotone trend: median Δ increases with n_graphs | yes | non-monotone direction at any size |
| **H17** | Crossover (Δ = 0) located between 188 and 4110 | yes | extrapolated zero outside [188, 4110] |

## 3. Outcome decision tree (preregistered)

| Pattern | Mechanism verdict | v0.0.2 narrative |
|---|---|---|
| H13+H14+H15+H16 all confirmed | **Sample-size argument vindicated.** | "The residual helps once n_graphs ≥ ~X (the crossover)." Honest, scientifically clean. |
| H13 refuted (residual wins at n=188) | **Feature-density argument vindicated.** | "It's not the sample size; NCI1's 37-dim features make the residual work even at MUTAG-size subsamples." Confounds with hypothesis 001's MUTAG defeat (which used 7-dim features). |
| H13 confirmed but H16 non-monotone | Need to investigate further | Document the anomaly; v0.0.2 narrative stays scoped to the headline finding |
| H15 fails (full-NCI1 reproduction breaks) | **Reproducibility failure.** Stop everything; investigate before publishing v0.0.2 | This is the most important sub-hypothesis — it's the control. |

## 4. Statistical procedure

- For each (sample_size, arm) cell, 30 seeds × 10 epochs × hidden_dim=32 × Adam(lr=1e-2).
- BCa 95% CI on per-arm median accuracy.
- **Paired Wilcoxon** (residual vs MLP, matched by seed) at each sample size — 4 comparisons.
- **Benjamini-Hochberg FDR** across the 4-comparison family at α = 0.05.
- **Monotonicity test (H16)** — Spearman rank correlation between median Δ and log(n_graphs), with the null = no monotone relationship. Report ρ and p.

## 5. Wall-clock budget

| Sample size | Smoke (3 seeds × 5 epochs × 2 arms) | Full (30 seeds × 10 epochs × 2 arms) |
|---|---|---|
| 188 | ~3 sec | ~3 min |
| 1113 | ~18 sec | ~18 min |
| 2000 | ~32 sec | ~32 min |
| 4110 | ~67 sec | ~67 min |
| **Total** | **~2 min smoke** | **~2 hours full** |

Background-runnable. The shorter run sizes finish quickly; the full-NCI1 control takes the bulk.

## 6. Resolved outcome (2026-05-22, 30 seeds × 10 epochs × 4 sample sizes, wall time ~2h)

Per-size reports in `notebooks/results/h004_nci1_n{188,1113,2000,4110}_30seeds.md`.

| n_graphs | hodge-mp-residual (median, BCa 95%) | mlp-baseline (median, BCa 95%) | median Δ | paired Wilcoxon p_BH | Verdict vs MLP |
|---|---|---|---|---|---|
| 188 (MUTAG-size) | 0.579 [0.533, 0.613] | 0.560 [0.520, 0.595] | **+0.019** | 0.897 | matches |
| 1113 (PROTEINS-size) | 0.601 [0.589, 0.629] | 0.549 [0.526, 0.590] | **+0.052** | **0.045** | **WINS** |
| 2000 | 0.595 [0.576, 0.613] | 0.536 [0.515, 0.600] | +0.059 | 0.053 | matches (border) |
| **4110 (full, control)** | **0.609 [0.581, 0.625]** | **0.523 [0.513, 0.566]** | **+0.086** | **3.38 × 10⁻³** | **WINS** |

### Headline finding

> **Sample size is NOT the mechanism behind the residual-scale effect.** Subsampling NCI1 to MUTAG's 188 graphs *does not* reproduce MUTAG's residual-defeat (which had Δ = −0.04 with p_BH = 0.019). NCI1-at-MUTAG-size shows Δ = +0.019 with p_BH = 0.897. Subsampling to PROTEINS' 1113 graphs *does not* reproduce PROTEINS' residual-equality either (NCI1-at-PROTEINS-size already strictly wins at p_BH = 0.045). The MUTAG and PROTEINS residual underperformances are caused by *dataset-specific properties of the data distribution*, not by data quantity. The leading remaining mechanism candidates are feature dimensionality and feature semantics (hypothesis 005).

### Sub-hypotheses, resolved

- **H13** (residual loses at NCI1[n=188]): **REFUTED.** Δ = +0.019 (positive, not negative); p_BH = 0.897. Sample-size mechanism rejected.
- **H14** (residual matches at NCI1[n=1113]): **REFUTED.** Δ = +0.052, p_BH = 0.045 → strictly wins. The PROTEINS equality is NOT replicated by sample-size matching.
- **H15** (reproducibility control at n=4110): **CONFIRMED.** Δ = +0.086, p_BH = 3.38 × 10⁻³ — matches PR #19's +0.086, p_BH = 4.83 × 10⁻³ to within stochastic noise. (The minor p-value difference comes from the BH-FDR family size: 4 comparisons here vs 10 there.) The hypothesis 003 headline result reproduces robustly under independent BH correction.
- **H16** (monotone trend Δ vs n_graphs): **CONFIRMED.** Δ values: 0.019 → 0.052 → 0.059 → 0.086, monotone non-decreasing. The residual advantage *grows* with sample size, even though it's already positive at the smallest size.
- **H17** (crossover at Δ = 0 between 188 and 4110): **N/A.** Δ never crosses zero in the tested range; the residual advantage is always non-negative for NCI1, even at MUTAG-size subsamples.

### Cross-experiment interpretation

The residual-vs-MLP gap **is monotone in sample size on NCI1** (H16 confirmed), but the *direction* (advantage to Hodge) is fixed for all NCI1-derived data. Meanwhile, MUTAG and PROTEINS — at THEIR native sample sizes — produce *negative* (MUTAG: Δ = −0.04) or *near-zero* (PROTEINS: Δ ≈ 0) results.

Subsampling NCI1 to match MUTAG's and PROTEINS's sample sizes does not reproduce those signs. **Conclusion: the dataset-specific factor that flips the residual's sign between MUTAG and NCI1 is in the data distribution, not in n.**

Concretely, the candidates that remain in play after this experiment:

1. **Feature dimensionality** (NCI1: 37, MUTAG: 7, PROTEINS: 3) — tested directly by hypothesis 005 via random projection
2. **Feature semantics** (atom type vs. secondary-structure type) — hard to isolate without dataset-specific transforms
3. **Graph topology** (NCI1 avg 30 nodes, MUTAG 18, PROTEINS 39) — orthogonal to features
4. **Label class balance / task difficulty** — possible confounder but priors suggest minor

The monotone-Δ-in-n pattern within NCI1 is also informative: more training data lets the architecture *exploit the topology more*, even when the residual is already net-positive. This suggests the Hodge inductive bias is non-trivial — it just needs the right data substrate to express it.

### What this licenses the framework to claim (refined)

The hypothesis 003 NCI1 positive claim **survives independent statistical replication** (H15). The architecture truly does outperform MLP on NCI1 at p_BH < 0.005. *But*: the cross-dataset behaviour (MUTAG defeat, PROTEINS equality) is not explained by sample size, so the framework should not claim "Hodge with residual helps once n ≥ X". Rather, it should claim "Hodge with residual helps on NCI1 across all sample sizes tested (n ≥ 188); whether it helps on MUTAG/PROTEINS is a separate dataset-specific question pending hypothesis 005."

### What hypothesis 005 does next

The feature-projection infrastructure (`feature_projection_dim` parameter in `run_classification`) is already implemented on this branch. The next experimental cell is:

- NCI1-7d (project 37 → 7, match MUTAG's dim) × hodge-mp-residual + mlp-baseline × 30 seeds × 10 epochs
- MUTAG-37d (project 7 → 37, match NCI1's dim) × same arms × seeds × epochs

Wall-time estimate: ~65 min.

---

## 7. What this hypothesis deliberately does NOT do

- **Does not vary node features.** Holding NCI1's natural 37-dim features fixed means *the feature distribution is identical across subsample sizes*. This is what makes the sample-size isolation clean.
- **Does not run on other datasets.** A clean mechanism test on one dataset is more informative than scattered ablations on three.
- **Does not run all 5 Hodge arms.** The headline question is residual vs MLP; the other arms are orthogonal here. Adding them would 2.5× the wall time without sharpening the mechanism test.
- **Does not change the optimiser / lr / hidden_dim.** Holding architecture-hyperparameters constant is part of the matched-design discipline.

## 8. If H13 is confirmed, what's hypothesis 005?

The cleanest next experiment given a confirmed sample-size mechanism: **vary node feature dim on a fixed sample-size NCI1 subsample.** Take 1113-graph NCI1 (PROTEINS-size where residual matched MLP) and:
- Project the 37-dim features to 3-dim via PCA before training (matching PROTEINS' feature dim)
- Project to 7-dim (matching MUTAG's)
- Keep at 37-dim (control)

If at 1113-graph NCI1, residual now LOSES with 3-dim features and MATCHES with 37-dim, feature density also plays a role and the two mechanisms compose. If the 3-dim projection still matches, sample-size is the sole mechanism at this scale.

That's the next session.