# Hypothesis 004: Is sample size the mechanism behind the residual-scale effect? NCI1 subsampling study

**Status.** Preregistered 2026-05-21. Test in progress (this branch). The 30-seed × 4-subsample-size run lands the resolved outcome in §6.
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

## 6. Resolved outcome (filled in when the multi-size run completes)

*Pending — preregistered before any execution.*

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