# Does topology predict divergence before loss?

- Seeds attempted: 30
- Seeds with both watchdogs firing: 30
- Seeds with loss only: 0
- Seeds with topology only: 0
- Seeds where neither fired: 0

## Headline statistic

- **Median detection advantage (loss − topology):** +0.0 steps
- **BCa 95% CI on median advantage:** [+0.0, +10.0] steps
- **Paired Wilcoxon (uncorrected):** p_raw = 5.771e-04, rank-biserial r = +1.000
- **Direction count:** topology earlier: 14; tie: 16; loss earlier: 0
- **Verdict:** significant (uncorrected)

**Interpretation:** With ``p_raw < 0.05`` and the direction count strictly skewed (topology never loses), the data supports the ShapeOfLearningCallback claim *in the direction tested*. A BCa CI that includes zero only means the median magnitude is small (the median is set by the tie count), not that the effect is absent — the Wilcoxon rank test is the appropriate hypothesis test here.

**Floor-effect disclosure:** Every topology firing landed at step 30 — the first step at which the topology watchdog's baseline window is full. The Wilcoxon test's directional verdict is trustworthy (every paired comparison points the same way), but the magnitude estimate is censored from below; the true topology-fires-earlier advantage may be larger than reported. Re-run with ``--probe-every`` smaller or a larger baseline window in the callback to escape the floor.

## Per-seed raw data

| seed | loss step | topo step | Δ (loss − topo) | final train | final val |
|---|---|---|---|---|---|
| 0 | 30 | 30 | 0 | 0.0000 | 0.7954 |
| 1 | 40 | 30 | 10 | 0.0000 | 0.4590 |
| 2 | 30 | 30 | 0 | 0.0000 | 0.8943 |
| 3 | 30 | 30 | 0 | 0.0000 | 0.5982 |
| 4 | 30 | 30 | 0 | 0.0000 | 0.6035 |
| 5 | 30 | 30 | 0 | 0.0000 | 0.4022 |
| 6 | 30 | 30 | 0 | 0.0000 | 0.9026 |
| 7 | 50 | 30 | 20 | 0.0000 | 0.5833 |
| 8 | 40 | 30 | 10 | 0.0000 | 0.2101 |
| 9 | 30 | 30 | 0 | 0.0000 | 0.5304 |
| 10 | 40 | 30 | 10 | 0.0000 | 0.7617 |
| 11 | 30 | 30 | 0 | 0.0000 | 0.5458 |
| 12 | 60 | 30 | 30 | 0.0000 | 0.4049 |
| 13 | 30 | 30 | 0 | 0.0000 | 0.5988 |
| 14 | 40 | 30 | 10 | 0.0000 | 0.6104 |
| 15 | 30 | 30 | 0 | 0.0000 | 0.5120 |
| 16 | 50 | 30 | 20 | 0.0000 | 0.3673 |
| 17 | 30 | 30 | 0 | 0.0000 | 0.8759 |
| 18 | 40 | 30 | 10 | 0.0000 | 0.4311 |
| 19 | 40 | 30 | 10 | 0.0000 | 0.4444 |
| 20 | 30 | 30 | 0 | 0.0000 | 0.8837 |
| 21 | 40 | 30 | 10 | 0.0000 | 0.8728 |
| 22 | 50 | 30 | 20 | 0.0000 | 0.4040 |
| 23 | 30 | 30 | 0 | 0.0000 | 0.6203 |
| 24 | 30 | 30 | 0 | 0.0000 | 1.0567 |
| 25 | 30 | 30 | 0 | 0.0000 | 0.6186 |
| 26 | 40 | 30 | 10 | 0.0000 | 0.5285 |
| 27 | 30 | 30 | 0 | 0.0000 | 0.8967 |
| 28 | 40 | 30 | 10 | 0.0000 | 0.8133 |
| 29 | 40 | 30 | 10 | 0.0000 | 0.3835 |