# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-22T01:14:45Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `proteins` | 0.684 [0.648, 0.704] | 30 |
| `mlp-baseline` | `proteins` | 0.596 [0.596, 0.596] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| proteins | `hodge-mp-residual` | `mlp-baseline` | +0.0874 | 5.833e-04 | 5.833e-04 | 0.571 | **hodge-mp-residual ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._