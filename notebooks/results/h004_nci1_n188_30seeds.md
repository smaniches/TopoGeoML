# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-21T23:33:39Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `nci1` | 0.579 [0.533, 0.613] | 30 |
| `mlp-baseline` | `nci1` | 0.560 [0.520, 0.595] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0188 | 8.967e-01 | 8.967e-01 | -0.034 | no diff |

_No claim made without a statistically significant result after BH correction at α=0.05._