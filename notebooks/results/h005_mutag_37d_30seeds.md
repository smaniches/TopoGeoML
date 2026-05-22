# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-22T01:07:50Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `mutag` | 0.776 [0.737, 0.816] | 30 |
| `mlp-baseline` | `mutag` | 0.789 [0.763, 0.829] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| mutag | `hodge-mp-residual` | `mlp-baseline` | -0.0132 | 2.458e-01 | 2.458e-01 | -0.231 | no diff |

_No claim made without a statistically significant result after BH correction at α=0.05._