# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-21T17:39:39Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-classifier` | `mutag` | 0.697 [0.658, 0.750] | 30 |
| `mlp-baseline` | `mutag` | 0.789 [0.763, 0.816] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| mutag | `hodge-mp-classifier` | `mlp-baseline` | -0.0921 | 5.657e-05 | 5.657e-05 | -0.760 | **hodge-mp-classifier ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._