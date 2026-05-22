# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-22T01:28:52Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `nci1` | 0.571 [0.543, 0.598] | 30 |
| `mlp-baseline` | `nci1` | 0.500 [0.500, 0.500] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0712 | 1.769e-05 | 1.769e-05 | 0.840 | **hodge-mp-residual ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._