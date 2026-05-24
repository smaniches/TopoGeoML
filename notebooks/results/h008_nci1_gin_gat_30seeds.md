# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-24T12:58:32Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `nci1` | 0.609 [0.581, 0.625] | 30 |
| `gin-baseline` | `nci1` | 0.500 [0.500, 0.505] | 30 |
| `gat-baseline` | `nci1` | 0.500 [0.500, 0.500] | 30 |
| `mlp-baseline` | `nci1` | 0.523 [0.513, 0.566] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `hodge-mp-residual` | `gin-baseline` | +0.1095 | 2.121e-06 | 6.364e-06 | 0.933 | **hodge-mp-residual ≠ gin-baseline** |
| nci1 | `hodge-mp-residual` | `gat-baseline` | +0.1095 | 1.732e-06 | 6.364e-06 | 1.000 | **hodge-mp-residual ≠ gat-baseline** |
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0864 | 3.378e-03 | 4.054e-03 | 0.533 | **hodge-mp-residual ≠ mlp-baseline** |
| nci1 | `gin-baseline` | `gat-baseline` | +0.0000 | 1.327e-02 | 1.327e-02 | 0.368 | **gin-baseline ≠ gat-baseline** |
| nci1 | `gin-baseline` | `mlp-baseline` | -0.0231 | 1.972e-03 | 2.958e-03 | -0.600 | **gin-baseline ≠ mlp-baseline** |
| nci1 | `gat-baseline` | `mlp-baseline` | -0.0231 | 5.266e-05 | 1.053e-04 | -0.833 | **gat-baseline ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._