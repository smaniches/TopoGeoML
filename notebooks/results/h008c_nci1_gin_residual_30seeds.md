# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-24T18:53:56Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15
- Git commit: `ae8040de714a8b906e69ed29a4f239aa3ba016a4`
- Dependencies: numpy=2.4.6, scipy=1.17.1, torch=2.12.0+cu130, torch_geometric=2.7.0

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `nci1` | 0.609 [0.581, 0.625] | 30 |
| `gin-residual` | `nci1` | 0.629 [0.607, 0.641] | 30 |
| `gin-normalised` | `nci1` | 0.500 [0.500, 0.500] | 30 |
| `mlp-baseline` | `nci1` | 0.523 [0.513, 0.566] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `hodge-mp-residual` | `gin-residual` | -0.0195 | 1.014e-02 | 1.014e-02 | -0.400 | **hodge-mp-residual ≠ gin-residual** |
| nci1 | `hodge-mp-residual` | `gin-normalised` | +0.1095 | 1.732e-06 | 5.197e-06 | 1.000 | **hodge-mp-residual ≠ gin-normalised** |
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0864 | 3.378e-03 | 4.054e-03 | 0.533 | **hodge-mp-residual ≠ mlp-baseline** |
| nci1 | `gin-residual` | `gin-normalised` | +0.1290 | 1.731e-06 | 5.197e-06 | 1.000 | **gin-residual ≠ gin-normalised** |
| nci1 | `gin-residual` | `mlp-baseline` | +0.1058 | 4.031e-04 | 6.047e-04 | 0.600 | **gin-residual ≠ mlp-baseline** |
| nci1 | `gin-normalised` | `mlp-baseline` | -0.0231 | 2.664e-05 | 5.328e-05 | -0.833 | **gin-normalised ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._