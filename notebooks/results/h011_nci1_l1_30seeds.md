# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-25T03:51:28Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15
- Git commit: `7076c1097c28d08eb99351abec693d3c7d8086f3`
- Dependencies: numpy=2.4.6, scipy=1.17.1, torch=2.12.0+cu130, torch_geometric=2.7.0

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `l1-hodge-residual` | `nci1` | 0.590 [0.525, 0.615] | 30 |
| `hodge-mp-residual` | `nci1` | 0.609 [0.581, 0.625] | 30 |
| `gin-residual` | `nci1` | 0.629 [0.607, 0.641] | 30 |
| `mlp-baseline` | `nci1` | 0.523 [0.513, 0.566] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `l1-hodge-residual` | `hodge-mp-residual` | -0.0195 | 6.556e-02 | 7.867e-02 | -0.308 | no diff |
| nci1 | `l1-hodge-residual` | `gin-residual` | -0.0389 | 3.161e-03 | 6.756e-03 | -0.533 | **l1-hodge-residual ≠ gin-residual** |
| nci1 | `l1-hodge-residual` | `mlp-baseline` | +0.0669 | 9.570e-02 | 9.570e-02 | 0.267 | no diff |
| nci1 | `hodge-mp-residual` | `gin-residual` | -0.0195 | 1.014e-02 | 1.520e-02 | -0.400 | **hodge-mp-residual ≠ gin-residual** |
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0864 | 3.378e-03 | 6.756e-03 | 0.533 | **hodge-mp-residual ≠ mlp-baseline** |
| nci1 | `gin-residual` | `mlp-baseline` | +0.1058 | 4.031e-04 | 2.419e-03 | 0.600 | **gin-residual ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._