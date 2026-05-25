# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-24T23:07:31Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15
- Git commit: `328a67933f981975cd967a3f0a9c45a451d4d511`
- Dependencies: numpy=2.4.6, scipy=1.17.1, torch=2.12.0+cu130, torch_geometric=2.7.0

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `sheaf-residual` | `nci1` | 0.604 [0.564, 0.619] | 30 |
| `hodge-mp-residual` | `nci1` | 0.609 [0.581, 0.625] | 30 |
| `gin-residual` | `nci1` | 0.629 [0.607, 0.641] | 30 |
| `mlp-baseline` | `nci1` | 0.523 [0.513, 0.566] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `sheaf-residual` | `hodge-mp-residual` | -0.0055 | 7.971e-01 | 7.971e-01 | 0.133 | no diff |
| nci1 | `sheaf-residual` | `gin-residual` | -0.0249 | 6.826e-03 | 1.365e-02 | -0.467 | **sheaf-residual ≠ gin-residual** |
| nci1 | `sheaf-residual` | `mlp-baseline` | +0.0809 | 1.397e-02 | 1.676e-02 | 0.333 | **sheaf-residual ≠ mlp-baseline** |
| nci1 | `hodge-mp-residual` | `gin-residual` | -0.0195 | 1.014e-02 | 1.520e-02 | -0.400 | **hodge-mp-residual ≠ gin-residual** |
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0864 | 3.378e-03 | 1.013e-02 | 0.533 | **hodge-mp-residual ≠ mlp-baseline** |
| nci1 | `gin-residual` | `mlp-baseline` | +0.1058 | 4.031e-04 | 2.419e-03 | 0.600 | **gin-residual ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._