# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-08-10T18:27:08Z
- Platform: Linux-6.17.0-1020-azure-x86_64-with-glibc2.39
- Python: 3.12.13
- Git commit: `79329df1b49b15867daf6a7959acb4911d994d6a`
- Dependencies: numpy=2.4.4, scipy=1.18.0, torch=2.13.0+cpu, torch_geometric=2.8.0.post1

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `sheaf-residual` | `nci1` | 0.604 [0.580, 0.624] | 30 |
| `hodge-mp-residual` | `nci1` | 0.605 [0.572, 0.612] | 30 |
| `gin-residual` | `nci1` | 0.630 [0.605, 0.648] | 30 |
| `mlp-baseline` | `nci1` | 0.523 [0.513, 0.566] | 30 |

## Paired Wilcoxon plus Benjamini-Hochberg correction

The final column reports generic BH significance at alpha=0.05 for this requested model family. It is not a preregistered hypothesis verdict.

| Dataset | A | B | median delta | p_raw | p_BH | Effect (r) | BH at 0.05 |
|---|---|---|---|---|---|---|---|
| nci1 | `sheaf-residual` | `hodge-mp-residual` | -0.0006 | 4.284e-01 | 4.284e-01 | 0.133 | not significant |
| nci1 | `sheaf-residual` | `gin-residual` | -0.0262 | 2.846e-02 | 3.416e-02 | -0.467 | significant |
| nci1 | `sheaf-residual` | `mlp-baseline` | +0.0809 | 2.367e-03 | 4.735e-03 | 0.333 | significant |
| nci1 | `hodge-mp-residual` | `gin-residual` | -0.0255 | 1.809e-04 | 5.426e-04 | -0.733 | significant |
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0815 | 1.737e-02 | 2.605e-02 | 0.379 | significant |
| nci1 | `gin-residual` | `mlp-baseline` | +0.1071 | 8.857e-05 | 5.314e-04 | 0.600 | significant |

_Use the corresponding preregistration to determine the scientific decision threshold. Some TopoGeoML sub-hypotheses require p_BH < 0.01._