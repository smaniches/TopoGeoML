# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-25T00:30:45Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15
- Git commit: `290c70f7254860a113a1add37535a8a4a9d3dac9`
- Dependencies: numpy=2.4.6, scipy=1.17.1, torch=2.12.0+cu130, torch_geometric=2.7.0

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `mutag` | 0.750 [0.724, 0.789] | 30 |
| `gin-residual` | `mutag` | 0.789 [0.763, 0.816] | 30 |
| `mlp-baseline` | `mutag` | 0.789 [0.763, 0.816] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| mutag | `hodge-mp-residual` | `gin-residual` | -0.0395 | 2.481e-03 | 7.442e-03 | -0.545 | **hodge-mp-residual ≠ gin-residual** |
| mutag | `hodge-mp-residual` | `mlp-baseline` | -0.0395 | 5.742e-03 | 8.613e-03 | -0.310 | **hodge-mp-residual ≠ mlp-baseline** |
| mutag | `gin-residual` | `mlp-baseline` | +0.0000 | 4.377e-01 | 4.377e-01 | -0.185 | no diff |

_No claim made without a statistically significant result after BH correction at α=0.05._