# H007 structural-signal decomposition

## Per-(dataset × proxy) class separability (max |rank-biserial r|)

| Dataset | Proxy | Feature dim | Max separability | Best component idx |
|---|---|---|---|---|
| mutag | size | 1 | 0.7634 | 0 |
| mutag | degree | 5 | 0.7722 | 0 |
| mutag | wl | 32 | 0.6721 | 24 |
| mutag | cycle | 4 | 0.8083 | 0 |
| mutag | spectral | 5 | 0.7431 | 3 |
| proteins | size | 1 | 0.5226 | 0 |
| proteins | degree | 5 | 0.5001 | 4 |
| proteins | wl | 32 | 0.2067 | 11 |
| proteins | cycle | 4 | 0.5485 | 0 |
| proteins | spectral | 5 | 0.4656 | 4 |
| nci1 | size | 1 | 0.3683 | 0 |
| nci1 | degree | 5 | 0.3658 | 4 |
| nci1 | wl | 32 | 0.1808 | 24 |
| nci1 | cycle | 4 | 0.2977 | 0 |
| nci1 | spectral | 5 | 0.3176 | 4 |

## Cross-dataset correlation (n=3, descriptive only)

| Proxy | mutag | proteins | nci1 | ρ vs H006 const-gap | ρ vs H006 full-gain |
|---|---|---|---|---|---|
| size | 0.7634 | 0.5226 | 0.3683 | +1.0000 | -1.0000 |
| degree | 0.7722 | 0.5001 | 0.3658 | +1.0000 | -1.0000 |
| wl | 0.6721 | 0.2067 | 0.1808 | +1.0000 | -1.0000 |
| cycle | 0.8083 | 0.5485 | 0.2977 | +1.0000 | -1.0000 |
| spectral | 0.7431 | 0.4656 | 0.3176 | +1.0000 | -1.0000 |

## Scoped interpretation

Each entry above is a *graph-structural proxy*.  The cycle-basis-size component of the `cycle` proxy is the only entry that specifically isolates a topological invariant (β₁, the rank of the first homology group).  The other proxies (size, degree, WL subtree, spectral) may co-vary with topology but do not isolate it.  With n=3 datasets, the Spearman ρ values are reported descriptively only — they carry no inferential power.