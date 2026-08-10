# Project Status

## What this project is

TopoGeoML is a topology-aware scientific software library with an attached preregistered empirical research record. The installed package provides persistent-homology features, differentiable topology primitives, simplicial and Hodge operators, signal-topology features, embedding diagnostics, and reproducible experiment metadata. The graph-classification investigation is one application of the library, not the definition of the package.

TopoGeoML 0.0.6 is beta research software, not a production training framework. Public APIs can change before 1.0.

## Investigation summary

The current research record contains 14 preregistered hypothesis documents spanning H001-H011b and 53 falsifiable sub-predictions. The statistical record covers MUTAG, PROTEINS, NCI1, and a directional COLLAB smoke run across 11 registered model variants.

H009 is invalidated by implementation audit. Its archived numerical artifact is retained for provenance but is not valid evidence about a cellular-sheaf Laplacian. H39-H41 are unresolved until the operator is corrected, mathematically validated, and rerun. Because the previous investigation-wide 59-comparison and 76-entry sensitivity tables included H009 comparisons, those tables are withdrawn as current project-level claims pending a corrected H009 rerun.

H011 is resolved on NCI1 as a negative/inconclusive higher-order test under a triangle-sparse regime. H011b remains unresolved at statistical rigor on triangle-rich COLLAB.

### Findings

1. **A narrow positive NCI1 difference is real within the tested regime.** In H003, `hodge-mp-residual` reaches 0.609 versus 0.523 for the matched-capacity MLP, a median difference of +0.086 with within-experiment p_BH = 4.83 x 10^-3. This is a mechanism study at fixed capacity, not a benchmark-performance claim. The previously reported investigation-wide 59-comparison sensitivity statement is withdrawn until the validated comparison family is rebuilt after H009 repair.

2. **No unique `L_0` Hodge advantage is supported once the architecture is matched.** In H008c on NCI1, the external-residual normalized-adjacency arm reaches 0.629 versus 0.609 for Hodge and 0.523 for MLP. The adjacency-versus-Hodge median difference is +0.0195 with p_BH = 0.0101. H010 then finds a significant adjacency advantage on MUTAG (0.789 versus 0.750, p_BH = 7.44 x 10^-3), no significant operator difference on PROTEINS (p_BH = 0.292), and the same favorable adjacency direction on NCI1. No tested dataset provides evidence of Hodge superiority over the matched normalized-adjacency arm.

3. **The external-residual formulation is sufficient to recover the normalized-adjacency arm on NCI1, but the causal claim must remain scoped.** `gin-normalised` uses a trainable self contribution inside the affine/nonlinear update, whereas `gin-residual` uses an identity skip outside the activation. Changing to the external-residual formulation raises the median from 0.500 to 0.629. This identifies the successful tested formulation and removes any need to invoke a unique Hodge operator effect. It does not prove that residual connections alone, independent of placement and self-path parameterization, are the sole mechanism in other models or datasets.

4. **H009 is invalidated, not negative evidence about sheaf methods.** The historical `sheaf-residual` implementation processed both orientations of each undirected edge independently, so its learned matrix was not guaranteed symmetric or positive semidefinite and was not, in general, the claimed `delta.T @ delta` cellular-sheaf Laplacian. The historical model also had 2,468 trainable parameters on NCI1 versus 2,338 for the fixed-operator controls, about 5.56% larger and slightly outside the stated 5% capacity tolerance. The old H009 numbers remain public only as provenance. H39-H41 have no current inferential status until a corrected rerun is completed.

5. **The NCI1 `L_1` experiment does not establish a higher-order advantage.** In H011, `l1-hodge-residual` reaches 0.590. It does not have a significant positive difference from MLP (p_BH = 0.0957) or from the `L_0` Hodge arm (p_BH = 0.0787), and it is lower than gin-residual (p_BH = 0.00676). More importantly, 96% of NCI1 graphs contain no triangles, so the triangle-based up-Laplacian term is absent for almost all graphs. H011b on triangle-rich COLLAB is the correct higher-order follow-up and remains unresolved beyond a one-seed directional smoke result.

6. **Several reusable library components are independent of the graph-classification result.** The package exposes scikit-learn persistent-homology features, differentiable Vietoris-Rips and cubical topology primitives, `CubicalTopologyLoss`, simplicial/Hodge algebra, signal-topology features, and provenance utilities. Their software correctness is tested separately from claims of downstream task improvement.

### What is not claimed

- Topology improves graph classification or machine learning in general.
- Hodge propagation is better than well-tuned graph neural networks.
- H008c proves that residual connections are the sole causal mechanism in arbitrary architectures.
- The historical H009 run is valid evidence about a cellular-sheaf Laplacian.
- Non-significant comparisons establish statistical equivalence.
- The tested `L_1` architecture provides unique higher-order signal on triangle-rich data.
- `CubicalTopologyLoss` has been shown to improve an end-to-end segmentation benchmark at statistical rigor.
- Any empirical result generalizes beyond its stated datasets, model family, capacity, training budget, and statistical design.

## Quality

| Metric | Value |
|---|---|
| Test suite | Required CI matrix; the exact verified full-dependency snapshot and reproduction command are recorded in [docs/CLAIMS_TO_EVIDENCE.md](docs/CLAIMS_TO_EVIDENCE.md) |
| Coverage | 100% line and 100% branch coverage on the `topogeoml` package with full dependencies (`.[all]`), enforced by the full-deps `coverage-gate` CI job (`--cov-branch --cov-fail-under=100`); the `benchmarks/` research harness is outside the gated package scope |
| Type checking | mypy strict enforced in CI on `topogeoml/` |
| Lint | ruff enforced in CI |
| DOI | [10.5281/zenodo.20365816](https://doi.org/10.5281/zenodo.20365816) |
| Statistical analysis | Within-experiment preregistered decision rules remain current; the former investigation-wide 59/76 sensitivity table is withdrawn pending corrected H009 evidence |
| Preregistration | 14 hypothesis documents with git-timestamped history |
| Invalidated evidence | H009 historical artifact retained for provenance and excluded from current scientific claims |

## Open items

| Item | Current state | Required next evidence |
|---|---|---|
| H009 repaired sheaf operator | Historical implementation invalidated; H39-H41 unresolved | Correct the operator from one undirected edge set, prove symmetry/PSD/identity/gradient invariants, satisfy exact capacity accounting, then run a fresh 30-seed NCI1 corrective replication |
| H011b (`L_1` on COLLAB) | One-seed, one-epoch smoke run only; an 18-seed compute attempt exceeded the GitHub Actions time limit | Complete the preregistered 30-seed experiment on adequate compute before making a higher-order cross-domain claim |
| Cubical topology loss | Differentiable primitive implemented and tested | Run a preregistered, powered end-to-end segmentation study before claiming downstream improvement |
| Historical research report | [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) is the Version 0.0.2 snapshot through H008c | Preserve it as history; use this status file, [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md), and [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) for current state |
| Broader architecture claims | Current graph results are mostly one-layer, hidden_dim=32, short-budget mechanism studies | Any deeper or differently normalized architecture study should be preregistered as a new experiment rather than retroactively generalized |

## How to verify

See [REVIEWER.md](REVIEWER.md) for the verification path, [LEADERBOARD.md](LEADERBOARD.md) for the empirical evidence index, and [docs/CLAIMS_TO_EVIDENCE.md](docs/CLAIMS_TO_EVIDENCE.md) for claim-level evidence and reproduction commands.
