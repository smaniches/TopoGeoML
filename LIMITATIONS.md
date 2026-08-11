# Limitations of TopoGeoML v0.0.6

TopoGeoML is a pre-stable scientific software library and an accompanying empirical research record. This document states the limits of the current implementation and of the claims supported by the repository. It is not a roadmap.

If observed behavior contradicts the public API documentation, treat that as a defect and open an issue.

## 1. Scientific claims are configuration-bound

The graph-classification investigation uses a deliberately constrained matched-capacity regime: primarily one message-passing layer, `hidden_dim=32`, 10 to 20 epochs, Adam at `1e-2`, no batch normalisation, and deterministic stratified train/test splits generated separately for each seed. Model arms within the same seed share the same split, while different seeds use different deterministic splits. Those experiments are mechanism studies, not competitive benchmark submissions.

The strongest current architecture conclusion is negative and specific: on the tested graph-classification configurations, the Hodge `L_0` operator does not provide a unique advantage over a matched normalized-adjacency external-residual control. H008c and H010 support that scoped conclusion. They do not establish that one propagation operator is universally superior.

The NCI1 Hodge-residual versus MLP comparison is a positive-difference result at the tested configuration (`median delta = +0.086`, within-experiment paired Wilcoxon `p_BH = 4.83e-3`). It does not establish that Hodge propagation is generally superior to graph neural networks or to adjacency propagation. The previous investigation-wide 59-comparison and 76-entry sensitivity tables are withdrawn as current evidence because they included comparisons from the invalidated H009 experiment.

H009 is invalidated by implementation audit. The historical `sheaf-residual` matrix was not guaranteed to be the claimed symmetric positive-semidefinite cellular-sheaf Laplacian, and the historical parameter count was also misstated. The old H009 artifact remains public for provenance only. The implementation has since been repaired (`sheaf-residual` 2.0.0) and the experiment rerun under the preregistered H009-R protocol (`notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}`), which resolves H39-H41: the repaired arm is above the matched MLP, shows no significant difference from the fixed Hodge operator (not an equivalence finding), and is directionally below the matched normalized-adjacency arm without crossing H41's preregistered 0.01 falsification threshold.

Non-significant pairwise tests are reported as failures to detect a difference at the tested power. They are not equivalence results unless an equivalence procedure is explicitly performed. See [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md) and [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md).

H011b, the triangle-rich COLLAB test of edge-level `L_1` propagation, has only a directional smoke result. A separate 18-seed compute attempt exceeded the GitHub Actions time limit, but the preregistered confirmatory design is 30 seeds. No cross-domain `L_1` claim is licensed until that design is completed.

The topology-divergence callback result is exploratory. In the published experiment the topology trigger fires at its earliest possible step on every seed, and no non-overfitting negative control was run. The result shows that the trigger was not later than the loss trigger under that experiment; it does not establish anticipatory prediction of divergence.

## 2. Persistent-homology scope

### Vietoris-Rips

`RipsFiltration` uses `ripser` on CPU. Rips-complex growth is combinatorial in the worst case. Large or dense point clouds can become expensive rapidly, especially when `max_homology_dim` increases or `max_edge_length` is unconstrained. For large clouds, set a finite threshold based on the data scale or subsample deliberately.

The public filtration API currently exposes Vietoris-Rips persistence. Alpha, Cech, witness, and general lower-star filtrations are not exposed through `RipsFiltration`.

### Cubical persistence

Differentiable cubical persistence is implemented in `topogeoml.nn.cubical_diff_ph` using GUDHI to identify persistence pairings and PyTorch indexing to route gradients to critical pixels. The combinatorial pairing step is CPU-bound and discrete. Gradients are valid for the selected critical pairing, but pairing changes create the expected piecewise-smooth behavior of topology-based losses. This is not a fully GPU-native cubical persistence implementation.

`CubicalTopologyLoss` is implemented and tested as a differentiable loss primitive. The repository does not yet contain a completed, statistically powered end-to-end segmentation study demonstrating that it improves a production segmentation model on DRIVE or another imaging benchmark.

The separate `cubical_mask_diagnostic` is a lightweight diagnostic and should not be confused with the differentiable cubical persistence backend.

### Diagram representations and distances

The public feature API ships persistence images and Betti curves. Persistence landscapes, silhouettes, and a general public kernel family are not currently part of the package API.

Bottleneck distance is used internally by the benchmark stability machinery through GUDHI, but TopoGeoML does not currently expose a general-purpose persistence-diagram distance API. Wasserstein distance is not part of the public package API.

## 3. Differentiable persistence limits

`topogeoml.nn.diff_ph` reconstructs differentiable Vietoris-Rips bar values by indexing critical distances back into an autograd-connected PyTorch distance matrix.

For `H_0`, finite deaths are tied to minimum-spanning-tree edges and the thresholded essential-component behavior is reconstructed explicitly.

For `H_1`, the implementation uses ripser cocycle information for birth routing and a critical-distance routing rule for finite deaths. This provides a useful piecewise subgradient path, but persistent-pair identities can change under perturbation and equal or nearly equal filtration values can make the selected route non-unique. Treat this layer as a differentiable research primitive, not as a claim of globally smooth or uniquely defined persistence gradients.

The repository checks gradient flow, selected small-input `torch.autograd.gradcheck` cases, and perturbation stability in the benchmark harness. Those checks are evidence for the exercised regimes, not a proof that every degenerate filtration or every large-scale training trajectory has numerically well-conditioned gradients.

`TopologyRegularizer` subsamples when the point count exceeds `max_points`. That changes the topology being regularized. Use the subsampling setting as an explicit modeling choice, not as a transparent acceleration.

## 4. Feature-pipeline limits

`TopologyFeaturePipeline` is a scikit-learn transformer for batches of point clouds, but each sample is currently processed serially. The public `n_jobs` constructor argument is reserved and does not provide parallel persistence computation in v0.0.6.

The persistence-image representation depends on the fitted filtration scale, image resolution, and Gaussian bandwidth. Distribution shift in geometric scale can therefore move test points outside the scale represented well by the training grid. Fit the transformer inside each cross-validation fold and normalize geometry deliberately when scale is not itself a feature.

Betti curves and persistence images are summaries, not lossless representations of persistence diagrams. Their discriminative value is task-dependent and can degrade under noise or inappropriate scale choices.

## 5. Simplicial and Hodge limits

`SimplicialComplex` enumerates faces of supplied facets. Clique-complex construction can grow exponentially with clique size, so dense graphs require conservative `max_dim` choices.

`betti_numbers` converts Hodge Laplacians to dense arrays and calls `numpy.linalg.eigvalsh`. It is intended for small complexes. It is not a scalable sparse homology solver.

`HodgeMessagePassing` is a minimal fixed-complex layer. It stores one Laplacian as a module buffer and does not implement variable-topology graph batching, multi-rank message passing, attention, learned incidence maps, or a complete simplicial neural-network architecture.

At `k=0`, TopoGeoML's normalized Hodge operator is a normalized graph Laplacian, not the standard Kipf-Welling GCN propagation operator. On positive-degree vertices, the usual identity is `I - D^{-1/2} A D^{-1/2}`; isolated vertices follow the implementation's zero degree-support convention and retain zero normalized rows. The distinction between high-pass Laplacian propagation and adjacency-based propagation is central to H008c and should not be collapsed in interpretation.

## 6. Embedding-audit limits

`audit_embedding` is a diagnostic prototype. Its `beta_0_estimate` and `beta_1_estimate` fields count persistent features whose lifetime exceeds a heuristic threshold. They are not exact Betti numbers at a specified filtration scale.

The default threshold, twice the median nearest-neighbor distance, is a heuristic noise floor. A single global threshold can be inappropriate for embeddings with strongly heterogeneous density. Audit outputs should be interpreted comparatively or diagnostically unless the threshold has been calibrated for the domain.

The audit subsamples to bound Rips cost, so results can vary with the chosen `max_points` and seed when the original embedding is larger than that limit.

## 7. Signal-analysis limits

The signal utilities provide Takens delay embedding, an autocorrelation-based delay heuristic, and sliding-window persistent-homology features. The delay estimator is a baseline heuristic, not a substitute for domain-specific embedding-dimension and delay selection.

Sliding-window persistence runs `ripser` independently across windows. Long signals, large windows, or higher homology dimensions can therefore be computationally expensive. The provided pooled topology statistics are invariant under the documented geometric transformations, but they are not guaranteed to improve downstream prediction on every signal domain.

## 8. Engineering and platform limits

The required CI matrix covers Python 3.11 and 3.12 on Linux and macOS, with a separate full-dependency package-coverage gate. Windows is declared as a supported package target but is not part of the current required CI matrix.

Mypy strict mode is enforced on `topogeoml/`, while missing third-party type stubs are ignored. A clean type-check therefore establishes consistency of the library's own annotations, not completeness of external package typing.

The benchmark harness is research infrastructure and is not part of the 100% package-coverage invariant. The maintained coverage claim is 100% line and 100% branch coverage for the importable `topogeoml` package under the required full-dependency gate.

`write_results` uses a temporary file followed by `os.replace`. This provides strong same-filesystem behavior on normal POSIX use, but the package does not claim transactional durability under every filesystem, process-crash, or concurrent-writer scenario.

## 9. What TopoGeoML does not claim

TopoGeoML does not claim to replace GUDHI, ripser, PyTorch Geometric, or broad topological-deep-learning frameworks. It composes narrower primitives into a typed, inspectable research workflow with provenance, scikit-learn integration, differentiable topology losses, simplicial operators, signal features, and explicit claim-to-evidence discipline.

It does not claim that topology improves every machine-learning task, that Hodge propagation is universally superior, that the historical H009 run is valid sheaf evidence, that its differentiable persistence backend is the fastest available implementation, or that the current graph-classification results generalize beyond the tested configurations.

## 10. Version stability

`0.0.6` is pre-stable. Public APIs can change before `1.0`. Pin exact versions in downstream research, record configuration and dependency versions, and rerun validation when upgrading.

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)).
