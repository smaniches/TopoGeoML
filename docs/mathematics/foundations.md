---
title: Mathematical foundations
nav_order: 6
---

# Mathematical Foundations for Topology-Aware Signal Analysis

**TopoGeoML** — Santiago Maniches (ORCID: 0009-0005-6480-1987), TOPOLOGICA LLC.

This document specifies the mathematical framework underlying the `topogeoml.signal` subpackage and the empirical validation in `experiments/ucr_topology_validation/`. Every statement is classified. Every theorem either contains a proof or cites a primary source by theorem number. Algorithms specify input space, output space, correctness criterion, and complexity. The reader is assumed to be familiar with the standard definitions of simplicial complex, filtration, and persistence module (Edelsbrunner & Harer, 2010, Chapters III, VII).

## 1. Setup

### 1.1 Conventions

Throughout this document:

- $\mathbb{R}^d$ is real $d$-space equipped with the standard Euclidean inner product $\langle \cdot, \cdot \rangle$ and metric $\|x - y\|_2 = \sqrt{\langle x - y, x - y \rangle}$.
- $(X, d_X)$ denotes a finite metric space with cardinality $|X| = n$.
- $\mathcal{D} = \{(b_i, d_i)\}_{i \in I}$ denotes a persistence diagram: a multiset of points in $\overline{\mathbb{R}}^2_{\le}$, the set $\{(b, d) \in \overline{\mathbb{R}}^2 : b \le d\}$, where $\overline{\mathbb{R}} = \mathbb{R} \cup \{+\infty\}$.
- $H_k$ denotes singular homology in dimension $k$ with coefficients in a fixed field $\mathbb{F}$ (we take $\mathbb{F} = \mathbb{F}_2$ unless stated otherwise).

### 1.2 Signals and point clouds

**DEFINITION 1.2.1** (Discrete signal). A *discrete signal of length $T$ in $\mathbb{R}^k$* is a map $s: \{0, 1, \ldots, T-1\} \to \mathbb{R}^k$. We identify $s$ with the matrix $S \in \mathbb{R}^{T \times k}$ whose row $t$ is $s(t)$. The space of all such signals is denoted $\mathcal{S}_{T,k} = \mathbb{R}^{T \times k}$.

**DEFINITION 1.2.2** (Sliding-window point cloud). Let $s \in \mathcal{S}_{T,k}$, $W \in \mathbb{N}$ with $W \le T$, and $t_0 \in \{0, 1, \ldots, T - W\}$. The *sliding-window point cloud at offset $t_0$ with window length $W$* is the finite metric subspace
$$P_W(s, t_0) = \{s(t_0), s(t_0+1), \ldots, s(t_0 + W - 1)\} \subset \mathbb{R}^k,$$
equipped with the metric inherited from $\mathbb{R}^k$.

**REMARK 1.2.3**. $P_W(s, t_0)$ has cardinality at most $W$; equality holds when all $s(t_0 + i)$ for $i = 0, \ldots, W-1$ are distinct.

## 2. The Vietoris–Rips Filtration

**DEFINITION 2.1** (Vietoris–Rips complex). Let $(X, d_X)$ be a finite metric space and $r \ge 0$. The *Vietoris–Rips complex at scale $r$* is the abstract simplicial complex
$$\operatorname{VR}_r(X) = \{\sigma \subseteq X : \sigma \neq \emptyset,\ \operatorname{diam}_{d_X}(\sigma) \le r\},$$
where $\operatorname{diam}_{d_X}(\sigma) = \max_{x, y \in \sigma} d_X(x, y)$.

**DEFINITION 2.2** (Vietoris–Rips filtration). The family $\{\operatorname{VR}_r(X)\}_{r \ge 0}$ is the *Vietoris–Rips filtration* of $X$. For $r \le r'$ we have inclusion $\operatorname{VR}_r(X) \hookrightarrow \operatorname{VR}_{r'}(X)$, making the family a filtration in the sense of Edelsbrunner & Harer (2010), Chapter VII §1.

**DEFINITION 2.3** (Persistent homology). For each integer $k \ge 0$, applying singular homology $H_k(-; \mathbb{F})$ to the Vietoris–Rips filtration yields a persistence module
$$\operatorname{PH}_k(X) := \bigl\{H_k(\operatorname{VR}_r(X); \mathbb{F})\bigr\}_{r \ge 0}$$
together with the maps induced by inclusion. By Crawley-Boevey (2015, Theorem 1.1) — see also Edelsbrunner & Harer (2010), Chapter VII §3, Theorem (Structure Theorem for Persistence Modules) — this module decomposes uniquely up to isomorphism as a direct sum of interval modules. The multiset of intervals is the *persistence diagram* $\operatorname{Dgm}_k(X)$, identified with a multiset in $\overline{\mathbb{R}}^2_{\le}$ by $(b, d) \mapsto $ the interval $[b, d)$.

**THEOREM 2.4** (Stability of persistence diagrams). *Let $(X, d_X)$ and $(Y, d_Y)$ be finite metric spaces. Then for every $k \ge 0$,*
$$d_B(\operatorname{Dgm}_k(X), \operatorname{Dgm}_k(Y)) \le 2 \cdot d_{GH}(X, Y),$$
*where $d_B$ is the bottleneck distance on persistence diagrams and $d_{GH}$ is the Gromov–Hausdorff distance between finite metric spaces.*

**Reference.** Chazal, de Silva, Glisse, & Oudot (2016), *The structure and stability of persistence modules*, Theorem 5.2 (the factor of 2 is tight for the Rips filtration; for the Čech filtration the constant is 1). The original result of Cohen-Steiner, Edelsbrunner, & Harer (2007, Theorem 4.4) gives the constant 1 for tame functions on the same domain; the metric version above is the standard generalization in the source above.

**REMARK 2.5** (Why this matters here). Theorem 2.4 is the foundation that makes persistence diagrams a usable feature in machine learning: small perturbations of input data produce small (in bottleneck distance) perturbations of the diagram. Without this theorem, $\operatorname{Dgm}_k$ could not be used as a feature extractor in a noisy empirical pipeline.

## 3. Takens Delay Embedding

**DEFINITION 3.1** (Delay embedding). Let $s: [0, T] \to \mathbb{R}$ be a continuous signal, $m \in \mathbb{N}$ (embedding dimension), and $\tau > 0$ (delay). The *delay embedding* is the map
$$\Phi_{m, \tau} : [(m-1)\tau, T] \to \mathbb{R}^m, \qquad t \mapsto \bigl(s(t),\ s(t - \tau),\ \ldots,\ s(t - (m-1)\tau)\bigr).$$

**THEOREM 3.2** (Takens' embedding theorem). *Let $M$ be a compact $C^2$ manifold of dimension $d$ and $\varphi: M \to M$ a $C^2$ diffeomorphism. For a generic $C^2$ observable $y: M \to \mathbb{R}$ and any integer $m \ge 2d + 1$, the map*
$$\Phi: M \to \mathbb{R}^m, \qquad p \mapsto (y(p), y(\varphi(p)), \ldots, y(\varphi^{m-1}(p)))$$
*is a $C^2$ embedding.*

**Reference.** Takens, F. (1981), *Detecting strange attractors in turbulence*, in: D. Rand & L.-S. Young (eds.), *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898, Springer, pp. 366–381. Theorem 1 (p. 366). The statement here is the discrete-time version; the continuous-time version (with $\varphi$ replaced by a flow) is in the same paper.

**REMARK 3.3** (Genericity and what "generic" means here). "Generic" in Takens' theorem means: the set of $(y, \varphi)$ pairs for which $\Phi$ is an embedding contains an open dense subset of $C^2(M, \mathbb{R}) \times \operatorname{Diff}^2(M)$ in the Whitney $C^2$ topology. The theorem does *not* assert that any particular observable–dynamics pair admits an embedding; this must be verified case by case or assumed.

**REMARK 3.4** (Application to discrete signals). For a finite discrete signal $s: \{0, \ldots, T-1\} \to \mathbb{R}$, the delay embedding is the finite point cloud
$$\Phi_{m,\tau}(s) = \bigl\{\bigl(s(t),\ s(t-\tau),\ \ldots,\ s(t-(m-1)\tau)\bigr) : t = (m-1)\tau, \ldots, T-1\bigr\} \subset \mathbb{R}^m,$$
of cardinality $T - (m-1)\tau$. This is the object we feed to the Vietoris–Rips filtration in the implementation. Takens' theorem provides the asymptotic justification for using delay embeddings to recover dynamical structure; in finite-sample practice we treat $m$ and $\tau$ as hyperparameters and validate their choice empirically.

## 4. Group Actions and Invariances

The Erlangen viewpoint of Klein (1872) directs attention to invariants of group actions. We make explicit which group acts on signal space and which features we extract are equivariant or invariant.

**DEFINITION 4.1** (Natural symmetry group of discrete signals). Let
$$G = \operatorname{Trans}(\mathbb{Z}) \times \mathbb{R}_{>0} \times \mathbb{R} \times \{\pm 1\},$$
acting on $\mathcal{S}_{T, k}$ (extended to bi-infinite signals when necessary; for finite $T$ the translation action is defined modulo boundary effects) by
$$(\Delta t,\ \alpha,\ \beta,\ \epsilon) \cdot s(t) = \epsilon \cdot \alpha \cdot s(t - \Delta t) + \beta \cdot \mathbf{1}_k,$$
where $\mathbf{1}_k = (1, \ldots, 1) \in \mathbb{R}^k$, $\alpha \in \mathbb{R}_{>0}$ is a global scaling, $\beta \in \mathbb{R}$ is a global offset, $\epsilon \in \{\pm 1\}$ is a sign flip, and $\Delta t \in \mathbb{Z}$ is a temporal translation.

**PROPOSITION 4.2** (Invariances of Vietoris–Rips persistence on sliding windows). *Let $s \in \mathcal{S}_{T,k}$ and let $P_W(s, t_0) \subset \mathbb{R}^k$ be a sliding-window point cloud (Definition 1.2.2). Let $\Phi: \mathbb{R}^k \to \mathbb{R}^k$ be an isometry of $(\mathbb{R}^k, \|\cdot\|_2)$ (orthogonal transformation composed with a translation). Then*
$$\operatorname{Dgm}_k(\Phi(P_W(s, t_0))) = \operatorname{Dgm}_k(P_W(s, t_0))$$
*for every $k$.*

**Proof.** The Vietoris–Rips filtration depends only on pairwise distances. Isometries preserve all pairwise distances. Therefore the entire filtered simplicial complex is isomorphic as a filtered complex, and persistence diagrams of isomorphic filtered complexes coincide (this is the universality of the bar-code decomposition; Crawley-Boevey 2015, Theorem 1.1, gives this as an isomorphism of decompositions). $\square$

**COROLLARY 4.3**. *Persistence diagrams of sliding-window point clouds are invariant under global temporal translation $\Delta t$ and (in dimension $k = 1$, where sign flip is realized by reflection through the origin only when composed with an appropriate offset) under reflection. They are not invariant under global temporal scaling $t \mapsto t/c$ nor under signal scaling $s \mapsto \alpha s$ for $\alpha \ne 1$.*

**REMARK 4.4** (Achieving scale invariance). Scale invariance under $s \mapsto \alpha s$ (which rescales all pairwise distances by $\alpha$) can be obtained by normalizing the persistence diagram: divide all bar endpoints by the diameter $\operatorname{diam}(P_W(s, t_0))$ before further processing. This is the standard scale-normalization choice in the TDA literature (Bubenik, 2015, §2.2 in the context of persistence landscapes). We adopt this normalization in `signal/invariant_features.py`.

## 5. Vectorization of Persistence Diagrams

**DEFINITION 5.1** (Total persistence). For a persistence diagram $\mathcal{D} = \{(b_i, d_i)\}_{i \in I}$ and $p \in [1, \infty)$, the *$p$-th total persistence* is
$$\operatorname{Pers}_p(\mathcal{D}) = \sum_{i \in I,\ d_i < \infty} (d_i - b_i)^p.$$
Infinite bars are excluded by convention (they would dominate any finite-$p$ sum).

**DEFINITION 5.2** (Persistence entropy). For a persistence diagram $\mathcal{D} = \{(b_i, d_i)\}_{i \in I_{\text{fin}}}$ with only the finite bars, let $L = \sum_{i} (d_i - b_i)$. If $L > 0$, define $p_i = (d_i - b_i) / L$. The *persistence entropy* is
$$E(\mathcal{D}) = -\sum_{i \in I_{\text{fin}}} p_i \log p_i,$$
with the convention $0 \log 0 := 0$. If $L = 0$ (all finite bars degenerate), $E(\mathcal{D}) := 0$.

**Reference.** Chintakunta, Gentimis, Gonzalez-Diaz, Jimenez, & Krim (2015), *An entropy-based persistence barcode*, Pattern Recognition 48(2), 391–401. Definition 3.1.

**PROPOSITION 5.3** (Stability of persistence entropy under bottleneck perturbations). *Persistence entropy is not Lipschitz under the bottleneck distance. A counterexample: let $\mathcal{D}_\varepsilon$ have two bars of lifetimes $1$ and $\varepsilon$, and $\mathcal{D}_0$ have one bar of lifetime $1$. As $\varepsilon \downarrow 0$, $d_B(\mathcal{D}_\varepsilon, \mathcal{D}_0) = \varepsilon \to 0$, but $E(\mathcal{D}_\varepsilon) \to \log\bigl(\tfrac{1}{1}\bigr) = 0 = E(\mathcal{D}_0)$ — so in this case continuous. A genuine counterexample: $\mathcal{D}_\varepsilon$ has two bars of lifetimes $1$ and $1 - \varepsilon$; $E(\mathcal{D}_\varepsilon) \to \log 2$ as $\varepsilon \to 0$. While $\mathcal{D}_0$ with one bar of lifetime $1$ has $E = 0$. Bottleneck distance $\to 0$ while entropy jumps by $\log 2$.* $\square$

**REMARK 5.4** (Consequence for empirical work). Proposition 5.3 implies that persistence entropy can change discontinuously under small perturbations of the input. In practice this introduces variance in the entropy feature across noise realizations. The empirical experiment of §7 reports cross-validated variance to quantify this.

**DEFINITION 5.5** (Topology feature vector of a signal). Given a signal $s \in \mathcal{S}_{T,k}$, window length $W$, stride $\Delta$, embedding dimension $m$, delay $\tau$, and maximum homology dimension $K$, the *topology feature vector* $\mathbf{F}(s) \in \mathbb{R}^{q}$ is constructed by the following procedure:

1. (If $k = 1$) compute the delay embedding $\Phi_{m,\tau}(s)$, yielding a point cloud in $\mathbb{R}^m$. If $k > 1$ the signal is already in $\mathbb{R}^k$ and we skip this step.
2. For each window offset $t_0 \in \{0, \Delta, 2\Delta, \ldots\}$, compute the sliding-window point cloud $P_W(\Phi_{m,\tau}(s), t_0)$.
3. Compute the Vietoris–Rips persistence diagrams $\operatorname{Dgm}_0, \operatorname{Dgm}_1, \ldots, \operatorname{Dgm}_K$ for each window.
4. From each diagram extract: $|\mathcal{D}_{\text{finite}}|$ (count of finite bars), $\operatorname{Pers}_1(\mathcal{D})$, $\operatorname{Pers}_2(\mathcal{D})$, $E(\mathcal{D})$, $\max_i(d_i - b_i)$ (longest lifetime).
5. Concatenate the per-window per-dimension statistics into a single vector, optionally pooled (mean and max) across windows to produce a fixed-length vector.

This construction is implemented in `topogeoml.signal.invariant_features.signal_topology_features`.

## 6. Algorithm Specifications

### 6.1 Delay embedding

**Input space**: $\mathcal{S}_{T,1} = \mathbb{R}^T$ (univariate discrete signal); $m \in \mathbb{N}_{\ge 1}$; $\tau \in \mathbb{N}_{\ge 1}$, with $(m-1)\tau \le T - 1$.

**Output space**: $\mathbb{R}^{(T - (m-1)\tau) \times m}$ (matrix whose rows are points in $\mathbb{R}^m$).

**Correctness criterion**: row $t$ of the output (for $t = 0, \ldots, T - (m-1)\tau - 1$) equals $(s((m-1)\tau + t),\ s((m-1)\tau + t - \tau),\ \ldots,\ s((m-1)\tau + t - (m-1)\tau))$ pointwise.

**Complexity**: $O(T m)$ time, $O(T m)$ space.

**Reference implementation**: `topogeoml.signal.delay_embedding.takens_embedding`.

### 6.2 Sliding-window topology features

**Input space**: Point cloud $X \in \mathbb{R}^{N \times k}$; $W, \Delta \in \mathbb{N}_{\ge 1}$ with $W \le N$; $K \in \mathbb{N}_{\ge 0}$ (maximum homology dimension); $\theta \in \mathbb{R}_{\ge 0}$ (Vietoris–Rips edge cutoff).

**Output space**: $\mathbb{R}^{(K+1) \times 5}$ (pooled per-dimension feature matrix; 5 statistics per homology dimension as in Definition 5.5 step 4) or $\mathbb{R}^{w \times (K+1) \times 5}$ if not pooled, where $w = \lfloor (N - W)/\Delta \rfloor + 1$.

**Correctness criterion**: For each window $j = 0, 1, \ldots, w-1$ at offset $t_0 = j \Delta$, the output statistics for dimension $k$ equal those computed from $\operatorname{Dgm}_k(P_W(X, t_0))$ via Definition 5.5 step 4. Where pooling is applied, the output is the elementwise mean and max of the per-window statistics. Determinism: identical inputs produce bitwise identical outputs.

**Complexity**: Per window: ripser is $O(W^{2K+2})$ worst case for $K+1$-dimensional persistence (Bauer, 2021, §4 — "Ripser: efficient computation of Vietoris-Rips persistence barcodes", *J. Appl. Comput. Topology* 5, 391–423). For $K = 1$ and $W \le 200$ in practice the computation is bounded by clearing-and-twisting reduction and is empirically near-quadratic in $W$.

**Reference implementation**: `topogeoml.signal.sliding_window.sliding_window_topology_features`.

## 7. Empirical Validation Protocol

**DEFINITION 7.1** (Topology-augmentation classification task). Let $(\mathcal{X}, \mathcal{Y})$ be a dataset of $n$ signals $\mathcal{X} = \{s_1, \ldots, s_n\} \subset \mathcal{S}_{T,1}$ with class labels $\mathcal{Y} = \{y_1, \ldots, y_n\} \in \{0, 1\}^n$. Let $\mathbf{B}: \mathcal{S}_{T,1} \to \mathbb{R}^{d_B}$ denote a *baseline* feature map (e.g., statistical summaries: mean, standard deviation, percentiles), $\mathbf{F}: \mathcal{S}_{T,1} \to \mathbb{R}^{d_F}$ denote the topology feature map of Definition 5.5, and $\mathbf{BF}: \mathcal{S}_{T,1} \to \mathbb{R}^{d_B + d_F}$ denote their concatenation.

The *topology-augmentation null hypothesis* $\mathcal{H}_0$: for a fixed classifier $C$ (logistic regression in our experiment), the cross-validated balanced accuracy of $C$ trained on $\mathbf{BF}(\mathcal{X})$ is equal to that of $C$ trained on $\mathbf{B}(\mathcal{X})$.

The *one-sided alternative* $\mathcal{H}_1$: the balanced accuracy of $C$ on $\mathbf{BF}$ exceeds that on $\mathbf{B}$.

**REMARK 7.2** (Why balanced accuracy). UCR datasets vary in class balance. Balanced accuracy $= \tfrac{1}{2}(\operatorname{TPR} + \operatorname{TNR})$ removes the dependence on class prior and prevents trivially high accuracy on imbalanced datasets dominated by one class.

**PROTOCOL 7.3** (Cross-validation, permutation test, multiple comparison correction). The complete experiment is:

1. Fix the random seed $42$ for all randomness (`numpy.random.default_rng(42)` and equivalent for `scikit-learn`).
2. For each dataset $\mathcal{D}_j$ from the UCR archive: load with stratified 5-fold cross-validation, repeated $R = 3$ times with seeds $42, 43, 44$, yielding $15$ fold balanced-accuracy scores per feature set.
3. Within each fold, all feature computation, scaling, and classifier fitting use *training-fold data only*. Test-fold data is never observed during preprocessing. This satisfies the correction-audit gate of `statistical-rigor-engine` §2 step 2.
4. Compute the per-fold accuracy difference $\Delta_i = \operatorname{acc}_{\mathbf{BF}, i} - \operatorname{acc}_{\mathbf{B}, i}$ for $i = 1, \ldots, 15$.
5. Compute observed mean difference $\bar\Delta = \tfrac{1}{15}\sum_i \Delta_i$ and Cohen's $d_z = \bar\Delta / \operatorname{sd}(\Delta_i)$.
6. Compute a sign-flip permutation test on the paired differences ($\Delta_i$, $-\Delta_i$ are exchangeable under $\mathcal{H}_0$): $10{,}000$ permutations of sign assignments, yielding an exact one-sided $p$-value.
7. Compute a BCa bootstrap 95% confidence interval for $\bar\Delta$ with $10{,}000$ resamples (following the implementation in `statistical-rigor-engine` §3.0 lines 102–122).
8. Apply Benjamini–Hochberg correction (Benjamini & Hochberg, 1995, *J. Royal Stat. Soc. B* 57, 289–300) across all datasets to control FDR at $\alpha = 0.05$.

**PROPOSITION 7.4** (Validity of the sign-flip permutation test). *Under $\mathcal{H}_0$ with exchangeable folds and an additive treatment-effect model $\Delta_i \stackrel{d}{=} \mu + \varepsilon_i$ with $\varepsilon_i \stackrel{d}{=} -\varepsilon_i$ (symmetric error under treatment swap), the distribution of $\bar\Delta$ is symmetric about $0$. The sign-flip null distribution coincides with the true null distribution of $\bar\Delta$. Therefore the empirical $p$-value from the sign-flip permutation test is exactly valid (not asymptotically).*

**Proof sketch.** Under $\mathcal{H}_0$ with symmetric errors, each $\Delta_i$ is exchangeable with $-\Delta_i$. The full collection $\{\Delta_i\}_{i=1}^{15}$ is jointly exchangeable with any sign-flipped version. Therefore the empirical distribution of $\bar\Delta$ under all $2^{15} = 32{,}768$ sign assignments is the exact null distribution. Sampling $10{,}000$ random sign assignments approximates this exact distribution; the resulting Monte Carlo $p$-value is unbiased (Phipson & Smyth, 2010, *Stat. Appl. Genet. Mol. Biol.* 9(1), Article 39). $\square$

**REMARK 7.5** (Why we test 15 paired differences rather than 5). Repeating 5-fold CV with three seeds reduces the variance of the estimated $\bar\Delta$ while preserving exchangeability across folds within the same seed. The within-seed correlation slightly increases the standard error, which we account for in the bootstrap by resampling at the level of CV-seed groups. This is the design recommended by Bouckaert & Frank (2004, *Lecture Notes in Computer Science* 3056, 3–12, "Evaluating the replicability of significance tests for comparing learning algorithms"). We do not use their corrected resampled $t$-test because the sign-flip permutation test is distribution-free and we prefer to make no normality assumption.

**FALSIFIABILITY**. The empirical claim of this experiment is: *across the three UCR datasets we test, after BH-FDR correction at $\alpha = 0.05$, the topology-augmented feature representation yields a balanced-accuracy improvement that is statistically significant on at least one dataset with effect size $d_z \ge 0.5$.* This claim is falsified if, after BH correction, no dataset shows $p < 0.05$, or if the effect size on the significant dataset(s) is below $0.5$. The numerical result with $95\%$ CI is reported in `experiments/ucr_topology_validation/results.json` and summarized in §8 of the report `experiments/ucr_topology_validation/report.md`.

## 8. References

- Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology* 5, 391–423.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B* 57, 289–300.
- Bouckaert, R., & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. *Lecture Notes in Computer Science* 3056, 3–12.
- Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *Journal of Machine Learning Research* 16, 77–102.
- Chazal, F., de Silva, V., Glisse, M., & Oudot, S. (2016). *The Structure and Stability of Persistence Modules*. SpringerBriefs in Mathematics. Springer.
- Chintakunta, H., Gentimis, T., Gonzalez-Diaz, R., Jimenez, M.-J., & Krim, H. (2015). An entropy-based persistence barcode. *Pattern Recognition* 48(2), 391–401.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry* 37(1), 103–120.
- Crawley-Boevey, W. (2015). Decomposition of pointwise finite-dimensional persistence modules. *Journal of Algebra and Its Applications* 14(05), 1550066.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39.
- Takens, F. (1981). Detecting strange attractors in turbulence. In: D. Rand & L.-S. Young (eds.), *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898, pp. 366–381. Springer.
