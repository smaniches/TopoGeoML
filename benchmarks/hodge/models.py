"""
Graph-classifier wrappers for the Hodge bench.

Each model implements the :class:`GraphClassifier` protocol. The runner
constructs a fresh instance per (seed, dataset) cell, trains it for a
fixed budget, then measures test accuracy.

Architectural ablation (hypothesis 001, see
``docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md``)
--------------------------------------------------
Four classifiers share a sum-pool + linear-head tail. They differ in the
middle propagation step:

  - ``hodge-mp-classifier``           combinatorial L_0, 1 layer, no residual
  - ``hodge-mp-normalised``           symmetric L̃ = D^-1/2 L D^-1/2, 1 layer, no residual
  - ``hodge-mp-residual``             symmetric L̃, 1 layer, **+ residual**
  - ``hodge-mp-deep-residual``        symmetric L̃, **2 layers**, residual on each
  - ``mlp-baseline``                  no Laplacian; matched-capacity control

The ablation tests three predictions: (H1) normalisation alone helps,
(H2) residual helps on top of normalisation, (H3) depth helps on top
of both. All four arms use ``hidden_dim=32``, the same Adam(lr=1e-2)
optimiser, and the same 20-epoch budget so the comparison isolates
the architectural change.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import nn


def _symmetric_normalize_sparse(
    laplacian: torch.sparse.Tensor, epsilon: float = 1e-6,
) -> torch.sparse.Tensor:
    """Symmetric normalisation L̃ = D^{-1/2} L D^{-1/2}.

    For a graph combinatorial Laplacian ``L = D - A``, the diagonal of
    ``L`` is exactly the degree, so we read ``D`` off the diagonal and
    form ``D^{-1/2}`` with an ``epsilon`` floor to avoid division by zero
    on isolated nodes (degree 0).

    Returns the result as a coalesced sparse COO tensor with the same
    dtype as the input. The operation is differentiable but the
    Laplacian is treated as a buffer (no gradient flows back through
    the degree computation), matching the convention in
    ``topogeoml.nn.hodge.normalize_hodge_laplacian``.
    """
    L = laplacian.coalesce()
    indices = L.indices()
    values = L.values()
    n = L.shape[0]
    # Read the diagonal: for entries (i, i), values[i] is the degree.
    diag = torch.zeros(n, dtype=values.dtype, device=values.device)
    diag_mask = indices[0] == indices[1]
    diag.index_add_(0, indices[0][diag_mask], values[diag_mask])
    d_inv_sqrt = 1.0 / torch.sqrt(diag + epsilon)
    # Scale each off-diagonal value by D^{-1/2}_i * D^{-1/2}_j.
    scaled_values = values * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices, scaled_values, L.shape,
    ).coalesce()


@runtime_checkable
class GraphClassifier(Protocol):
    """A graph classifier the Hodge bench can train and evaluate."""

    name: ClassVar[str]
    version: ClassVar[str]

    @staticmethod
    def available() -> bool: ...

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module: ...


class _HodgeGraphClassifier(nn.Module):
    """HodgeMP layer + sum-pool + linear head.

    For each graph we precompute the L_0 Hodge Laplacian (clique
    complex of the graph), apply one Hodge message-passing step to
    the node features, sum-pool across nodes, and run a linear
    classifier.

    The Hodge propagation step replicates ``HodgeMessagePassing``
    inline — ``activation(L @ x @ W + b)`` — but with **shared**
    learnable ``W`` and ``b`` across graphs. Previously the layer was
    constructed inside ``forward_one`` per graph, which (a) reset its
    Xavier-initialised weights on every call and (b) left those weights
    out of ``model.parameters()`` so the optimizer never saw them.
    Caught by Gemini's PR #6 review; this rewrite fixes it.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head = nn.Linear(hidden_dim, num_classes)
        self._activation = nn.ReLU()
        self._proj_in = nn.Linear(input_dim, hidden_dim)
        # Shared Hodge-MP parameters. Xavier init for ``W`` matches the
        # default in ``topogeoml.nn.hodge.HodgeMessagePassing``; ``b``
        # starts at zero per the original layer's convention.
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64)
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        # Project the head to float64 so the dtype chain stays consistent
        # with the inline Hodge propagation.
        self.head = self.head.to(torch.float64)
        self._proj_in = self._proj_in.to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        """Forward for a single graph: x is (n_nodes, input_dim), output is
        (num_classes,) logits.

        Inline Hodge propagation step (mirroring
        ``HodgeMessagePassing.forward``):
            h ← activation( L_norm @ proj_in(x) @ W_shared + b_shared )
        with ``W_shared`` and ``b_shared`` defined in ``__init__`` so
        the optimizer actually updates them.
        """
        h = self._proj_in(x)
        propagated = torch.sparse.mm(laplacian, h)
        h = self._activation(propagated @ self._mp_weight + self._mp_bias)
        graph_emb = h.sum(dim=0)
        return self.head(graph_emb)


class HodgeClassifier:
    """HodgeMP-based graph classifier (combinatorial Laplacian, baseline).

    Empirical status (MUTAG, 30 seeds, 20 epochs of Adam(lr=1e-2)):
    this combinatorial-L baseline arm **underperforms** the MLP
    baseline by ~9 percentage points with paired Wilcoxon p_BH =
    5.66e-04 and rank-biserial r = -0.643 (vs the normalised arm).

    The architectural fix is *symmetric Laplacian normalisation*, not
    depth or residual — see ``HodgeNormalisedClassifier`` for the
    matching-MLP arm and ``docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md``
    for the full ablation. This combinatorial variant is kept in the
    registry as the *control* of the ablation, not as a working
    classifier.
    """

    name: ClassVar[str] = "hodge-mp-classifier"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import topogeoml.nn.hodge  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _HodgeGraphClassifier(input_dim, num_classes)


class _HodgeNormalisedGraphClassifier(nn.Module):
    """Hypothesis 001, arm H1: symmetric-normalised Laplacian + 1 Hodge step.

    The combinatorial Laplacian ``L = D - A`` makes propagation magnitude
    scale with node degree; high-degree nodes dominate the forward pass
    and the MUTAG mutagenicity signal (small functional groups like
    -NO_2 attached to aromatic rings) gets buried. Kipf & Welling 2017
    Lemma 1: the symmetrically-normalised ``L̃ = D^{-1/2} L D^{-1/2}``
    has eigenvalues bounded to [0, 2], balancing propagation across
    degrees. This arm changes only that one variable.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64)
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        l_norm = _symmetric_normalize_sparse(laplacian)
        h = self._proj_in(x)
        propagated = torch.sparse.mm(l_norm, h)
        h = self._activation(propagated @ self._mp_weight + self._mp_bias)
        return self.head(h.sum(dim=0))


class HodgeNormalisedClassifier:
    """Hodge classifier with symmetric-normalised Laplacian.

    Architectural ablation arm H1 of hypothesis 001. See
    ``docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md``.
    """

    name: ClassVar[str] = "hodge-mp-normalised"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import topogeoml.nn.hodge  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _HodgeNormalisedGraphClassifier(input_dim, num_classes)


class _HodgeResidualGraphClassifier(nn.Module):
    """Hypothesis 001, arm H2: H1 + residual connection.

    A random-init ``W`` early in training would otherwise destroy the
    per-node features the projection layer just learned. The residual
    ``out = activation(L̃ @ proj(x) @ W + b) + proj(x)`` preserves the
    projection through the Hodge step, letting the model learn what to
    *add* to the per-node features rather than what to *replace* them
    with. Standard ResNet-style identity skip (He et al. 2016 §3.1).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64)
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        l_norm = _symmetric_normalize_sparse(laplacian)
        proj = self._proj_in(x)
        propagated = torch.sparse.mm(l_norm, proj)
        h = self._activation(propagated @ self._mp_weight + self._mp_bias) + proj
        return self.head(h.sum(dim=0))


class HodgeResidualClassifier:
    """Hodge classifier with symmetric-normalised Laplacian + residual.

    Architectural ablation arm H2 of hypothesis 001.
    """

    name: ClassVar[str] = "hodge-mp-residual"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import topogeoml.nn.hodge  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _HodgeResidualGraphClassifier(input_dim, num_classes)


class _HodgeDeepResidualGraphClassifier(nn.Module):
    """Hypothesis 001, arm H3: H2 with 2 stacked Hodge propagation steps.

    Aromatic rings in MUTAG span 5-6 hops in molecular graph distance.
    A single Hodge propagation step is a 1-hop neighbourhood operator;
    two stacked steps reach 2-hop, which is the minimum to detect a
    benzene ring's *adjacency* structure (one edge of the ring) — still
    short of the full 5-6 hops, but a step in the right direction. Each
    step has its own learnable weights to avoid the degenerate ``W^2``
    composition.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._mp_weight_1 = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64)
        )
        self._mp_bias_1 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        self._mp_weight_2 = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64)
        )
        self._mp_bias_2 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight_1)
        nn.init.xavier_uniform_(self._mp_weight_2)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        l_norm = _symmetric_normalize_sparse(laplacian)
        h0 = self._proj_in(x)
        # Layer 1 with residual.
        h1 = self._activation(
            torch.sparse.mm(l_norm, h0) @ self._mp_weight_1 + self._mp_bias_1
        ) + h0
        # Layer 2 with residual.
        h2 = self._activation(
            torch.sparse.mm(l_norm, h1) @ self._mp_weight_2 + self._mp_bias_2
        ) + h1
        return self.head(h2.sum(dim=0))


class HodgeDeepResidualClassifier:
    """Hodge classifier with symmetric L̃, 2 stacked layers, residuals.

    Architectural ablation arm H3 of hypothesis 001.
    """

    name: ClassVar[str] = "hodge-mp-deep-residual"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import topogeoml.nn.hodge  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        # ``hidden_dim=24`` makes the 2-layer arm's total parameter count
        # comparable to the 1-layer arms (1442 vs 1378 — within 5%), so
        # the comparison isolates the depth+residual effect rather than
        # capacity. The MLP baseline also runs at ``hidden_dim=32``;
        # see ``docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md`` for the
        # capacity-matching argument.
        return _HodgeDeepResidualGraphClassifier(
            input_dim, num_classes, hidden_dim=24,
        )


class _MLPGraphClassifier(nn.Module):
    """No-topology baseline: feature-MLP applied per-node, then sum-pool.

    This is the natural control for ``HodgeClassifier`` — both apply a
    per-node transformation and then sum-pool to a graph embedding. The
    Hodge model differs only in that its per-node transformation
    incorporates the Laplacian (= the graph structure).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        # Baseline ignores the laplacian — that's the point.
        del laplacian
        h = self.net(x)
        return self.head(h.sum(dim=0))


class MLPBaseline:
    """No-topology baseline (feature MLP + sum-pool)."""

    name: ClassVar[str] = "mlp-baseline"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _MLPGraphClassifier(input_dim, num_classes)


def _adj_matmul_from_laplacian(
    laplacian: torch.sparse.Tensor, h: torch.Tensor,
) -> torch.Tensor:
    """Compute A @ H from the Laplacian L = D - A, without forming A explicitly.

    A @ H = D @ H - L @ H, where D is the degree diagonal read from L.
    """
    L = laplacian.coalesce()
    LH = torch.sparse.mm(L, h)
    indices = L.indices()
    values = L.values()
    n = L.shape[0]
    diag = torch.zeros(n, dtype=values.dtype, device=values.device)
    diag_mask = indices[0] == indices[1]
    diag.index_add_(0, indices[0][diag_mask], values[diag_mask])
    DH = diag.unsqueeze(1) * h
    return DH - LH


class _GINGraphClassifier(nn.Module):
    """GIN (Graph Isomorphism Network, Xu et al. 2019) baseline.

    Update: h' = MLP((1 + eps) * h + A @ h)
    where A @ h is computed from the Laplacian via A = D - L.
    Matched-capacity design: proj_in + gin_nn + head ≈ 2339 params on NCI1.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._eps = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self._gin_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim).to(torch.float64),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
    ) -> torch.Tensor:
        h = self._proj_in(x)
        agg = _adj_matmul_from_laplacian(laplacian, h)
        h = self._gin_nn((1.0 + self._eps) * h + agg)
        return self.head(h.sum(dim=0))


class GINBaseline:
    """GIN baseline (Xu et al. 2019) — the WL-1 upper bound."""

    name: ClassVar[str] = "gin-baseline"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _GINGraphClassifier(input_dim, num_classes)


class _GATGraphClassifier(nn.Module):
    """GAT (Graph Attention Network, Velickovic et al. 2018) baseline.

    Single-head attention with LeakyReLU gating. Attention is computed
    over the Laplacian's off-diagonal sparsity pattern (= edge set).
    Matched-capacity design: proj_in + W + attn + head ≈ 2340 params on NCI1.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._W = nn.Linear(hidden_dim, hidden_dim, bias=False).to(torch.float64)
        self._attn_src = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float64))
        self._attn_dst = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._W.weight)
        nn.init.xavier_uniform_(self._attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self._attn_dst.unsqueeze(0))
        self._leaky_relu = nn.LeakyReLU(0.2)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
    ) -> torch.Tensor:
        h = self._proj_in(x)
        Wh = self._W(h)

        L = laplacian.coalesce()
        indices = L.indices()
        off_diag = indices[0] != indices[1]
        src, dst = indices[0][off_diag], indices[1][off_diag]

        n = h.shape[0]
        agg = torch.zeros_like(Wh)

        if src.numel() > 0:
            e_src = (Wh[src] * self._attn_src).sum(dim=-1)
            e_dst = (Wh[dst] * self._attn_dst).sum(dim=-1)
            e = self._leaky_relu(e_src + e_dst)
            exp_e = torch.exp(e - e.max())
            exp_sum = torch.zeros(n, dtype=h.dtype, device=h.device)
            exp_sum.index_add_(0, dst, exp_e)
            exp_sum = exp_sum.clamp(min=1e-8)
            attn_weights = exp_e / exp_sum[dst]
            agg.index_add_(0, dst, attn_weights.unsqueeze(1) * Wh[src])

        h = self._activation(agg)
        return self.head(h.sum(dim=0))


class GATBaseline:
    """GAT baseline (Velickovic et al. 2018) — attention-based aggregation."""

    name: ClassVar[str] = "gat-baseline"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _GATGraphClassifier(input_dim, num_classes)


class _GINNormalisedGraphClassifier(nn.Module):
    """GIN with symmetric degree normalisation (GCN-style aggregation).

    Update: h' = MLP((1 + eps) * h + D^{-1/2} A D^{-1/2} @ h)
    where D^{-1/2} A D^{-1/2} @ h = (I - L_tilde) @ h = h - L_tilde @ h,
    with L_tilde the symmetrically normalised Laplacian.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._eps = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self._gin_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim).to(torch.float64),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
    ) -> torch.Tensor:
        h = self._proj_in(x)
        l_norm = _symmetric_normalize_sparse(laplacian)
        norm_adj_h = h - torch.sparse.mm(l_norm, h)
        h = self._gin_nn((1.0 + self._eps) * h + norm_adj_h)
        return self.head(h.sum(dim=0))


class GINNormalisedBaseline:
    """GIN with symmetric degree normalisation — tests whether normalisation
    alone closes the gap observed in H008."""

    name: ClassVar[str] = "gin-normalised"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _GINNormalisedGraphClassifier(input_dim, num_classes)


class _GINResidualGraphClassifier(nn.Module):
    """Normalised adjacency aggregation with external residual.

    Matches the Hodge-MP-residual architecture exactly except for the
    operator: uses the normalised adjacency (I - L_tilde, low-pass) instead
    of the normalised Laplacian (L_tilde, high-pass).

    Forward: h' = act(A_norm @ proj(x) @ W + b) + proj(x)
    where A_norm = I - L_tilde = D^{-1/2} A D^{-1/2}.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64),
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
    ) -> torch.Tensor:
        l_norm = _symmetric_normalize_sparse(laplacian)
        proj = self._proj_in(x)
        norm_adj_proj = proj - torch.sparse.mm(l_norm, proj)
        h = self._activation(norm_adj_proj @ self._mp_weight + self._mp_bias) + proj
        return self.head(h.sum(dim=0))


class GINResidualBaseline:
    """Normalised adjacency + external residual — isolates operator choice
    from residual placement in the Hodge-GIN comparison."""

    name: ClassVar[str] = "gin-residual"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _GINResidualGraphClassifier(input_dim, num_classes)


class _SheafResidualGraphClassifier(nn.Module):
    """Learned sheaf Laplacian with external residual (scalar stalks).

    For each edge e={i,j}, a linear layer predicts restriction scalars
    f_{i<-e}, f_{j<-e} from concatenated projected features [h_i || h_j].
    The sheaf Laplacian L_F is PSD by construction (L_F = delta^T delta):
      L_F[i,j] = -f_{i<-e} * f_{j<-e}  (off-diagonal)
      L_F[i,i] = sum_{e containing i} f_{i<-e}^2  (diagonal)

    Propagation: h' = act(L_F_tilde @ proj(x) @ W + b) + proj(x)
    where L_F_tilde is the symmetrically normalised sheaf Laplacian.
    Generalises the Hodge arm (special case: all f = 1).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._sheaf_learner = nn.Linear(2 * hidden_dim, 2, bias=True).to(torch.float64)
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64),
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
    ) -> torch.Tensor:
        proj = self._proj_in(x)
        n = proj.shape[0]

        L = laplacian.coalesce()
        indices = L.indices()
        off_diag = indices[0] != indices[1]
        src, dst = indices[0][off_diag], indices[1][off_diag]

        if src.numel() == 0:
            h = self._activation(proj @ self._mp_weight + self._mp_bias) + proj
            return self.head(h.sum(dim=0))

        edge_features = torch.cat([proj[src], proj[dst]], dim=-1)
        restrictions = self._sheaf_learner(edge_features)
        f_src = restrictions[:, 0]
        f_dst = restrictions[:, 1]

        L_F = torch.zeros(n, n, dtype=proj.dtype, device=proj.device)
        L_F[src, dst] = -f_src * f_dst
        L_F_diag = torch.zeros(n, dtype=proj.dtype, device=proj.device)
        L_F_diag.index_add_(0, src, f_src ** 2)
        L_F_diag.index_add_(0, dst, f_dst ** 2)
        L_F = L_F + torch.diag(L_F_diag)

        d_inv_sqrt = 1.0 / torch.sqrt(L_F_diag.clamp(min=1e-6))
        L_F_norm = L_F * d_inv_sqrt.unsqueeze(1) * d_inv_sqrt.unsqueeze(0)

        propagated = L_F_norm @ proj
        h = self._activation(propagated @ self._mp_weight + self._mp_bias) + proj
        return self.head(h.sum(dim=0))


class SheafResidualBaseline:
    """Learned sheaf Laplacian + external residual (Bodnar et al. 2022)."""

    name: ClassVar[str] = "sheaf-residual"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _SheafResidualGraphClassifier(input_dim, num_classes)


class _L1HodgeResidualGraphClassifier(nn.Module):
    """Edge-level message passing on the 1-Hodge Laplacian L_1.

    L_1 operates on edge features and encodes shared-triangle adjacency —
    two edges are L_1-adjacent if they are co-faces of a common triangle.
    This captures cycle and ring topology that L_0 (node-level) cannot access.

    Forward:
      1. proj = proj_in(x)  (n_nodes, d)
      2. e_{ij} = proj[i] + proj[j]  for each edge (i,j)  (n_edges, d)
      3. e' = act(L_1_tilde @ e @ W + b) + e  (external residual on edges)
      4. graph_emb = sum(e')
      5. logits = head(graph_emb)
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._proj_in = nn.Linear(input_dim, hidden_dim).to(torch.float64)
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64),
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        self._activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, num_classes).to(torch.float64)

    def _compute_l1(
        self, n_nodes: int, laplacian: torch.sparse.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Compute L_1 from L_0 by reconstructing the graph and building
        the clique complex with max_dim=2."""
        import networkx as nx

        from topogeoml.core.complexes import hodge_laplacian
        from topogeoml.data.graph_to_complex import graph_to_clique_complex
        from topogeoml.nn.hodge import sparse_scipy_to_torch

        L = laplacian.coalesce()
        indices = L.indices()
        off_diag = indices[0] != indices[1]
        src = indices[0][off_diag].tolist()
        dst = indices[1][off_diag].tolist()

        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        seen: set[tuple[int, int]] = set()
        for s, d in zip(src, dst, strict=True):
            edge = (min(s, d), max(s, d))
            if edge not in seen:
                g.add_edge(*edge)
                seen.add(edge)

        sc = graph_to_clique_complex(g, max_dim=2, include_isolated_vertices=True)
        n_edges = sc.n_simplices(1)
        if n_edges == 0:
            return torch.zeros(0, 0, dtype=torch.float64), []

        L1 = hodge_laplacian(sc, k=1)
        L1_torch = sparse_scipy_to_torch(L1, dtype=torch.float64)

        edge_list = sorted(sc.simplices.get(1, []))
        edge_pairs = [(int(e[0]), int(e[1])) for e in edge_list]
        return L1_torch, edge_pairs

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
    ) -> torch.Tensor:
        proj = self._proj_in(x)
        n_nodes = x.shape[0]

        L1, edge_pairs = self._compute_l1(n_nodes, laplacian)

        if len(edge_pairs) == 0:
            return self.head(proj.sum(dim=0))

        edge_src = [e[0] for e in edge_pairs]
        edge_dst = [e[1] for e in edge_pairs]
        edge_features = proj[edge_src] + proj[edge_dst]

        L1_norm = _symmetric_normalize_sparse(L1)
        propagated = torch.sparse.mm(L1_norm, edge_features)
        edge_out = self._activation(
            propagated @ self._mp_weight + self._mp_bias
        ) + edge_features

        return self.head(edge_out.sum(dim=0))


class L1HodgeResidualClassifier:
    """L_1 Hodge message passing on edges — tests higher-order topology."""

    name: ClassVar[str] = "l1-hodge-residual"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import topogeoml.nn.hodge  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _L1HodgeResidualGraphClassifier(input_dim, num_classes)


REGISTERED: dict[str, type[GraphClassifier]] = {
    HodgeClassifier.name: HodgeClassifier,
    HodgeNormalisedClassifier.name: HodgeNormalisedClassifier,
    HodgeResidualClassifier.name: HodgeResidualClassifier,
    HodgeDeepResidualClassifier.name: HodgeDeepResidualClassifier,
    MLPBaseline.name: MLPBaseline,
    GINBaseline.name: GINBaseline,
    GINNormalisedBaseline.name: GINNormalisedBaseline,
    GINResidualBaseline.name: GINResidualBaseline,
    GATBaseline.name: GATBaseline,
    SheafResidualBaseline.name: SheafResidualBaseline,
    L1HodgeResidualClassifier.name: L1HodgeResidualClassifier,
}


def get_model(name: str) -> type[GraphClassifier]:
    if name not in REGISTERED:
        known = ", ".join(sorted(REGISTERED))
        raise KeyError(f"unknown model {name!r}; registered: {known}")
    return REGISTERED[name]
