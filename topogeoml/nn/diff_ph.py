"""
Differentiable Persistent Homology via critical-edge distance indexing.

The key insight: ripser identifies WHICH edge distances are the birth/death
values (via Elder Lemma for H_0, cocycle representatives for H_1). We index
those values back into the PyTorch distance matrix — which carries full
autograd. No custom backward required; PyTorch handles gradient propagation
automatically through the indexing operations.

Mathematical basis
------------------
H_0 (connected components):
    Elder Lemma: H_0 deaths are exactly the minimum spanning tree edge weights.
    scipy.sparse.csgraph.minimum_spanning_tree gives us the exact critical edges.
    Gradient of death_i w.r.t. X flows through (X[u] - X[v]) / ||X[u] - X[v]||.

H_1 (loops):
    Birth: the youngest edge in the ripser cocycle representative whose distance
    equals the birth filtration value. This is the standard subgradient choice
    (Hofer et al. 2017, Clough et al. 2020).
    Death: the edge in the upper-star whose distance equals the death value.
    Both birth and death gradients are subgradients — correct for descent.

References
----------
Hofer et al., "Deep Learning with Topological Signatures", NeurIPS 2017.
Clough et al., "A Topological Loss Function for Deep-Learning Based Image
  Segmentation Using Persistent Homology", IEEE TPAMI 2022.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as scg
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "topogeoml.nn.diff_ph requires PyTorch. "
        "Install with `pip install torch`."
    ) from exc

from ripser import ripser


# ---------------------------------------------------------------------------
# Core: differentiable diagram construction
# ---------------------------------------------------------------------------

def pairwise_distances(X: torch.Tensor) -> torch.Tensor:
    """
    Compute (n, n) Euclidean distance matrix from (n, d) point cloud.

    Uses ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 <x_i, x_j> for efficiency,
    then clips negatives (elite-code-standards §1.2) before sqrt.

    Returns float64 tensor on same device as X.
    """
    X = X.to(torch.float64)
    dot = X @ X.t()                                  # (n, n)
    sq = (X * X).sum(dim=1, keepdim=True)            # (n, 1)
    D_sq = sq + sq.t() - 2.0 * dot                  # (n, n)
    D_sq = torch.clamp(D_sq, min=0.0)               # §1.2: no negative before sqrt
    return torch.sqrt(D_sq + 1e-300)                 # §1.4: division/sqrt safety


def _critical_edges_h0(
    D_np: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """
    Return (n-1) critical edge indices for H_0 bars via Elder Lemma.

    The H_0 persistence diagram of a Rips filtration is exactly the
    set of MST edge weights (each giving a bar (0, weight)). Sorted
    ascending, these match ripser's dgms[0] finite deaths.

    Returns list of (i, j) tuples in increasing MST weight order.
    """
    n = D_np.shape[0]
    mst = scg.minimum_spanning_tree(sp.csr_matrix(D_np)).tocoo()
    # Sort by weight ascending to match ripser's bar order
    order = np.argsort(mst.data)
    rows = mst.row[order].tolist()
    cols = mst.col[order].tolist()
    return list(zip(rows, cols))


def _critical_edges_h1(
    D_np: NDArray[np.float64],
    cocycles: list[NDArray[np.integer[Any]]],
    dgm_h1: NDArray[np.float64],
) -> tuple[list[tuple[int, int] | None], list[tuple[int, int] | None]]:
    """
    Return (birth_edges, death_edges) for H_1 bars.

    Birth edge: the edge in the cocycle representative whose distance
    is closest to the birth filtration value.

    Death edge: the edge in the full upper-star distance matrix whose
    distance is closest to the death filtration value (and is not the
    birth edge). This is a subgradient approximation.

    Returns lists of (i, j) tuples or None for infinite deaths.
    """
    n = D_np.shape[0]
    birth_edges: list[tuple[int, int] | None] = []
    death_edges: list[tuple[int, int] | None] = []

    for bar_idx, (birth_val, death_val) in enumerate(dgm_h1):
        # --- Birth ---
        if bar_idx < len(cocycles):
            cocycle = cocycles[bar_idx]  # (n_edges, 3): [i, j, coeff]
            cocycle_dists = np.array(
                [D_np[int(e[0]), int(e[1])] for e in cocycle], dtype=np.float64
            )
            best = int(np.argmin(np.abs(cocycle_dists - birth_val)))
            birth_edges.append((int(cocycle[best, 0]), int(cocycle[best, 1])))
        else:
            birth_edges.append(None)

        # --- Death ---
        if not np.isfinite(death_val):
            death_edges.append(None)
            continue

        # Search upper triangle for edge distance closest to death_val.
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists_upper = D_np[i_idx, j_idx]
        candidates = np.abs(dists_upper - death_val)
        # Exclude the birth edge to avoid routing both gradients to the same edge.
        b_edge = birth_edges[-1]
        if b_edge is not None:
            b_flat = b_edge[0] * n + b_edge[1]
            flat_idx = i_idx * n + j_idx
            mask = flat_idx != b_flat
            if mask.any():
                candidates_masked = np.where(mask, candidates, np.inf)
                best_d = int(np.argmin(candidates_masked))
            else:
                best_d = int(np.argmin(candidates))
        else:
            best_d = int(np.argmin(candidates))

        death_edges.append((int(i_idx[best_d]), int(j_idx[best_d])))

    return birth_edges, death_edges


def rips_diagram_torch(
    X: torch.Tensor,
    max_dim: int = 1,
    max_edge_length: float | None = None,
) -> list[torch.Tensor]:
    """
    Compute Rips persistence diagram as a list of torch tensors WITH gradients.

    Each tensor has shape (n_bars, 2) where columns are (birth, death).
    Infinite deaths are represented as torch.inf.

    The gradient of any scalar function of these bars flows back to X via
    the PyTorch distance matrix (no custom backward required).

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_points, ambient_dim). Should have requires_grad=True
        for gradient flow.
    max_dim : int
        Highest homology dimension (default 1: H_0 and H_1).
    max_edge_length : float, optional
        Maximum edge length passed to ripser.

    Returns
    -------
    list[torch.Tensor]
        diagrams[k] has shape (n_k, 2): (birth, death) pairs for H_k.

    Notes
    -----
    The function is NOT a torch.autograd.Function. Instead, birth/death
    values are constructed by indexing into the differentiable distance
    matrix D_torch. This means PyTorch's standard autograd handles all
    gradient computation automatically.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_points, dim); got shape {tuple(X.shape)}")
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 points; got {n}")

    # Step 1: Differentiable distance matrix (carries gradients back to X).
    D_torch = pairwise_distances(X)  # (n, n), float64, autograd-connected

    # Step 2: Ripser on detached numpy copy.
    D_np = D_torch.detach().cpu().numpy().astype(np.float64, copy=False)
    ripser_kwargs: dict[str, Any] = {
        "maxdim": int(max_dim),
        "distance_matrix": True,
        "do_cocycles": max_dim >= 1,
    }
    if max_edge_length is not None:
        ripser_kwargs["thresh"] = float(max_edge_length)

    result = ripser(D_np, **ripser_kwargs)
    dgms = result["dgms"]
    cocycles = result.get("cocycles", [])

    diagrams: list[torch.Tensor] = []
    device = X.device
    dtype = torch.float64  # §1.3: explicit dtype

    # --- H_0 diagram ---
    critical_edges_h0 = _critical_edges_h0(D_np)
    n_h0_finite = len(critical_edges_h0)  # n - 1 finite bars + 1 infinite

    births_h0 = torch.zeros(n_h0_finite + 1, dtype=dtype, device=device)

    if n_h0_finite > 0:
        deaths_finite = torch.stack([D_torch[i, j] for i, j in critical_edges_h0])
        deaths_h0 = torch.cat([
            deaths_finite,
            torch.tensor([torch.inf], dtype=dtype, device=device),
        ])
    else:
        deaths_h0 = torch.tensor([torch.inf], dtype=dtype, device=device)

    diagrams.append(torch.stack([births_h0, deaths_h0], dim=1))

    # --- H_1 diagram ---
    if max_dim >= 1:
        dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2), dtype=np.float64)
        cocycles_h1 = cocycles[1] if len(cocycles) > 1 else []
        n_h1 = len(dgm_h1)

        if n_h1 == 0:
            diagrams.append(torch.empty((0, 2), dtype=dtype, device=device))
        else:
            birth_edges, death_edges = _critical_edges_h1(D_np, cocycles_h1, dgm_h1)

            birth_tensors: list[torch.Tensor] = []
            death_tensors: list[torch.Tensor] = []

            for bar_idx in range(n_h1):
                b_edge = birth_edges[bar_idx]
                d_edge = death_edges[bar_idx]

                if b_edge is not None:
                    birth_tensors.append(D_torch[b_edge[0], b_edge[1]])
                else:
                    birth_tensors.append(
                        torch.tensor(dgm_h1[bar_idx, 0], dtype=dtype, device=device)
                    )

                if d_edge is not None:
                    death_tensors.append(D_torch[d_edge[0], d_edge[1]])
                else:
                    death_tensors.append(
                        torch.tensor(torch.inf, dtype=dtype, device=device)
                    )

            diagram_h1 = torch.stack([
                torch.stack(birth_tensors),
                torch.stack(death_tensors),
            ], dim=1)
            diagrams.append(diagram_h1)

    return diagrams


# ---------------------------------------------------------------------------
# Topology loss functions
# ---------------------------------------------------------------------------

def finite_lifetimes(diagram: torch.Tensor) -> torch.Tensor:
    """
    Return lifetimes (death - birth) for finite bars.
    Drops infinite deaths. Returns empty tensor if none finite.
    """
    mask = torch.isfinite(diagram[:, 1])
    if not mask.any():
        return torch.empty(0, dtype=diagram.dtype, device=diagram.device)
    finite = diagram[mask]
    return finite[:, 1] - finite[:, 0]


def total_persistence_loss(
    diagram: torch.Tensor,
    p: float = 2.0,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    L_p total persistence loss: Σ_{lifetime > threshold} lifetime^p.

    Minimizing this shrinks all bars (pushes toward trivial topology).
    Maximizing it grows bars (encourages topological richness).
    Typically used as a REGULARIZER added to the training loss.
    """
    lifetimes = finite_lifetimes(diagram)
    if lifetimes.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    significant = lifetimes[lifetimes > threshold]
    if significant.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    return significant.pow(p).sum()


def persistence_entropy_loss(diagram: torch.Tensor) -> torch.Tensor:
    """
    Persistence entropy: -Σ p_i log(p_i) where p_i = lifetime_i / Σ lifetimes.

    Maximizing entropy encourages uniform lifetime distribution (rich topology).
    Minimizing entropy pushes mass toward a few dominant bars.
    """
    lifetimes = finite_lifetimes(diagram)
    if lifetimes.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    total = lifetimes.sum() + 1e-300  # §1.4: division safety
    p = lifetimes / total
    p = torch.clamp(p, min=1e-300)  # prevent log(0)
    return -(p * p.log()).sum()


def betti_regularization_loss(
    diagram: torch.Tensor,
    target_n_components: int,
    prominence_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Penalize the excess Betti number above target_n_components.

    For H_0: penalizes fragmentation — pushes the representation toward
    target_n_components connected clusters by shrinking excess H_0 bars.

    The loss is Σ_{i > target} lifetime_i for bars sorted by lifetime descending.
    This is differentiable: it pushes excess bars toward zero lifetime.

    Parameters
    ----------
    diagram : torch.Tensor
        (n_bars, 2) persistence diagram for H_0 or H_1.
    target_n_components : int
        Desired number of topological features (β_k target).
    prominence_threshold : float
        Minimum lifetime to count as a "real" component (filters noise).
    """
    lifetimes = finite_lifetimes(diagram)
    if lifetimes.numel() == 0:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    # Sort descending: most prominent first
    sorted_lifetimes, _ = lifetimes.sort(descending=True)
    # Include the "infinite" component implicitly (it has ∞ lifetime)
    # Real components = 1 (infinite bar) + significant finite bars
    significant = sorted_lifetimes[sorted_lifetimes > prominence_threshold]
    n_real = 1 + significant.numel()  # +1 for the persistent infinite H_0 component
    if n_real <= target_n_components:
        return torch.tensor(0.0, dtype=diagram.dtype, device=diagram.device)
    # Penalize excess bars: push them toward zero lifetime
    excess = sorted_lifetimes[:max(0, n_real - target_n_components)]
    return excess.sum()


# ---------------------------------------------------------------------------
# Module interface
# ---------------------------------------------------------------------------

class TopologyRegularizer(nn.Module):
    """
    Differentiable topology regularizer for use in training loops.

    Computes Rips persistence of a batch of embeddings and returns a
    scalar loss term. Add to your main loss:

        loss = task_loss + lambda_topo * topo_regularizer(embeddings)

    Parameters
    ----------
    max_dim : int
        Highest homology dimension to compute (default 1).
    loss_type : str
        'total_persistence', 'entropy', or 'betti_regularization'.
    p : float
        Power for total_persistence loss.
    target_betti : dict[int, int], optional
        Target Betti numbers per dimension for betti_regularization loss.
        Example: {0: 3, 1: 0} for 3 clusters with no loops.
    max_points : int
        Subsample to this many points before computing PH (for speed).
    seed : int
        RNG seed for subsampling reproducibility.
    """

    def __init__(
        self,
        max_dim: int = 1,
        loss_type: str = "total_persistence",
        p: float = 2.0,
        target_betti: dict[int, int] | None = None,
        max_points: int = 500,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if loss_type not in ("total_persistence", "entropy", "betti_regularization"):
            raise ValueError(
                f"loss_type must be 'total_persistence', 'entropy', or "
                f"'betti_regularization'; got {loss_type!r}"
            )
        self.max_dim = max_dim
        self.loss_type = loss_type
        self.p = p
        self.target_betti = target_betti or {}
        self.max_points = max_points
        self._rng = np.random.default_rng(seed)  # §6: seeded RNG

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape (n_points, embedding_dim). Should have requires_grad=True
            if you need gradients w.r.t. the embeddings.

        Returns
        -------
        torch.Tensor
            Scalar topology loss.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_points, dim); got {tuple(X.shape)}")

        # Subsample for speed (§3: performance)
        n = X.shape[0]
        if n > self.max_points:
            idx = self._rng.choice(n, size=self.max_points, replace=False)
            idx_t = torch.from_numpy(idx.astype(np.intp)).to(X.device)
            X_sub = X[idx_t]
        else:
            X_sub = X

        diagrams = rips_diagram_torch(X_sub, max_dim=self.max_dim)

        total_loss = torch.tensor(0.0, dtype=torch.float64, device=X.device)

        for k, dgm in enumerate(diagrams):
            if dgm.numel() == 0:
                continue
            if self.loss_type == "total_persistence":
                total_loss = total_loss + total_persistence_loss(dgm, p=self.p)
            elif self.loss_type == "entropy":
                # Negate entropy to get a minimizable loss:
                # maximizing entropy = minimizing negative entropy
                total_loss = total_loss - persistence_entropy_loss(dgm)
            elif self.loss_type == "betti_regularization":
                target = self.target_betti.get(k, 0)
                total_loss = total_loss + betti_regularization_loss(dgm, target)

        return total_loss.to(X.dtype)
