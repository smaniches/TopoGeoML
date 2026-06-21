"""
Minimal Hodge message passing layer for PyTorch.

Implements the simplest Hodge-Laplacian-based propagation on k-simplices:

        x' = σ( L̃_k @ x @ W + b )

where L̃_k is a (sym-)normalized Hodge Laplacian of dimension k built from a
SimplicialComplex, x is an (n_k, in_features) tensor of features on the
k-simplices, and W is a learnable (in_features, out_features) weight matrix.

For k = 0 with a graph-like complex (vertices + edges only), this reduces to
Kipf-Welling GCN propagation on the graph. For k = 1 (features on edges) it
propagates through the up-Laplacian (triangles) and down-Laplacian (shared
vertices), the basic building block of simplicial neural networks (Ebli et al.
2020; Bunch et al. 2020).

Item 8 of the v0.0.1 scope. This is a minimal layer — full SCN architecture,
batching across complexes, and equivariance machinery land in v0.1.

Notes
-----
This module requires PyTorch. Install via:

    pip install "topogeoml[torch]"

Without torch, importing this module raises ImportError on the torch line.
The rest of topogeoml does NOT require torch.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sp

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "topogeoml.nn.hodge requires PyTorch. Install with "
        "`pip install \"topogeoml[torch]\"` or `pip install torch`."
    ) from exc

from topogeoml.core.complexes import SimplicialComplex, hodge_laplacian


def normalize_hodge_laplacian(
    L: sp.spmatrix,
    epsilon: float = 1e-6,
) -> sp.csr_matrix:
    """
    Symmetric normalization L̃ = D^{-1/2} L D^{-1/2} where D = diag(deg).

    For combinatorial Laplacians "degree" is the diagonal of L. The standard
    symmetric normalization sets D^{-1/2}_ii = 0 when the degree is 0, leaving
    isolated simplices (whose Laplacian row is already all-zero) unchanged. We
    apply that convention directly rather than perturbing the diagonal.

    Parameters
    ----------
    L : sp.spmatrix
        Combinatorial Hodge Laplacian (symmetric, PSD).
    epsilon : float
        Retained for API back-compatibility. Zero-degree simplices are handled
        exactly via the D^{-1/2}_ii = 0 convention, so this argument no longer
        participates in the normalization.

    Returns
    -------
    sp.csr_matrix
        Symmetrically normalized Laplacian.
    """
    L = sp.csr_matrix(L)
    diag = np.asarray(L.diagonal(), dtype=np.float64)
    # D^{-1/2}_ii = 1/sqrt(diag_ii) for positive degree, 0 for isolated simplices
    # (degree 0). Computing the reciprocal only on the nonzero entries avoids a
    # spurious divide-by-zero warning on the zero-degree rows.
    d_inv_sqrt = np.zeros_like(diag)
    nonzero = diag > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(diag[nonzero])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ L @ D_inv_sqrt).tocsr()


def sparse_scipy_to_torch(
    matrix: sp.spmatrix,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse COO tensor."""
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    # Standard float32/float64 data passes straight through (no float32
    # intermediate — that would truncate precision for float64 callers,
    # elite-code §1.3); any other input dtype that torch.from_numpy cannot wrap
    # (e.g. np.longdouble, np.uint16) is normalised to float64 first, then the
    # final .to(dtype) applies the caller's requested precision.
    data = coo.data if coo.data.dtype in (np.float32, np.float64) else coo.data.astype(np.float64)
    values = torch.from_numpy(np.ascontiguousarray(data)).to(dtype)
    shape = torch.Size(coo.shape)
    tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class HodgeMessagePassing(nn.Module):  # type: ignore[misc]
    """
    One layer of Hodge-Laplacian message passing on k-simplices.

    Parameters
    ----------
    in_features : int
        Input feature dimension per k-simplex.
    out_features : int
        Output feature dimension.
    laplacian : torch.sparse.Tensor | sp.spmatrix | np.ndarray
        Pre-computed normalized Laplacian of shape (n_k, n_k). Will be cast to
        a torch sparse tensor at construction time.
    activation : callable, optional
        Nonlinearity (default torch.relu). Pass None for linear layer.
    bias : bool
        Whether to add a learnable bias (default True).
    dtype : torch.dtype
        Parameter dtype (default torch.float32).

    Examples
    --------
    >>> from topogeoml.core.complexes import SimplicialComplex, hodge_laplacian
    >>> from topogeoml.nn.hodge import HodgeMessagePassing, normalize_hodge_laplacian
    >>> # Triangle (2-simplex) — three edges, β_1 = 0.
    >>> sc = SimplicialComplex(facets=[(0, 1, 2)])
    >>> L0 = hodge_laplacian(sc, 0)
    >>> L0_norm = normalize_hodge_laplacian(L0)
    >>> layer = HodgeMessagePassing(in_features=4, out_features=8, laplacian=L0_norm)
    >>> x = torch.randn(sc.n_simplices(0), 4)
    >>> out = layer(x)
    >>> out.shape
    torch.Size([3, 8])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        laplacian: torch.Tensor | sp.spmatrix | np.ndarray,
        activation: Any = None,  # use torch.relu by default in forward
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in/out features must be positive; got {in_features}, {out_features}"
            )

        # Cast laplacian to torch sparse tensor and register as buffer (non-trainable).
        if isinstance(laplacian, torch.Tensor):
            L = laplacian
        elif sp.issparse(laplacian):
            L = sparse_scipy_to_torch(laplacian, dtype=dtype)
        elif isinstance(laplacian, np.ndarray):
            L = sparse_scipy_to_torch(sp.csr_matrix(laplacian), dtype=dtype)
        else:
            raise TypeError(
                f"laplacian must be torch.Tensor, scipy.sparse, or ndarray; "
                f"got {type(laplacian).__name__}"
            )

        if L.shape[0] != L.shape[1]:
            raise ValueError(f"Laplacian must be square; got shape {tuple(L.shape)}")

        self.register_buffer("laplacian", L)
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=dtype)) if bias else None
        )
        self.activation = activation if activation is not None else torch.relu
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot init.
        nn.init.xavier_uniform_(self.weight)

    @property
    def _laplacian(self) -> torch.Tensor:
        """Typed accessor for the registered Laplacian buffer.

        ``nn.Module.__getattr__`` returns ``Size | Tensor | Module`` for
        any attribute (buffers, parameters, submodules all share the
        dispatch). This property narrows the type to ``torch.Tensor`` so
        mypy can verify the indexing in the shape-check assertions and
        the ``torch.sparse.mm`` call below. The actual buffer is still
        ``self.laplacian`` and continues to move with ``.to(device)``.
        """
        return cast(torch.Tensor, self.laplacian)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one round of Hodge propagation.

        Parameters
        ----------
        x : torch.Tensor
            Shape (n_k, in_features).

        Returns
        -------
        torch.Tensor
            Shape (n_k, out_features).
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (n_k, in_features); got shape {tuple(x.shape)}")
        if x.shape[0] != self._laplacian.shape[0]:
            raise ValueError(
                f"x has {x.shape[0]} rows but Laplacian has {self._laplacian.shape[0]} "
                f"k-simplices — shape mismatch"
            )
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"x has {x.shape[1]} features but layer expects {self.in_features}"
            )

        # Propagate: x' = activation( L @ x @ W + b )
        # torch.sparse.mm is the operator for sparse @ dense.
        propagated = torch.sparse.mm(self._laplacian, x)
        transformed = propagated @ self.weight
        if self.bias is not None:
            transformed = transformed + self.bias
        return self.activation(transformed)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_simplices={self._laplacian.shape[0]}"
        )


def build_hodge_layer_from_complex(
    complex_: SimplicialComplex,
    k: int,
    in_features: int,
    out_features: int,
    **kwargs: Any,
) -> HodgeMessagePassing:
    """
    Convenience constructor: build the k-th normalized Hodge Laplacian
    from a SimplicialComplex and wrap a HodgeMessagePassing layer around it.
    """
    L = hodge_laplacian(complex_, k)
    L_norm = normalize_hodge_laplacian(L)
    return HodgeMessagePassing(
        in_features=in_features,
        out_features=out_features,
        laplacian=L_norm,
        **kwargs,
    )
