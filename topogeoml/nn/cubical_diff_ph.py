"""
Differentiable Cubical Persistent Homology via critical-pixel indexing.

The key insight is the same as for the Vietoris–Rips case
(``topogeoml.nn.diff_ph``): gudhi identifies *which input pixel values*
realize each (birth, death) bar via its lower-star cubical persistence
computation; we index those pixel values back into the input PyTorch
tensor — which carries full autograd — so no custom backward is
required.

Mathematical basis
------------------
Lower-star cubical filtration on a regular grid: vertices carry filtration
values equal to the pixel intensity; higher-dimensional cells (edges,
squares, cubes) inherit the *maximum* over their incident vertices.

For every finite-persistence bar :math:`(b_i, d_i)` gudhi's
:func:`gudhi.CubicalComplex.vertices_of_persistence_pairs` returns the
pair :math:`(v_b, v_d)` of vertex indices whose filtration value realizes
the birth and the death (flat row-major indexing). We then construct
bars as

.. math::
    b_i = \\text{img.flat}[v_b], \\quad d_i = \\text{img.flat}[v_d],

which is a differentiable function of the input image: PyTorch's autograd
flows the gradient back through the indexing.

The standard subgradient choice (Hofer et al. 2017; Carrière et al. 2021)
is to push the gradient through the *single* critical vertex per bar
endpoint; ties are broken by gudhi's stable internal ordering, which
makes the output deterministic given the same input.

This implementation is the cubical-PH analogue of Clough et al. 2020
("A Topological Loss Function for Deep-Learning Based Image
Segmentation Using Persistent Homology", IEEE TPAMI), shipped as a
differentiable PyTorch ``nn.Module``.

References
----------
Clough, J. R., Byrne, N., Oksuz, I., Zimmer, V. A., Schnabel, J. A.,
  & King, A. P. (2020). "A Topological Loss Function for Deep-Learning
  Based Image Segmentation Using Persistent Homology." *IEEE TPAMI*.
Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). "Stability of
  persistence diagrams." *Discrete & Computational Geometry*, 37(1).
Hofer, C., Kwitt, R., Niethammer, M., & Uhl, A. (2017). "Deep Learning
  with Topological Signatures." *NeurIPS 2017*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "topogeoml.nn.cubical_diff_ph requires PyTorch. "
        "Install with `pip install torch`."
    ) from exc


def _gudhi_cubical_vertex_pairs(
    img_np: NDArray[np.float64], max_dim: int
) -> tuple[list[NDArray[np.int64]], NDArray[np.int64]]:
    """Run gudhi's cubical persistence and return the (vertex, vertex) pairs
    for every finite bar plus the essential-class birth vertices.

    Returns
    -------
    finite_by_dim : list[ndarray of shape (n_bars, 2)]
        ``finite_by_dim[k]`` is a 2-column array of ``(birth_vertex,
        death_vertex)`` indices for the finite H_k bars, in flat row-major
        order over ``img_np``.
    essential_h0_vertices : ndarray of shape (n_essential,)
        Flat indices of the vertices realizing essential H_0 classes
        (typically a single global minimum). We expose only essential H_0
        because higher-dim essential classes do not exist for compact
        domains in cubical-PH at the default filtration.
    """
    import gudhi

    if img_np.ndim < 2:  # pragma: no cover
        # Defensive: ``cubical_diagram_torch`` raises ``ValueError`` for
        # ``ndim < 2`` before reaching this internal helper.
        raise ValueError(
            f"image must be at least 2D; got shape {img_np.shape}"
        )

    cc = gudhi.CubicalComplex(vertices=img_np.astype(np.float64, copy=False))
    cc.persistence()
    verts = cc.vertices_of_persistence_pairs()

    finite_by_dim: list[NDArray[np.int64]] = []
    finite_pairs_per_dim = verts[0]
    for k in range(max_dim + 1):
        if k < len(finite_pairs_per_dim):
            arr = np.asarray(finite_pairs_per_dim[k], dtype=np.int64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                finite_by_dim.append(arr)
            elif arr.size == 0:  # pragma: no cover
                # gudhi 3.12 always returns a (n, 2) array even when n=0;
                # we fall through here only on a future schema change.
                finite_by_dim.append(np.zeros((0, 2), dtype=np.int64))
            else:  # pragma: no cover  -- defensive against future gudhi schema changes.
                finite_by_dim.append(np.zeros((0, 2), dtype=np.int64))
        else:
            finite_by_dim.append(np.zeros((0, 2), dtype=np.int64))

    essential_per_dim = verts[1]
    essential_h0 = (
        np.asarray(essential_per_dim[0], dtype=np.int64)
        if len(essential_per_dim) > 0
        else np.zeros((0,), dtype=np.int64)
    )
    return finite_by_dim, essential_h0


def _assemble_bars(
    flat: torch.Tensor,
    finite_by_dim: list[NDArray[np.int64]],
    essential_h0: NDArray[np.int64],
    max_dim: int,
    include_essential: bool,
) -> list[torch.Tensor]:
    """Reconstruct differentiable bars from precomputed gudhi vertex indices.

    Pure tensor indexing — autograd flows through ``flat[v_b]`` back to the
    original image. Factored out so callers that already have a CPU view
    of the image (e.g. a batched loss that wants one bulk transfer) can
    skip the per-image ``detach().cpu()`` round-trip.
    """
    diagrams: list[torch.Tensor] = []
    for k in range(max_dim + 1):
        pairs = finite_by_dim[k]
        if pairs.shape[0] > 0:
            v_b = torch.from_numpy(pairs[:, 0]).to(flat.device)
            v_d = torch.from_numpy(pairs[:, 1]).to(flat.device)
            births = flat[v_b]
            deaths = flat[v_d]
            bars = torch.stack([births, deaths], dim=1)
        else:
            bars = flat.new_empty((0, 2))
        diagrams.append(bars)

    if include_essential and essential_h0.size > 0:
        v_ess = torch.from_numpy(essential_h0).to(flat.device)
        births = flat[v_ess]
        deaths = torch.full_like(births, float("inf"))
        essential_bars = torch.stack([births, deaths], dim=1)
        diagrams[0] = torch.cat([diagrams[0], essential_bars], dim=0)

    return diagrams


def cubical_diagram_torch(
    img: torch.Tensor, max_dim: int = 1, include_essential: bool = True
) -> list[torch.Tensor]:
    """
    Differentiable cubical persistence diagram of a single image.

    Parameters
    ----------
    img : torch.Tensor
        ``(H, W)`` or ``(D, H, W)`` float64 grid. The filtration is the
        lower-star filtration with pixel intensities at vertices.
    max_dim : int
        Highest homology dimension to compute. For 2D images use 1 (H_0,
        H_1); for 3D, up to 2 (H_0, H_1, H_2).
    include_essential : bool
        If True, the essential H_0 class is emitted as a single bar
        ``(birth=img.flat[v_essential], death=inf)`` so the output schema
        matches the Rips backend. Set False to omit it (e.g. for losses
        that only care about finite bars).

    Returns
    -------
    list[torch.Tensor]
        ``[H_0_bars, H_1_bars, ..., H_max_dim_bars]``. Each entry is a
        ``(n_bars, 2)`` float64 tensor with columns ``(birth, death)``.
        Gradients flow back to ``img`` through the indexing.

    Notes
    -----
    The combinatorial step (which vertex realizes each bar) is
    non-differentiable by nature: gudhi computes it on a detached
    ``numpy`` view. The differentiable forward then reconstructs each
    bar value as ``img.flat[v_b]`` and ``img.flat[v_d]``. Standard
    PyTorch autograd handles the rest.

    Determinism
    -----------
    Given the same input, gudhi returns the same critical-vertex
    indices. The forward is thus byte-deterministic.
    """
    if img.dtype != torch.float64:
        raise TypeError(
            f"input image must be float64, got {img.dtype}"
        )
    if img.ndim < 2:
        raise ValueError(
            f"image must be at least 2D (HxW); got shape {tuple(img.shape)}"
        )

    img_np = img.detach().cpu().numpy().astype(np.float64, copy=False)
    finite_by_dim, essential_h0 = _gudhi_cubical_vertex_pairs(img_np, max_dim)

    flat = img.reshape(-1)
    return _assemble_bars(
        flat, finite_by_dim, essential_h0, max_dim, include_essential
    )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def finite_lifetimes_cubical(diagram: torch.Tensor) -> torch.Tensor:
    """Return lifetimes ``death - birth`` for finite bars."""
    if diagram.numel() == 0:
        return diagram.new_empty((0,))
    mask = torch.isfinite(diagram[:, 1])
    if not mask.any():
        return diagram.new_empty((0,))
    finite = diagram[mask]
    return finite[:, 1] - finite[:, 0]


def betti_matching_loss(
    diagram: torch.Tensor,
    target_n_bars: int,
    prominence_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Clough et al. 2020 style Betti-matching loss.

    The full diagram is interpreted as the union of essential bars
    (``death = +inf``) and finite bars. Both contribute toward the
    target Betti count:

    1. Essential bars cannot be shrunk (infinite lifetime → no
       gradient), so they always count as preserved features. They
       consume the first ``n_essential`` slots of the
       ``target_n_bars`` budget.
    2. Remaining budget ``max(0, target_n_bars - n_essential)`` is
       allocated to the longest-lifetime finite bars.
    3. Finite bars beyond the remaining budget with lifetime greater
       than ``prominence_threshold`` are *excess*; their lifetimes
       are summed into the loss. Descent then shrinks each toward
       the diagonal.

    This convention matches the standard interpretation of
    ``target_betti``: for H_0 it is the desired number of connected
    components (every component contributes one essential bar in
    lower-star cubical persistence); for H_1 it is the desired number
    of loops (no essential H_1 on a compact 2-D grid, so all bars are
    finite). The same code path handles both.

    Parameters
    ----------
    diagram : torch.Tensor
        ``(n_bars, 2)`` persistence diagram for one homology dimension.
        Essential bars are encoded with ``death = +inf``.
    target_n_bars : int
        Expected Betti number for this dimension (e.g. 1 for a single
        connected component in H_0, or 1 for a single loop in H_1).
    prominence_threshold : float
        Finite bars with lifetime <= this are not counted as significant.

    Returns
    -------
    torch.Tensor
        Scalar loss; minimizing it pushes the network toward the
        target topology.
    """
    if diagram.numel() == 0:
        return torch.zeros((), dtype=diagram.dtype, device=diagram.device)
    finite_mask = torch.isfinite(diagram[:, 1])
    n_essential = int((~finite_mask).sum().item())
    n_finite_keep = max(0, target_n_bars - n_essential)

    finite = diagram[finite_mask]
    if finite.numel() == 0:
        return torch.zeros((), dtype=diagram.dtype, device=diagram.device)
    lifetimes = finite[:, 1] - finite[:, 0]
    sorted_lifetimes, _ = lifetimes.sort(descending=True)
    significant = sorted_lifetimes[sorted_lifetimes > prominence_threshold]
    if significant.numel() <= n_finite_keep:
        # Within budget after accounting for essential bars: no penalty.
        return torch.zeros((), dtype=diagram.dtype, device=diagram.device)
    excess = significant[n_finite_keep:]
    return excess.sum()


# ---------------------------------------------------------------------------
# Module interface
# ---------------------------------------------------------------------------

class CubicalTopologyLoss(nn.Module):  # type: ignore[misc]
    """
    Cubical-PH topology loss for image segmentation training.

    Takes the predicted segmentation map (soft mask in [0, 1]) and
    penalizes the deviation from the target Betti numbers per homology
    dimension. Designed to be added as an auxiliary loss to a U-Net's
    Dice/BCE objective:

        loss = dice_loss(pred, target) + lambda_topo * topo_loss(pred)

    Parameters
    ----------
    target_betti : dict[int, int]
        ``{k: target_beta_k}``. The loss penalizes excess bars in each
        dimension beyond the target. The standard Betti convention is
        used: for H_0, ``target_beta_0`` is the number of desired
        connected components (each contributes one essential bar in
        lower-star cubical persistence and is counted toward the target);
        for H_1 it is the number of desired loops.
    prominence_threshold : float
        Finite bars with lifetime ``<= prominence_threshold`` are not
        counted as significant. Essential bars (death = ∞) are always
        kept and are not subject to this threshold.
    invert : bool
        If True, invert the prediction (``1 - pred``) before computing
        persistence. Use this when foreground = high intensity (the
        standard for vessel/cell segmentation): the lower-star filtration
        then captures the *foreground* topology rather than the
        background.
    """

    def __init__(
        self,
        target_betti: dict[int, int],
        prominence_threshold: float = 0.0,
        invert: bool = True,
    ) -> None:
        super().__init__()
        if not target_betti:
            raise ValueError("target_betti must be non-empty")
        for k, v in target_betti.items():
            if k < 0:
                raise ValueError(f"target_betti keys must be >= 0, got {k}")
            if v < 0:
                raise ValueError(f"target_betti values must be >= 0, got {v}")
        self.target_betti = dict(target_betti)
        self.prominence_threshold = float(prominence_threshold)
        self.invert = bool(invert)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            ``(H, W)`` or ``(B, H, W)`` or ``(B, 1, H, W)`` float64
            predicted segmentation map. For batched inputs the loss
            averages over the batch.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if pred.ndim == 4 and pred.shape[1] == 1:
            # (B, 1, H, W) -- squeeze the channel dim and recurse.
            return self.forward(pred.squeeze(1))
        if pred.ndim == 2:
            return self._forward_single_image(pred)
        if pred.ndim == 3:
            return self._forward_batched(pred)
        raise ValueError(
            f"pred must be (H, W) or (B, H, W) or (B, 1, H, W); got shape {tuple(pred.shape)}"
        )

    def _img_for_filtration(self, pred: torch.Tensor) -> torch.Tensor:
        """Apply the invert convention and cast to float64."""
        pred64 = pred.to(torch.float64)
        return 1.0 - pred64 if self.invert else pred64

    def _loss_from_diagrams(
        self,
        diagrams: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        loss = torch.zeros((), dtype=dtype, device=device)
        for k, target in self.target_betti.items():
            if k < len(diagrams):  # pragma: no branch
                # max_dim == max(target_betti) and cubical_diagram_torch returns
                # max_dim + 1 diagrams, so every key is in range; defensive edge.
                loss = loss + betti_matching_loss(
                    diagrams[k],
                    target_n_bars=target,
                    prominence_threshold=self.prominence_threshold,
                )
        return loss

    def _forward_single_image(self, pred: torch.Tensor) -> torch.Tensor:
        img = self._img_for_filtration(pred)
        max_dim = max(self.target_betti) if self.target_betti else 1
        diagrams = cubical_diagram_torch(
            img, max_dim=max_dim, include_essential=True,
        )
        return self._loss_from_diagrams(diagrams, img.device, img.dtype)

    def _forward_batched(self, pred: torch.Tensor) -> torch.Tensor:
        """Batched path with a single bulk CPU transfer.

        The naive per-image loop incurs *B* blocking GPU→CPU transfers
        because each ``cubical_diagram_torch`` call detaches its input
        independently. We instead detach the whole batch once, run
        gudhi per slice on the CPU view (gudhi is CPU-only by
        construction), and reassemble the differentiable bars by
        indexing the original GPU tensor.
        """
        img = self._img_for_filtration(pred)
        # Single bulk transfer of the whole batch.
        img_cpu = img.detach().cpu().numpy().astype(np.float64, copy=False)
        max_dim = max(self.target_betti) if self.target_betti else 1

        per_image_losses: list[torch.Tensor] = []
        for b in range(img.shape[0]):
            finite_by_dim, essential_h0 = _gudhi_cubical_vertex_pairs(
                img_cpu[b], max_dim,
            )
            flat = img[b].reshape(-1)
            diagrams = _assemble_bars(
                flat, finite_by_dim, essential_h0, max_dim,
                include_essential=True,
            )
            per_image_losses.append(
                self._loss_from_diagrams(diagrams, img.device, img.dtype)
            )
        return torch.stack(per_image_losses).mean()


def cubical_correctness_vs_gudhi(
    img_np: NDArray[np.float64], max_dim: int
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:  # pragma: no cover
    """Convenience function for the bench: compute the cubical diagram
    once via the torch backend (detached) and once via gudhi directly,
    returning both sets so a caller can compare.

    Excluded from coverage because it's purely a debugging convenience.
    """
    import gudhi

    img_torch = torch.from_numpy(img_np).to(torch.float64)
    torch_diagrams = cubical_diagram_torch(img_torch, max_dim=max_dim)
    torch_arrays = [d.detach().numpy() for d in torch_diagrams]

    cc = gudhi.CubicalComplex(vertices=img_np.astype(np.float64, copy=False))
    pers = cc.persistence()
    gudhi_arrays: list[list[tuple[float, float]]] = [[] for _ in range(max_dim + 1)]
    for d, (b, dth) in pers:
        if d <= max_dim:
            gudhi_arrays[d].append((float(b), float(dth)))
    gudhi_out = [
        np.asarray(bars, dtype=np.float64).reshape(-1, 2)
        for bars in gudhi_arrays
    ]
    return torch_arrays, gudhi_out


__all__ = [
    "CubicalTopologyLoss",
    "betti_matching_loss",
    "cubical_correctness_vs_gudhi",
    "cubical_diagram_torch",
    "finite_lifetimes_cubical",
]
