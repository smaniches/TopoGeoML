"""
Backend wrapper: gudhi-python.

gudhi (Inria) is the reference C++ persistence library with Python
bindings. The wheel that ships on PyPI does not include gudhi's
TensorFlow / autograd integration — only the core persistence
computation. We therefore register gudhi as a **non-differentiable**
backend: it participates in the correctness axis as a third reference
(alongside ripser) but is skipped by axes that require ``loss_longest_h1``
(speed, stability, optimization).

Why register a non-differentiable backend at all
------------------------------------------------
The correctness axis asks "does this library's persistence diagram match
ripser to atol?" — having a *second* independent persistence library
strengthens that check considerably. ripser and gudhi were developed
independently from different algorithmic backbones (matrix reduction
vs. Bauer's algorithm), so if both agree on a diagram we have stronger
confidence in the diagram itself.

References
----------
Maria, C., Boissonnat, J.-D., Glisse, M., & Yvinec, M. (2014). "The GUDHI
  library: simplicial complexes and persistent homology." *ICMS 2014*.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch

from benchmarks.backends import register_backend


@register_backend
class GudhiPython:
    """gudhi-python — Inria's reference C++/Python persistence library."""

    name: ClassVar[str] = "gudhi-python"
    version: ClassVar[str] = ""
    differentiable: ClassVar[bool] = False

    @staticmethod
    def available() -> bool:
        try:
            import gudhi
        except ImportError:  # pragma: no cover  -- defensive: bench extras install gudhi.
            return False
        GudhiPython.version = str(gudhi.__version__)
        return True

    @staticmethod
    def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
        """Compute Rips persistence via gudhi.

        gudhi computes the full diagram in one call; we split by dimension
        to match the protocol's ``[H_0, H_1, ...]`` layout.
        """
        import gudhi

        if X.dtype != torch.float64:
            raise TypeError(
                f"{GudhiPython.name}: input X must be float64, got {X.dtype}"
            )
        pts = X.detach().cpu().numpy().astype(np.float64, copy=False)
        # max_edge_length: use the maximum pairwise distance so we capture
        # all bars. For a unit-bounded point cloud this is at most 2 * sqrt(d).
        # We compute the actual diameter to avoid producing artifacts from
        # an unnecessarily large filtration parameter.
        if pts.shape[0] >= 2:
            from scipy.spatial.distance import pdist
            diameter = float(pdist(pts).max())
            # Vietoris-Rips persistence is invariant for filtration
            # parameters >= diameter; the 5% headroom that lived here
            # was therefore arithmetic-noise on top of an already-saturated
            # parameter (caught by Gemini PR #4 review).
            max_edge_length = max(diameter, 1e-10)
        else:  # pragma: no cover  -- bench always uses n >= 2.
            max_edge_length = 1.0

        rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
        # max_dimension here is the *simplex* dimension, which is the
        # homology dimension + 1 (a k-simplex computes H_{k-1} and H_k).
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)
        persistence = simplex_tree.persistence()

        # Bucket bars by homology dimension.
        by_dim: dict[int, list[tuple[float, float]]] = {k: [] for k in range(max_dim + 1)}
        for dim, (birth, death) in persistence:
            if dim in by_dim:
                by_dim[dim].append((float(birth), float(death)))

        # Convert each list to a (n_bars, 2) float64 tensor on the same
        # device as X. Non-differentiable — these tensors do not require grad.
        out: list[torch.Tensor] = []
        for k in range(max_dim + 1):
            bars = by_dim[k]
            if not bars:
                out.append(torch.empty((0, 2), dtype=torch.float64, device=X.device))
            else:
                out.append(torch.tensor(bars, dtype=torch.float64, device=X.device))
        return out

    @staticmethod
    def loss_longest_h1(X: torch.Tensor) -> torch.Tensor:
        """gudhi-python is non-differentiable; this is a no-op stub.

        Raises ``NotImplementedError`` so the runner can detect the
        capability gap explicitly. Callers should consult
        ``GudhiPython.differentiable`` before invoking this method.
        """
        raise NotImplementedError(
            f"{GudhiPython.name} is non-differentiable; consult "
            "`differentiable` class attribute before calling loss_longest_h1"
        )
