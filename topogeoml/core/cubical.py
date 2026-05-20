"""
Cubical mask topology diagnostic.

For binary 2D/3D masks, this module reports topological summary statistics
(β_0, β_1, Euler characteristic) without invoking a full persistent homology
backend. Implementation uses scipy.ndimage connected-component labeling on the
foreground and on the background, exploiting Alexander duality on a closed
ambient grid.

Item 4 of the v0.0.1 scope. The full cubical filtration (sub-level sets of a
real-valued image producing a persistence diagram) is deferred to v0.1.

Conventions
-----------
* Input: 2D or 3D boolean / 0-1 integer ndarray.
* Connectivity: 1 = face-connected (4-conn in 2D, 6-conn in 3D),
                2 = edge-connected (8-conn in 2D, 18-conn in 3D — 2D supported).
* β_0 = # foreground connected components.
* β_1 (2D only) = # holes in foreground = # background components fully enclosed
  by foreground. With background padded by 1 on all sides, the outer infinite
  background component is excluded.
* Euler characteristic for 2D: χ = β_0 - β_1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray


@dataclass(frozen=True)
class CubicalDiagnostic:
    """Topological summary of a binary mask."""

    betti_0: int
    """Number of foreground connected components."""

    betti_1: int
    """Number of 1-dimensional holes (2D inputs only; -1 for 3D until v0.1)."""

    euler_characteristic: int
    """χ = β_0 - β_1 for 2D inputs. -1 for higher dims until extended."""

    n_foreground_pixels: int
    """Count of True / non-zero pixels."""

    ndim: int
    """Spatial dimension of the input (2 or 3)."""

    connectivity: int
    """Foreground connectivity used (1 or 2)."""

    extras: dict[str, Any]
    """Backend-specific diagnostics."""


def cubical_mask_diagnostic(
    mask: NDArray[Any],
    connectivity: int = 1,
) -> CubicalDiagnostic:
    """
    Compute topology of a binary mask.

    Parameters
    ----------
    mask : NDArray
        2D or 3D array. Cast to bool; non-zero = foreground.
    connectivity : int
        scipy.ndimage connectivity:
          1 = face-connected (4-conn 2D / 6-conn 3D),
          2 = edge+face (8-conn 2D / 18-conn 3D).

    Returns
    -------
    CubicalDiagnostic
        β_0 always reported. β_1 reported for 2D inputs (via Alexander duality
        on the padded background); reported as -1 for 3D pending v0.1 cubical
        persistence backend.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((10, 10), dtype=bool)
    >>> mask[2:8, 2:8] = True
    >>> mask[4:6, 4:6] = False  # one hole
    >>> diag = cubical_mask_diagnostic(mask)
    >>> diag.betti_0, diag.betti_1, diag.euler_characteristic
    (1, 1, 0)
    """
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim not in (2, 3):
        raise ValueError(f"mask must be 2D or 3D; got ndim={arr.ndim}")
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")

    n_fg = int(arr.sum())

    # Foreground connected components: β_0.
    fg_structure = ndi.generate_binary_structure(arr.ndim, connectivity)
    _, n_components = ndi.label(arr, structure=fg_structure)
    betti_0 = int(n_components)

    extras: dict[str, Any] = {"shape": arr.shape}

    if arr.ndim == 2:
        # β_1 via Alexander duality on the padded complement:
        # Pad background by 1 pixel on each side so the outer region is one
        # connected component. Count background components and subtract 1 for
        # the outer region.
        # Use complementary connectivity (face-adjacent for background when
        # foreground is 8-conn, and vice versa) to satisfy the digital
        # Jordan curve theorem: foreground 8-conn ↔ background 4-conn,
        # foreground 4-conn ↔ background 8-conn.
        bg_connectivity = 3 - connectivity  # 1 ↔ 2
        bg_structure = ndi.generate_binary_structure(arr.ndim, bg_connectivity)
        padded_bg = np.pad(~arr, 1, mode="constant", constant_values=True)
        _, n_bg_components = ndi.label(padded_bg, structure=bg_structure)
        betti_1 = int(n_bg_components - 1)  # subtract outer background
        euler = betti_0 - betti_1
        extras["bg_connectivity"] = bg_connectivity
    else:
        # 3D β_1 / β_2 require a cubical persistence backend (GUDHI's
        # cubical_complex or cripser). Deferred to v0.1.
        betti_1 = -1
        euler = -1
        extras["note"] = "3D β_1 not computed; pending v0.1 cubical PH backend"

    return CubicalDiagnostic(
        betti_0=betti_0,
        betti_1=betti_1,
        euler_characteristic=euler,
        n_foreground_pixels=n_fg,
        ndim=arr.ndim,
        connectivity=connectivity,
        extras=extras,
    )
