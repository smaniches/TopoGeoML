"""Tests for cubical mask topology diagnostic (item 4)."""

from __future__ import annotations

import numpy as np
import pytest

from topogeoml.core.cubical import cubical_mask_diagnostic


def test_disk_no_holes() -> None:
    """Filled disk: β_0=1, β_1=0, χ=1."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    d = cubical_mask_diagnostic(mask)
    assert d.betti_0 == 1
    assert d.betti_1 == 0
    assert d.euler_characteristic == 1


def test_annulus_one_hole() -> None:
    """Annulus: β_0=1, β_1=1, χ=0."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:17, 3:17] = True
    mask[7:13, 7:13] = False  # central hole
    d = cubical_mask_diagnostic(mask)
    assert d.betti_0 == 1
    assert d.betti_1 == 1
    assert d.euler_characteristic == 0


def test_two_disjoint_disks() -> None:
    """Two disjoint filled rectangles: β_0=2, β_1=0, χ=2."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:8, 2:8] = True
    mask[12:18, 12:18] = True
    d = cubical_mask_diagnostic(mask)
    assert d.betti_0 == 2
    assert d.betti_1 == 0
    assert d.euler_characteristic == 2


def test_disk_with_two_holes() -> None:
    """One component with two holes: β_0=1, β_1=2, χ=-1."""
    mask = np.zeros((25, 25), dtype=bool)
    mask[3:22, 3:22] = True
    mask[6:10, 6:10] = False
    mask[15:19, 15:19] = False
    d = cubical_mask_diagnostic(mask)
    assert d.betti_0 == 1
    assert d.betti_1 == 2
    assert d.euler_characteristic == -1


def test_empty_mask() -> None:
    """All-False mask: β_0=0, β_1=0."""
    mask = np.zeros((10, 10), dtype=bool)
    d = cubical_mask_diagnostic(mask)
    assert d.betti_0 == 0
    assert d.betti_1 == 0
    assert d.n_foreground_pixels == 0


def test_full_mask() -> None:
    """All-True mask: β_0=1 (entire grid is foreground)."""
    mask = np.ones((10, 10), dtype=bool)
    d = cubical_mask_diagnostic(mask)
    assert d.betti_0 == 1
    # No background interior holes (the outer infinite background is excluded).
    assert d.betti_1 == 0


def test_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        cubical_mask_diagnostic(np.array([1, 0, 1, 0]))


def test_rejects_4d_input() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        cubical_mask_diagnostic(np.zeros((3, 3, 3, 3), dtype=bool))


def test_3d_input_betti_0_only() -> None:
    """3D inputs report β_0 only in v0.0.1; β_1 is -1 (pending v0.1)."""
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    d = cubical_mask_diagnostic(mask)
    assert d.ndim == 3
    assert d.betti_0 == 1
    assert d.betti_1 == -1  # explicit deferred sentinel
    assert d.euler_characteristic == -1


def test_provenance_fields() -> None:
    mask = np.eye(5, dtype=bool)
    d = cubical_mask_diagnostic(mask, connectivity=2)
    assert d.connectivity == 2
    assert d.n_foreground_pixels == 5
    assert d.ndim == 2
    assert "shape" in d.extras


def test_connectivity_affects_count() -> None:
    """A 5-pixel diagonal: 4-connectivity → 5 components; 8-connectivity → 1."""
    mask = np.eye(5, dtype=bool)
    d4 = cubical_mask_diagnostic(mask, connectivity=1)
    d8 = cubical_mask_diagnostic(mask, connectivity=2)
    assert d4.betti_0 == 5
    assert d8.betti_0 == 1
