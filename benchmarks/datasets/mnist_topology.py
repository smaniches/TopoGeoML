"""
MNIST topology dataset — real-data point clouds for the bench.

Each MNIST digit is a 28x28 grayscale image. We convert each sample to a
2-D point cloud by extracting the (row, column) coordinates of pixels
that exceed a threshold, optionally subsampled to a fixed point count
for deterministic per-sample cost. The result is a real-data fixture
with naturally non-trivial topology:

  - digits 0, 6, 9 have an expected :math:`\\beta_1 = 1` (one loop),
  - digit 8 has expected :math:`\\beta_1 = 2` (two loops),
  - digits 1, 2, 3, 5, 7 have expected :math:`\\beta_1 = 0`,
  - digit 4 is ambiguous (open or closed top) so we do not assert
    a specific :math:`\\beta_1`.

Caching
-------
The torchvision MNIST loader writes its files into ``XDG_CACHE_HOME``
(or ``~/.cache``) at ``topogeoml/benchmarks/mnist``. On first use the
data is downloaded (~12 MB); subsequent uses are offline. The CI
workflow caches this directory between runs via ``actions/cache``.

Test strategy
-------------
``MNISTPointCloud.generate`` is exercised end-to-end against a real
MNIST download in the integration test path, but the unit-test suite
uses :class:`MNISTPointCloudMock`, an in-memory replacement that
returns a synthetic-but-fixed-shape tensor satisfying the same
contract. This lets the unit tests pass without network access while
still verifying the generation contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from benchmarks.datasets import register_dataset

# Per-digit expected H_1 Betti numbers used by the correctness axis to
# sanity-check that the topology of the point cloud is what the dataset
# label implies. Digit 4 has historically been ambiguous (open vs closed
# top variants are both common in MNIST); we omit it.
_EXPECTED_H1: dict[int, int] = {
    0: 1, 1: 0, 2: 0, 3: 0, 5: 0, 6: 1, 7: 0, 8: 2, 9: 1,
}


def _cache_root() -> Path:
    """Resolve the cache directory for MNIST downloads.

    XDG-conforming: prefers ``$XDG_CACHE_HOME``, falls back to ``~/.cache``.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    root = Path(base) / "topogeoml" / "benchmarks" / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _digit_to_point_cloud(
    img: NDArray[np.floating],
    *,
    threshold: float,
    n_points: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Convert a single 28x28 MNIST image to a 2-D point cloud.

    Steps:
      1. Active pixels = pixels with intensity > ``threshold * max(img)``.
      2. Coordinates are returned as ``(row, col)`` in pixel space, then
         centered and normalized to a unit bounding box for scale
         invariance across digits.
      3. If the active-pixel count is less than ``n_points`` we resample
         with replacement; otherwise we draw a uniform random subsample
         without replacement. Either case yields exactly ``n_points``.

    The ``rng`` argument seeds the subsampling so the resulting point
    cloud is deterministic given the same ``(digit_index, seed)``.
    """
    if img.shape != (28, 28):
        raise ValueError(f"MNIST image must be 28x28, got {img.shape}")

    img = np.asarray(img, dtype=np.float64)
    img_max = float(img.max())
    if img_max <= 0.0:
        # All-black image — return a degenerate single-point cloud at the
        # origin, replicated to ``n_points``. The diff-PH layer will
        # produce zero H_1 signal, which is the honest answer.
        return np.zeros((n_points, 2), dtype=np.float64)

    active = img > (threshold * img_max)
    rows, cols = np.where(active)
    if rows.size == 0:
        return np.zeros((n_points, 2), dtype=np.float64)

    coords = np.column_stack([rows.astype(np.float64), cols.astype(np.float64)])
    # Center and normalize to unit bounding box.
    coords -= coords.mean(axis=0, keepdims=True)
    span = float(np.abs(coords).max())
    if span > 0:
        coords /= span

    n_active = coords.shape[0]
    if n_active >= n_points:
        idx = rng.choice(n_active, size=n_points, replace=False)
    else:
        idx = rng.choice(n_active, size=n_points, replace=True)
    return np.ascontiguousarray(coords[idx], dtype=np.float64)


@dataclass(frozen=True)
class MNISTPointCloud:
    """MNIST digits, each converted to a fixed-size 2-D point cloud.

    Parameters
    ----------
    digit
        Which MNIST digit to draw from. Must be in 0..9.
    threshold
        Fraction of the per-image maximum intensity above which a pixel
        is considered "active". 0.5 is a robust default for MNIST.
    download
        If True, allow torchvision to download MNIST on first use. CI
        sets this to True (cached by ``actions/cache``); offline test
        runs should pre-populate the cache directory and set
        ``download=False``.
    """

    digit: int = 0
    threshold: float = 0.5
    download: bool = True

    @property
    def name(self) -> str:
        return f"mnist_digit_{self.digit}"

    @property
    def version(self) -> str:
        # Bumping invalidates leaderboard entries for this dataset.
        # Bump when the generation algorithm changes (threshold semantics,
        # normalization, subsampling strategy).
        return "1.0.0"

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        from torchvision import datasets

        if not 0 <= self.digit <= 9:
            raise ValueError(f"digit must be in 0..9, got {self.digit}")
        if n_points < 4:
            raise ValueError(f"n_points must be >= 4 for meaningful PH, got {n_points}")

        rng = np.random.default_rng(seed)
        root = _cache_root()

        mnist = datasets.MNIST(root=str(root), train=True, download=self.download)
        # MNIST stores PIL images; convert to numpy.
        # Find the (seed mod len) sample of the requested digit; this gives
        # deterministic but seed-varying selection.
        targets = mnist.targets.numpy() if hasattr(mnist.targets, "numpy") else np.asarray(mnist.targets)
        digit_indices = np.where(targets == self.digit)[0]
        if digit_indices.size == 0:
            raise RuntimeError(f"MNIST contains no samples of digit {self.digit}")
        pick = int(digit_indices[seed % digit_indices.size])
        pil_img, _ = mnist[pick]
        img = np.asarray(pil_img, dtype=np.float64)

        cloud = _digit_to_point_cloud(
            img,
            threshold=self.threshold,
            n_points=n_points,
            rng=rng,
        )
        return torch.from_numpy(cloud).to(torch.float64)

    def expected_h1(self, n_points: int) -> int:
        # Returning -1 sentinels the "unknown" case for digits whose
        # topology is ambiguous (digit 4). Callers that consult this
        # attribute should treat negative values as "do not assert".
        return _EXPECTED_H1.get(self.digit, -1)


@dataclass(frozen=True)
class MNISTPointCloudMock:
    """In-memory MNIST replacement for offline unit tests.

    Produces a fixed-shape point cloud with controllable topology so the
    test suite does not depend on the MNIST download. The protocol
    contract matches :class:`MNISTPointCloud`; the data does not.
    """

    digit: int = 0
    _name_suffix: str = "mock"

    @property
    def name(self) -> str:
        return f"mnist_mock_digit_{self.digit}"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        # Produce a configurable-loop point cloud for the requested digit.
        # Digits 0/6/9 -> one ring; 8 -> two rings; everything else -> blob.
        h1 = self.expected_h1(n_points)
        if h1 == 1:
            theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
            pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        elif h1 == 2:
            half = n_points // 2
            theta_a = np.linspace(0.0, 2.0 * np.pi, half, endpoint=False)
            theta_b = np.linspace(0.0, 2.0 * np.pi, n_points - half, endpoint=False)
            pts = np.concatenate([
                np.stack([np.cos(theta_a) - 2.0, np.sin(theta_a)], axis=1),
                np.stack([np.cos(theta_b) + 2.0, np.sin(theta_b)], axis=1),
            ], axis=0)
        else:
            pts = rng.standard_normal((n_points, 2))
        pts += 0.05 * rng.standard_normal(pts.shape)
        return torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float64))

    def expected_h1(self, n_points: int) -> int:
        return _EXPECTED_H1.get(self.digit, -1)


# Register the real-data datasets that the bench will run by default.
# We register one digit per Betti-number class so each axis exercises
# diverse topology without ballooning the cell count.
register_dataset(MNISTPointCloud(digit=0))  # expected beta_1 = 1
register_dataset(MNISTPointCloud(digit=1))  # expected beta_1 = 0
register_dataset(MNISTPointCloud(digit=8))  # expected beta_1 = 2

# Mocks for offline test runs — registered so the test suite can refer
# to them by name without requiring an MNIST download.
register_dataset(MNISTPointCloudMock(digit=0))
register_dataset(MNISTPointCloudMock(digit=1))
register_dataset(MNISTPointCloudMock(digit=8))
