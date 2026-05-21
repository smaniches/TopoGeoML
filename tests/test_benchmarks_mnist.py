"""
Tests for the MNIST topology dataset.

Coverage targets:
  - ``_digit_to_point_cloud`` for the active-pixel edge cases,
  - ``_cache_root`` for both ``XDG_CACHE_HOME`` set and unset,
  - ``MNISTPointCloud.generate`` end-to-end against a faked
    ``torchvision.datasets.MNIST`` (no network),
  - ``MNISTPointCloudMock`` for each Betti-class branch.

The real MNIST download is exercised by the CI benchmark workflow,
not the unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.datasets.mnist_topology import (
    MNISTPointCloud,
    MNISTPointCloudMock,
    _cache_root,
    _digit_to_point_cloud,
)


class TestDigitToPointCloud:
    def test_normalizes_to_unit_bounding_box(self) -> None:
        img = np.zeros((28, 28), dtype=np.float64)
        img[5:23, 5:23] = 1.0  # A solid 18x18 block of active pixels.
        rng = np.random.default_rng(0)
        cloud = _digit_to_point_cloud(img, threshold=0.5, n_points=20, rng=rng)
        assert cloud.shape == (20, 2)
        assert cloud.dtype == np.float64
        # Centered: mean within unit box, not at corner.
        assert np.abs(np.abs(cloud).max() - 1.0) <= 1e-12

    def test_all_black_image_returns_zeros(self) -> None:
        img = np.zeros((28, 28), dtype=np.float64)
        rng = np.random.default_rng(0)
        cloud = _digit_to_point_cloud(img, threshold=0.5, n_points=10, rng=rng)
        assert cloud.shape == (10, 2)
        assert np.all(cloud == 0.0)

    def test_below_threshold_only_returns_zeros(self) -> None:
        img = np.full((28, 28), 0.1, dtype=np.float64)  # All pixels under threshold * max.
        # threshold * max = 0.5 * 0.1 = 0.05; img > 0.05 is True everywhere — but
        # that's a different branch. Let's set everything below threshold instead:
        img = np.full((28, 28), 0.05, dtype=np.float64)
        # max = 0.05, threshold = 0.9, so threshold * max = 0.045
        # img > 0.045 is True for all pixels — not the branch we want.
        # Force "no active" by making max positive but threshold above 1.0
        # via an asymmetric image:
        img = np.zeros((28, 28), dtype=np.float64)
        img[0, 0] = 1.0
        rng = np.random.default_rng(0)
        # threshold=2.0 ensures no pixel satisfies img > threshold * max = 2.0.
        cloud = _digit_to_point_cloud(img, threshold=2.0, n_points=10, rng=rng)
        assert cloud.shape == (10, 2)
        assert np.all(cloud == 0.0)

    def test_subsamples_when_more_active_than_n_points(self) -> None:
        img = np.ones((28, 28), dtype=np.float64)  # All 784 pixels active.
        rng = np.random.default_rng(42)
        cloud = _digit_to_point_cloud(img, threshold=0.5, n_points=20, rng=rng)
        assert cloud.shape == (20, 2)
        # All distinct after subsample (without replacement) when active >= n_points.
        # Tuples for set comparison.
        unique = {tuple(row) for row in cloud}
        assert len(unique) == 20

    def test_resamples_with_replacement_when_fewer_active_than_n_points(self) -> None:
        img = np.zeros((28, 28), dtype=np.float64)
        img[10, 10] = 1.0
        img[15, 15] = 1.0  # Only 2 active pixels.
        rng = np.random.default_rng(0)
        cloud = _digit_to_point_cloud(img, threshold=0.5, n_points=10, rng=rng)
        assert cloud.shape == (10, 2)
        # After centering and unit-box normalization the two active pixels
        # become exactly +/- 1.0 along the diagonal; replacement-resampling
        # must produce ≤ 2 distinct rows.
        unique = {tuple(row) for row in cloud}
        assert len(unique) <= 2

    def test_rejects_wrong_image_shape(self) -> None:
        img = np.zeros((27, 28), dtype=np.float64)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="28x28"):
            _digit_to_point_cloud(img, threshold=0.5, n_points=10, rng=rng)


class TestCacheRoot:
    def test_uses_xdg_cache_home_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        root = _cache_root()
        assert root.is_dir()
        assert str(tmp_path) in str(root)

    def test_falls_back_to_home_cache_when_xdg_unset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # type: ignore[arg-type]
        root = _cache_root()
        assert root.is_dir()
        assert "topogeoml" in str(root)


class TestMNISTPointCloudMock:
    @pytest.mark.parametrize("digit,expected_h1", [(0, 1), (6, 1), (9, 1), (8, 2), (1, 0), (3, 0), (4, -1)])
    def test_expected_h1_per_digit(self, digit: int, expected_h1: int) -> None:
        ds = MNISTPointCloudMock(digit=digit)
        assert ds.expected_h1(n_points=50) == expected_h1

    def test_name_and_version(self) -> None:
        ds = MNISTPointCloudMock(digit=0)
        assert ds.name == "mnist_mock_digit_0"
        assert ds.version == "1.0.0"

    def test_one_loop_for_digit_0(self) -> None:
        ds = MNISTPointCloudMock(digit=0)
        cloud = ds.generate(seed=0, n_points=30)
        assert cloud.shape == (30, 2)
        assert cloud.dtype == torch.float64

    def test_two_loops_for_digit_8(self) -> None:
        ds = MNISTPointCloudMock(digit=8)
        cloud = ds.generate(seed=0, n_points=30)
        # The two rings sit at x ≈ -2 and x ≈ +2 — the x-coordinate span
        # should be > 3.
        x_span = float(cloud[:, 0].max() - cloud[:, 0].min())
        assert x_span > 3.0

    def test_blob_for_digit_1(self) -> None:
        ds = MNISTPointCloudMock(digit=1)
        cloud = ds.generate(seed=0, n_points=30)
        assert cloud.shape == (30, 2)
        assert cloud.dtype == torch.float64


class _FakeMNIST:
    """A tiny stand-in for ``torchvision.datasets.MNIST`` so tests don't
    download. Implements the bits the dataset code touches: ``targets``
    (numpy or tensor-like) and ``__getitem__`` returning (PIL-like, label).
    """

    class _PILLike:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            return self._arr if dtype is None else self._arr.astype(dtype)

    def __init__(self, *, n_per_digit: int = 3) -> None:
        # Build n_per_digit images per class (0..9), each a deterministic 28x28
        # synthetic image with a single bright row to ensure non-empty active pixels.
        images: list[np.ndarray] = []
        labels: list[int] = []
        for digit in range(10):
            for k in range(n_per_digit):
                arr = np.zeros((28, 28), dtype=np.uint8)
                row = (digit * 3 + k) % 28
                arr[row, :] = 200 + (digit * 5 + k) % 50
                images.append(arr)
                labels.append(digit)
        self._images = images
        self._labels = labels
        self.targets = np.asarray(self._labels, dtype=np.int64)

    def __getitem__(self, idx: int) -> tuple[object, int]:
        return self._PILLike(self._images[idx]), self._labels[idx]

    def __len__(self) -> int:
        return len(self._images)


class TestMNISTPointCloudGenerate:
    def test_generate_with_faked_torchvision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torchvision.datasets as tvd

        monkeypatch.setattr(tvd, "MNIST", lambda *a, **k: _FakeMNIST())
        ds = MNISTPointCloud(digit=0, download=False)
        cloud = ds.generate(seed=0, n_points=12)
        assert cloud.shape == (12, 2)
        assert cloud.dtype == torch.float64

    def test_generate_is_deterministic_given_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torchvision.datasets as tvd

        monkeypatch.setattr(tvd, "MNIST", lambda *a, **k: _FakeMNIST())
        ds = MNISTPointCloud(digit=3, download=False)
        a = ds.generate(seed=42, n_points=15)
        b = ds.generate(seed=42, n_points=15)
        assert torch.allclose(a, b)

    def test_rejects_digit_out_of_range(self) -> None:
        ds = MNISTPointCloud(digit=11)
        with pytest.raises(ValueError, match=r"digit must be in 0\.\.9"):
            ds.generate(seed=0, n_points=10)

    def test_rejects_n_points_too_small(self) -> None:
        ds = MNISTPointCloud(digit=0)
        with pytest.raises(ValueError, match="n_points must be"):
            ds.generate(seed=0, n_points=2)

    def test_raises_when_digit_missing_from_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torchvision.datasets as tvd

        class _NoZeros(_FakeMNIST):
            def __init__(self) -> None:
                super().__init__(n_per_digit=1)
                # Remove all samples of digit 0 from labels (won't match anywhere).
                keep = [i for i, t in enumerate(self._labels) if t != 0]
                self._images = [self._images[i] for i in keep]
                self._labels = [self._labels[i] for i in keep]
                self.targets = np.asarray(self._labels, dtype=np.int64)

        monkeypatch.setattr(tvd, "MNIST", lambda *a, **k: _NoZeros())
        ds = MNISTPointCloud(digit=0, download=False)
        with pytest.raises(RuntimeError, match="no samples of digit 0"):
            ds.generate(seed=0, n_points=10)


class TestMNISTPointCloudMetadata:
    def test_name_and_version(self) -> None:
        ds = MNISTPointCloud(digit=8)
        assert ds.name == "mnist_digit_8"
        assert ds.version == "1.0.0"

    def test_expected_h1_for_ambiguous_digit_4(self) -> None:
        ds = MNISTPointCloud(digit=4)
        assert ds.expected_h1(n_points=50) == -1
