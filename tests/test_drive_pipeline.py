"""
Tests for the DRIVE U-Net topology-loss training pipeline.

The full pipeline is exercised only on GPU in the integration tests;
these are lightweight unit tests covering imports, the synthetic-data
generator, and the helper functions.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestDrivePipelineImports:
    def test_module_imports(self) -> None:
        import notebooks.drive_unet_topology_loss as mod

        assert callable(mod.main)
        assert callable(mod._build_unet)
        assert callable(mod._dice_bce_loss)


class TestSyntheticDataGenerator:
    def test_synthetic_shape_contract(self) -> None:
        from notebooks.drive_unet_topology_loss import _synthetic_vessel_data

        train, test = _synthetic_vessel_data(seed=0, n_train=4, n_test=2, image_size=16)
        assert len(train) == 4
        assert len(test) == 2
        img, mask = train[0]
        assert img.shape == (16, 16)
        assert mask.shape == (16, 16)
        assert img.dtype == np.float32
        assert mask.dtype == np.float32
        # Mask is binarized in [0, 1].
        assert ((mask == 0.0) | (mask == 1.0)).all()


def _has_gudhi() -> bool:
    try:
        import gudhi  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestUNetForward:
    def test_unet_forward_shape(self) -> None:
        from notebooks.drive_unet_topology_loss import _build_unet

        model = _build_unet(in_channels=1, base=4).to(torch.float32)
        # 16x16 is the smallest size that keeps the 3-level U-Net well-defined.
        x = torch.randn(1, 1, 16, 16, dtype=torch.float32)
        y = model(x)
        assert y.shape == (1, 1, 16, 16)
        # Output is sigmoid-activated, so within [0, 1].
        assert float(y.min().item()) >= 0.0
        assert float(y.max().item()) <= 1.0


class TestLossHelpers:
    def test_dice_bce_loss_on_perfect_prediction(self) -> None:
        from notebooks.drive_unet_topology_loss import _dice_bce_loss

        target = torch.zeros(1, 1, 8, 8)
        target[..., 2:6, 2:6] = 1.0
        # Predict exactly the target (clamped away from 0/1 for BCE stability).
        pred = torch.clamp(target, 1e-6, 1.0 - 1e-6)
        loss = _dice_bce_loss(pred, target)
        # Lower than the loss on a uniform 0.5 prediction.
        uniform = torch.full_like(target, 0.5)
        loss_uniform = _dice_bce_loss(uniform, target)
        assert float(loss.item()) < float(loss_uniform.item())

    def test_iou_perfect_and_zero(self) -> None:
        from notebooks.drive_unet_topology_loss import _iou

        target = torch.zeros(8, 8)
        target[2:6, 2:6] = 1.0
        # Perfect overlap.
        assert _iou(target, target) == pytest.approx(1.0)
        # No overlap.
        zero = torch.zeros(8, 8)
        assert _iou(zero, target) == 0.0
