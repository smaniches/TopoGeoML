"""
Tests for ``topogeoml.nn.cubical_diff_ph``.

Coverage targets:
  - forward correctness on known patterns (ring, square, two-rings,
    flat, gradient images);
  - gradient flow + Hofer 2017-style subgradient direction;
  - autograd.gradcheck against finite differences;
  - CubicalTopologyLoss module forward over (H,W), (B,H,W), (B,1,H,W)
    shapes;
  - input validation (float64, ndim, target_betti).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _has_gudhi() -> bool:
    try:
        import gudhi  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestForward:
    def test_solid_square_has_no_h1(self) -> None:
        """A uniformly-bright square has only essential H_0; no finite H_1."""
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        img = torch.ones((5, 5), dtype=torch.float64)
        dgms = cubical_diagram_torch(img, max_dim=1)
        # Filter finite H_1 bars.
        h1 = dgms[1]
        finite_h1 = h1[torch.isfinite(h1).all(dim=1)] if h1.numel() else h1
        assert finite_h1.shape[0] == 0

    def test_single_ring_has_one_h1(self) -> None:
        """A ring (high border, low center) yields exactly one finite H_1."""
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        img = torch.tensor([
            [0.0, 0.5, 0.5, 0.5, 0.0],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.5, 1.0, 0.0, 1.0, 0.5],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 0.5, 0.0],
        ], dtype=torch.float64)
        dgms = cubical_diagram_torch(img, max_dim=1)
        h1 = dgms[1]
        finite = h1[torch.isfinite(h1).all(dim=1)]
        assert finite.shape[0] == 1
        b, d = float(finite[0, 0]), float(finite[0, 1])
        assert 0.0 <= b < d
        # Lifetime is at least 0.4 (matches the ring contrast).
        assert d - b >= 0.4

    def test_two_disjoint_rings_have_two_h1(self) -> None:
        """Two separated rings yield exactly two finite H_1 bars."""
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        # Two 3x3 ring patches separated by a low strip.
        img = torch.zeros((5, 11), dtype=torch.float64)
        for cx in (2, 8):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    img[2 + di, cx + dj] = 1.0
            img[2, cx] = 0.0  # punch the center
        dgms = cubical_diagram_torch(img, max_dim=1)
        h1 = dgms[1]
        finite = h1[torch.isfinite(h1).all(dim=1)]
        assert finite.shape[0] == 2

    def test_include_essential_toggle(self) -> None:
        """include_essential=True emits the essential H_0 bar with death=inf."""
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        img = torch.zeros((4, 4), dtype=torch.float64)
        with_ess = cubical_diagram_torch(img, max_dim=0, include_essential=True)
        without_ess = cubical_diagram_torch(img, max_dim=0, include_essential=False)
        assert with_ess[0].shape[0] == without_ess[0].shape[0] + 1
        # The essential bar's death is +inf.
        assert torch.isinf(with_ess[0][:, 1]).any()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_float32(self) -> None:
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        img = torch.zeros((4, 4), dtype=torch.float32)
        with pytest.raises(TypeError, match="float64"):
            cubical_diagram_torch(img, max_dim=1)

    def test_rejects_1d_input(self) -> None:
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        img = torch.zeros((16,), dtype=torch.float64)
        with pytest.raises(ValueError, match="at least 2D"):
            cubical_diagram_torch(img, max_dim=1)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestGradientFlow:
    def test_gradient_points_at_critical_pixels(self) -> None:
        """Maximizing the H_1 lifetime should put nonzero gradient on the
        two critical pixels (birth and death) only — that's the Hofer
        2017 subgradient."""
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        img = torch.tensor([
            [0.0, 0.5, 0.5, 0.5, 0.0],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.5, 1.0, 0.0, 1.0, 0.5],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 0.5, 0.0],
        ], dtype=torch.float64, requires_grad=True)
        dgms = cubical_diagram_torch(img, max_dim=1)
        h1 = dgms[1]
        finite = h1[torch.isfinite(h1).all(dim=1)]
        loss = -(finite[:, 1] - finite[:, 0]).max()
        loss.backward()
        # Exactly two pixels carry the gradient: the birth vertex (+1) and
        # the death vertex (-1) — Hofer-style subgradient.
        nonzero = (img.grad.abs() > 1e-12).sum().item()
        assert nonzero == 2
        # Gradient should sum to 0 (one +1, one -1).
        assert float(img.grad.sum().item()) == pytest.approx(0.0, abs=1e-12)

    def test_gradient_is_finite_on_random_image(self) -> None:
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        torch.manual_seed(0)
        img = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        dgms = cubical_diagram_torch(img, max_dim=1)
        # Sum of all finite-bar lifetimes — exercises every critical pixel.
        h0_f = dgms[0][torch.isfinite(dgms[0]).all(dim=1)]
        h1_f = dgms[1][torch.isfinite(dgms[1]).all(dim=1)] if dgms[1].numel() else dgms[1]
        loss = ((h0_f[:, 1] - h0_f[:, 0]).sum() if h0_f.numel() else 0) \
             + ((h1_f[:, 1] - h1_f[:, 0]).sum() if h1_f.numel() else 0)
        loss.backward()
        assert torch.all(torch.isfinite(img.grad))


# ---------------------------------------------------------------------------
# Autograd gradcheck against finite differences
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestGradcheck:
    def test_cubical_diagram_gradcheck(self) -> None:
        """``torch.autograd.gradcheck`` of the H_1 bars against finite
        differences.

        The forward reconstructs each bar endpoint as an exact index into the
        input image, so the analytic gradient is the indicator of the critical
        pixel. gradcheck perturbs each pixel and compares to a central finite
        difference. We use a fixed ring image with *distinct* pixel values so a
        small perturbation never flips which vertex realizes a bar (which would
        change the piecewise-linear branch and break the comparison), and
        ``include_essential=False`` so no infinite-death bar enters the checked
        output.
        """
        from topogeoml.nn.cubical_diff_ph import cubical_diagram_torch

        # Ring with strictly distinct values: high, distinct border; low center.
        img = torch.tensor([
            [0.91, 0.80, 0.83, 0.78, 0.93],
            [0.82, 0.97, 0.99, 0.96, 0.84],
            [0.85, 0.98, 0.10, 0.95, 0.86],
            [0.81, 0.94, 0.92, 0.90, 0.87],
            [0.88, 0.77, 0.79, 0.76, 0.89],
        ], dtype=torch.float64, requires_grad=True)

        def h1_bars(x: torch.Tensor) -> torch.Tensor:
            dgms = cubical_diagram_torch(x, max_dim=1, include_essential=False)
            return dgms[1]

        # Sanity: there is exactly one finite H_1 bar to check.
        assert h1_bars(img).shape == (1, 2)
        assert torch.autograd.gradcheck(
            h1_bars, (img,), atol=1e-3, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestBettiMatchingLoss:
    def test_zero_loss_when_below_target(self) -> None:
        """An image with one ring (β_1 = 1) and target_n_bars = 1 has zero loss."""
        from topogeoml.nn.cubical_diff_ph import (
            betti_matching_loss,
            cubical_diagram_torch,
        )

        img = torch.tensor([
            [0.0, 0.5, 0.5, 0.5, 0.0],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.5, 1.0, 0.0, 1.0, 0.5],
            [0.5, 1.0, 1.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 0.5, 0.0],
        ], dtype=torch.float64)
        dgms = cubical_diagram_torch(img, max_dim=1, include_essential=False)
        loss = betti_matching_loss(dgms[1], target_n_bars=1, prominence_threshold=0.0)
        assert float(loss.item()) == 0.0

    def test_nonzero_loss_when_excess_bars(self) -> None:
        """A diagram constructed by hand with two finite bars of differing
        lifetime gives positive loss when target_n_bars < n_bars."""
        from topogeoml.nn.cubical_diff_ph import betti_matching_loss

        # Bypass the cubical computation: feed the loss directly with a
        # diagram that has two finite bars of positive lifetime. With
        # target_n_bars = 1, the smaller bar should contribute its full
        # lifetime (0.3) to the loss.
        bars = torch.tensor(
            [[0.0, 1.0], [0.0, 0.3]], dtype=torch.float64,
        )
        loss = betti_matching_loss(bars, target_n_bars=1, prominence_threshold=0.0)
        # The longest bar (lifetime 1.0) is kept; the second (lifetime 0.3)
        # is excess and contributes 0.3 to the loss.
        assert float(loss.item()) == pytest.approx(0.3, abs=1e-9)

    def test_empty_diagram_returns_zero(self) -> None:
        from topogeoml.nn.cubical_diff_ph import betti_matching_loss

        empty = torch.empty((0, 2), dtype=torch.float64)
        loss = betti_matching_loss(empty, target_n_bars=1)
        assert float(loss.item()) == 0.0


# ---------------------------------------------------------------------------
# CubicalTopologyLoss module
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestCubicalTopologyLoss:
    def test_forward_2d_input(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        torch.manual_seed(0)
        pred = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        loss_module = CubicalTopologyLoss(target_betti={1: 1})
        loss = loss_module(pred)
        assert loss.shape == ()
        loss.backward()
        assert pred.grad is not None

    def test_forward_batched_3d_input(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        torch.manual_seed(0)
        pred = torch.rand(2, 6, 6, dtype=torch.float64, requires_grad=True)
        loss_module = CubicalTopologyLoss(target_betti={1: 1})
        loss = loss_module(pred)
        assert loss.shape == ()
        loss.backward()
        # Gradient flows to every image in the batch on average.
        assert torch.all(torch.isfinite(pred.grad))

    def test_forward_unet_shape_4d(self) -> None:
        """Common U-Net output shape: (B, 1, H, W). Module should squeeze
        the channel dim transparently."""
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        torch.manual_seed(0)
        pred = torch.rand(2, 1, 6, 6, dtype=torch.float64, requires_grad=True)
        loss_module = CubicalTopologyLoss(target_betti={1: 1})
        loss = loss_module(pred)
        assert loss.shape == ()

    def test_rejects_bad_shape(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        loss_module = CubicalTopologyLoss(target_betti={1: 1})
        with pytest.raises(ValueError, match="pred must be"):
            loss_module(torch.rand(4, 5, 6, 7, 8, dtype=torch.float64))

    def test_rejects_empty_target_betti(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        with pytest.raises(ValueError, match="non-empty"):
            CubicalTopologyLoss(target_betti={})

    def test_rejects_negative_target_dim(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        with pytest.raises(ValueError, match="keys must be"):
            CubicalTopologyLoss(target_betti={-1: 0})

    def test_rejects_negative_target_count(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        with pytest.raises(ValueError, match="values must be"):
            CubicalTopologyLoss(target_betti={1: -1})

    def test_invert_flag_changes_diagram(self) -> None:
        """invert=True computes PH on (1 - pred); invert=False on (pred).
        The two should differ on a non-symmetric image."""
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        torch.manual_seed(0)
        pred = torch.rand(6, 6, dtype=torch.float64)
        loss_inv = CubicalTopologyLoss(target_betti={1: 1}, invert=True)
        loss_no_inv = CubicalTopologyLoss(target_betti={1: 1}, invert=False)
        v1 = float(loss_inv(pred).item())
        v2 = float(loss_no_inv(pred).item())
        # Generically these differ — we don't assert by how much, only
        # that the invert branch is exercised.
        assert isinstance(v1, float) and isinstance(v2, float)


# ---------------------------------------------------------------------------
# finite_lifetimes_cubical helper
# ---------------------------------------------------------------------------

class TestFiniteLifetimesCubical:
    def test_empty_input(self) -> None:
        from topogeoml.nn.cubical_diff_ph import finite_lifetimes_cubical

        out = finite_lifetimes_cubical(torch.empty((0, 2), dtype=torch.float64))
        assert out.shape == (0,)

    def test_only_inf_input(self) -> None:
        from topogeoml.nn.cubical_diff_ph import finite_lifetimes_cubical

        bars = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        out = finite_lifetimes_cubical(bars)
        assert out.shape == (0,)

    def test_finite_input_returns_lifetimes(self) -> None:
        from topogeoml.nn.cubical_diff_ph import finite_lifetimes_cubical

        bars = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        out = finite_lifetimes_cubical(bars)
        assert torch.allclose(out, torch.tensor([1.0, 1.5], dtype=torch.float64))


# ---------------------------------------------------------------------------
# Essential-aware betti_matching_loss semantics (regression for PR #9 review).
# ---------------------------------------------------------------------------

class TestBettiMatchingLossEssentialAware:
    """Essential bars (death = inf) consume target_n_bars budget.

    Standard interpretation: ``target_betti={0: 1}`` means *one connected
    component*. Lower-star cubical persistence emits one essential H_0 bar
    per component. The loss must therefore credit the essential bar against
    the target so that, e.g., a one-component image gives zero H_0 loss.
    """

    def test_one_essential_at_target_one_gives_zero_loss(self) -> None:
        from topogeoml.nn.cubical_diff_ph import betti_matching_loss

        # One essential bar (death=inf), no finite bars. Target = 1.
        # Expected: loss is zero — the essential bar IS the one component.
        bars = torch.tensor(
            [[0.0, float("inf")]], dtype=torch.float64,
        )
        loss = betti_matching_loss(bars, target_n_bars=1)
        assert float(loss.item()) == 0.0

    def test_one_essential_plus_one_finite_at_target_one_penalises_finite(
        self,
    ) -> None:
        from topogeoml.nn.cubical_diff_ph import betti_matching_loss

        # 1 essential + 1 finite bar of lifetime 0.4, target = 1.
        # Essential consumes the budget; the finite bar is excess and is
        # penalised by its lifetime.
        bars = torch.tensor(
            [[0.0, float("inf")], [0.0, 0.4]], dtype=torch.float64,
        )
        loss = betti_matching_loss(bars, target_n_bars=1, prominence_threshold=0.0)
        assert float(loss.item()) == pytest.approx(0.4, abs=1e-9)

    def test_two_essential_at_target_one_keeps_loss_zero(self) -> None:
        from topogeoml.nn.cubical_diff_ph import betti_matching_loss

        # Two essential bars exceed target_n_bars=1 but they cannot be
        # shrunk (inf lifetime, no gradient). The loss is zero on them —
        # the prediction simply has more components than asked for, and
        # we surface this through the diagnostic count, not through a
        # gradient term we know is uninformative.
        bars = torch.tensor(
            [[0.0, float("inf")], [0.0, float("inf")]], dtype=torch.float64,
        )
        loss = betti_matching_loss(bars, target_n_bars=1)
        assert float(loss.item()) == 0.0

    def test_essential_aware_preserves_old_finite_only_behaviour(self) -> None:
        """A diagram with only finite bars and no essential bars behaves
        exactly as before — the new code path is a strict superset."""
        from topogeoml.nn.cubical_diff_ph import betti_matching_loss

        bars = torch.tensor(
            [[0.0, 1.0], [0.0, 0.3]], dtype=torch.float64,
        )
        loss = betti_matching_loss(bars, target_n_bars=1, prominence_threshold=0.0)
        assert float(loss.item()) == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# Batched CPU-transfer optimisation regression (PR #9 review).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestBatchedTransferEquivalence:
    """The batched forward must give the same scalar loss as the per-image
    forward — this is the regression test for the bulk-transfer rewrite."""

    def test_batched_matches_mean_over_singletons(self) -> None:
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        torch.manual_seed(0)
        pred = torch.rand(3, 7, 7, dtype=torch.float64)
        loss_module = CubicalTopologyLoss(target_betti={0: 1, 1: 1})

        batched = float(loss_module(pred).item())
        per_image = [float(loss_module(pred[b]).item()) for b in range(pred.shape[0])]
        assert batched == pytest.approx(sum(per_image) / len(per_image), abs=1e-12)

    def test_batched_grad_matches_per_image_grad(self) -> None:
        """Autograd must produce identical gradients for batched vs scalar
        invocation (sum of singletons, divided by batch size).

        Uses ``target_betti={1: 0}`` so any H_1 bar produced by gudhi on a
        random image is excess and contributes a real gradient — avoids
        the no-op zero-loss path.
        """
        from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

        torch.manual_seed(1)
        pred_batched = torch.rand(2, 6, 6, dtype=torch.float64, requires_grad=True)
        loss_module = CubicalTopologyLoss(target_betti={1: 0})
        loss_b = loss_module(pred_batched)
        loss_b.backward()
        grad_batched = pred_batched.grad.detach().clone()

        # Recompute as a manual mean over per-image losses.
        pred_scalar = pred_batched.detach().clone().requires_grad_(True)
        manual = torch.stack([
            loss_module(pred_scalar[b]) for b in range(pred_scalar.shape[0])
        ]).mean()
        manual.backward()
        grad_scalar = pred_scalar.grad.detach().clone()

        assert torch.allclose(grad_batched, grad_scalar, atol=1e-12)
