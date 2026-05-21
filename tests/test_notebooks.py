"""
Tests for the GPU bench + MNIST topology demo scripts.

The scripts are designed to be runnable end-to-end (Colab / Modal / CLI),
so the tests focus on import correctness, schema contracts, and the
cheap helper functions. The full training loop is exercised in the
``[gpu]`` integration tests when GPU is available.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestGPUBenchScript:
    def test_module_imports(self) -> None:
        """``notebooks.diff_ph_bench_gpu`` should import without side effects."""
        import notebooks.diff_ph_bench_gpu as mod

        assert callable(mod.main)


class TestModalScript:
    def test_module_imports_without_modal_installed(self) -> None:
        """The script should be importable even if modal-client is absent;
        the modal import happens lazily inside the helper functions."""
        import scripts.modal_diff_ph_bench as mod

        assert callable(mod.main)
        assert callable(mod.define_app)


class TestMNISTTopologyScript:
    def test_module_imports(self) -> None:
        import notebooks.mnist_topology_classification as mod

        assert callable(mod.main)
        assert callable(mod.build_models)
        assert callable(mod.train_eval_one_seed)

    def test_seed_result_dataclass(self) -> None:
        from notebooks.mnist_topology_classification import SeedResult

        r = SeedResult(
            seed=0, topology_accuracy=0.5, baseline_accuracy=0.4,
            topology_train_loss=1.0, baseline_train_loss=1.1,
        )
        assert r.seed == 0
        assert r.topology_accuracy == 0.5

    def test_build_models_constructs_both(self) -> None:
        """Smoke check that the model factory returns two parameterized modules
        with sensible param counts."""
        from notebooks.mnist_topology_classification import build_models

        topo, base = build_models(input_dim=2, num_classes=3, seed=0)
        n_topo = sum(p.numel() for p in topo.parameters())
        n_base = sum(p.numel() for p in base.parameters())
        # Same parameter budget (32 hidden dims + 3->16 vs 2->16) — within 10%.
        assert 0.9 <= n_topo / n_base <= 1.1, (
            f"param counts not comparable: topo={n_topo}, base={n_base}"
        )

    def test_topology_aware_forward_shape(self) -> None:
        """Topology-aware model forward returns a 3-class logit vector."""
        from notebooks.mnist_topology_classification import build_models

        topo, _ = build_models(input_dim=2, num_classes=3, seed=0)
        # 10-point unit circle: should have a finite H_1 bar.
        import numpy as np

        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False, dtype=np.float64)
        cloud = torch.from_numpy(
            np.stack([np.cos(theta), np.sin(theta)], axis=1)
        )
        logits = topo(cloud)
        assert logits.shape == (3,)
