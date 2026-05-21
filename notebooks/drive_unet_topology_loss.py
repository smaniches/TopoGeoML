"""
DRIVE retinal-vessel segmentation with diff-PH cubical topology loss.

The killer demo for ``topogeoml.nn.cubical_diff_ph``. Trains a small
U-Net on the DRIVE benchmark (Staal et al. 2004; 40 fundus images,
binary vessel segmentation) under two losses:

  1. Dice + BCE baseline.
  2. Dice + BCE + lambda * CubicalTopologyLoss.

Reports the IoU difference with bootstrap CI and paired Wilcoxon
significance test, using the same ``benchmarks.stats`` machinery the
rest of the framework uses.

Why DRIVE
---------
DRIVE is the canonical published benchmark for topology-aware
segmentation methods (Clough et al. 2020; Hu et al. 2019; Mosinska et
al. 2018). Vessels are connected, thin, branching structures — the
exact topology a Dice/BCE loss tends to break (vessel disconnections,
fictitious loops). Adding a topology loss should *improve* IoU on the
vessel-skeleton metric.

How to get DRIVE
----------------
DRIVE is hosted at https://drive.grand-challenge.org/ and requires
registering with an institutional email. Once you have the data, unzip
into ``$XDG_CACHE_HOME/topogeoml/drive/`` (or pass ``--data-root``).
Expected layout::

    drive/
    ├── training/
    │   ├── images/      *.tif
    │   ├── 1st_manual/  *.gif
    └── test/
        ├── images/
        ├── 1st_manual/

A synthetic fallback (``--synthetic``) generates noisy circle-with-vessel
images so the pipeline can be smoke-tested without DRIVE.

Invocation
----------
    # Real DRIVE (assumes data is downloaded into ~/.cache/topogeoml/drive/):
    python notebooks/drive_unet_topology_loss.py \\
        --seeds 0 1 2 3 4 \\
        --n-epochs 50 \\
        --topo-weight 0.1 \\
        --output /tmp/drive_results.json

    # Synthetic fallback (smoke test):
    python notebooks/drive_unet_topology_loss.py --synthetic \\
        --seeds 0 1 --n-epochs 2

Cost
----
On a free Colab T4: ~40 min for 50 epochs, 5 seeds, both models.
On Modal a10g: ~20 min, ~$0.40.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch as _torch_typing


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

def _drive_root(override: Path | None) -> Path:
    if override is not None:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "topogeoml" / "drive"


def _load_drive(
    data_root: Path, image_size: int = 128
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Load DRIVE training and test sets as (image, mask) numpy pairs.

    Images are resized to ``image_size × image_size`` and normalized to
    float32 in [0, 1]. Masks are binarized.
    """
    from PIL import Image

    def _load_dir(split: str) -> list[tuple[np.ndarray, np.ndarray]]:
        img_dir = data_root / split / "images"
        mask_dir = data_root / split / "1st_manual"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"DRIVE {split} images not found at {img_dir}. See module "
                "docstring for download instructions."
            )
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for img_path in sorted(img_dir.glob("*.tif")):
            stem = img_path.stem.split("_")[0]
            mask_paths = sorted(mask_dir.glob(f"{stem}_*"))
            if not mask_paths:
                continue
            img = Image.open(img_path).convert("L").resize((image_size, image_size))
            mask = Image.open(mask_paths[0]).convert("L").resize((image_size, image_size))
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
            mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
            pairs.append((img_arr, mask_arr))
        return pairs

    return _load_dir("training"), _load_dir("test")


def _synthetic_vessel_data(
    seed: int, n_train: int = 16, n_test: int = 4, image_size: int = 32,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Synthetic vessel-like images for smoke-testing the pipeline.

    Each image has a few branching curves (the 'vessels') on a noisy
    background. Mask = pixels with vessel.

    NOT a substitute for DRIVE — only for verifying the training loop
    runs end-to-end before invoking on real data.
    """
    rng = np.random.default_rng(seed)

    def _one(s: int) -> tuple[np.ndarray, np.ndarray]:
        rng_l = np.random.default_rng(s)
        mask = np.zeros((image_size, image_size), dtype=np.float32)
        n_curves = rng_l.integers(2, 4)
        for _ in range(int(n_curves)):
            x = rng_l.uniform(2, image_size - 2)
            y = rng_l.uniform(2, image_size - 2)
            theta = rng_l.uniform(0, 2 * np.pi)
            length = rng_l.integers(image_size // 4, image_size // 2 + 1)
            for _t in range(int(length)):
                x += np.cos(theta)
                y += np.sin(theta)
                theta += rng_l.normal(0, 0.1)
                xi, yi = round(x), round(y)
                if 0 <= xi < image_size and 0 <= yi < image_size:
                    mask[xi, yi] = 1.0
        img = mask * 0.8 + 0.1 + rng_l.normal(0, 0.05, mask.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img, mask

    train = [_one(int(rng.integers(0, 2**31 - 1))) for _ in range(n_train)]
    test = [_one(int(rng.integers(0, 2**31 - 1))) for _ in range(n_test)]
    return train, test


# ---------------------------------------------------------------------------
# Small U-Net
# ---------------------------------------------------------------------------

def _build_unet(in_channels: int = 1, base: int = 16) -> _torch_typing.nn.Module:
    """A minimal 3-level U-Net suitable for the DRIVE bench.

    Channel progression: in_channels -> base -> 2*base -> 4*base -> ... -> 1.
    """
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self, in_c: int, out_c: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = _Block(in_channels, base)
            self.enc2 = _Block(base, base * 2)
            self.enc3 = _Block(base * 2, base * 4)
            self.bottle = _Block(base * 4, base * 8)
            self.dec3 = _Block(base * 8 + base * 4, base * 4)
            self.dec2 = _Block(base * 4 + base * 2, base * 2)
            self.dec1 = _Block(base * 2 + base, base)
            self.out = nn.Conv2d(base, 1, kernel_size=1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottle(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
            return torch.sigmoid(self.out(d1))

    return _UNet()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _dice_bce_loss(pred: _torch_typing.Tensor, target: _torch_typing.Tensor) -> _torch_typing.Tensor:
    """Standard Dice + BCE loss. Both inputs in [0, 1]."""
    from torch import nn

    bce = nn.functional.binary_cross_entropy(pred, target)
    # Dice loss with smoothing.
    eps = 1e-6
    p = pred.flatten()
    t = target.flatten()
    intersection = (p * t).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


def _iou(pred_binary: _torch_typing.Tensor, target: _torch_typing.Tensor) -> float:
    """IoU on a binarized prediction."""
    p = pred_binary.flatten().bool()
    t = target.flatten().bool()
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedResult:
    seed: int
    baseline_iou: float
    topology_iou: float
    baseline_dice_loss_final: float
    topology_loss_final: float


def _train_one_seed(
    *,
    seed: int,
    train: list[tuple[np.ndarray, np.ndarray]],
    test: list[tuple[np.ndarray, np.ndarray]],
    n_epochs: int,
    learning_rate: float,
    topo_weight: float,
    topo_resolution: int,
) -> SeedResult:
    import torch

    from topogeoml.nn.cubical_diff_ph import CubicalTopologyLoss

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[_torch_typing.Tensor, _torch_typing.Tensor]:
        x = torch.from_numpy(np.stack([p[0] for p in pairs])).unsqueeze(1).to(device).float()
        y = torch.from_numpy(np.stack([p[1] for p in pairs])).unsqueeze(1).to(device).float()
        return x, y

    x_train, y_train = _to_tensor(train)
    x_test, y_test = _to_tensor(test)

    def _train(use_topology: bool) -> tuple[float, float]:
        torch.manual_seed(seed)
        model = _build_unet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        topo_loss = CubicalTopologyLoss(
            target_betti={1: 0}, prominence_threshold=0.05, invert=True,
        ).to(device) if use_topology else None
        final_loss = float("nan")
        for _ in range(n_epochs):
            model.train()
            opt.zero_grad()
            pred = model(x_train)
            dice = _dice_bce_loss(pred, y_train)
            if use_topology and topo_loss is not None:
                # Resolution at which the topology loss is computed. For
                # fine structures (retinal vessels, 1-3 px wide at 128x128)
                # aggressive downsampling collapses the foreground and
                # invents fictitious topology, so we keep it tunable and
                # default to the full prediction resolution above.
                if topo_resolution < pred.shape[-1]:
                    topo_input = torch.nn.functional.interpolate(
                        pred, size=(topo_resolution, topo_resolution),
                        mode="bilinear", align_corners=False,
                    )
                else:
                    topo_input = pred
                topo = topo_loss(topo_input.to(torch.float64)).to(dice.dtype)
                loss = dice + topo_weight * topo
            else:
                loss = dice
            loss.backward()
            opt.step()
            final_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            pred_test = model(x_test)
            binary = (pred_test > 0.5).float()
            ious = [
                _iou(binary[i], y_test[i]) for i in range(binary.shape[0])
            ]
        return float(np.mean(ious)), final_loss

    base_iou, base_loss = _train(use_topology=False)
    topo_iou, topo_loss_final = _train(use_topology=True)

    return SeedResult(
        seed=seed,
        baseline_iou=base_iou,
        topology_iou=topo_iou,
        baseline_dice_loss_final=base_loss,
        topology_loss_final=topo_loss_final,
    )


def _render_markdown(results: list[SeedResult]) -> str:
    from benchmarks.stats import bootstrap_ci, compare_paired

    base = np.asarray([r.baseline_iou for r in results])
    topo = np.asarray([r.topology_iou for r in results])

    lines = [
        "# DRIVE retinal-vessel segmentation: diff-PH topology loss vs Dice+BCE baseline",
        "",
        f"- Seeds: {len(results)}",
    ]
    if base.size >= 2:
        b_ci = bootstrap_ci(base, statistic="median", n_resamples=10_000, seed=0)
        t_ci = bootstrap_ci(topo, statistic="median", n_resamples=10_000, seed=0)
        lines.append(
            f"- Baseline (Dice+BCE) IoU: {b_ci.point_estimate:.4f} "
            f"[{b_ci.ci_low:.4f}, {b_ci.ci_high:.4f}]"
        )
        lines.append(
            f"- + CubicalTopologyLoss IoU: {t_ci.point_estimate:.4f} "
            f"[{t_ci.ci_low:.4f}, {t_ci.ci_high:.4f}]"
        )
        cmp = compare_paired(
            topo, base,
            arm_a_name="+CubicalTopologyLoss", arm_b_name="Dice+BCE-baseline",
        )
        kind = cmp.kind.value if hasattr(cmp.kind, "value") else cmp.kind
        lines.append(
            f"- Paired Wilcoxon: median Δ = {cmp.median_diff:+.4f}, "
            f"p_raw = {cmp.p_value_raw:.3e}, effect (r) = {cmp.effect_size:+.3f}, "
            f"verdict = {kind}"
        )
    else:
        lines.append(f"- Baseline median IoU: {float(np.median(base)):.4f}")
        lines.append(f"- + topo median IoU: {float(np.median(topo)):.4f}")

    lines.append("")
    lines.append("## Per-seed IoU")
    lines.append("")
    lines.append("| seed | baseline IoU | +topo IoU | Δ |")
    lines.append("|---|---|---|---|")
    for r in results:
        delta = r.topology_iou - r.baseline_iou
        lines.append(f"| {r.seed} | {r.baseline_iou:.4f} | {r.topology_iou:.4f} | {delta:+.4f} |")
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser. Factored out so tests can introspect the
    argument set without running the training loop."""
    parser = argparse.ArgumentParser(description="DRIVE U-Net + topology loss training")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--topo-weight", type=float, default=0.1)
    parser.add_argument(
        "--topo-resolution",
        type=int,
        default=64,
        help=(
            "Resolution at which the cubical topology loss is computed. "
            "Set to >= --image-size to skip downsampling. The default 64 "
            "balances per-iteration cost against the loss of fine "
            "vessel structure that aggressive downsampling causes "
            "(retinal vessels are 1-3 px wide at 128x128)."
        ),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic vessel-like data instead of DRIVE (smoke test)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/drive_topology_results.json"))
    parser.add_argument("--markdown", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    import torch
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}; CUDA={torch.cuda.is_available()}")

    if args.synthetic:
        print("Using synthetic data (smoke-test path; NOT DRIVE results).")
        train, test = _synthetic_vessel_data(seed=0, image_size=args.image_size)
    else:
        data_root = _drive_root(args.data_root)
        print(f"Loading DRIVE from {data_root}")
        train, test = _load_drive(data_root, image_size=args.image_size)
    print(f"Train: {len(train)}, Test: {len(test)}")

    results: list[SeedResult] = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        r = _train_one_seed(
            seed=seed,
            train=train, test=test,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            topo_weight=args.topo_weight,
            topo_resolution=args.topo_resolution,
        )
        dt = time.perf_counter() - t0
        results.append(r)
        print(
            f"seed={seed:3d}  base={r.baseline_iou:.4f}  "
            f"+topo={r.topology_iou:.4f}  Δ={r.topology_iou - r.baseline_iou:+.4f}"
            f"  ({dt:.1f}s)"
        )

    md = _render_markdown(results)
    print()
    print(md)

    payload: dict[str, Any] = {
        "config": {
            "seeds": args.seeds, "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate, "topo_weight": args.topo_weight,
            "topo_resolution": args.topo_resolution,
            "image_size": args.image_size, "synthetic": args.synthetic,
        },
        "results": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
