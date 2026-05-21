"""
GPU runner for the TopoGeoML differentiable-PH benchmark.

This file is the source-of-truth for the GPU bench. It can be:

  1. Invoked directly:   python notebooks/diff_ph_bench_gpu.py
  2. Pasted into Colab:  see notebooks/diff_ph_bench_gpu.ipynb (auto-generated)
  3. Run on Modal:       see scripts/modal_diff_ph_bench.py

All three paths execute the same code. The script invokes
``benchmarks.runner.run`` with CUDA-aware seeds, writes a JSON
leaderboard, and renders a markdown summary.

Why a GPU run is interesting
----------------------------
Both registered diff-PH backends (`topogeoml-diff-ph` and
`torch-topological`) target GPU. The CPU numbers in
``benchmarks/leaderboard/current.json`` are not portable to real
workloads where the diff-PH layer is invoked inside a training loop
on GPU tensors. This script produces the corresponding GPU numbers
and writes them alongside.

The Colab notebook variant uses a free T4; the Modal variant lets the
user pay for an A10/L4 if they want lower-variance timings.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU runner for the diff-PH bench")
    parser.add_argument("--output", type=Path, default=Path("/tmp/diff_ph_bench_gpu.json"))
    parser.add_argument("--markdown", type=Path, default=Path("/tmp/diff_ph_bench_gpu.md"))
    parser.add_argument("--datasets", nargs="+", default=["mnist_mock_digit_0"])
    parser.add_argument("--axes", nargs="+", default=["correctness", "stability", "speed"])
    args = parser.parse_args(argv)

    # Import the framework lazily — these imports can be slow on Colab
    # cold-start, but only fire once.
    import torch

    from benchmarks.report import render_markdown
    from benchmarks.runner import run, write_result

    print("=" * 70)
    print("GPU bench environment")
    print("=" * 70)
    print(f"Python:        {platform.python_version()}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA avail:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
        print(f"CUDA capable:  compute {torch.cuda.get_device_capability(0)}")
    else:
        print("Running on CPU — for a real GPU bench, invoke this script in an")
        print("environment with CUDA (free Colab T4, Modal A10, Lambda, etc.).")
    print()

    result = run(
        dataset_names=args.datasets,
        axis_names=args.axes,
    )
    write_result(result, args.output)
    md = render_markdown(result.as_dict())
    args.markdown.write_text(md)
    print(md)
    print(f"\nJSON leaderboard written to: {args.output}")
    print(f"Markdown report written to: {args.markdown}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md)

    failed = [c for c in result.cells if not c.success]
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
