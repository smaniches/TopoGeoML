"""
Modal script for the GPU diff-PH bench.

Modal (modal.com) lets you run code on rented GPUs for cents-per-run with
zero infrastructure setup. This script defines a Modal function that:

  1. Boots a container with PyTorch CUDA pre-installed
  2. Installs TopoGeoML's [bench] extras
  3. Runs notebooks/diff_ph_bench_gpu.py
  4. Returns the JSON + markdown artifacts

Invocation
----------
    pip install modal-client
    modal token new  # one-time auth
    modal run scripts/modal_diff_ph_bench.py

Cost
----
At a10g.us-east (~$1.10/hr, as of 2026-05) the typical bench run takes
~10 minutes, i.e. ~$0.20 per run. Use ``--gpu T4`` for the cheaper
(~$0.40/hr) option if Modal supports it for your account.

Output
------
By default writes results into the calling process's working directory
under ``modal_outputs/``. Pass ``--output-dir`` to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal as _modal_typing


def _build_image() -> _modal_typing.Image:
    """Build the Modal container image: CUDA torch + TopoGeoML bench deps."""
    import modal

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install(
            "numpy>=1.24",
            "scipy>=1.10",
            "scikit-learn>=1.3",
            "ripser>=0.6.4",
            "networkx>=3.0",
            "pyyaml>=6.0",
            "matplotlib>=3.7",
            "deprecated>=1.2",
        )
        # CUDA-enabled torch wheel + torch-topological + gudhi + torchvision.
        .pip_install(
            "torch>=2.0",
            extra_index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "torch-topological>=0.1.9",
            "torchvision>=0.15",
            "gudhi>=3.7",
            "torch-geometric>=2.4",
        )
    )


def define_app() -> tuple[_modal_typing.App, Any]:
    """Define the Modal app + GPU function. Importable from other scripts."""
    import modal

    app = modal.App("topogeoml-diff-ph-bench")
    image = _build_image()

    @app.function(  # type: ignore[misc]
        image=image,
        gpu="a10g",
        timeout=1800,  # 30 minutes
    )
    def run_bench(  # type: ignore[no-untyped-def]
        repo_url: str = "https://github.com/smaniches/TopoGeoML.git",
        ref: str = "main",
    ):
        """Run the bench on the Modal GPU and return the JSON + markdown."""
        import subprocess
        import sys as _sys
        from pathlib import Path

        # Clone the repository.
        subprocess.run(
            ["git", "clone", repo_url, "/repo"], check=True,
        )
        subprocess.run(
            ["git", "-C", "/repo", "checkout", ref], check=True,
        )
        subprocess.run(
            [_sys.executable, "-m", "pip", "install", "-e", "/repo[bench]", "--quiet"],
            check=True,
        )
        # Run the bench.
        _sys.path.insert(0, "/repo")
        result = subprocess.run(
            [
                _sys.executable, "/repo/notebooks/diff_ph_bench_gpu.py",
                "--output", "/tmp/result.json",
                "--markdown", "/tmp/result.md",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "exit_code": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "json": Path("/tmp/result.json").read_text() if Path("/tmp/result.json").exists() else "",
            "markdown": Path("/tmp/result.md").read_text() if Path("/tmp/result.md").exists() else "",
        }

    return app, run_bench


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modal_diff_ph_bench")
    parser.add_argument("--ref", default="main", help="git ref (branch/tag/SHA) to bench")
    parser.add_argument("--repo-url", default="https://github.com/smaniches/TopoGeoML.git")
    parser.add_argument("--output-dir", type=Path, default=Path("modal_outputs"))
    args = parser.parse_args(argv)

    try:
        import modal  # noqa: F401
    except ImportError:
        print("ERROR: Modal is not installed. `pip install modal-client` first.",
              file=sys.stderr)
        return 1

    app, run_bench = define_app()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with app.run():  # type: ignore[attr-defined]
        result = run_bench.remote(repo_url=args.repo_url, ref=args.ref)

    json_out = args.output_dir / f"diff_ph_bench_{args.ref}.json"
    md_out = args.output_dir / f"diff_ph_bench_{args.ref}.md"
    json_out.write_text(result["json"])
    md_out.write_text(result["markdown"])

    print(result["markdown"])
    print(f"\nArtifacts:\n  {json_out}\n  {md_out}")
    print(f"\nBench process exit code: {result['exit_code']}")
    return int(result["exit_code"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
