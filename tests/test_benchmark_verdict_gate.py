"""CLI-level scientific verdict gate contract.

The benchmark may run successfully yet report incorrect persistence diagrams.
Only explicitly required backends must fail CI on that scientific verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize(
    ("overall_pass", "require_verdict", "expected_rc"),
    [
        (True, True, 0),
        (False, True, 1),
        (False, False, 0),
    ],
)
def test_correctness_verdict_is_enforced_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overall_pass: bool,
    require_verdict: bool,
    expected_rc: int,
) -> None:
    from benchmarks import runner
    from benchmarks.cli import main

    def _fixed_correctness(backend, dataset, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "n_points": 50,
            "atol": 1e-6,
            "overall_pass": overall_pass,
            "per_seed": [{
                "max_abs_diff_h0": 0.0,
                "max_abs_diff_h1": 0.0,
                "bottleneck_h0": 0.0,
                "bottleneck_h1": 0.0,
            }],
        }

    monkeypatch.setitem(runner.AXES, "correctness", _fixed_correctness)
    out = tmp_path / "result.json"
    args = [
        "--backends", "topogeoml-diff-ph",
        "--datasets", "mnist_mock_digit_0",
        "--axes", "correctness",
        "--output", str(out),
    ]
    if require_verdict:
        args.extend(["--require-correctness-backend", "topogeoml-diff-ph"])

    result = main(args)
    assert result == expected_rc
    parsed = json.loads(out.read_text())
    assert parsed["cells"][0]["success"] is True
    assert parsed["cells"][0]["payload"]["overall_pass"] is overall_pass


def test_missing_required_backend_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.backends import get_backend
    from benchmarks.cli import main

    backend = get_backend("topogeoml-diff-ph")
    monkeypatch.setattr(backend, "available", staticmethod(lambda: False))
    out = tmp_path / "missing.json"
    rc = main([
        "--backends", "topogeoml-diff-ph",
        "--datasets", "mnist_mock_digit_0",
        "--axes", "correctness",
        "--require-correctness-backend", "topogeoml-diff-ph",
        "--output", str(out),
    ])
    assert rc == 1
    parsed = json.loads(out.read_text())
    assert parsed["cells"][0]["error_kind"] == "UnavailableBackend"
