"""Regression tests for the Hodge benchmark command-line interface."""

import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize(
    "abbreviated_option",
    [
        "--out=/tmp/result.json",
        "--mark=/tmp/result.md",
        "--n-e=1",
    ],
)
def test_rejects_abbreviated_long_options(abbreviated_option: str) -> None:
    """Workflow-managed arguments must not be overridable by abbreviations."""
    from benchmarks.hodge.__main__ import main

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--models",
                "mlp-baseline",
                "--datasets",
                "mutag",
                "--seeds",
                "0",
                abbreviated_option,
            ]
        )

    assert exc_info.value.code == 2
