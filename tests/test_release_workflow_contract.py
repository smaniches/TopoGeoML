from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


def _jobs() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict)
    return jobs


def _run_commands(job: dict) -> str:
    steps = job.get("steps", [])
    return "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))


def test_runtime_dependency_resolution_is_unprivileged() -> None:
    jobs = _jobs()
    build = jobs["build"]
    sbom = jobs["sbom"]

    assert build["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert sbom["permissions"] == {"contents": "read"}

    build_runs = _run_commands(build)
    sbom_runs = _run_commands(sbom)
    assert "cyclonedx-py" not in build_runs
    assert "pip --python" not in build_runs
    assert 'pip --python .sbom-runtime install "$wheel"' in sbom_runs
    assert "scripts/bind_release_sbom.py" in sbom_runs


def test_sbom_attestation_executes_no_project_or_dependency_code() -> None:
    jobs = _jobs()
    attest = jobs["attest-sbom"]

    assert attest["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert _run_commands(attest).strip() == ""

    steps = attest["steps"]
    action_step = next(step for step in steps if str(step.get("uses", "")).startswith("actions/attest@"))
    assert action_step["with"]["subject-path"] == "dist/*.whl"
    assert action_step["with"]["predicate-type"] == "https://cyclonedx.org/bom"
    assert action_step["with"]["predicate-path"] == "sbom.cdx.json"


def test_publication_waits_for_verified_sbom_attestation() -> None:
    jobs = _jobs()
    assert set(jobs["publish-pypi"]["needs"]) == {"build", "sbom", "attest-sbom"}
    assert set(jobs["sign-and-release"]["needs"]) == {"build", "sbom", "attest-sbom"}
