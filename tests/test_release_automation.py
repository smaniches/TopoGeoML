"""Regression contracts for automatic, exact-commit releases.

These tests verify the GitHub Actions wiring. Live permission and network
behavior is verified by a real tagged release, not by static assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflow(name: str) -> dict[str, Any]:
    # BaseLoader preserves the YAML key "on" instead of treating it as boolean.
    result = yaml.load((ROOT / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(result, dict)
    return result


def _commands(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_tagging_requires_successful_main_ci_on_the_same_repository() -> None:
    workflow = _workflow("auto-tag-release.yml")
    assert workflow["on"] == {
        "workflow_run": {"workflows": ["CI"], "types": ["completed"]}
    }
    assert workflow["permissions"] == {}
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    job = workflow["jobs"]["tag-and-dispatch"]
    assert job["permissions"] == {"contents": "write", "actions": "write"}
    for required in (
        "workflow_run.conclusion == 'success'",
        "workflow_run.event == 'push'",
        "workflow_run.head_branch == 'main'",
        "workflow_run.head_repository.full_name == github.repository",
    ):
        assert required in job["if"]


def test_tagger_uses_validated_commit_and_never_overwrites_existing_tags() -> None:
    job = _workflow("auto-tag-release.yml")["jobs"]["tag-and-dispatch"]
    checkout, publish = job["steps"]
    assert checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    assert publish["env"]["CI_SHA"] == "${{ github.event.workflow_run.head_sha }}"
    assert publish["env"]["GH_TOKEN"] == "${{ github.token }}"
    script = _commands(job)
    assert 'git rev-parse HEAD' in script
    assert script.count('git/ref/heads/main') == 2
    assert 'git show-ref --verify --quiet "refs/tags/$tag"' in script
    assert "sort -V" in script
    assert 'object="$CI_SHA"' in script
    assert 'ref="refs/tags/$tag"' in script
    assert "gh workflow run release.yml" in script
    assert '--ref "$tag"' in script and '--field tag="$tag"' in script
    assert "git push --force" not in script


def test_release_accepts_only_matching_exact_version_tag() -> None:
    workflow = _workflow("release.yml")
    assert workflow["on"]["push"]["tags"] == ["v[0-9]*"]
    manual = workflow["on"]["workflow_dispatch"]["inputs"]["tag"]
    assert manual["required"] == "true"
    assert manual["type"] == "string"
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    guard = workflow["jobs"]["validate-release"]
    assert guard["permissions"] == {"contents": "read"}
    commands = _commands(guard)
    assert '[[ "$REF_TYPE" == "tag" ]]' in commands
    assert '"$INPUT_TAG" != "$REF_NAME"' in commands
    assert '"$GITHUB_REF_NAME" == "v$version"' in commands
    assert 'grep -Fq "## [$version]" CHANGELOG.md' in commands
    assert workflow["jobs"]["build"]["needs"] == "validate-release"


def test_existing_integrity_and_publication_chain_is_preserved() -> None:
    jobs = _workflow("release.yml")["jobs"]
    assert set(jobs["publish-pypi"]["needs"]) == {
        "build", "sbom", "attest-sbom"
    }
    assert set(jobs["sign-and-release"]["needs"]) == {
        "build", "sbom", "attest-sbom", "publish-pypi"
    }
    assert jobs["publish-pypi"]["environment"]["name"] == "pypi"
    publisher = _commands(jobs["publish-pypi"])
    assert "verify_pypi_files.py" in publisher
    assert "--require-complete" in publisher
    assert "scripts/bind_release_sbom.py" in _commands(jobs["sbom"])

    assert set(jobs["verify-released-artifacts"]["needs"]) == {
        "publish-pypi", "sign-and-release"
    }
    verifier = _commands(jobs["verify-released-artifacts"])
    assert "verify_pypi_files.py" in verifier
    assert "releases/tags/$GITHUB_REF_NAME" in verifier
    assert "sha256:" in verifier
    assert "sigstore.json" in verifier
