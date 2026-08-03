#!/usr/bin/env python3
"""Verify a release's local distributions against PyPI metadata.

Before publication, existing PyPI files must match the local release artifacts
byte-for-byte. Only files absent from PyPI are copied into an upload directory.
After publication, --require-complete verifies that PyPI exposes exactly the
local filename set with matching SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_distributions(dist_dir: Path) -> dict[str, tuple[Path, str]]:
    files = sorted(
        path
        for path in dist_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not files:
        raise RuntimeError(f"no wheel or sdist files found in {dist_dir}")
    return {path.name: (path, sha256_file(path)) for path in files}


def pypi_metadata(project: str, version: str) -> dict[str, Any] | None:
    project_path = urllib.parse.quote(project, safe="")
    version_path = urllib.parse.quote(version, safe="")
    url = f"https://pypi.org/pypi/{project_path}/{version_path}/json"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "topogeoml-release-verifier/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"PyPI metadata request failed with HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PyPI metadata request failed: {error}") from error


def remote_distributions(metadata: dict[str, Any] | None) -> dict[str, str]:
    if metadata is None:
        return {}

    remote: dict[str, str] = {}
    for entry in metadata.get("urls", []):
        filename = entry.get("filename")
        sha256 = entry.get("digests", {}).get("sha256")
        if not isinstance(filename, str) or not isinstance(sha256, str):
            raise RuntimeError("PyPI metadata contains a file without a SHA-256 digest")
        remote[filename] = sha256.lower()
    return remote


def compare(
    local: dict[str, tuple[Path, str]], remote: dict[str, str]
) -> list[str]:
    local_names = set(local)
    remote_names = set(remote)
    unexpected = sorted(remote_names - local_names)
    if unexpected:
        raise RuntimeError(
            "PyPI contains unexpected files for this version: " + ", ".join(unexpected)
        )

    for filename in sorted(remote_names):
        local_digest = local[filename][1]
        if remote[filename] != local_digest:
            raise RuntimeError(
                f"SHA-256 mismatch for existing PyPI file {filename}: "
                f"remote={remote[filename]} local={local_digest}"
            )

    return sorted(local_names - remote_names)


def prepare_upload_directory(
    missing: list[str],
    local: dict[str, tuple[Path, str]],
    upload_dir: Path,
) -> None:
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)
    for filename in missing:
        shutil.copy2(local[filename][0], upload_dir / filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--prepare-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.require_complete and args.prepare_dir is not None:
        parser.error("--prepare-dir and --require-complete are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    local = local_distributions(args.dist_dir)
    last_error: RuntimeError | None = None

    for attempt in range(1, args.attempts + 1):
        try:
            remote = remote_distributions(pypi_metadata(args.project, args.version))
            missing = compare(local, remote)
            if args.require_complete and missing:
                raise RuntimeError(
                    "PyPI is still missing release files: " + ", ".join(missing)
                )

            if args.prepare_dir is not None:
                prepare_upload_directory(missing, local, args.prepare_dir)

            print(len(missing))
            print(
                f"verified {len(remote)} existing PyPI file(s); "
                f"{len(missing)} file(s) missing",
                file=sys.stderr,
            )
            return 0
        except RuntimeError as error:
            last_error = error
            if attempt == args.attempts:
                break
            print(
                f"attempt {attempt}/{args.attempts} failed: {error}; retrying",
                file=sys.stderr,
            )
            time.sleep(args.delay_seconds)

    assert last_error is not None
    print(f"verification failed: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
