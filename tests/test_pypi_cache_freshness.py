"""Prevent reintroducing stale PyPI JSON CDN lookups in publication checks."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/verify_pypi_files.py"
SPEC = importlib.util.spec_from_file_location("verify_pypi_files", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_each_pypi_lookup_uses_fresh_cdn_key_and_no_cache_header() -> None:
    response = b'{"urls": []}'
    with (
        patch.object(VERIFY.time, "time_ns", side_effect=[100, 101]),
        patch.object(
            VERIFY.urllib.request,
            "urlopen",
            side_effect=lambda request, timeout: io.BytesIO(response),
        ) as fetch,
    ):
        assert VERIFY.pypi_metadata("topogeoml", "0.0.8") == {"urls": []}
        assert VERIFY.pypi_metadata("topogeoml", "0.0.8") == {"urls": []}

    calls = fetch.call_args_list
    assert len(calls) == 2
    urls = [call.args[0].full_url for call in calls]
    assert urls[0] != urls[1]
    for index, call in enumerate(calls):
        request = call.args[0]
        parsed = urllib.parse.urlparse(request.full_url)
        assert parsed.path == "/pypi/topogeoml/0.0.8/json"
        assert urllib.parse.parse_qs(parsed.query) == {
            "topogeoml_fresh": [str(100 + index)]
        }
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["cache-control"] == "no-cache"
        assert headers["accept"] == "application/json"
        assert call.kwargs["timeout"] == 30


def test_404_remains_missing_without_weakening_hash_or_filename_gates(tmp_path: Path) -> None:
    wheel = tmp_path / "topogeoml-0.0.8-py3-none-any.whl"
    wheel.write_bytes(b"expected release wheel bytes")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    local = VERIFY.local_distributions(tmp_path)

    error = urllib.error.HTTPError(
        "https://pypi.org/pypi/topogeoml/0.0.8/json", 404,
        "Not Found", None, None
    )
    with patch.object(VERIFY.urllib.request, "urlopen", side_effect=error):
        missing = VERIFY.remote_distributions(
            VERIFY.pypi_metadata("topogeoml", "0.0.8")
        )
        assert VERIFY.compare(local, missing) == [wheel.name]

    assert VERIFY.compare(local, {wheel.name: digest}) == []
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        VERIFY.compare(local, {wheel.name: "0" * 64})
    with pytest.raises(RuntimeError, match="unexpected files"):
        VERIFY.compare(local, {wheel.name: digest, "extra.whl": digest})
