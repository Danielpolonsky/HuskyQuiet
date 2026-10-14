import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from huskyquiet.bundle import _extract_missing, ensure_runtime
from huskyquiet.config import Paths


def test_committed_bundle_source_is_a_public_https_release():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "deployment" / "bundle_source.json").read_text(encoding="utf-8"))
    assert payload["url"].startswith("https://github.com/Danielpolonsky/HuskyQuiet/releases/download/")
    assert len(payload["sha256"]) == 64
    int(payload["sha256"], 16)


def test_extract_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "no")
    with pytest.raises(RuntimeError, match="Unsafe"):
        _extract_missing(archive_path, tmp_path / "root")


def test_extract_does_not_replace_an_existing_file(tmp_path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("reports/metrics.json", '{"new": 1}')
    root = tmp_path / "app"
    target = root / "reports" / "metrics.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"old": 1}', encoding="utf-8")
    _extract_missing(archive_path, root)
    assert target.read_text(encoding="utf-8") == '{"old": 1}'


def test_missing_runtime_without_a_url_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("HUSKYQUIET_BUNDLE_URL", raising=False)
    with pytest.raises(FileNotFoundError, match="HUSKYQUIET_BUNDLE_URL"):
        ensure_runtime(Paths(root=tmp_path))


def test_file_source_must_be_https(tmp_path, monkeypatch):
    monkeypatch.delenv("HUSKYQUIET_BUNDLE_URL", raising=False)
    monkeypatch.delenv("HUSKYQUIET_BUNDLE_SHA256", raising=False)
    monkeypatch.setattr("huskyquiet.bundle._secrets_bundle", lambda: ("", ""))
    root = tmp_path / "app"
    source = root / "deployment"
    source.mkdir(parents=True)
    (source / "bundle_source.json").write_text(
        '{"url": "http://example.invalid/bundle.zip", "sha256": ""}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="HTTPS"):
        ensure_runtime(Paths(root=root))


def test_download_checks_sha256(tmp_path, monkeypatch):
    class Response:
        content = b"not-a-real-bundle"
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setenv("HUSKYQUIET_BUNDLE_URL", "https://example.invalid/huskyquiet-runtime.zip")
    monkeypatch.setenv("HUSKYQUIET_BUNDLE_SHA256", "0" * 64)
    monkeypatch.setattr("huskyquiet.bundle.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="SHA-256"):
        ensure_runtime(Paths(root=tmp_path))
    assert hashlib.sha256(b"not-a-real-bundle").hexdigest() != "0" * 64
