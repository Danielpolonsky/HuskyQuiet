import json

import pandas as pd
import pytest

from huskyquiet.config import Paths
from huskyquiet.data import ingest
from huskyquiet.data.schema import EXPECTED_FIELDS


def _record(i: int) -> dict:
    rec = {f: "x" for f in EXPECTED_FIELDS if f not in ("location1", "link")}
    rec.update({":id": f"row-{i}", ":updated_at": "2026-01-01T00:00:00.000Z", "permitnum": f"{i}-CN",
                "link": {"url": f"https://example/{i}"}, "location1": {"latitude": "47.6"}})
    return rec


def test_records_to_frame_flattens_nested_fields():
    df = ingest.records_to_frame([_record(1)])
    assert "location1" not in df.columns
    assert df.loc[0, "link"] == "https://example/1"
    assert all(str(t) == "string" for t in df.dtypes)


def test_check_schema_requires_core_fields():
    with pytest.raises(ingest.IngestionError):
        ingest.check_schema(["permitnum", "applieddate"])


class _FakeSession:
    pass


def test_download_paginates_and_validates(tmp_path, monkeypatch):
    rows = [_record(i) for i in range(7)]
    monkeypatch.setattr(ingest, "build_session", lambda token=None: _FakeSession())
    monkeypatch.setattr(ingest, "fetch_metadata",
                        lambda s: {"name": "Building Permits", "rows_updated_at": 1700000000, "columns": []})
    monkeypatch.setattr(ingest, "fetch_row_count", lambda s: len(rows))
    calls = []

    def fake_page(session, offset, limit):
        calls.append(offset)
        return rows[offset: offset + limit]

    monkeypatch.setattr(ingest, "fetch_page", fake_page)
    paths = Paths(root=tmp_path)
    df = ingest.download_permits(paths=paths, page_size=3)
    assert calls == [0, 3, 6]
    assert len(df) == 7 and df["socrata_id"].is_unique
    assert pd.read_parquet(paths.raw_permits).shape[0] == 7
    meta = json.loads(paths.raw_metadata.read_text())
    assert meta["n_rows"] == 7 and len(meta["raw_parquet_sha256"]) == 64


def test_download_fails_on_persistent_count_mismatch(tmp_path, monkeypatch):
    rows = [_record(i) for i in range(4)]
    monkeypatch.setattr(ingest, "build_session", lambda token=None: _FakeSession())
    monkeypatch.setattr(ingest, "fetch_metadata",
                        lambda s: {"name": "x", "rows_updated_at": 1, "columns": []})
    monkeypatch.setattr(ingest, "fetch_row_count", lambda s: 5)
    monkeypatch.setattr(ingest, "fetch_page", lambda s, o, n: rows[o: o + n])
    with pytest.raises(ingest.IngestionError):
        ingest.download_permits(paths=Paths(root=tmp_path), page_size=10, max_attempts=2)
