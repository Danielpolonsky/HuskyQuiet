"""Paginated download of the Seattle Building Permits dataset from the Socrata SODA API.

Design notes
------------
* Pages are ordered by the Socrata system row id (``:id``), which gives a stable total
  order so that ``$offset`` pagination neither skips nor duplicates rows.
* The portal performs a full daily reload (every row shares one ``:updated_at``). If a
  reload happens mid-download, row ids can change, so the dataset's ``rowsUpdatedAt``
  is checked before and after paging and the download is retried if it moved.
* The downloaded row count is validated against a server-side ``count(*)``.
* Raw values are stored as strings exactly as delivered; typing happens in ``clean.py``.
* An optional free Socrata app token (env ``SOCRATA_APP_TOKEN``) raises rate limits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from huskyquiet.config import DATASET_ID, METADATA_URL, RESOURCE_URL, Paths, get_paths
from huskyquiet.data.schema import EXPECTED_FIELDS, REQUIRED_FIELDS

log = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 50_000
SYSTEM_FIELDS = (":id", ":created_at", ":updated_at", ":version")


class IngestionError(RuntimeError):
    """Raised when the downloaded data fails integrity checks."""


def build_session(app_token: str | None = None) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "application/json", "User-Agent": "HuskyQuiet/0.1"})
    if app_token:
        session.headers["X-App-Token"] = app_token
    return session


def fetch_metadata(session: requests.Session, timeout: float = 60) -> dict[str, Any]:
    resp = session.get(METADATA_URL, timeout=timeout)
    resp.raise_for_status()
    meta = resp.json()
    return {
        "name": meta.get("name"),
        "rows_updated_at": meta.get("rowsUpdatedAt"),
        "columns": [c["fieldName"] for c in meta.get("columns", [])],
    }


def fetch_row_count(session: requests.Session, timeout: float = 60) -> int:
    resp = session.get(RESOURCE_URL, params={"$select": "count(*) AS n"}, timeout=timeout)
    resp.raise_for_status()
    return int(resp.json()[0]["n"])


def fetch_page(
    session: requests.Session, offset: int, limit: int, timeout: float = 300
) -> list[dict[str, Any]]:
    params = {
        # SoQL does not accept individual system fields alongside "*"; ":*" selects all of them.
        "$select": ":*,*",
        "$order": ":id",
        "$limit": str(limit),
        "$offset": str(offset),
    }
    resp = session.get(RESOURCE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten Socrata JSON records into an all-string DataFrame."""
    rows = []
    for rec in records:
        row: dict[str, Any] = {}
        for key, value in rec.items():
            if key == "location1":
                continue  # duplicate of latitude/longitude
            if key == "link" and isinstance(value, dict):
                value = value.get("url")
            elif isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            row[key] = None if value is None else str(value)
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    return df.astype("string")


def check_schema(columns: list[str]) -> None:
    present = set(columns)
    missing_required = REQUIRED_FIELDS - present
    if missing_required:
        raise IngestionError(f"API response is missing required fields: {sorted(missing_required)}")
    unexpected = present - EXPECTED_FIELDS - set(SYSTEM_FIELDS)
    if unexpected:
        log.warning("API returned fields not in the verified schema: %s", sorted(unexpected))
    missing_optional = EXPECTED_FIELDS - present - {"location1"}
    if missing_optional:
        log.warning("Verified fields absent from response (all null?): %s", sorted(missing_optional))


def _download_once(session: requests.Session, page_size: int) -> tuple[pd.DataFrame, dict]:
    meta_before = fetch_metadata(session)
    expected = fetch_row_count(session)
    log.info("Dataset '%s': %d rows reported by server", meta_before["name"], expected)

    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        t0 = time.perf_counter()
        records = fetch_page(session, offset, page_size)
        if not records:
            break
        frames.append(records_to_frame(records))
        log.info(
            "Fetched rows %d-%d (%.1fs)", offset, offset + len(records) - 1, time.perf_counter() - t0
        )
        offset += len(records)
        if len(records) < page_size:
            break

    meta_after = fetch_metadata(session)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    info = {
        "expected_rows": expected,
        "rows_updated_at_before": meta_before["rows_updated_at"],
        "rows_updated_at_after": meta_after["rows_updated_at"],
        "api_columns": meta_before["columns"],
    }
    return df, info


def download_permits(
    paths: Paths | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_attempts: int = 3,
    app_token: str | None = None,
) -> pd.DataFrame:
    """Download the full dataset, validate it, and write raw Parquet plus metadata JSON."""
    paths = paths or get_paths()
    paths.ensure()
    session = build_session(app_token or os.environ.get("SOCRATA_APP_TOKEN"))

    for attempt in range(1, max_attempts + 1):
        started = datetime.now(UTC)
        df, info = _download_once(session, page_size)
        if info["rows_updated_at_before"] != info["rows_updated_at_after"]:
            log.warning("Dataset was reloaded during download (attempt %d); retrying", attempt)
            continue
        if len(df) != info["expected_rows"]:
            log.warning(
                "Row count mismatch: got %d, expected %d (attempt %d)",
                len(df), info["expected_rows"], attempt,
            )
            continue
        break
    else:
        raise IngestionError(f"Download failed integrity checks after {max_attempts} attempts")

    check_schema(list(df.columns))
    if ":id" not in df.columns:
        raise IngestionError("Socrata row id ':id' missing from response")
    n_dup_ids = int(df[":id"].duplicated().sum())
    if n_dup_ids:
        raise IngestionError(f"{n_dup_ids} duplicated Socrata row ids; pagination is unstable")

    df = df.rename(columns={c: "socrata_" + c[1:] for c in SYSTEM_FIELDS if c in df.columns})
    df.to_parquet(paths.raw_permits, index=False)
    digest = hashlib.sha256(paths.raw_permits.read_bytes()).hexdigest()

    metadata = {
        "dataset_id": DATASET_ID,
        "source_url": RESOURCE_URL,
        "downloaded_at_utc": started.isoformat(timespec="seconds"),
        "rows_updated_at_epoch": info["rows_updated_at_after"],
        "rows_updated_at_utc": (
            datetime.fromtimestamp(info["rows_updated_at_after"], UTC).isoformat(timespec="seconds")
            if info["rows_updated_at_after"]
            else None
        ),
        "n_rows": len(df),
        "columns": sorted(df.columns),
        "raw_parquet_sha256": digest,
    }
    paths.raw_metadata.write_text(json.dumps(metadata, indent=2))
    log.info("Wrote %d raw rows to %s", len(df), paths.raw_permits)
    return df
