"""Pack and fetch the runtime files the web app needs.

Model binaries and the cleaned permit table are git-ignored. A deployment host that
only has the git tree can download a zip produced by ``python -m huskyquiet bundle``
when ``HUSKYQUIET_BUNDLE_URL`` is set. The zip is checked against
``HUSKYQUIET_BUNDLE_SHA256`` when that variable is set. Existing local files are not
replaced, so a validated production model is left alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import requests

from huskyquiet.config import Paths, get_paths

_BUNDLE_SOURCE = Path("deployment") / "bundle_source.json"
_USER_AGENT = "HuskyQuiet/0.2 (public runtime bundle)"

SLIM_COLUMNS = (
    "permit_num",
    "applied_date",
    "issued_date",
    "is_model_eligible",
    "permit_type_mapped",
    "permit_class",
    "permit_type_desc",
    "permit_suffix",
    "description",
    "est_project_cost",
    "related_mup",
    "latitude",
    "longitude",
    "zip",
    "address_key",
)

BUNDLE_MEMBERS = (
    "artifacts/models/production/model.joblib",
    "artifacts/models/production/metadata.json",
    "data/interim/permits_clean.parquet",
    "reports/metrics.json",
    "reports/model_report.md",
    "reports/figures/reliability_test.png",
    "reports/test_month_calibration.json",
)


def write_runtime_bundle(paths: Paths | None = None, dest: Path | None = None) -> Path:
    """Write a zip of the files required to score and to render the model page."""
    import pandas as pd

    paths = paths or get_paths()
    model = paths.model_dir / "production" / "model.joblib"
    if not model.exists() or not paths.clean_permits.exists():
        raise FileNotFoundError("Build the pipeline first: python -m huskyquiet all")
    dest = dest or (paths.root / "dist" / "huskyquiet-runtime.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    clean = pd.read_parquet(paths.clean_permits)
    missing = [col for col in SLIM_COLUMNS if col not in clean.columns]
    if missing:
        raise RuntimeError(f"cleaned permits are missing columns {missing}")
    slim = clean.loc[clean["is_model_eligible"], list(SLIM_COLUMNS)]
    staging = dest.parent / "_runtime_staging"
    if staging.exists():
        for child in staging.rglob("*"):
            if child.is_file():
                child.unlink()
    slim_path = staging / "data" / "interim" / "permits_clean.parquet"
    slim_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(slim_path, index=False)

    copies = {
        "artifacts/models/production/model.joblib": model,
        "artifacts/models/production/metadata.json": paths.model_dir / "production" / "metadata.json",
        "data/interim/permits_clean.parquet": slim_path,
        "reports/metrics.json": paths.reports / "metrics.json",
        "reports/model_report.md": paths.reports / "model_report.md",
        "reports/figures/reliability_test.png": paths.reports / "figures" / "reliability_test.png",
        "reports/test_month_calibration.json": paths.reports / "test_month_calibration.json",
    }
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in copies.items():
            if source.exists():
                archive.write(source, name)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return dest


def _secrets_bundle() -> tuple[str, str]:
    """Read Cloud or local Streamlit secrets. Missing secrets are not an error."""
    try:
        import streamlit as st

        url = str(st.secrets.get("HUSKYQUIET_BUNDLE_URL", "")).strip()
        digest = str(st.secrets.get("HUSKYQUIET_BUNDLE_SHA256", "")).strip().lower()
    except Exception:
        return "", ""
    return url, digest


def _file_bundle(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    url = str(payload.get("url", "")).strip()
    digest = str(payload.get("sha256", "")).strip().lower()
    return url, digest


def configured_bundle(paths: Paths) -> tuple[str, str]:
    """URL and optional SHA-256. Environment and Streamlit secrets override the repo file."""
    env_url = os.environ.get("HUSKYQUIET_BUNDLE_URL", "").strip()
    if env_url:
        return env_url, os.environ.get("HUSKYQUIET_BUNDLE_SHA256", "").strip().lower()
    secret_url, secret_sha = _secrets_bundle()
    if secret_url:
        return secret_url, secret_sha
    return _file_bundle(paths.root / _BUNDLE_SOURCE)


def ensure_runtime(paths: Paths | None = None) -> None:
    """No-op when the production model and cleaned permits are already on disk."""
    paths = paths or get_paths()
    model = paths.model_dir / "production" / "model.joblib"
    if model.exists() and paths.clean_permits.exists():
        return
    url, expected = configured_bundle(paths)
    if not url:
        raise FileNotFoundError(
            "The production model or cleaned permits are not on disk. "
            "Run `python -m huskyquiet all`, or set HUSKYQUIET_BUNDLE_URL to a runtime zip "
            "from `python -m huskyquiet bundle`."
        )
    if not url.startswith("https://"):
        raise RuntimeError("Runtime bundle URL must be HTTPS.")
    cache = paths.data / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "runtime_bundle.zip"
    try:
        response = requests.get(
            url,
            timeout=180,
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download the runtime bundle ({exc.__class__.__name__}).") from exc
    actual = hashlib.sha256(response.content).hexdigest()
    if expected and actual != expected:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded bundle SHA-256 does not match HUSKYQUIET_BUNDLE_SHA256.")
    if not response.content.startswith(b"PK"):
        raise RuntimeError("Downloaded runtime bundle is not a zip file.")
    zip_path.write_bytes(response.content)
    _extract_missing(zip_path, paths.root)


def _extract_missing(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            relative = _safe_member(info.filename)
            if relative is None:
                continue
            target = root / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))


def _safe_member(name: str) -> Path | None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe path in runtime bundle: {name}")
    if name.endswith("/"):
        return None
    return path
