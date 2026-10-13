"""Validation, typing, normalization, and deduplication of raw permit records.

Produces ``data/interim/permits_clean.parquet``: one row per permit with typed columns,
snake_case names, and explicit data-quality flags. Values are never silently "fixed";
invalid values are nulled and flagged so downstream stages can decide what to exclude.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from huskyquiet.config import MIN_VALID_DATE, SEATTLE_BBOX, Paths, get_paths
from huskyquiet.data.schema import DATE_FIELDS, NUMBER_FIELDS, TEXT_FIELDS

log = logging.getLogger(__name__)

RENAME: dict[str, str] = {
    "permitnum": "permit_num",
    "permitclass": "permit_class",
    "permitclassmapped": "permit_class_mapped",
    "permittypemapped": "permit_type_mapped",
    "permittypedesc": "permit_type_desc",
    "housingunits": "housing_units",
    "housingunitsremoved": "housing_units_removed",
    "housingunitsadded": "housing_units_added",
    "estprojectcost": "est_project_cost",
    "applieddate": "applied_date",
    "issueddate": "issued_date",
    "expiresdate": "expires_date",
    "completeddate": "completed_date",
    "statuscurrent": "status_current",
    "relatedmup": "related_mup",
    "originaladdress1": "address",
    "originalcity": "city",
    "originalstate": "state",
    "originalzip": "zip",
    "contractorcompanyname": "contractor_company_name",
    "totaldaysplanreview": "total_days_plan_review",
    "daysinitialplanreview": "days_initial_plan_review",
    "daysplanreviewcity": "days_plan_review_city",
    "daysoutcorrections": "days_out_corrections",
    "numberreviewcycles": "number_review_cycles",
    "initialreviewcompletedate": "initial_review_complete_date",
    "planreviewcompletedate": "plan_review_complete_date",
    "daysissuepermitcity": "days_issue_permit_city",
    "readytoissuedate": "ready_to_issue_date",
    "dwellingunittype": "dwelling_unit_type",
    "standardplan": "standard_plan",
    "dependentbuilding": "dependent_building",
    "parentpermitnum": "parent_permit_num",
    "housingcategory": "housing_category",
}

_WS = re.compile(r"\s+")


def _norm_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(_WS, " ", regex=True).str.strip()
    return s.mask(s == "", pd.NA)


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse Socrata floating timestamps (``YYYY-MM-DDTHH:MM:SS.fff``) to midnight dates."""
    parsed = pd.to_datetime(s, errors="coerce", format="ISO8601")
    return parsed.dt.tz_localize(None).dt.normalize() if parsed.dt.tz is not None else parsed.dt.normalize()


def clean_permits(raw: pd.DataFrame, data_as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe, quality report). ``data_as_of`` bounds plausible dates."""
    report: dict[str, int | float | str] = {"n_raw": len(raw)}
    df = raw.copy()

    for col in TEXT_FIELDS:
        if col in df.columns:
            df[col] = _norm_text(df[col])
    if "permitnum" in df.columns:
        df["permitnum"] = df["permitnum"].str.upper()

    # Dates: parse, then null out sentinels (e.g. 1900-01-01) and impossible future dates.
    min_valid = pd.Timestamp(MIN_VALID_DATE)
    for col in DATE_FIELDS:
        if col not in df.columns:
            continue
        raw_nonnull = df[col].notna()
        parsed = parse_dates(df[col])
        report[f"{col}_unparseable"] = int((raw_nonnull & parsed.isna()).sum())
        sentinel = parsed < min_valid
        report[f"{col}_sentinel_nulled"] = int(sentinel.sum())
        df[f"q_{col}_sentinel"] = sentinel
        df[col] = parsed.mask(sentinel)
    for col in ("applieddate", "issueddate"):
        future = df[col] > data_as_of
        report[f"{col}_future_nulled"] = int(future.sum())
        df[col] = df[col].mask(future)

    for col in NUMBER_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    neg_cost = df["estprojectcost"] < 0
    report["estprojectcost_negative_nulled"] = int(neg_cost.sum())
    df["estprojectcost"] = df["estprojectcost"].mask(neg_cost)

    lat, lon = df["latitude"], df["longitude"]
    in_bbox = lat.between(SEATTLE_BBOX["lat_min"], SEATTLE_BBOX["lat_max"]) & lon.between(
        SEATTLE_BBOX["lon_min"], SEATTLE_BBOX["lon_max"]
    )
    bad_coords = lat.notna() & lon.notna() & ~in_bbox
    report["coords_outside_seattle_nulled"] = int(bad_coords.sum())
    df.loc[bad_coords, ["latitude", "longitude"]] = np.nan
    df["q_coords_missing"] = df["latitude"].isna() | df["longitude"].isna()

    df = df.rename(columns=RENAME)
    df["permit_suffix"] = df["permit_num"].str.extract(r"-([A-Z]+)$", expand=False).astype("string")
    df["city"] = df["city"].str.upper()
    df["zip"] = df["zip"].str.extract(r"^(\d{5})", expand=False).astype("string")
    df["address_key"] = df["address"].str.upper()
    for col in ("dependent_building",):
        df[col] = df[col].map({"1": True, "0": False}).astype("boolean")

    # Deduplicate on the business key. Prefer the most recently updated, most complete row.
    n_before = len(df)
    df["_completeness"] = df.notna().sum(axis=1)
    sort_cols = ["permit_num", "_completeness"]
    if "socrata_updated_at" in df.columns:
        sort_cols.insert(1, "socrata_updated_at")
    df = (
        df.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
        .drop_duplicates("permit_num", keep="first")
        .drop(columns="_completeness")
    )
    report["duplicates_removed"] = n_before - len(df)
    report["missing_permit_num"] = int(df["permit_num"].isna().sum())
    df = df[df["permit_num"].notna()]

    df["q_issued_before_applied"] = (
        df["issued_date"].notna() & df["applied_date"].notna() & (df["issued_date"] < df["applied_date"])
    )
    report["issued_before_applied"] = int(df["q_issued_before_applied"].sum())
    report["applied_date_missing"] = int(df["applied_date"].isna().sum())

    # Eligibility for the prediction problem: a formally applied application with a
    # consistent timeline. Records without an application date never entered the review
    # queue in this dataset (mostly exemption requests and intake-stage records).
    df["is_model_eligible"] = df["applied_date"].notna() & ~df["q_issued_before_applied"]
    report["model_eligible"] = int(df["is_model_eligible"].sum())
    report["n_clean"] = len(df)
    report["data_as_of"] = str(data_as_of.date())

    df = df.sort_values("permit_num").reset_index(drop=True)
    return df, report


def determine_data_as_of(raw: pd.DataFrame, metadata: dict | None) -> pd.Timestamp:
    """Last date through which outcomes are observed.

    Uses the latest issuance/application date present in the extract, bounded by the
    portal refresh timestamp. Labels are only trusted for windows ending on/before this date.
    """
    dates = pd.concat([parse_dates(raw["issueddate"]), parse_dates(raw["applieddate"])])
    latest = dates.max()
    if metadata and metadata.get("rows_updated_at_utc"):
        refreshed = pd.Timestamp(metadata["rows_updated_at_utc"]).tz_localize(None).normalize()
        latest = min(latest, refreshed)
    return pd.Timestamp(latest).normalize()


def run_clean(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or get_paths()
    paths.ensure()
    if not paths.raw_permits.exists():
        raise FileNotFoundError(f"{paths.raw_permits} not found; run `huskyquiet download` first")
    raw = pd.read_parquet(paths.raw_permits)
    metadata = json.loads(paths.raw_metadata.read_text()) if paths.raw_metadata.exists() else None
    data_as_of = determine_data_as_of(raw, metadata)
    log.info("Cleaning %d raw rows (data as of %s)", len(raw), data_as_of.date())

    df, report = clean_permits(raw, data_as_of)
    df.to_parquet(paths.clean_permits, index=False)
    (paths.reports / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        log.info("  %-36s %s", k, v)
    log.info("Wrote %d clean rows to %s", len(df), paths.clean_permits)
    return df
