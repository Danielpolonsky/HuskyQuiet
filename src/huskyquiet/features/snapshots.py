"""Historical forecast snapshots (the unit of analysis for the prediction problem).

Definitions (all dates are calendar days; a forecast is made at the *start* of day ``t``,
so only events dated strictly before ``t`` are known):

* **Risk set at t**: model-eligible permits with ``applied_date < t`` that were not issued
  before ``t`` (``issued_date IS NULL OR issued_date >= t``) and whose age
  ``t - applied_date`` is at most ``MAX_APPLICATION_AGE_DAYS``.
* **Label**: 1 if ``t <= issued_date < t + HORIZON_DAYS`` else 0. Never-issued
  applications (withdrawn, cancelled, still pending) are negatives.
* **Complete observation window**: only forecast dates with ``t + HORIZON_DAYS <=
  data_as_of`` receive labels, so no label is right-censored.

Permit status is intentionally NOT used to define the risk set: the dataset only stores
the *current* status, and there is no record of when a withdrawal/cancellation happened.
The age cap bounds how long a never-issued application can remain in the risk set.
"""

from __future__ import annotations

import json
import logging

import duckdb
import pandas as pd

from huskyquiet.config import (
    FIRST_FORECAST_DATE,
    HORIZON_DAYS,
    MAX_APPLICATION_AGE_DAYS,
    Paths,
    get_paths,
)
from huskyquiet.features.engineering import PERMIT_ATTRIBUTE_COLUMNS, build_feature_frame

log = logging.getLogger(__name__)

HISTORY_COLUMNS = (
    "permit_num",
    "applied_date",
    "issued_date",
    "permit_type_mapped",
    "permit_class",
    "address_key",
)


def monthly_forecast_dates(
    first: str | pd.Timestamp, data_as_of: pd.Timestamp, horizon_days: int = HORIZON_DAYS
) -> pd.DatetimeIndex:
    """First-of-month forecast dates whose full label window is observed."""
    last_allowed = pd.Timestamp(data_as_of) - pd.Timedelta(days=horizon_days)
    return pd.date_range(pd.Timestamp(first), last_allowed, freq="MS")


def eligible_history(clean: pd.DataFrame) -> pd.DataFrame:
    """Permit timeline table used for risk sets and point-in-time context features."""
    cols = list(dict.fromkeys(HISTORY_COLUMNS + PERMIT_ATTRIBUTE_COLUMNS))
    hist = clean.loc[clean["is_model_eligible"], cols].copy()
    return hist.reset_index(drop=True)


def risk_set(
    history: pd.DataFrame,
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    horizon_days: int = HORIZON_DAYS,
    max_age_days: int = MAX_APPLICATION_AGE_DAYS,
    with_labels: bool = True,
) -> pd.DataFrame:
    """Return one row per (permit, forecast_date) in the risk set, optionally labeled."""
    d = pd.DataFrame({"forecast_date": pd.to_datetime(pd.Series(list(dates))).astype("datetime64[us]")})
    h = history[["permit_num", "applied_date", "issued_date"]].copy()
    h["applied_date"] = h["applied_date"].astype("datetime64[us]")
    h["issued_date"] = h["issued_date"].astype("datetime64[us]")
    label_sql = (
        f""",
        CAST(h.issued_date IS NOT NULL
             AND h.issued_date < d.forecast_date + INTERVAL {int(horizon_days)} DAY AS TINYINT) AS label,
        datediff('day', d.forecast_date, h.issued_date) AS outcome_days_until_issue"""
        if with_labels
        else ""
    )
    con = duckdb.connect()
    try:
        con.register("d", d)
        con.register("h", h)
        out = con.execute(
            f"""
            SELECT h.permit_num, d.forecast_date,
                   datediff('day', h.applied_date, d.forecast_date) AS age_days
                   {label_sql}
            FROM d JOIN h
              ON h.applied_date < d.forecast_date
             AND h.applied_date >= d.forecast_date - INTERVAL {int(max_age_days)} DAY
             AND (h.issued_date IS NULL OR h.issued_date >= d.forecast_date)
            ORDER BY d.forecast_date, h.permit_num
            """
        ).df()
    finally:
        con.close()
    out["forecast_date"] = pd.to_datetime(out["forecast_date"])
    return out


def build_snapshots(
    clean: pd.DataFrame,
    data_as_of: pd.Timestamp,
    first_forecast: str = FIRST_FORECAST_DATE,
    horizon_days: int = HORIZON_DAYS,
    max_age_days: int = MAX_APPLICATION_AGE_DAYS,
) -> pd.DataFrame:
    history = eligible_history(clean)
    dates = monthly_forecast_dates(first_forecast, data_as_of, horizon_days)
    log.info(
        "Building snapshots for %d forecast dates (%s .. %s), horizon=%dd, max_age=%dd",
        len(dates), dates.min().date(), dates.max().date(), horizon_days, max_age_days,
    )
    rs = risk_set(history, dates, horizon_days, max_age_days, with_labels=True)
    assert (rs["forecast_date"] + pd.Timedelta(days=horizon_days) <= data_as_of).all()
    log.info("Risk set: %d snapshot rows covering %d permits", len(rs), rs["permit_num"].nunique())
    snaps = rs.merge(history[list(PERMIT_ATTRIBUTE_COLUMNS)], on="permit_num", how="left", validate="m:1")
    feats = build_feature_frame(snaps, history, max_age_days=max_age_days)
    return feats


def run_build_snapshots(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or get_paths()
    paths.ensure()
    clean = pd.read_parquet(paths.clean_permits)
    report = json.loads((paths.reports / "cleaning_report.json").read_text())
    data_as_of = pd.Timestamp(report["data_as_of"])
    snaps = build_snapshots(clean, data_as_of)
    snaps.to_parquet(paths.snapshots, index=False)
    summary = {
        "data_as_of": str(data_as_of.date()),
        "horizon_days": HORIZON_DAYS,
        "max_application_age_days": MAX_APPLICATION_AGE_DAYS,
        "n_snapshots": len(snaps),
        "n_permits": int(snaps["permit_num"].nunique()),
        "n_forecast_dates": int(snaps["forecast_date"].nunique()),
        "first_forecast_date": str(snaps["forecast_date"].min().date()),
        "last_forecast_date": str(snaps["forecast_date"].max().date()),
        "positive_rate": float(snaps["label"].mean()),
        "mean_snapshots_per_permit": float(snaps.groupby("permit_num").size().mean()),
    }
    (paths.reports / "snapshot_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Snapshot summary: %s", summary)
    return snaps
