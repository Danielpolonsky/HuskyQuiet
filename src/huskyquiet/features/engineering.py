"""Point-in-time feature engineering shared by training and inference.

Every feature for a snapshot ``(permit, t)`` is a function of (a) application-time permit
attributes and (b) the permit history restricted to events dated strictly before ``t``.
The same functions are used to build training snapshots and to score new requests, so
there is no train/serve skew.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from huskyquiet.config import HORIZON_DAYS, MAX_APPLICATION_AGE_DAYS, UW_LAT, UW_LON
from huskyquiet.features.text import KEYWORD_FEATURES, keyword_features

# Application-time attributes carried into snapshots. Outcome/current-state fields
# (issued_date, status_current, review metrics, contractor, zoning, housing_category, ...)
# are deliberately absent; tests assert this. housing_units_* are also excluded: they are
# filled in during plan review (null for 46% of pending vs 9% of issued new SF applications).
PERMIT_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "permit_num",
    "applied_date",
    "permit_class",
    "permit_type_mapped",
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

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "permit_class",
    "permit_type_mapped",
    "permit_type_desc",
    "permit_suffix",
    "zip",
)

CONTEXT_FEATURES: tuple[str, ...] = (
    "ctx_backlog_all",
    "ctx_backlog_grp",
    "ctx_applied_90d_grp",
    "ctx_issued_90d_grp",
    "ctx_issue_rate_lag_all",
    "ctx_issue_rate_lag_grp",
    "ctx_issue_rate_lag_age",
    "ctx_median_days_to_issue_365d_grp",
)
ADDRESS_FEATURES: tuple[str, ...] = ("addr_prior_apps", "addr_prior_issued_365d")

NUMERIC_FEATURES: tuple[str, ...] = (
    "age_days",
    "log_age_days",
    "log_est_cost",
    "est_cost_missing",
    "has_related_mup",
    "latitude",
    "longitude",
    "dist_uw_km",
    "applied_month",
    "forecast_month",
    *KEYWORD_FEATURES,
    "desc_missing",
    *CONTEXT_FEATURES,
    *ADDRESS_FEATURES,
)
# Heavy-tailed counts that linear models see on a log1p scale.
COUNT_FEATURES: tuple[str, ...] = (
    "ctx_backlog_all",
    "ctx_backlog_grp",
    "ctx_applied_90d_grp",
    "ctx_issued_90d_grp",
    "ctx_median_days_to_issue_365d_grp",
    "addr_prior_apps",
    "addr_prior_issued_365d",
)
FEATURE_COLUMNS: tuple[str, ...] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

AGE_BUCKET_EDGES: tuple[int, ...] = (0, 30, 60, 90, 180, 365, 10_000)
AGE_BUCKET_LABELS: tuple[str, ...] = ("0-30", "31-60", "61-90", "91-180", "181-365", "366+")


def age_bucket(age_days: pd.Series) -> pd.Series:
    return pd.cut(
        age_days, bins=list(AGE_BUCKET_EDGES), labels=list(AGE_BUCKET_LABELS), right=True
    ).astype("string")


def group_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["permit_type_mapped"].astype("string").fillna("NA")
        + "|"
        + df["permit_class"].astype("string").fillna("NA")
    )


def haversine_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    lat1, lon1 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lon1, float))
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def static_features(permits: pd.DataFrame) -> pd.DataFrame:
    """Application-time features, one row per permit (index aligned with input)."""
    out = pd.DataFrame(index=permits.index)
    for col in CATEGORICAL_FEATURES:
        out[col] = permits[col].astype("string")
    cost = pd.to_numeric(permits["est_project_cost"], errors="coerce")
    out["log_est_cost"] = np.log1p(cost.clip(lower=0, upper=5e9))
    out["est_cost_missing"] = cost.isna().astype("int8")
    out["has_related_mup"] = permits["related_mup"].notna().astype("int8")
    out["latitude"] = pd.to_numeric(permits["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(permits["longitude"], errors="coerce")
    out["dist_uw_km"] = haversine_km(out["latitude"], out["longitude"], UW_LAT, UW_LON)
    out["applied_month"] = pd.to_datetime(permits["applied_date"]).dt.month.astype("int8")
    out = out.join(keyword_features(permits["description"]))
    return out


def _as_us(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[us]")


def context_features(
    history: pd.DataFrame,
    dates: list[pd.Timestamp] | pd.DatetimeIndex,
    horizon_days: int = HORIZON_DAYS,
    max_age_days: int = MAX_APPLICATION_AGE_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Market/backlog context known at each forecast date ``t``.

    Returns three tables keyed by (forecast_date, grp), (forecast_date,) and
    (forecast_date, age_bucket). All aggregates use only events dated before ``t``:
    * backlog = risk-set size at t,
    * applied/issued counts over [t-90, t),
    * lagged issuance rate = share of the risk set at ``t - horizon`` issued in [t - horizon, t),
    * median applied->issued days for permits issued in [t-365, t) (durations >= 1 day).
    """
    from huskyquiet.features.snapshots import risk_set

    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique().sort_values()
    h = history[["permit_num", "applied_date", "issued_date"]].copy()
    h["grp"] = group_key(history)
    h["applied_date"] = _as_us(h["applied_date"])
    h["issued_date"] = _as_us(h["issued_date"])
    d = pd.DataFrame({"forecast_date": _as_us(pd.Series(dates))})

    grp_of = h.set_index("permit_num")["grp"]
    rs_now = risk_set(history, dates, horizon_days, max_age_days, with_labels=False)
    rs_now["grp"] = rs_now["permit_num"].map(grp_of)

    lag_dates = dates - pd.Timedelta(days=horizon_days)
    rs_lag = risk_set(history, lag_dates, horizon_days, max_age_days, with_labels=True)
    # Every label in rs_lag is observable at t: its window [t-h, t) ends at t.
    rs_lag["forecast_date"] = rs_lag["forecast_date"] + pd.Timedelta(days=horizon_days)
    rs_lag["grp"] = rs_lag["permit_num"].map(grp_of)
    rs_lag["age_bucket"] = age_bucket(rs_lag["age_days"])

    con = duckdb.connect()
    try:
        con.register("d", d)
        con.register("h", h)
        con.register("rs_now", rs_now)
        con.register("rs_lag", rs_lag)
        flows = con.execute(
            """
            SELECT d.forecast_date, h.grp,
                   count(*) FILTER (WHERE h.applied_date >= d.forecast_date - INTERVAL 90 DAY
                                     AND h.applied_date < d.forecast_date) AS ctx_applied_90d_grp,
                   count(*) FILTER (WHERE h.issued_date >= d.forecast_date - INTERVAL 90 DAY
                                     AND h.issued_date < d.forecast_date) AS ctx_issued_90d_grp,
                   median(datediff('day', h.applied_date, h.issued_date)) FILTER (
                       WHERE h.issued_date >= d.forecast_date - INTERVAL 365 DAY
                         AND h.issued_date < d.forecast_date
                         AND h.issued_date > h.applied_date) AS ctx_median_days_to_issue_365d_grp
            FROM d JOIN h
              ON (h.applied_date >= d.forecast_date - INTERVAL 365 DAY AND h.applied_date < d.forecast_date)
              OR (h.issued_date >= d.forecast_date - INTERVAL 365 DAY AND h.issued_date < d.forecast_date)
            GROUP BY ALL
            """
        ).df()
        backlog_grp = con.execute(
            "SELECT forecast_date, grp, count(*) AS ctx_backlog_grp FROM rs_now GROUP BY ALL"
        ).df()
        lag_grp = con.execute(
            "SELECT forecast_date, grp, avg(label) AS ctx_issue_rate_lag_grp FROM rs_lag GROUP BY ALL"
        ).df()
        overall = con.execute(
            """
            SELECT d.forecast_date,
                   coalesce(b.n, 0) AS ctx_backlog_all,
                   l.r AS ctx_issue_rate_lag_all
            FROM d
            LEFT JOIN (SELECT forecast_date, count(*) AS n FROM rs_now GROUP BY 1) b USING (forecast_date)
            LEFT JOIN (SELECT forecast_date, avg(label) AS r FROM rs_lag GROUP BY 1) l USING (forecast_date)
            """
        ).df()
        by_age = con.execute(
            """SELECT forecast_date, age_bucket, avg(label) AS ctx_issue_rate_lag_age
               FROM rs_lag GROUP BY ALL"""
        ).df()
    finally:
        con.close()

    by_grp = flows.merge(backlog_grp, on=["forecast_date", "grp"], how="outer").merge(
        lag_grp, on=["forecast_date", "grp"], how="outer"
    )
    for frame in (by_grp, overall, by_age):
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"]).astype("datetime64[ns]")
    for col in ("ctx_applied_90d_grp", "ctx_issued_90d_grp", "ctx_backlog_grp"):
        by_grp[col] = by_grp[col].fillna(0)
    return by_grp, overall, by_age


def address_features(snaps: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Prior activity at the same street address, using only events before ``t``."""
    s = snaps[["permit_num", "forecast_date", "address_key"]].dropna(subset=["address_key"]).copy()
    s["forecast_date"] = _as_us(s["forecast_date"])
    h = history[["permit_num", "address_key", "applied_date", "issued_date"]].dropna(
        subset=["address_key"]
    ).copy()
    h["applied_date"] = _as_us(h["applied_date"])
    h["issued_date"] = _as_us(h["issued_date"])
    con = duckdb.connect()
    try:
        con.register("s", s)
        con.register("h", h)
        out = con.execute(
            """
            SELECT s.permit_num, s.forecast_date,
                   count(h.permit_num) FILTER (WHERE h.applied_date < s.forecast_date) AS addr_prior_apps,
                   count(h.permit_num) FILTER (WHERE h.issued_date >= s.forecast_date - INTERVAL 365 DAY
                                                AND h.issued_date < s.forecast_date) AS addr_prior_issued_365d
            FROM s LEFT JOIN h
              ON h.address_key = s.address_key AND h.permit_num <> s.permit_num
             AND h.applied_date < s.forecast_date
            GROUP BY ALL
            """
        ).df()
    finally:
        con.close()
    out["forecast_date"] = pd.to_datetime(out["forecast_date"]).astype("datetime64[ns]")
    return out


def build_feature_frame(
    snaps: pd.DataFrame,
    history: pd.DataFrame,
    max_age_days: int = MAX_APPLICATION_AGE_DAYS,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Attach all model features to snapshot rows.

    ``snaps`` needs ``permit_num``, ``forecast_date``, ``age_days`` and the
    ``PERMIT_ATTRIBUTE_COLUMNS``; label columns, if present, are passed through untouched.
    """
    snaps = snaps.copy()
    snaps["forecast_date"] = pd.to_datetime(snaps["forecast_date"]).astype("datetime64[ns]")
    base_cols = [c for c in snaps.columns if c in ("permit_num", "forecast_date", "age_days", "label",
                                                     "outcome_days_until_issue")]
    out = snaps[base_cols].copy()

    perm = snaps.drop_duplicates("permit_num")[list(PERMIT_ATTRIBUTE_COLUMNS)].set_index("permit_num")
    static = static_features(perm)
    out = out.merge(static, left_on="permit_num", right_index=True, how="left", validate="m:1")

    out["log_age_days"] = np.log1p(out["age_days"].astype(float))
    out["forecast_month"] = out["forecast_date"].dt.month.astype("int8")
    out["grp"] = group_key(snaps)
    out["age_bucket"] = age_bucket(out["age_days"])

    by_grp, overall, by_age = context_features(
        history, out["forecast_date"].unique(), horizon_days, max_age_days
    )
    out = out.merge(by_grp, on=["forecast_date", "grp"], how="left", validate="m:1")
    out = out.merge(overall, on="forecast_date", how="left", validate="m:1")
    out = out.merge(by_age, on=["forecast_date", "age_bucket"], how="left", validate="m:1")
    for col in ("ctx_applied_90d_grp", "ctx_issued_90d_grp", "ctx_backlog_grp"):
        out[col] = out[col].fillna(0)

    addr = address_features(snaps, history)
    out = out.merge(addr, on=["permit_num", "forecast_date"], how="left", validate="1:1")

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string")
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    ordered = [c for c in base_cols if c not in FEATURE_COLUMNS] + ["grp", "age_bucket"]
    ordered += list(FEATURE_COLUMNS)
    return out[ordered]
