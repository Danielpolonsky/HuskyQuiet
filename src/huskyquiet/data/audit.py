"""Reproducible data-quality and leakage-diagnostic statistics.

Writes ``reports/data_audit_stats.json`` and ``reports/data_audit_stats.md``. The narrative
in ``docs/data_audit.md`` cites these generated numbers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from huskyquiet.config import Paths, get_paths

log = logging.getLogger(__name__)

# Applications in this window have had >= ~21 months for outcomes to materialize.
DIAG_WINDOW = ("2010-01-01", "2024-12-31")


def _q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def compute_audit(clean: pd.DataFrame) -> dict[str, Any]:
    con = duckdb.connect()
    con.register("p", clean)
    lo, hi = DIAG_WINDOW
    out: dict[str, Any] = {}
    out["n_records"] = len(clean)
    out["null_rate"] = clean.isna().mean().round(4).sort_values(ascending=False).to_dict()
    out["date_coverage"] = _q(con, """
        SELECT min(applied_date)::DATE AS min_applied, max(applied_date)::DATE AS max_applied,
               min(issued_date)::DATE AS min_issued, max(issued_date)::DATE AS max_issued
        FROM p""").astype(str).to_dict(orient="records")[0]
    out["applications_by_year"] = _q(con, """
        SELECT year(applied_date) AS year, count(*) AS n,
               avg((issued_date IS NULL)::INT) AS share_never_issued
        FROM p WHERE applied_date IS NOT NULL GROUP BY 1 ORDER BY 1""").to_dict(orient="records")
    out["status_counts"] = _q(con, """
        SELECT status_current, count(*) AS n FROM p GROUP BY 1 ORDER BY 2 DESC""").to_dict(orient="records")
    out["permit_type_counts"] = _q(con, """
        SELECT permit_type_mapped, permit_class, count(*) AS n FROM p GROUP BY ALL ORDER BY n DESC
        """).to_dict(orient="records")
    out["by_suffix"] = _q(con, """
        SELECT permit_suffix, count(*) AS n,
               avg((applied_date IS NULL)::INT) AS share_applied_missing,
               avg((issued_date IS NULL)::INT) AS share_issued_missing,
               avg(is_model_eligible::INT) AS share_model_eligible
        FROM p GROUP BY 1 ORDER BY 2 DESC""").to_dict(orient="records")
    out["applied_missing_by_status"] = _q(con, """
        SELECT status_current, count(*) AS n FROM p WHERE applied_date IS NULL
        GROUP BY 1 ORDER BY 2 DESC""").to_dict(orient="records")
    dur = _q(con, """
        SELECT datediff('day', applied_date, issued_date) AS d FROM p
        WHERE is_model_eligible AND issued_date IS NOT NULL AND applied_date >= '2005-01-01'""")["d"]
    out["applied_to_issued_days"] = {
        "n": int(dur.size),
        "quantiles": {str(q): float(dur.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "share_same_day": float((dur == 0).mean()),
        "share_within_90d": float((dur < 90).mean()),
        "share_within_730d": float((dur <= 730).mean()),
    }
    # Leakage diagnostic: if a field's missingness/value depends strongly on the eventual
    # outcome, it was probably populated at or after issuance.
    out["missingness_by_eventual_issuance"] = _q(con, f"""
        SELECT (issued_date IS NOT NULL) AS eventually_issued, count(*) AS n,
          avg((est_project_cost IS NULL)::INT) AS est_project_cost,
          avg((housing_units_added IS NULL)::INT) AS housing_units_added,
          avg((related_mup IS NULL)::INT) AS related_mup,
          avg((latitude IS NULL)::INT) AS latitude,
          avg((permit_type_desc IS NULL)::INT) AS permit_type_desc,
          avg((contractor_company_name IS NULL)::INT) AS contractor_company_name,
          avg((parent_permit_num IS NULL)::INT) AS parent_permit_num,
          avg((zoning IS NULL)::INT) AS zoning,
          avg((dwelling_unit_type IS NULL)::INT) AS dwelling_unit_type,
          avg((standard_plan IS NULL)::INT) AS standard_plan,
          avg((development_site IS NULL)::INT) AS development_site,
          avg((ready_to_issue_date IS NULL)::INT) AS ready_to_issue_date,
          avg((total_days_plan_review IS NULL)::INT) AS total_days_plan_review,
          avg(dependent_building::INT) AS dependent_building_true
        FROM p WHERE applied_date BETWEEN '{lo}' AND '{hi}' GROUP BY 1 ORDER BY 1""").to_dict(orient="records")
    out["housing_category_by_eventual_issuance"] = _q(con, f"""
        SELECT housing_category, count(*) FILTER (WHERE issued_date IS NOT NULL) AS issued,
               count(*) FILTER (WHERE issued_date IS NULL) AS never_issued
        FROM p WHERE applied_date BETWEEN '{lo}' AND '{hi}' GROUP BY 1 ORDER BY 2 DESC
        """).to_dict(orient="records")
    out["description_admin_keywords_by_eventual_issuance"] = _q(con, f"""
        SELECT (issued_date IS NOT NULL) AS eventually_issued, count(*) AS n,
          avg(regexp_matches(upper(description), 'CANCEL')::INT) AS cancel,
          avg(regexp_matches(upper(description), 'WITHDR')::INT) AS withdraw,
          avg(regexp_matches(upper(description), 'EXPIR')::INT) AS expire,
          avg(regexp_matches(upper(description), 'EXTEN[DS]')::INT) AS extension,
          avg(regexp_matches(description, '\\d{{1,2}}/\\d{{1,2}}/\\d{{2,4}}')::INT) AS embedded_date
        FROM p WHERE applied_date BETWEEN '{lo}' AND '{hi}' GROUP BY 1 ORDER BY 1""").to_dict(orient="records")
    out["est_project_cost"] = _q(con, """
        SELECT count(*) FILTER (WHERE est_project_cost = 0) AS n_zero,
               quantile_cont(est_project_cost, 0.5) AS median,
               quantile_cont(est_project_cost, 0.99) AS p99, max(est_project_cost) AS max
        FROM p""").to_dict(orient="records")[0]
    out["cost_missing_by_applied_year"] = _q(con, """
        SELECT year(applied_date) AS year, avg((est_project_cost IS NULL)::INT) AS share_missing
        FROM p WHERE applied_date >= '2005-01-01' GROUP BY 1 ORDER BY 1""").to_dict(orient="records")
    out["distinct_addresses"] = int(clean["address_key"].nunique())
    con.close()
    out["recent_cohort_pending_vs_issued"] = recent_cohort_diagnostic(clean)
    return out


def recent_cohort_diagnostic(clean: pd.DataFrame, months: int = 15) -> list[dict[str, Any]]:
    """Compare *currently pending* recent applications with recently issued ones.

    Pending records show what a record looks like before issuance. If a field is filled in
    during review/issuance, it is missing far more often among pending records. This is a
    sharper test than comparing old cohorts, where few never-issued records survive.
    Same-day issuances are excluded (they never enter a risk set).
    """
    from huskyquiet.features.text import KEYWORD_FEATURES, keyword_features

    as_of = clean["applied_date"].max()
    lo = as_of - pd.DateOffset(months=months)
    c = clean[clean["is_model_eligible"] & (clean["applied_date"] >= lo)
              & (clean["issued_date"].isna() | (clean["issued_date"] > clean["applied_date"]))].copy()
    c["group"] = np.where(c["issued_date"].isna(), "pending_or_never", "issued")
    kw = keyword_features(c["description"])
    fields = ["est_project_cost", "housing_units_added", "housing_units_removed", "related_mup",
              "permit_type_desc", "latitude", "zip", "address_key", "description"]
    rows = []
    for (grp, cls), g in c.groupby(["group", "permit_class"]):
        row: dict[str, Any] = {"group": grp, "permit_class": cls, "n": len(g)}
        row.update({f"null_{f}": round(float(g[f].isna().mean()), 4) for f in fields})
        row["median_est_project_cost"] = float(g["est_project_cost"].median())
        row.update({k: round(float(kw.loc[g.index, k].mean()), 4) for k in KEYWORD_FEATURES})
        rows.append(row)
    return rows


def _md(audit: dict[str, Any]) -> str:
    def table(records: list[dict]) -> str:
        return pd.DataFrame(records).to_string(index=False)

    parts = ["# Generated data-audit statistics (`huskyquiet audit`)", "",
             f"Records: {audit['n_records']:,}", "", "## Date coverage", "",
             json.dumps(audit["date_coverage"], indent=2)]
    for key in ("applications_by_year", "status_counts", "by_suffix", "applied_missing_by_status",
                "missingness_by_eventual_issuance", "housing_category_by_eventual_issuance",
                "description_admin_keywords_by_eventual_issuance", "recent_cohort_pending_vs_issued"):
        parts += ["", f"## {key}", "", "```", table(audit[key]), "```"]
    parts += ["", "## applied_to_issued_days", "", "```", json.dumps(audit["applied_to_issued_days"],
                                                                      indent=2), "```"]
    parts += ["", "## Null rate by field", "", "```",
              pd.Series(audit["null_rate"]).to_string(), "```"]
    return "\n".join(parts) + "\n"


def run_audit(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or get_paths()
    paths.ensure()
    clean = pd.read_parquet(paths.clean_permits)
    audit = compute_audit(clean)
    (paths.reports / "data_audit_stats.json").write_text(json.dumps(audit, indent=2, default=str))
    (paths.reports / "data_audit_stats.md").write_text(_md(audit), encoding="utf-8")
    log.info("Wrote audit statistics to %s", paths.reports)
    return audit
