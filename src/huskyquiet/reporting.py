"""Read saved evaluation reports. The web app displays these values; it does not recompute them."""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from huskyquiet.config import Paths, get_paths

MODEL_NAMES = {
    "hgb_calibrated_platt": "Gradient-boosted trees with Platt calibration",
    "historical_rate_baseline": "Historical-rate baseline",
    "logistic_calibrated_isotonic": "Logistic regression, isotonic-calibrated",
    "recent_rate_baseline": "Recent-rate baseline",
    "hgb_raw": "Gradient-boosted trees, uncalibrated",
}

FEATURE_LABELS = {
    "age_days": "Application age",
    "permit_type_desc": "Permit category",
    "log_est_cost": "Estimated project cost",
    "ctx_issue_rate_lag_age": "Recent issuance rate, same age band",
    "zip": "ZIP code",
    "ctx_backlog_grp": "Pending applications of the same type",
    "longitude": "Longitude",
    "latitude": "Latitude",
    "addr_prior_issued_365d": "Issuances at the address, prior year",
    "kw_establish_use": "Description mentions establishing a use",
    "permit_class": "Permit class",
    "has_related_mup": "Linked land-use (MUP) application",
    "kw_stfi": "Subject to field inspection",
    "dist_uw_km": "Distance to UW",
    "forecast_month": "Calendar month",
    "kw_tenant_improvement": "Description mentions tenant improvement",
    "ctx_median_days_to_issue_365d_grp": "Typical days to issue, same type",
    "kw_multifamily": "Description mentions multifamily",
    "kw_demolition": "Description mentions demolition",
    "kw_earthwork": "Description mentions earthwork",
    "kw_adu": "Description mentions an ADU",
    "kw_deck_porch": "Description mentions a deck or porch",
    "ctx_issue_rate_lag_grp": "Recent issuance rate, same type",
    "permit_type_mapped": "Permit type",
    "ctx_applied_90d_grp": "Recent applications of the same type",
}


@dataclass(frozen=True)
class Period:
    role: str
    model: str
    start: str
    end: str
    note: str


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(paths: Paths | None = None) -> dict:
    paths = paths or get_paths()
    path = paths.reports / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics at {path}. Run `python -m huskyquiet train`.")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a metrics object")
    return payload


def load_model_metadata(paths: Paths | None = None) -> dict:
    paths = paths or get_paths()
    path = paths.model_dir / "production" / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No production metadata at {path}.")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a metadata object")
    return payload


def evaluation_headline(metrics: dict) -> dict[str, float | str | int]:
    """Pull the locked test-set comparison out of ``reports/metrics.json``."""
    key = str(metrics["selection"]["selected_test_key"])
    chosen = metrics["test"][key]
    baseline = metrics["test"]["historical_rate_baseline"]
    return {
        "model_key": key,
        "roc_auc": float(chosen["roc_auc"]),
        "pr_auc": float(chosen["pr_auc"]),
        "brier": float(chosen["brier"]),
        "ece": float(chosen["ece"]),
        "baseline_roc_auc": float(baseline["roc_auc"]),
        "baseline_pr_auc": float(baseline["pr_auc"]),
        "baseline_brier": float(baseline["brier"]),
        "baseline_ece": float(baseline["ece"]),
        "n": int(chosen["n"]),
        "prevalence": float(chosen["prevalence"]),
    }


def model_name(key: str) -> str:
    return MODEL_NAMES.get(key, key)


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def model_timeline(metrics: dict, metadata: dict | None, first_forecast: str) -> list[Period]:
    """Forecast-date ranges for each fitting step, derived from the saved split boundaries.

    Forecast dates fall on the first of each month. A training snapshot needs its whole
    90-day outcome window to end on or before the next period's start.
    """
    split = metrics["split"]
    horizon = int(split.get("horizon_days", 90))
    val_start = date.fromisoformat(split["val_start"])
    test_start = date.fromisoformat(split["test_start"])
    last_test = date.fromisoformat(split["last_forecast"])
    periods = [
        Period(
            "Training",
            "Evaluation copy",
            _month(first_forecast),
            _month(_last_month_before(val_start, horizon)),
            f"Outcome windows close by {_day(val_start)}.",
        ),
        Period(
            "Calibration",
            "Evaluation copy",
            _month(val_start),
            _month(_last_month_before(test_start, horizon)),
            "Platt scaling fitted on this period.",
        ),
        Period(
            "Held-out test",
            "Evaluation copy",
            _month(test_start),
            _month(last_test),
            "Never used to fit the evaluation copy. All reported metrics come from here.",
        ),
    ]
    if metadata:
        data_as_of = date.fromisoformat(str(metadata["data_as_of"]))
        method = str(metadata.get("calibration", "isotonic")).capitalize()
        periods += [
            Period(
                "Training",
                "Deployed model",
                _month(first_forecast),
                _month(_last_month_before(test_start, horizon)),
                f"Outcome windows close by {_day(test_start)}.",
            ),
            Period(
                "Calibration",
                "Deployed model",
                _month(test_start),
                _month(last_test),
                f"{method} calibration refitted on the test period, so there is no untouched test set for it.",
            ),
            Period(
                "Scoring",
                "Deployed model",
                _day(data_as_of + timedelta(days=1)),
                "",
                f"Applications still pending at the start of this day (permit data through {_day(data_as_of)}).",
            ),
        ]
    return periods


def data_counts(paths: Paths | None = None) -> dict[str, int | str]:
    """Row counts from the saved cleaning and snapshot reports, when present."""
    paths = paths or get_paths()
    out: dict[str, int | str] = {}
    for name, keys in (
        ("cleaning_report.json", ("n_raw", "model_eligible", "data_as_of")),
        ("snapshot_summary.json", ("n_snapshots", "n_permits", "first_forecast_date", "last_forecast_date")),
    ):
        path = paths.reports / name
        if not path.exists():
            continue
        payload = read_json(path)
        if isinstance(payload, dict):
            out.update({key: payload[key] for key in keys if key in payload})
    return out


def _last_month_before(boundary: date, horizon_days: int) -> date:
    day = date(boundary.year, boundary.month, 1)
    while day + timedelta(days=horizon_days) > boundary:
        year, month = (day.year, day.month - 1) if day.month > 1 else (day.year - 1, 12)
        day = date(year, month, 1)
    return day


def _month(value: date | str) -> str:
    day = date.fromisoformat(value) if isinstance(value, str) else value
    return f"{calendar.month_abbr[day.month]} {day.year}"


def _day(value: date) -> str:
    return f"{calendar.month_abbr[value.month]} {value.day}, {value.year}"
