"""Deterministic summaries of scored permits around a location or inside an area.

Every sentence in ``difference_points`` is computed from two summaries. There is no
language model, and no sentence ranks homes or claims anything about noise,
construction starts, or disturbance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from huskyquiet.geo import with_distance

ISSUANCE_NOTE = (
    "Issuance is the City's approval of a permit application. "
    "It is not a construction start date, and HuskyQuiet does not predict noise."
)
EXPECTED_HELP = (
    "Adds the issuance probabilities of nearby applications. "
    "This counts permits, not unique construction projects."
)
LIKELY_THRESHOLD = 0.5

# Sequential teal. Darker means a higher predicted issuance probability, not more danger.
PROBABILITY_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (214, 224, 226)),
    (0.2, (168, 198, 202)),
    (0.4, (110, 164, 170)),
    (0.6, (54, 122, 130)),
    (0.8, (18, 84, 94)),
    (1.0, (8, 48, 58)),
)
RAMP_HEX: tuple[str, ...] = ("#D6E0E2", "#A8C6CA", "#6EA4AA", "#367A82", "#12545E", "#08303A")

KEYWORD_LABELS: dict[str, str] = {
    "kw_new_construction": "New construction",
    "kw_demolition": "Demolition",
    "kw_adu": "Accessory dwelling unit",
    "kw_townhouse": "Townhouse",
    "kw_multifamily": "Apartment or multifamily",
    "kw_tenant_improvement": "Tenant improvement",
    "kw_stfi": "Subject to field inspection",
    "kw_change_of_use": "Change of use",
    "kw_establish_use": "Establish use",
    "kw_earthwork": "Grading or earthwork",
    "kw_seismic": "Seismic retrofit",
    "kw_solar": "Solar",
    "kw_deck_porch": "Deck or porch",
    "kw_basement": "Basement",
    "kw_addition": "Addition",
}

PROBABILITY_COLUMN = "Chance issued within 90 days (%)"


def probability_color(probability: float) -> list[int]:
    """RGBA color for a predicted issuance probability in ``[0, 1]``."""
    p = min(1.0, max(0.0, float(probability)))
    left_p, left_c = PROBABILITY_RAMP[0]
    right_p, right_c = PROBABILITY_RAMP[-1]
    for stop_p, stop_c in PROBABILITY_RAMP[1:]:
        right_p, right_c = stop_p, stop_c
        if p <= stop_p:
            break
        left_p, left_c = stop_p, stop_c
    span = right_p - left_p
    t = 0.0 if span == 0 else (p - left_p) / span
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(left_c, right_c, strict=True))
    return [int(rgb[0]), int(rgb[1]), int(rgb[2]), 214]


def format_cost(value) -> str:
    if value is None or pd.isna(value):
        return "Not reported"
    return f"${float(value):,.0f}"


def format_probability(value: float) -> str:
    return f"{float(value):.0%}"


def format_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return "—"
    if distance_m < 1000:
        return f"{distance_m:,.0f} m"
    return f"{distance_m / 1000:.2f} km"


@dataclass(frozen=True)
class LocationSummary:
    """Aggregate of one scored set of pending permits."""

    label: str
    scope: str
    n_permits: int
    expected_issuances_90d: float
    nearest_m: float | None
    median_probability: float | None
    n_likely: int
    by_description: dict[str, int]
    by_class: dict[str, int]
    top: tuple[dict, ...]


def apply_filters(
    frame: pd.DataFrame,
    classes: Sequence[str] | None = None,
    min_probability: float = 0.0,
) -> pd.DataFrame:
    """Keep permits in ``classes`` (all when ``None``) at or above ``min_probability``."""
    out = frame
    if classes is not None:
        out = out.loc[out["permit_class"].isin(list(classes))]
    if min_probability > 0:
        out = out.loc[pd.to_numeric(out["probability_issued_90d"], errors="coerce") >= min_probability]
    return out


def summarize_permits(
    frame: pd.DataFrame,
    label: str,
    scope: str,
    top_n: int = 5,
    distance_col: str | None = "distance_m",
) -> LocationSummary:
    """Summarize a set of already-scored permits.

    ``expected_issuances_90d`` is the sum of predicted issuance probabilities: an
    expected count of permits issued, not a count of projects that start work.
    """
    work = frame
    if work.empty:
        return LocationSummary(label, scope, 0, 0.0, None, None, 0, {}, {}, ())
    probabilities = pd.to_numeric(work["probability_issued_90d"], errors="coerce").fillna(0.0)
    nearest = None
    if distance_col and distance_col in work.columns and work[distance_col].notna().any():
        nearest = float(work[distance_col].min())
    ranked = work.assign(_p=probabilities).sort_values(["_p", "permit_num"], ascending=[False, True])
    top_rows = []
    for row in ranked.head(top_n).to_dict(orient="records"):
        top_rows.append({
            "permit_num": row.get("permit_num"),
            "address": _text(row.get("address_key"), "Address not recorded"),
            "permit_class": _text(row.get("permit_class"), "Class not recorded"),
            "permit_type_desc": _text(row.get("permit_type_desc"), "Type not recorded"),
            "description": _text(row.get("description"), "No description on file."),
            "est_project_cost": None if _missing(row.get("est_project_cost")) else float(row["est_project_cost"]),
            "probability_issued_90d": float(row["_p"]),
            "distance_m": None if _missing(row.get("distance_m")) else float(row["distance_m"]),
            "age_days": None if _missing(row.get("age_days")) else int(row["age_days"]),
        })
    return LocationSummary(
        label=label,
        scope=scope,
        n_permits=int(len(work)),
        expected_issuances_90d=float(probabilities.sum()),
        nearest_m=nearest,
        median_probability=float(probabilities.median()),
        n_likely=int((probabilities >= LIKELY_THRESHOLD).sum()),
        by_description=_counts(work, "permit_type_desc"),
        by_class=_counts(work, "permit_class"),
        top=tuple(top_rows),
    )


def nearby_summary(
    permits: pd.DataFrame,
    latitude: float,
    longitude: float,
    radius_m: float,
    label: str,
) -> LocationSummary:
    nearby = with_distance(permits, latitude, longitude, radius_m)
    return summarize_permits(nearby, label, f"within {format_distance(radius_m)}")


def difference_points(
    left: LocationSummary,
    right: LocationSummary,
    left_name: str = "the first location",
    right_name: str = "the second location",
    identical: bool = False,
) -> list[str]:
    """Plain-language facts that differ between two summaries. Computed, never generated."""
    if identical:
        return [
            "Both entries resolve to the same point, so every figure below is the same: "
            f"{left.n_permits} pending {_permits(left.n_permits)} {left.scope} and "
            f"{left.expected_issuances_90d:.1f} expected to be issued within 90 days."
        ]
    points = [_count_point(left, right, left_name, right_name)]
    points.append(_expected_point(left, right, left_name, right_name))
    nearest = _nearest_point(left, right, left_name, right_name)
    if nearest:
        points.append(nearest)
    if left.n_permits and right.n_permits:
        points.append(
            f"{left.n_likely} of {left.n_permits} near {left_name} and {right.n_likely} of {right.n_permits} "
            f"near {right_name} have at least a {LIKELY_THRESHOLD:.0%} predicted chance of issuance."
        )
    for name, summary in ((left_name, left), (right_name, right)):
        mix = _mix_point(name, summary)
        if mix:
            points.append(mix)
    return points


def explanatory_rows(row: pd.Series) -> list[tuple[str, str]]:
    """Readable model inputs for one scored permit. Missing values are omitted."""
    rows: list[tuple[str, str]] = []
    if not _missing(row.get("age_days")):
        rows.append(("Application age", f"{int(row['age_days'])} days"))
    if not _missing(row.get("dist_uw_km")):
        rows.append(("Distance to UW", f"{float(row['dist_uw_km']):.2f} km"))
    if not _missing(row.get("est_project_cost")):
        rows.append(("Estimated project cost", format_cost(row["est_project_cost"])))
    elif _flag(row.get("est_cost_missing")):
        rows.append(("Estimated project cost", "Not reported"))
    if not _missing(row.get("ctx_issue_rate_lag_age")):
        rows.append((
            "Recent issuance rate, same age band",
            format_probability(float(row["ctx_issue_rate_lag_age"])),
        ))
    if not _missing(row.get("ctx_backlog_grp")):
        rows.append(("Pending applications of this type and class", f"{float(row['ctx_backlog_grp']):.0f}"))
    if not _missing(row.get("ctx_applied_90d_grp")):
        rows.append(("Applications of this type and class, prior 90 days", f"{float(row['ctx_applied_90d_grp']):.0f}"))
    if not _missing(row.get("addr_prior_apps")):
        rows.append(("Earlier applications at this address", f"{float(row['addr_prior_apps']):.0f}"))
    if not _missing(row.get("addr_prior_issued_365d")):
        rows.append(("Issuances at this address, prior year", f"{float(row['addr_prior_issued_365d']):.0f}"))
    keywords = [label for key, label in KEYWORD_LABELS.items() if _flag(row.get(key))]
    if keywords:
        rows.append(("Scope keywords used by the model", ", ".join(keywords)))
    elif _flag(row.get("desc_missing")):
        rows.append(("Scope keywords used by the model", "Description missing"))
    else:
        rows.append(("Scope keywords used by the model", "None of the tracked scope terms"))
    return rows


def permit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The permits in ``frame`` with readable column names, in the order given."""
    out = pd.DataFrame({
        "Permit number": _column(frame, "permit_num"),
        "Address": _column(frame, "address_key").fillna("Address not recorded"),
        "Permit type": _column(frame, "permit_type_mapped"),
        "Class": _column(frame, "permit_class"),
        "Category": _column(frame, "permit_type_desc").fillna("Not recorded"),
        "Description": _column(frame, "description"),
        "Applied": pd.to_datetime(_column(frame, "applied_date"), errors="coerce").dt.date,
        "Age (days)": pd.to_numeric(_column(frame, "age_days"), errors="coerce"),
        "Estimated cost (USD)": pd.to_numeric(_column(frame, "est_project_cost"), errors="coerce"),
    })
    if "distance_m" in frame.columns:
        out["Distance (m)"] = pd.to_numeric(frame["distance_m"], errors="coerce").round(0).to_numpy()
    probability = pd.to_numeric(_column(frame, "probability_issued_90d"), errors="coerce")
    out[PROBABILITY_COLUMN] = (probability * 100).round(1).to_numpy()
    out["Latitude"] = pd.to_numeric(_column(frame, "latitude"), errors="coerce").to_numpy()
    out["Longitude"] = pd.to_numeric(_column(frame, "longitude"), errors="coerce").to_numpy()
    return out


def permits_csv(frame: pd.DataFrame) -> bytes:
    return permit_rows(frame).to_csv(index=False, lineterminator="\n").encode("utf-8")


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name].reset_index(drop=True)
    return pd.Series([None] * len(frame), dtype="object")


def _count_point(left: LocationSummary, right: LocationSummary, left_name: str, right_name: str) -> str:
    if left.n_permits == right.n_permits:
        return f"Both have {left.n_permits} pending {_permits(left.n_permits)} {left.scope}."
    more, less, more_name = (left, right, left_name) if left.n_permits > right.n_permits else (right, left, right_name)
    gap = more.n_permits - less.n_permits
    return (
        f"{_capital(more_name)} has {gap} more pending {_permits(gap)} {more.scope} "
        f"({more.n_permits} compared with {less.n_permits})."
    )


def _expected_point(left: LocationSummary, right: LocationSummary, left_name: str, right_name: str) -> str:
    a, b = round(left.expected_issuances_90d, 1), round(right.expected_issuances_90d, 1)
    if a == b:
        return f"Both are expected to see about {a:.1f} of those permits issued within 90 days."
    return (
        f"Expected permits issued within 90 days: {a:.1f} near {left_name} "
        f"and {b:.1f} near {right_name}."
    )


def _mix_point(name: str, summary: LocationSummary) -> str:
    if summary.n_permits == 0 or not summary.by_description:
        return ""
    kind, count = max(summary.by_description.items(), key=lambda item: (item[1], item[0]))
    return f"The most common category near {name} is {kind} ({count} of {summary.n_permits})."


def _nearest_point(left: LocationSummary, right: LocationSummary, left_name: str, right_name: str) -> str:
    if left.nearest_m is not None and right.nearest_m is not None:
        return (
            f"The nearest pending permit is {format_distance(left.nearest_m)} from {left_name} "
            f"and {format_distance(right.nearest_m)} from {right_name}."
        )
    for name, summary, other_name, other in (
        (left_name, left, right_name, right),
        (right_name, right, left_name, left),
    ):
        if summary.nearest_m is not None:
            return (
                f"The nearest pending permit to {name} is {format_distance(summary.nearest_m)} away. "
                f"{_capital(other_name)} has none {other.scope}."
            )
    return ""


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    series = frame[column].fillna("Not recorded").astype(str)
    return {str(k): int(v) for k, v in series.value_counts().items()}


def _permits(n: int) -> str:
    return "permit" if n == 1 else "permits"


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _missing(value) -> bool:
    return value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value))


def _flag(value) -> bool:
    if _missing(value):
        return False
    return float(value) >= 0.5


def _text(value, fallback: str) -> str:
    if _missing(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()
