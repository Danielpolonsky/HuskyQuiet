"""Deterministic summaries of scored permits around a home or inside an area.

Every sentence in ``comparison_text`` is computed from the summary objects. There is no
language model and no claim about noise, construction starts, or disturbance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from huskyquiet.geo import in_seattle, with_distance

ISSUANCE_NOT_NOISE = (
    "These figures are forecasts of permit issuance, a regulatory milestone. "
    "They are a proxy for nearby development activity in the City's permit record. "
    "They do not measure construction starts, noise, or the chance a residence will be disturbed."
)

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
    return f"{float(value):.1%}"


@dataclass(frozen=True)
class LocationSummary:
    """Aggregate of one scored set of pending permits."""

    label: str
    scope: str
    n_permits: int
    expected_issuances_90d: float
    nearest_m: float | None
    median_probability: float | None
    by_description: dict[str, int]
    by_class: dict[str, int]
    top: tuple[dict, ...]


def summarize_permits(
    frame: pd.DataFrame,
    label: str,
    scope: str,
    top_n: int = 5,
    distance_col: str | None = "distance_m",
) -> LocationSummary:
    """Summarize a set of already-scored permits.

    ``expected_issuances_90d`` is the sum of predicted issuance probabilities. It is an
    expected count of issuances, not a count of projects that will start construction.
    """
    work = frame
    if work.empty:
        return LocationSummary(label, scope, 0, 0.0, None, None, {}, {}, ())
    probabilities = pd.to_numeric(work["probability_issued_90d"], errors="coerce").fillna(0.0)
    nearest = None
    if distance_col and distance_col in work.columns and work[distance_col].notna().any():
        nearest = float(work[distance_col].min())
    by_description = _counts(work, "permit_type_desc")
    by_class = _counts(work, "permit_class")
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
        by_description=by_description,
        by_class=by_class,
        top=tuple(top_rows),
    )


@dataclass(frozen=True)
class Pin:
    label: str
    latitude: float
    longitude: float
    source: str


def resolve_pin(address: str, latitude: float | None, longitude: float | None, manual: bool, geocoder) -> Pin:
    """Resolve a form submission to a Seattle point, or raise ``ValueError`` / ``GeocodeError``."""
    from huskyquiet.geocode import GeocodeError

    text = " ".join((address or "").split())
    if manual:
        if latitude is None or longitude is None:
            raise ValueError("Enter both latitude and longitude.")
        if not in_seattle(latitude, longitude):
            raise ValueError("Those coordinates are outside the Seattle bounding box used by HuskyQuiet.")
        return Pin(text or f"{latitude:.5f}, {longitude:.5f}", float(latitude), float(longitude), "coordinates")
    if not text:
        raise ValueError("Enter a Seattle address, or choose coordinates instead.")
    try:
        hit = geocoder.geocode(text)
    except GeocodeError:
        raise
    return Pin(hit.display_name, hit.latitude, hit.longitude, hit.source)


def nearby_summary(
    permits: pd.DataFrame,
    latitude: float,
    longitude: float,
    radius_m: float,
    label: str,
) -> LocationSummary:
    nearby = with_distance(permits, latitude, longitude, radius_m)
    scope = f"within {int(radius_m)} m"
    return summarize_permits(nearby, label, scope)


def comparison_text(left: LocationSummary, right: LocationSummary) -> str:
    """Plain-language comparison using only counts computed from the two summaries."""
    sentences = [
        (
            f"{_quote(left.label)} ({left.scope}) has {left.n_permits} pending "
            f"{_permit_word(left.n_permits)} with usable coordinates. "
            f"{_quote(right.label)} ({right.scope}) has {right.n_permits}."
        )
    ]
    sentences.append(
        "The expected number of those permits issued in the next 90 days is "
        f"{left.expected_issuances_90d:.1f} near the first place and "
        f"{right.expected_issuances_90d:.1f} near the second. "
        "Each figure is the sum of the individual predicted issuance probabilities."
    )
    if left.nearest_m is not None and right.nearest_m is not None:
        sentences.append(
            f"The nearest pending application is {_meters(left.nearest_m)} from the first place "
            f"and {_meters(right.nearest_m)} from the second."
        )
    elif left.median_probability is not None and right.median_probability is not None:
        sentences.append(
            "The median predicted issuance probability is "
            f"{format_probability(left.median_probability)} in the first set and "
            f"{format_probability(right.median_probability)} in the second."
        )
    sentences.append(_mix_sentence("first", left))
    sentences.append(_mix_sentence("second", right))
    sentences.append(ISSUANCE_NOT_NOISE)
    return " ".join(part for part in sentences if part)


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


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    series = frame[column].fillna("Not recorded").astype(str)
    return {str(k): int(v) for k, v in series.value_counts().items()}


def _mix_sentence(which: str, summary: LocationSummary) -> str:
    if summary.n_permits == 0 or not summary.by_description:
        return ""
    kind, count = max(summary.by_description.items(), key=lambda item: item[1])
    return (
        f"In the {which} set the most common description category is {kind} "
        f"({count} of {summary.n_permits})."
    )


def _permit_word(n: int) -> str:
    return "application" if n == 1 else "applications"


def _quote(text: str) -> str:
    return text if len(text) < 80 else text[:77] + "..."


def _meters(distance_m: float) -> str:
    if distance_m < 1000:
        return f"{distance_m:.0f} m"
    return f"{distance_m / 1000:.2f} km"


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
