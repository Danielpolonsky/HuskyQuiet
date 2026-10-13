"""Verified schema of the Seattle Building Permits dataset (Socrata id 76t5-zqzr).

Field names were verified against ``https://data.seattle.gov/api/views/76t5-zqzr.json``.
Each field is classified by whether its value could have been known at a historical
forecast date. This registry is the single source of truth used by cleaning, feature
engineering, and tests that guard against leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Availability(StrEnum):
    """When a field's value is known relative to a historical forecast date."""

    APPLICATION_TIME = "application_time"  # set at intake; usable as a feature
    APPLICATION_TIME_CAVEAT = "application_time_caveat"  # usable, but may be revised later
    OUTCOME = "outcome"  # reveals or is derived from the outcome; never a feature
    CURRENT_STATE = "current_state"  # present-day snapshot value; leaks the future
    IDENTIFIER = "identifier"  # keys / links; not predictive features
    SPARSE_RECENT = "sparse_recent"  # only populated for a subset of recent records


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str  # "text" | "number" | "date" | "url" | "location"
    availability: Availability
    note: str = ""


A = Availability

FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("permitnum", "text", A.IDENTIFIER, "Unique permit/application number, e.g. 6665840-CN."),
    FieldSpec("permitclass", "text", A.APPLICATION_TIME, "Single Family/Duplex, Commercial, ..."),
    FieldSpec("permitclassmapped", "text", A.APPLICATION_TIME, "Residential / Non-Residential."),
    FieldSpec("permittypemapped", "text", A.APPLICATION_TIME, "Building, Demolition, Roof, ..."),
    FieldSpec("permittypedesc", "text", A.APPLICATION_TIME, "New, Addition/Alteration, ..."),
    FieldSpec(
        "description",
        "text",
        A.APPLICATION_TIME_CAVEAT,
        "Free-text scope. Edited after intake in some records (e.g. 'PROJECT CANCELLED').",
    ),
    FieldSpec(
        "housingunits",
        "number",
        A.CURRENT_STATE,
        "Filled in during plan review: null for 46% of pending vs 9% of issued new SF applications.",
    ),
    FieldSpec("housingunitsremoved", "number", A.CURRENT_STATE, "See housingunits."),
    FieldSpec("housingunitsadded", "number", A.CURRENT_STATE, "See housingunits."),
    FieldSpec(
        "estprojectcost",
        "number",
        A.APPLICATION_TIME_CAVEAT,
        "Applicant-declared valuation; may be revised during review.",
    ),
    FieldSpec("applieddate", "date", A.APPLICATION_TIME, "Application date; defines risk-set entry."),
    FieldSpec("issueddate", "date", A.OUTCOME, "Used ONLY to construct labels / exit risk set."),
    FieldSpec("expiresdate", "date", A.OUTCOME, "Derived from issuance date."),
    FieldSpec("completeddate", "date", A.OUTCOME, "Post-issuance outcome."),
    FieldSpec("statuscurrent", "text", A.CURRENT_STATE, "Present-day status; leaks outcomes."),
    FieldSpec("relatedmup", "text", A.APPLICATION_TIME_CAVEAT, "Related Master Use Permit number."),
    FieldSpec("originaladdress1", "text", A.APPLICATION_TIME),
    FieldSpec("originalcity", "text", A.APPLICATION_TIME),
    FieldSpec("originalstate", "text", A.APPLICATION_TIME),
    FieldSpec("originalzip", "text", A.APPLICATION_TIME),
    FieldSpec(
        "contractorcompanyname",
        "text",
        A.CURRENT_STATE,
        "Typically recorded at/after issuance; missingness tracks the outcome.",
    ),
    FieldSpec("link", "url", A.IDENTIFIER, "Link to the Seattle Services Portal record."),
    FieldSpec("latitude", "number", A.APPLICATION_TIME),
    FieldSpec("longitude", "number", A.APPLICATION_TIME),
    FieldSpec("location1", "location", A.IDENTIFIER, "Duplicate of latitude/longitude."),
    FieldSpec("totaldaysplanreview", "number", A.CURRENT_STATE, "Cumulative review metric."),
    FieldSpec("daysinitialplanreview", "number", A.CURRENT_STATE),
    FieldSpec("daysplanreviewcity", "number", A.CURRENT_STATE),
    FieldSpec("daysoutcorrections", "number", A.CURRENT_STATE),
    FieldSpec("numberreviewcycles", "number", A.CURRENT_STATE),
    FieldSpec("initialreviewcompletedate", "date", A.CURRENT_STATE, "No history of when it was set."),
    FieldSpec("planreviewcompletedate", "date", A.CURRENT_STATE),
    FieldSpec("daysissuepermitcity", "number", A.OUTCOME),
    FieldSpec("readytoissuedate", "date", A.CURRENT_STATE, "Mostly populated near issuance."),
    FieldSpec("zoning", "text", A.SPARSE_RECENT),
    FieldSpec("dwellingunittype", "text", A.SPARSE_RECENT),
    FieldSpec("standardplan", "text", A.SPARSE_RECENT),
    FieldSpec("dependentbuilding", "text", A.APPLICATION_TIME_CAVEAT),
    FieldSpec("parentpermitnum", "text", A.APPLICATION_TIME_CAVEAT),
    FieldSpec("housingcategory", "text", A.APPLICATION_TIME_CAVEAT, "Derived category."),
    FieldSpec("development_site", "text", A.SPARSE_RECENT),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}
EXPECTED_FIELDS: frozenset[str] = frozenset(FIELD_BY_NAME)
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"permitnum", "applieddate", "issueddate", "permittypemapped", "permitclass", "statuscurrent"}
)
DATE_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.dtype == "date")
NUMBER_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.dtype == "number")
TEXT_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.dtype == "text")

# Fields that must never feed a model, directly or through derived features.
FORBIDDEN_FEATURE_SOURCES: frozenset[str] = frozenset(
    f.name for f in FIELDS if f.availability in (A.OUTCOME, A.CURRENT_STATE)
)
