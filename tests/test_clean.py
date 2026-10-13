import pandas as pd

from huskyquiet.data.clean import clean_permits
from huskyquiet.data.schema import EXPECTED_FIELDS


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {f: None for f in EXPECTED_FIELDS if f not in ("location1",)}
    base.update({"permitclass": "Commercial", "permittypemapped": "Building", "statuscurrent": "Issued",
                 "socrata_updated_at": "2026-01-01T00:00:00.000Z"})
    return pd.DataFrame([{**base, **r} for r in rows]).astype("string")


AS_OF = pd.Timestamp("2026-10-10")


def test_sentinel_and_future_dates_are_nulled():
    df, rep = clean_permits(_raw([
        {"permitnum": "1-CN", "applieddate": "1900-01-01T00:00:00.000", "issueddate": "2020-01-01T00:00:00.000"},
        {"permitnum": "2-CN", "applieddate": "2020-01-01T00:00:00.000", "issueddate": "2030-01-01T00:00:00.000"},
    ]), AS_OF)
    r1, r2 = df.set_index("permit_num").loc["1-CN"], df.set_index("permit_num").loc["2-CN"]
    assert pd.isna(r1["applied_date"]) and bool(r1["q_applieddate_sentinel"])
    assert pd.isna(r2["issued_date"]) and rep["issueddate_future_nulled"] == 1
    assert not bool(r1["is_model_eligible"]) and bool(r2["is_model_eligible"])


def test_issued_before_applied_is_flagged_and_ineligible():
    df, rep = clean_permits(_raw([
        {"permitnum": "1-CN", "applieddate": "2020-05-01T00:00:00.000", "issueddate": "2020-01-01T00:00:00.000"},
    ]), AS_OF)
    assert rep["issued_before_applied"] == 1
    assert not df["is_model_eligible"].iloc[0]


def test_dedup_keeps_most_recent_and_normalizes_fields():
    df, rep = clean_permits(_raw([
        {"permitnum": " 1-cn ", "applieddate": "2020-01-01T00:00:00.000", "estprojectcost": "-5",
         "socrata_updated_at": "2025-01-01T00:00:00.000Z", "originalzip": "98105-1234",
         "latitude": "10.0", "longitude": "-122.3", "description": "  old   text "},
        {"permitnum": "1-CN", "applieddate": "2020-01-02T00:00:00.000",
         "socrata_updated_at": "2026-01-01T00:00:00.000Z", "originalzip": "98105",
         "latitude": "47.65", "longitude": "-122.30", "description": "new"},
    ]), AS_OF)
    assert rep["duplicates_removed"] == 1 and len(df) == 1
    row = df.iloc[0]
    assert row["applied_date"] == pd.Timestamp("2020-01-02") and row["description"] == "new"
    assert row["permit_suffix"] == "CN" and row["zip"] == "98105"


def test_invalid_values_nulled_not_dropped():
    df, rep = clean_permits(_raw([
        {"permitnum": "9-DM", "applieddate": "2020-01-01T00:00:00.000", "estprojectcost": "-5",
         "originalzip": "98105-1234", "latitude": "10.0", "longitude": "-122.3",
         "description": "  two   spaces "},
    ]), AS_OF)
    row = df.iloc[0]
    assert pd.isna(row["est_project_cost"]) and rep["estprojectcost_negative_nulled"] == 1
    assert pd.isna(row["latitude"]) and bool(row["q_coords_missing"])
    assert row["zip"] == "98105" and row["description"] == "two spaces"
