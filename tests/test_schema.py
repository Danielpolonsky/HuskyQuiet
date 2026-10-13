from huskyquiet.data.clean import RENAME
from huskyquiet.data.schema import FIELD_BY_NAME, FORBIDDEN_FEATURE_SOURCES, Availability
from huskyquiet.features.engineering import FEATURE_COLUMNS, PERMIT_ATTRIBUTE_COLUMNS

# Fields the audit showed are populated around/after issuance, so they are excluded too.
AUDIT_EXCLUDED = {"housing_category", "zoning", "dwelling_unit_type", "standard_plan",
                  "dependent_building", "parent_permit_num", "development_site"}


def test_outcome_and_current_state_fields_are_forbidden():
    for name in ("issueddate", "statuscurrent", "completeddate", "expiresdate", "readytoissuedate",
                 "numberreviewcycles", "contractorcompanyname", "daysissuepermitcity",
                 "housingunits", "housingunitsadded", "housingunitsremoved"):
        assert name in FORBIDDEN_FEATURE_SOURCES
        assert FIELD_BY_NAME[name].availability in (Availability.OUTCOME, Availability.CURRENT_STATE)


def test_no_forbidden_field_reaches_features():
    forbidden_clean_names = {RENAME.get(f, f) for f in FORBIDDEN_FEATURE_SOURCES} | AUDIT_EXCLUDED
    assert not forbidden_clean_names & set(PERMIT_ATTRIBUTE_COLUMNS)
    assert not forbidden_clean_names & set(FEATURE_COLUMNS)
    assert "issued_date" not in PERMIT_ATTRIBUTE_COLUMNS
    assert "status_current" not in FEATURE_COLUMNS


def test_feature_columns_unique():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
