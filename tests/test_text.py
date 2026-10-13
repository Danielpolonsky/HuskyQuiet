import pandas as pd

from huskyquiet.features.text import KEYWORD_FEATURES, keyword_features, sanitize_description


def test_admin_annotations_are_removed():
    raw = "1/10/17 MD.CANCELLED IAW SBC 106.6.9. Construct new single family residence per plan."
    clean = sanitize_description(raw)
    assert "CANCEL" not in clean and "1/10/17" not in clean
    assert "CONSTRUCT NEW SINGLE FAMILY RESIDENCE" in clean


def test_outcome_words_never_create_features():
    s = pd.Series(["PROJECT CANCELLED -- withdrawn per applicant request", "Permit expired 2019-01-01"])
    f = keyword_features(s)
    assert f[list(KEYWORD_FEATURES)].to_numpy().sum() == 0


def test_scope_keywords_detected_and_missing_flag():
    s = pd.Series(["Demolish existing garage and construct new ADU", None, "Tenant improvement STFI"])
    f = keyword_features(s)
    assert f.loc[0, ["kw_demolition", "kw_adu", "kw_new_construction"]].tolist() == [1, 1, 1]
    assert f.loc[1, "desc_missing"] == 1 and f.loc[1, list(KEYWORD_FEATURES)].sum() == 0
    assert f.loc[2, ["kw_tenant_improvement", "kw_stfi"]].tolist() == [1, 1]
