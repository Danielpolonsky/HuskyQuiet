"""Property tests: features at forecast date t must not depend on anything after t."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from huskyquiet.features.engineering import FEATURE_COLUMNS, PERMIT_ATTRIBUTE_COLUMNS, build_feature_frame
from huskyquiet.features.snapshots import risk_set

T = pd.Timestamp("2010-03-01")


def _features(history: pd.DataFrame) -> pd.DataFrame:
    rs = risk_set(history, [T], with_labels=False)
    snaps = rs.merge(history[list(PERMIT_ATTRIBUTE_COLUMNS)], on="permit_num")
    f = build_feature_frame(snaps, history)
    return f.set_index("permit_num").sort_index()[list(FEATURE_COLUMNS)]


def test_features_invariant_to_future_events(history):
    base = _features(history)
    assert len(base) > 20

    perturbed = history.copy()
    rng = np.random.default_rng(1)
    future_issued = perturbed["issued_date"] >= T
    # Move every post-t issuance to a different post-t date, or erase it.
    shift = pd.to_timedelta(rng.integers(0, 400, future_issued.sum()), unit="D")
    perturbed.loc[future_issued, "issued_date"] = perturbed.loc[future_issued, "issued_date"] + shift
    perturbed.loc[future_issued & (rng.random(len(perturbed)) < 0.3), "issued_date"] = pd.NaT
    # Delete permits applied on/after t and add brand-new future ones.
    future_apps = perturbed["applied_date"] >= T
    extra = perturbed[future_apps].copy()
    extra["permit_num"] = extra["permit_num"].str.replace("-", "9-", regex=False)
    extra["applied_date"] = extra["applied_date"] + pd.Timedelta(days=30)
    perturbed = pd.concat([perturbed[~future_apps], extra], ignore_index=True)

    pdt.assert_frame_equal(base, _features(perturbed))


def test_features_change_when_past_events_change(history):
    """Sanity check that the invariance test above is not vacuous."""
    base = _features(history)
    altered = history.copy()
    past_issued = altered["issued_date"] < T
    altered.loc[past_issued, "issued_date"] = pd.NaT  # past issuances disappear
    changed = _features(altered)
    assert not base["ctx_issued_90d_grp"].equals(changed.reindex(base.index)["ctx_issued_90d_grp"])


def test_labels_never_in_features(history):
    rs = risk_set(history, [T], with_labels=True)
    snaps = rs.merge(history[list(PERMIT_ATTRIBUTE_COLUMNS)], on="permit_num")
    f = build_feature_frame(snaps, history)
    assert "label" in f.columns  # passed through for training ...
    assert "label" not in FEATURE_COLUMNS and "outcome_days_until_issue" not in FEATURE_COLUMNS
