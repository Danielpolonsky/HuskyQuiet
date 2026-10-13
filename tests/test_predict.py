from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from huskyquiet.config import Paths, get_paths
from huskyquiet.features.engineering import FEATURE_COLUMNS, PERMIT_ATTRIBUTE_COLUMNS, build_feature_frame
from huskyquiet.features.snapshots import risk_set
from huskyquiet.models.calibration import CalibratedModel, ProbabilityCalibrator
from huskyquiet.models.pipelines import build_hgb, model_matrix
from huskyquiet.models.registry import save_bundle
from huskyquiet.predict import IssuancePredictor, PermitApplication, PredictionInputError

DATA_AS_OF = "2011-12-31"


@pytest.fixture
def predictor(history, tmp_path):
    rs = risk_set(history, pd.date_range("2009-01-01", "2011-06-01", freq="MS"))
    snaps = rs.merge(history[list(PERMIT_ATTRIBUTE_COLUMNS)], on="permit_num")
    feats = build_feature_frame(snaps, history)
    est = build_hgb({"max_iter": 30}).fit(model_matrix(feats), feats["label"])
    model = CalibratedModel(est, ProbabilityCalibrator("none"))
    save_bundle(tmp_path / "m", model, FEATURE_COLUMNS,
                {"data_as_of": DATA_AS_OF, "model": "hgb", "role": "test"})
    return IssuancePredictor(model_dir=tmp_path / "m", history=history, paths=Paths(root=tmp_path))


def _app(**kw) -> PermitApplication:
    base = dict(applied_date=date(2011, 10, 1), permit_type_mapped="Building",
                permit_class="Single Family/Duplex", description="Construct addition per plan.",
                est_project_cost=120_000, latitude=47.66, longitude=-122.31, address="1200 NE 45TH ST")
    return PermitApplication(**{**base, **kw})


def test_schema_validation_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        _app(est_project_cost=-1)
    with pytest.raises(ValidationError):
        _app(latitude=40.0, longitude=-122.3)
    with pytest.raises(ValidationError):
        _app(latitude=47.6, longitude=None)
    with pytest.raises(ValidationError):
        _app(issued_date=date(2011, 11, 1))  # outcome fields are not accepted
    with pytest.raises(ValidationError):
        _app(permit_num="not a permit")


def test_predict_returns_probability(predictor):
    res = predictor.predict(_app(), as_of=date(2011, 12, 1), return_features=True)
    assert 0 < res.probability < 1 and res.horizon_days == 90
    assert res.application_age_days == 61
    assert "not a noise prediction" in res.interpretation
    assert set(res.features) == set(FEATURE_COLUMNS)
    assert any("permit_num" in w for w in res.warnings)


def test_predict_accepts_dicts_and_batches(predictor):
    rows = [_app().model_dump(), _app(permit_class="Commercial", permit_num="7000001-CN").model_dump()]
    out = predictor.predict_many(rows, as_of=date(2011, 12, 1))
    assert len(out) == 2 and out[1].permit_num == "7000001-CN"


@pytest.mark.parametrize("kwargs, as_of, match", [
    ({}, date(2011, 10, 1), "after applied_date"),
    ({"applied_date": date(2009, 1, 1)}, date(2011, 12, 1), "days old"),
    ({}, date(2012, 6, 1), "latest permit data"),
    ({"permit_type_mapped": "Spaceport"}, date(2011, 12, 1), "unknown permit_type_mapped"),
])
def test_predict_rejects_out_of_scope(predictor, kwargs, as_of, match):
    with pytest.raises(PredictionInputError, match=match):
        predictor.predict(_app(**kwargs), as_of=as_of)


def test_predict_rejects_already_issued_permit(predictor, history):
    rec = history.dropna(subset=["issued_date"]).iloc[0]
    app = _app(permit_num=rec["permit_num"], applied_date=rec["applied_date"].date())
    with pytest.raises(PredictionInputError, match="already issued"):
        predictor.predict(app, as_of=(rec["issued_date"] + pd.Timedelta(days=1)).date())


def test_score_active_permits(predictor):
    df = predictor.score_active_permits(date(2011, 6, 1))
    assert len(df) > 0 and df["probability_issued_90d"].between(0, 1).all()
    assert df["probability_issued_90d"].is_monotonic_decreasing


def test_score_active_features_match_the_probability(predictor):
    as_of = date(2011, 6, 1)
    plain = predictor.score_active_permits(as_of)
    rich = predictor.score_active_permits(as_of, return_features=True)
    assert "dist_uw_km" in rich.columns and "kw_addition" in rich.columns
    merged = plain.merge(rich[["permit_num", "probability_issued_90d"]], on="permit_num", suffixes=("", "_rich"))
    assert (merged["probability_issued_90d"] - merged["probability_issued_90d_rich"]).abs().max() < 1e-12


@pytest.mark.integration
def test_production_artifacts_end_to_end():
    paths = get_paths()
    if not (paths.model_dir / "production" / "model.joblib").exists() or not paths.clean_permits.exists():
        pytest.skip("production artifacts not built; run `python -m huskyquiet all`")
    pred = IssuancePredictor()
    as_of = pred.data_as_of.date()
    active = pred.score_active_permits()
    assert len(active) > 100 and active["probability_issued_90d"].between(0, 1).all()
    res = pred.predict(_app(applied_date=(pred.data_as_of - pd.Timedelta(days=40)).date()), as_of=as_of)
    assert 0 < res.probability < 1
