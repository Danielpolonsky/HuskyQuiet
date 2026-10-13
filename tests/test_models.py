import numpy as np
import pandas as pd
import pytest

from huskyquiet.features.engineering import PERMIT_ATTRIBUTE_COLUMNS, build_feature_frame
from huskyquiet.features.snapshots import risk_set
from huskyquiet.models.baseline import HistoricalRateModel, recent_rate_forecast
from huskyquiet.models.calibration import ProbabilityCalibrator, select_calibration_method
from huskyquiet.models.evaluate import (
    binary_metrics,
    cluster_bootstrap,
    expected_calibration_error,
)
from huskyquiet.models.pipelines import build_hgb, build_logistic, model_matrix
from huskyquiet.models.train import TemporalSplit


@pytest.fixture
def feature_frame(history):
    dates = pd.date_range("2009-01-01", "2011-06-01", freq="MS")
    rs = risk_set(history, dates)
    snaps = rs.merge(history[list(PERMIT_ATTRIBUTE_COLUMNS)], on="permit_num")
    return build_feature_frame(snaps, history)


def test_historical_rate_shrinks_and_falls_back():
    X = pd.DataFrame({"permit_type_mapped": ["A"] * 4 + ["B"], "permit_class": ["x"] * 5,
                      "age_bucket": ["0-30"] * 5})
    y = np.array([1, 1, 1, 1, 0])
    m = HistoricalRateModel(pseudo_count=1.0).fit(X, y)
    p = m.predict_proba(pd.DataFrame({"permit_type_mapped": ["A", "B", "Z"], "permit_class": ["x"] * 3,
                                      "age_bucket": ["0-30", "0-30", "unseen"]}))[:, 1]
    assert p[0] > p[1]  # group A had all positives
    assert 0 < p[1] < 0.8  # shrunk toward the parent rate
    assert p[2] == pytest.approx(0.8)  # unseen cell -> global rate


def test_recent_rate_baseline_handles_missing():
    p = recent_rate_forecast(pd.DataFrame({"ctx_issue_rate_lag_age": [0.2, np.nan]}))
    assert p.shape == (2, 2) and np.isclose(p[1, 1], 0.2)


@pytest.mark.parametrize("builder", [lambda: build_logistic(0.1), lambda: build_hgb({"max_iter": 20})])
def test_pipelines_fit_predict_with_missing_and_unseen(builder, feature_frame):
    X, y = model_matrix(feature_frame), feature_frame["label"].to_numpy()
    model = builder().fit(X, y)
    Xn = X.head(5).copy()
    Xn["zip"] = "99999"  # unseen category
    Xn.loc[Xn.index[0], "log_est_cost"] = np.nan
    p = model.predict_proba(Xn)[:, 1]
    assert p.shape == (5,) and np.all((p > 0) & (p < 1))


def test_ece_and_metrics():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    m = binary_metrics(y, p)
    assert 0.7 < m["roc_auc"] < 0.9 and abs(m["calibration_slope"] - 1) < 0.1
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2), rel=1e-6)


def test_calibrator_fixes_miscalibration():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(int)
    skewed = true_p ** 2  # systematically under-confident about positives
    groups = np.arange(true_p.size) // 4
    method, scores = select_calibration_method(skewed, y, groups)
    assert method in ("platt", "isotonic") and scores[method] < scores["none"]
    fixed = ProbabilityCalibrator(method).fit(skewed, y).transform(skewed)
    assert expected_calibration_error(y, fixed) < expected_calibration_error(y, skewed)


def test_cluster_bootstrap_moves_whole_permits():
    """Every snapshot of a permit shares one draw count, and permits are not drawn identically."""
    clusters = pd.Series([1, 1, 1, 2, 2, 3])
    y = np.zeros(len(clusters))
    permit_id = clusters.to_numpy(float)

    def same_weight(_yy, pp, ww):
        for cid in np.unique(pp):
            if len(set(np.round(ww[pp == cid], 6))) != 1:
                return 0.0
        return 1.0

    def weight_spread(_yy, pp, ww):
        means = [float(ww[pp == cid][0]) for cid in np.unique(pp)]
        return float(np.var(means))

    res = cluster_bootstrap(
        y, {"m": permit_id}, clusters,
        metric_fns={"together": same_weight, "spread": weight_spread},
        n_boot=40, seed=0,
    )
    assert res["m"]["together"]["mean"] == 1.0
    assert res["m"]["spread"]["mean"] > 0


def test_cluster_bootstrap_paired_difference():
    rng = np.random.default_rng(0)
    clusters = pd.Series(np.repeat(np.arange(300), 3))
    y = rng.integers(0, 2, clusters.size)
    p_good = np.clip(y * 0.6 + rng.uniform(0, 0.4, y.size), 0, 1)
    p_rand = rng.uniform(size=y.size)
    res = cluster_bootstrap(y, {"good": p_good, "rand": p_rand}, clusters, reference="rand", n_boot=50)
    lo, hi = res["good"]["diff_vs_rand_roc_auc"]["ci_low"], res["good"]["diff_vs_rand_roc_auc"]["ci_high"]
    assert 0 < lo <= hi


def test_temporal_split_embargo():
    t = pd.Series(pd.date_range("2020-01-01", "2026-07-01", freq="MS"))
    split = TemporalSplit.from_last_forecast(t.max())
    m = split.masks(t)
    h = pd.Timedelta(days=90)
    assert (t[m["train"]] + h <= split.val_start).all()
    assert (t[m["val"]] >= split.val_start).all() and (t[m["val"]] + h <= split.test_start).all()
    assert t[m["test"]].min() == split.test_start and m["test"].sum() == 12
    assert t[m["train"]].max() < t[m["val"]].min() < t[m["test"]].min()
