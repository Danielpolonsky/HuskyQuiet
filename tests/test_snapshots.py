import pandas as pd

from huskyquiet.features.snapshots import monthly_forecast_dates, risk_set


def _hist(rows):
    df = pd.DataFrame(rows, columns=["permit_num", "applied_date", "issued_date"])
    df["applied_date"] = pd.to_datetime(df["applied_date"])
    df["issued_date"] = pd.to_datetime(df["issued_date"])
    return df


T = pd.Timestamp("2020-06-01")


def test_risk_set_membership_and_label_boundaries():
    h = _hist([
        ("A", "2020-05-31", "2020-06-01"),  # issued on t -> in risk set, positive
        ("B", "2020-05-01", "2020-08-29"),  # t + 89 -> positive
        ("C", "2020-05-01", "2020-08-30"),  # t + 90 -> negative (window is [t, t+90))
        ("D", "2020-05-01", None),          # never issued -> negative
        ("E", "2020-05-01", "2020-05-31"),  # issued before t -> excluded
        ("F", "2020-06-01", "2020-06-05"),  # applied on t -> not yet known -> excluded
        ("G", "2018-06-01", None),          # older than age cap -> excluded
        ("H", "2020-05-15", "2020-05-15"),  # same-day issuance -> never in a risk set
    ])
    rs = risk_set(h, [T], horizon_days=90, max_age_days=730).set_index("permit_num")
    assert set(rs.index) == {"A", "B", "C", "D"}
    assert rs.loc[["A", "B", "C", "D"], "label"].tolist() == [1, 1, 0, 0]
    assert rs.loc["A", "age_days"] == 1


def test_age_cap_is_inclusive():
    h = _hist([("X", T - pd.Timedelta(days=730), None), ("Y", T - pd.Timedelta(days=731), None)])
    rs = risk_set(h, [T], max_age_days=730)
    assert rs["permit_num"].tolist() == ["X"]


def test_forecast_dates_have_complete_windows():
    dates = monthly_forecast_dates("2020-01-01", pd.Timestamp("2020-12-15"), horizon_days=90)
    assert dates.max() + pd.Timedelta(days=90) <= pd.Timestamp("2020-12-15")
    assert dates.min() == pd.Timestamp("2020-01-01") and (dates.day == 1).all()


def test_repeated_observations_over_dates():
    h = _hist([("A", "2020-01-15", "2020-04-10")])
    rs = risk_set(h, pd.date_range("2020-02-01", "2020-05-01", freq="MS"))
    assert rs["forecast_date"].dt.strftime("%m").tolist() == ["02", "03", "04"]
    assert rs["label"].tolist() == [1, 1, 1]
