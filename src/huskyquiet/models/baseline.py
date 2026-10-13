"""Non-ML reference forecasts that every model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


class HistoricalRateModel:
    """Empirical 90-day issuance rate by (type, class, age bucket) in the training period.

    Sparse cells are shrunk toward their parent level with an empirical-Bayes style
    pseudo-count ``m``: ``rate = (positives + m * parent_rate) / (n + m)``. The hierarchy
    is (type, class, age) -> (type, age) -> (age) -> global.
    """

    LEVELS: tuple[tuple[str, ...], ...] = (
        ("age_bucket",),
        ("permit_type_mapped", "age_bucket"),
        ("permit_type_mapped", "permit_class", "age_bucket"),
    )

    def __init__(self, pseudo_count: float = 50.0) -> None:
        self.pseudo_count = pseudo_count
        self.global_rate_: float | None = None
        self.tables_: list[pd.DataFrame] = []

    @staticmethod
    def _keys(X: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
        return X[list(cols)].astype("string").fillna("NA")

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> HistoricalRateModel:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_rate_ = float(y.mean())
        self.tables_ = []
        parent = pd.Series(self.global_rate_, index=X.index)
        for cols in self.LEVELS:
            keys = self._keys(X, cols)
            frame = keys.assign(_y=y.values, _parent=parent.values)
            agg = frame.groupby(list(cols), dropna=False).agg(
                n=("_y", "size"), pos=("_y", "sum"), parent=("_parent", "mean")
            )
            agg["rate"] = (agg["pos"] + self.pseudo_count * agg["parent"]) / (agg["n"] + self.pseudo_count)
            self.tables_.append(agg[["rate"]])
            parent = self._lookup(keys, agg["rate"], parent)
        return self

    @staticmethod
    def _lookup(keys: pd.DataFrame, rates: pd.Series, fallback: pd.Series) -> pd.Series:
        idx = pd.MultiIndex.from_frame(keys) if keys.shape[1] > 1 else pd.Index(keys.iloc[:, 0])
        vals = rates.reindex(idx).to_numpy(dtype=float)
        return pd.Series(np.where(np.isnan(vals), fallback.to_numpy(), vals), index=keys.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_rate_ is None:
            raise RuntimeError("HistoricalRateModel is not fitted")
        p = pd.Series(self.global_rate_, index=X.index)
        for cols, table in zip(self.LEVELS, self.tables_, strict=True):
            p = self._lookup(self._keys(X, cols), table["rate"], p)
        p = p.to_numpy()
        return np.column_stack([1 - p, p])


def recent_rate_forecast(X: pd.DataFrame) -> np.ndarray:
    """Persistence baseline: the issuance rate observed over the last 90 days for
    applications of the same age bucket (feature ``ctx_issue_rate_lag_age``)."""
    p = X["ctx_issue_rate_lag_age"].to_numpy(dtype=float)
    fallback = np.nanmean(p) if np.isfinite(p).any() else 0.3
    p = np.clip(np.where(np.isnan(p), fallback, p), 1e-4, 1 - 1e-4)
    return np.column_stack([1 - p, p])
