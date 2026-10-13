"""scikit-learn pipelines for the learned models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

from huskyquiet.config import RANDOM_SEED
from huskyquiet.features.engineering import (
    CATEGORICAL_FEATURES,
    COUNT_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


def model_matrix(df: pd.DataFrame, features: tuple[str, ...] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Select model features with uniform dtypes (object + np.nan for categoricals)."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    X = df[list(features)].copy()
    for col in features:
        if col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    return X


def build_logistic(C: float = 0.1, features: tuple[str, ...] = FEATURE_COLUMNS) -> Pipeline:
    cats = [c for c in CATEGORICAL_FEATURES if c in features]
    months = [c for c in ("forecast_month", "applied_month") if c in features]
    counts = [c for c in COUNT_FEATURES if c in features]
    age = ["log_age_days"] if "log_age_days" in features else []
    other = [
        c for c in NUMERIC_FEATURES
        if c in features and c not in set(counts) | set(months) | {"log_age_days", "age_days"}
    ]
    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100)),
                ]),
                cats + months,
            ),
            (
                "age_spline",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("spline", SplineTransformer(n_knots=8, degree=3, extrapolation="constant")),
                ]),
                age,
            ),
            (
                "counts",
                Pipeline([
                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]),
                counts,
            ),
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]),
                other,
            ),
        ],
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_hgb(params: dict[str, Any] | None = None, features: tuple[str, ...] = FEATURE_COLUMNS) -> Pipeline:
    cats = [c for c in CATEGORICAL_FEATURES if c in features]
    nums = [c for c in features if c not in CATEGORICAL_FEATURES]
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    min_frequency=20,
                ),
                cats,
            ),
            ("num", "passthrough", nums),
        ],
        verbose_feature_names_out=False,
    )
    defaults: dict[str, Any] = {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": RANDOM_SEED,
    }
    defaults.update(params or {})
    clf = HistGradientBoostingClassifier(
        categorical_features=[True] * len(cats) + [False] * len(nums), **defaults
    )
    return Pipeline([("pre", pre), ("clf", clf)])
