"""Reusable prediction interface for 90-day permit-issuance probabilities.

Example::

    from datetime import date
    from huskyquiet.predict import IssuancePredictor, PermitApplication

    predictor = IssuancePredictor()  # loads artifacts/models/production + clean history
    result = predictor.predict(
        PermitApplication(
            permit_num="7095626-CN",
            applied_date=date(2026, 8, 3),
            permit_type_mapped="Building",
            permit_class="Single Family/Duplex",
            permit_type_desc="Addition/Alteration",
            description="Construct addition to existing single family residence, per plan.",
            est_project_cost=150_000,
            latitude=47.6612, longitude=-122.3134,
        ),
        as_of=date(2026, 10, 10),
    )
    result.probability  # calibrated P(issued within 90 days of as_of)

Context features (city-wide backlog, recent issuance rates, prior activity at the same
address) are computed from the cleaned permit history using only events dated before
``as_of``, with exactly the code used to build training snapshots.

The output is a probability of *permit issuance*. It is not a prediction that
construction will start, nor a measurement or prediction of noise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from huskyquiet.config import (
    FIRST_FORECAST_DATE,
    HORIZON_DAYS,
    MAX_APPLICATION_AGE_DAYS,
    SEATTLE_BBOX,
    Paths,
    get_paths,
)
from huskyquiet.features.engineering import FEATURE_COLUMNS, PERMIT_ATTRIBUTE_COLUMNS, build_feature_frame
from huskyquiet.features.snapshots import eligible_history, risk_set
from huskyquiet.features.text import KEYWORD_FEATURES
from huskyquiet.models.pipelines import model_matrix
from huskyquiet.models.registry import load_bundle

DISCLAIMER = (
    "Probability that the City of Seattle issues this permit within the next "
    f"{HORIZON_DAYS} days. Permit issuance is not the same as construction starting, "
    "and this is not a noise prediction."
)
# Context features need permit history up to as_of; beyond this gap the request is refused.
MAX_STALENESS_DAYS = 14
# Columns worth showing next to a live score. They are already computed inside
# ``score_active_permits``; returning them does not change the probability.
EXPLAIN_COLUMNS: tuple[str, ...] = (
    "dist_uw_km",
    "log_est_cost",
    "est_cost_missing",
    "has_related_mup",
    "zip",
    "ctx_backlog_all",
    "ctx_backlog_grp",
    "ctx_issue_rate_lag_age",
    "ctx_applied_90d_grp",
    "ctx_issued_90d_grp",
    "addr_prior_apps",
    "addr_prior_issued_365d",
    "desc_missing",
    *KEYWORD_FEATURES,
)
_PERMIT_NUM = re.compile(r"^[0-9A-Z]+-[A-Z]{2,3}$")


class PredictionInputError(ValueError):
    """The request is invalid or outside the population the model was trained on."""


class PermitApplication(BaseModel):
    """Application-time attributes of a permit application (no outcome fields allowed)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    permit_num: str | None = Field(None, description="e.g. '7095626-CN'; enables history checks")
    applied_date: date
    permit_type_mapped: str = Field(..., min_length=1, description="e.g. 'Building', 'Demolition'")
    permit_class: str = Field(..., min_length=1, description="e.g. 'Single Family/Duplex'")
    permit_type_desc: str | None = None
    description: str | None = Field(None, max_length=10_000)
    est_project_cost: float | None = Field(None, ge=0)
    related_mup: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    zip: str | None = Field(None, pattern=r"^\d{5}$")
    address: str | None = None

    @field_validator("permit_num")
    @classmethod
    def _permit_num_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.upper()
        if not _PERMIT_NUM.match(v):
            raise ValueError("permit_num must look like '7095626-CN'")
        return v

    @model_validator(mode="after")
    def _coordinates(self) -> PermitApplication:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if self.latitude is not None and not (
            SEATTLE_BBOX["lat_min"] <= self.latitude <= SEATTLE_BBOX["lat_max"]
            and SEATTLE_BBOX["lon_min"] <= self.longitude <= SEATTLE_BBOX["lon_max"]
        ):
            raise ValueError("coordinates are outside the City of Seattle bounding box")
        return self

    @property
    def permit_suffix(self) -> str | None:
        if self.permit_num and "-" in self.permit_num:
            return self.permit_num.rsplit("-", 1)[1]
        return None


class PredictionResult(BaseModel):
    permit_num: str | None
    as_of: date
    horizon_days: int
    probability: float = Field(..., ge=0, le=1)
    application_age_days: int
    model: str
    model_role: str
    data_as_of: date
    warnings: list[str] = Field(default_factory=list)
    interpretation: str = DISCLAIMER
    features: dict[str, Any] | None = None


class IssuancePredictor:
    """Loads a model bundle and the permit history; scores applications as of a date."""

    def __init__(
        self,
        model_dir: Path | None = None,
        history: pd.DataFrame | None = None,
        paths: Paths | None = None,
    ) -> None:
        self.paths = paths or get_paths()
        bundle = load_bundle(model_dir or self.paths.model_dir / "production")
        self.model = bundle["model"]
        self.features: tuple[str, ...] = tuple(bundle["features"])
        stale = set(self.features) - set(FEATURE_COLUMNS)
        if stale:
            raise ValueError(
                f"Model bundle uses features no longer produced by the feature code {sorted(stale)}; "
                "retrain with `python -m huskyquiet train`"
            )
        self.metadata: dict[str, Any] = bundle["metadata"]
        self.data_as_of = pd.Timestamp(self.metadata["data_as_of"])
        if history is None:
            history = eligible_history(pd.read_parquet(self.paths.clean_permits))
        self.history = history

    @cached_property
    def known_categories(self) -> dict[str, set[str]]:
        seen = self.metadata.get("categories_seen_in_training", {})
        return {c: set(seen[c]) if c in seen else set(self.history[c].dropna().astype(str).unique())
                for c in ("permit_type_mapped", "permit_class")}

    @cached_property
    def _history_index(self) -> pd.DataFrame:
        return self.history.set_index("permit_num")

    # ------------------------------------------------------------------ validation
    def _validate(self, permit: PermitApplication, as_of: pd.Timestamp) -> tuple[int, list[str]]:
        warnings: list[str] = []
        applied = pd.Timestamp(permit.applied_date)
        if as_of < pd.Timestamp(FIRST_FORECAST_DATE):
            raise PredictionInputError(f"as_of must be on/after {FIRST_FORECAST_DATE}")
        if as_of <= applied:
            raise PredictionInputError(
                "as_of must be after applied_date (forecasts are made at the start of a day)"
            )
        age = int((as_of - applied).days)
        if age > MAX_APPLICATION_AGE_DAYS:
            raise PredictionInputError(
                f"application is {age} days old; the model covers applications up to "
                f"{MAX_APPLICATION_AGE_DAYS} days old"
            )
        gap = int((as_of - self.data_as_of).days)
        if gap > MAX_STALENESS_DAYS:
            raise PredictionInputError(
                f"as_of is {gap} days after the latest permit data ({self.data_as_of.date()}); "
                "refresh the data first"
            )
        if gap > 1:
            warnings.append(f"Permit history ends {gap} days before as_of; context features may be stale.")
        for col in ("permit_type_mapped", "permit_class"):
            value = getattr(permit, col)
            if value not in self.known_categories[col]:
                raise PredictionInputError(
                    f"unknown {col} {value!r}; expected one of {sorted(self.known_categories[col])}"
                )
        if permit.permit_num and permit.permit_num in self._history_index.index:
            rec = self._history_index.loc[permit.permit_num]
            issued = rec["issued_date"]
            if pd.notna(issued) and pd.Timestamp(issued) < as_of:
                raise PredictionInputError(
                    f"{permit.permit_num} was already issued on {pd.Timestamp(issued).date()}"
                )
            if pd.Timestamp(rec["applied_date"]) != applied:
                warnings.append("applied_date differs from the city record for this permit_num.")
        if permit.permit_num is None:
            warnings.append("permit_num not provided; permit-suffix feature treated as missing.")
        return age, warnings

    # ------------------------------------------------------------------ features
    def _snapshot_rows(self, permits: list[PermitApplication], as_of: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(permits):
            address_key = re.sub(r"\s+", " ", p.address).strip().upper() if p.address else None
            rows.append({
                "permit_num": p.permit_num or f"__query_{i}__",
                "forecast_date": as_of,
                "age_days": int((as_of - pd.Timestamp(p.applied_date)).days),
                "applied_date": pd.Timestamp(p.applied_date),
                "permit_class": p.permit_class,
                "permit_type_mapped": p.permit_type_mapped,
                "permit_type_desc": p.permit_type_desc,
                "permit_suffix": p.permit_suffix,
                "description": p.description,
                "est_project_cost": p.est_project_cost,
                "related_mup": p.related_mup,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "zip": p.zip,
                "address_key": address_key,
            })
        df = pd.DataFrame(rows)
        if df["permit_num"].duplicated().any():
            raise PredictionInputError("duplicate permit_num values in one request")
        return df[["forecast_date", "age_days", *PERMIT_ATTRIBUTE_COLUMNS]]

    def _score(self, snaps: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        feats = build_feature_frame(snaps, self.history)
        p = self.model.predict_proba(model_matrix(feats, self.features))[:, 1]
        return p, feats

    # ------------------------------------------------------------------ public API
    def predict_many(
        self,
        permits: Iterable[PermitApplication | dict[str, Any]],
        as_of: date,
        return_features: bool = False,
    ) -> list[PredictionResult]:
        parsed = [p if isinstance(p, PermitApplication) else PermitApplication(**p) for p in permits]
        if not parsed:
            return []
        t = pd.Timestamp(as_of).normalize()
        checks = [self._validate(p, t) for p in parsed]
        probs, feats = self._score(self._snapshot_rows(parsed, t))
        out = []
        for p, (age, warns), prob, (_, frow) in zip(parsed, checks, probs, feats.iterrows(), strict=True):
            out.append(PredictionResult(
                permit_num=p.permit_num,
                as_of=t.date(),
                horizon_days=HORIZON_DAYS,
                probability=float(prob),
                application_age_days=age,
                model=str(self.metadata.get("model")),
                model_role=str(self.metadata.get("role")),
                data_as_of=self.data_as_of.date(),
                warnings=warns,
                features={k: (None if pd.isna(frow[k]) else frow[k]) for k in self.features}
                if return_features else None,
            ))
        return out

    def predict(
        self, permit: PermitApplication | dict[str, Any], as_of: date, return_features: bool = False
    ) -> PredictionResult:
        return self.predict_many([permit], as_of, return_features)[0]

    def score_active_permits(self, as_of: date | None = None, return_features: bool = False) -> pd.DataFrame:
        """Score every application in the modeled risk set at ``as_of`` (default: the day
        after the latest data). Intended for map/dashboard use.

        With ``return_features=True``, explanatory columns from the same feature frame that
        produced the probabilities are attached. The probabilities themselves do not change.
        """
        t = pd.Timestamp(as_of) if as_of else self.data_as_of + pd.Timedelta(days=1)
        gap = int((t - self.data_as_of).days)
        if gap > MAX_STALENESS_DAYS:
            raise PredictionInputError(f"as_of is {gap} days after the latest permit data")
        rs = risk_set(self.history, [t], with_labels=False)
        snaps = rs.merge(self.history[list(PERMIT_ATTRIBUTE_COLUMNS)], on="permit_num", how="left")
        probs, feats = self._score(snaps)
        cols = ["permit_num", "applied_date", "permit_type_mapped", "permit_class", "permit_type_desc",
                "description", "est_project_cost", "latitude", "longitude", "address_key"]
        out = snaps[["permit_num", "age_days"]].merge(self.history[cols], on="permit_num", how="left")
        out["forecast_date"] = t
        out["probability_issued_90d"] = probs
        if return_features:
            if len(feats) != len(out):
                raise RuntimeError("feature row count does not match the active risk set")
            for col in EXPLAIN_COLUMNS:
                if col not in out.columns and col in feats.columns:
                    out[col] = feats[col].to_numpy()
        return out.sort_values("probability_issued_90d", ascending=False, ignore_index=True)
