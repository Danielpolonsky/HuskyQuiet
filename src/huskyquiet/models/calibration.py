"""Post-hoc probability calibration fitted on a later, held-out period."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

EPS = 1e-6
CalibrationMethod = Literal["none", "platt", "isotonic"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    """Maps raw scores to calibrated probabilities (Platt on the logit, or isotonic)."""

    def __init__(self, method: CalibrationMethod = "platt") -> None:
        self.method = method
        self._model: Any = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> ProbabilityCalibrator:
        y = np.asarray(y, int)
        if self.method == "platt":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw)[:, None], y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(
                np.asarray(p_raw, float), y
            )
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, float)
        if self.method == "none" or self._model is None:
            return p_raw
        if self.method == "platt":
            return self._model.predict_proba(_logit(p_raw)[:, None])[:, 1]
        return np.clip(self._model.predict(p_raw), EPS, 1 - EPS)


def select_calibration_method(
    p_raw: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> tuple[CalibrationMethod, dict[str, float]]:
    """Choose a calibrator by permit-grouped cross-validated log loss on the calibration set."""
    scores: dict[str, float] = {}
    y = np.asarray(y, int)
    for method in ("none", "platt", "isotonic"):
        losses = []
        for tr, te in GroupKFold(n_splits=n_splits).split(p_raw, y, groups):
            cal = ProbabilityCalibrator(method).fit(p_raw[tr], y[tr])  # type: ignore[arg-type]
            losses.append(log_loss(y[te], np.clip(cal.transform(p_raw[te]), EPS, 1 - EPS), labels=[0, 1]))
        scores[method] = float(np.mean(losses))
    best = min(scores, key=scores.get)  # type: ignore[arg-type]
    return best, scores  # type: ignore[return-value]


class CalibratedModel:
    """A fitted estimator plus calibrator exposing ``predict_proba`` like sklearn."""

    def __init__(self, estimator: Any, calibrator: ProbabilityCalibrator) -> None:
        self.estimator = estimator
        self.calibrator = calibrator

    def predict_raw(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.calibrator.transform(self.predict_raw(X))
        return np.column_stack([1 - p, p])
