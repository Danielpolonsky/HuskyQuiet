"""Extra evaluation tables derived from the saved model and snapshots.

These are not used to choose a model. They describe calibration inside the already
locked test period.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from huskyquiet.config import Paths, get_paths
from huskyquiet.models.pipelines import model_matrix
from huskyquiet.models.registry import load_bundle


def test_month_calibration(paths: Paths | None = None) -> pd.DataFrame:
    """Observed versus predicted issuance rate by forecast month on the test period.

    Uses the evaluation HistGradientBoosting bundle, which was not fit on test labels.
    The production bundle is deliberately not used here: its calibrator saw the test period.
    """
    paths = paths or get_paths()
    bundle = load_bundle(paths.model_dir / "evaluation" / "hgb")
    snaps = pd.read_parquet(paths.snapshots)
    metrics = json.loads((paths.reports / "metrics.json").read_text(encoding="utf-8"))
    test_start = pd.Timestamp(metrics["split"]["test_start"])
    test = snaps.loc[snaps["forecast_date"] >= test_start].copy()
    features = list(bundle["features"])
    test["predicted"] = bundle["model"].predict_proba(model_matrix(test, features))[:, 1]
    grouped = test.groupby(test["forecast_date"].dt.strftime("%Y-%m"), sort=True)
    out = grouped.agg(n=("label", "size"), observed=("label", "mean"), predicted=("predicted", "mean"))
    out = out.reset_index(names="month")
    out["gap"] = out["predicted"] - out["observed"]
    return out


def write_test_month_calibration(paths: Paths | None = None) -> Path:
    paths = paths or get_paths()
    table = test_month_calibration(paths)
    dest = paths.reports / "test_month_calibration.json"
    dest.write_text(table.to_json(orient="records", double_precision=5), encoding="utf-8")
    return dest
