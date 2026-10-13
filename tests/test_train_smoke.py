"""End-to-end smoke test of the training orchestrator on synthetic data (fast grids)."""

import json

import pandas as pd

from huskyquiet.config import Paths
from huskyquiet.features.snapshots import build_snapshots
from huskyquiet.models import train
from huskyquiet.models.registry import load_bundle
from huskyquiet.predict import IssuancePredictor, PermitApplication
from tests.conftest import make_history


def test_run_training_end_to_end(tmp_path, monkeypatch):
    history = make_history(n=900, seed=3)
    data_as_of = pd.Timestamp("2012-12-31")
    snaps = build_snapshots(history, data_as_of, first_forecast="2008-07-01")
    paths = Paths(root=tmp_path)
    paths.ensure()
    snaps.to_parquet(paths.snapshots, index=False)
    (paths.reports / "snapshot_summary.json").write_text(json.dumps({"data_as_of": str(data_as_of.date())}))

    monkeypatch.setattr(train, "LOGREG_GRID", (0.1,))
    monkeypatch.setattr(train, "HGB_GRID", ({"max_iter": 20, "max_leaf_nodes": 7},))
    monkeypatch.setattr(train, "BACKTEST_STARTS", ("2010-01-01",))
    monkeypatch.setattr(train, "N_BOOTSTRAP", 5)
    monkeypatch.setattr(train, "RECENT_WINDOW_START", "2009-06-01")

    metrics = train.run_training(paths=paths, run_backtest=True)

    assert metrics["selection"]["selected_model"] in ("logistic", "hgb")
    assert len(metrics["backtest"]) == 1
    sel = metrics["selection"]["selected_test_key"]
    assert 0 <= metrics["test"][sel]["brier"] <= 1
    assert (paths.reports / "metrics.json").exists() and (paths.reports / "model_report.md").exists()
    bundle = load_bundle(paths.model_dir / "production")
    assert bundle["metadata"]["role"] == "production"

    predictor = IssuancePredictor(history=history, paths=paths)
    res = predictor.predict(
        PermitApplication(applied_date="2012-11-01", permit_type_mapped="Building",
                          permit_class="Commercial"),
        as_of="2012-12-15",
    )
    assert 0 < res.probability < 1
