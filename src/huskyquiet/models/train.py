"""Training, model selection, calibration, evaluation, backtesting, and persistence.

Temporal design (``t`` = forecast date, ``h`` = 90-day horizon)::

    train:      t + h <= val_start                 (labels fully observed before validation)
    validation: val_start <= t,  t + h <= test_start
    test:       test_start <= t <= last labelled forecast date

The 90-day embargo before each boundary guarantees that no training label depends on
events inside a later evaluation period. Hyperparameters and the calibration method are
chosen on validation only; the test period is scored once per model.

The *production* model reruns the identical procedure shifted forward by one period:
fit on everything with ``t + h <= test_start`` and calibrate on the test period. Its
expected performance is the evaluated test performance of the procedure.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from huskyquiet import __version__
from huskyquiet.config import HORIZON_DAYS, MAX_APPLICATION_AGE_DAYS, Paths, get_paths
from huskyquiet.features.engineering import (
    CONTEXT_FEATURES,
    FEATURE_COLUMNS,
    KEYWORD_FEATURES,
)
from huskyquiet.models.baseline import HistoricalRateModel, recent_rate_forecast
from huskyquiet.models.calibration import (
    CalibratedModel,
    ProbabilityCalibrator,
    select_calibration_method,
)
from huskyquiet.models.evaluate import (
    binary_metrics,
    cluster_bootstrap,
    permutation_importance,
    plot_reliability,
    reliability_table,
)
from huskyquiet.models.pipelines import build_hgb, build_logistic, model_matrix
from huskyquiet.models.registry import save_bundle

log = logging.getLogger(__name__)

LOGREG_GRID: tuple[float, ...] = (0.01, 0.1, 1.0)
HGB_GRID: tuple[dict[str, Any], ...] = (
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 600},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 300},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 600},
    {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 300},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300},
)
TEST_MONTHS = 12
VAL_MONTHS = 12
BACKTEST_STARTS: tuple[str, ...] = ("2020-08-01", "2021-08-01", "2022-08-01", "2023-08-01")
N_BOOTSTRAP = 200
RECENT_WINDOW_START = "2016-01-01"


@dataclass(frozen=True)
class TemporalSplit:
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    last_forecast: pd.Timestamp
    horizon_days: int = HORIZON_DAYS

    @classmethod
    def from_last_forecast(
        cls, last_forecast: pd.Timestamp, test_months: int = TEST_MONTHS, val_months: int = VAL_MONTHS,
        horizon_days: int = HORIZON_DAYS,
    ) -> TemporalSplit:
        test_start = (last_forecast - pd.DateOffset(months=test_months - 1)).normalize()
        val_start = (test_start - pd.DateOffset(months=val_months)).normalize()
        return cls(pd.Timestamp(val_start), pd.Timestamp(test_start), pd.Timestamp(last_forecast),
                   horizon_days)

    def masks(self, t: pd.Series) -> dict[str, pd.Series]:
        h = pd.Timedelta(days=self.horizon_days)
        train = t + h <= self.val_start
        val = (t >= self.val_start) & (t + h <= self.test_start)
        test = (t >= self.test_start) & (t <= self.last_forecast)
        assert not (train & val).any() and not (val & test).any() and not (train & test).any()
        return {"train": train, "val": val, "test": test}

    def to_dict(self) -> dict[str, str | int]:
        return {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(self).items()}


def _p(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _fit_logistic(X: pd.DataFrame, y: np.ndarray, C: float, features=FEATURE_COLUMNS):
    return build_logistic(C, features).fit(X[list(features)], y)


def _fit_hgb(X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], features=FEATURE_COLUMNS):
    return build_hgb(params, features).fit(X[list(features)], y)


class _FeatureSubset:
    """Wraps an estimator trained on a subset of feature columns."""

    def __init__(self, estimator: Any, features: tuple[str, ...]) -> None:
        self.estimator, self.features = estimator, tuple(features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[list(self.features)])


def select_logistic(Xtr, ytr, Xva, yva) -> tuple[float, Any, dict[str, float]]:
    """Grid-search C on validation log loss; returns (best C, its fitted model, scores)."""
    scores, models = {}, {}
    for C in LOGREG_GRID:
        t0 = time.perf_counter()
        models[C] = _fit_logistic(Xtr, ytr, C)
        scores[str(C)] = float(log_loss(yva, _p(models[C], Xva)))
        log.info("  logistic C=%-5s val log loss %.4f (%.0fs)", C, scores[str(C)], time.perf_counter() - t0)
    best = min(LOGREG_GRID, key=lambda c: scores[str(c)])
    return best, models[best], scores


def select_hgb(Xtr, ytr, Xva, yva) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Grid-search HGB on validation log loss; returns (best params, fitted model, results)."""
    results, best_model, best_loss = [], None, np.inf
    for params in HGB_GRID:
        t0 = time.perf_counter()
        m = _fit_hgb(Xtr, ytr, params)
        loss = float(log_loss(yva, _p(m, Xva)))
        results.append({**params, "val_log_loss": loss})
        log.info("  hgb %s val log loss %.4f (%.0fs)", params, loss, time.perf_counter() - t0)
        if loss < best_loss:
            best_model, best_loss = m, loss
    best = min(results, key=lambda r: r["val_log_loss"])
    return {k: v for k, v in best.items() if k != "val_log_loss"}, best_model, results


def _calibrate(estimator: Any, Xva: pd.DataFrame, yva: np.ndarray, groups: pd.Series):
    p_raw = _p(estimator, Xva)
    method, cv_scores = select_calibration_method(p_raw, yva, groups.to_numpy())
    cal = ProbabilityCalibrator(method).fit(p_raw, yva)
    return CalibratedModel(estimator, cal), method, cv_scores


def _slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {
        **{f"type={v}": df["permit_type_mapped"] == v for v in ("Building", "Demolition")},
        **{f"age={v}": df["age_bucket"] == v for v in df["age_bucket"].dropna().unique()},
        "within_2km_of_UW": df["dist_uw_km"] <= 2.0,
        "class=Multifamily": df["permit_class"] == "Multifamily",
        "class=Single Family/Duplex": df["permit_class"] == "Single Family/Duplex",
        "class=Commercial": df["permit_class"] == "Commercial",
        "desc=New": df["permit_type_desc"] == "New",
        "desc=Addition/Alteration": df["permit_type_desc"] == "Addition/Alteration",
    }
    return {k: v.fillna(False).astype(bool) for k, v in masks.items()}


def rolling_backtest(snaps: pd.DataFrame, C: float, hgb_params: dict[str, Any], test_months: int = 12):
    """Rolling-origin evaluation of the modeling procedure (uncalibrated, fixed hyperparameters)."""
    h = pd.Timedelta(days=HORIZON_DAYS)
    t = snaps["forecast_date"]
    folds = []
    for start in BACKTEST_STARTS:
        s = pd.Timestamp(start)
        end = s + pd.DateOffset(months=test_months)
        tr, te = snaps[t + h <= s], snaps[(t >= s) & (t < end)]
        if te.empty:
            continue
        Xtr, Xte = model_matrix(tr), model_matrix(te)
        ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
        hist = HistoricalRateModel().fit(tr, ytr)
        preds = {
            "historical_rate": _p(hist, te),
            "recent_rate": recent_rate_forecast(te)[:, 1],
            "logistic": _p(_fit_logistic(Xtr, ytr, C), Xte),
            "hgb": _p(_fit_hgb(Xtr, ytr, hgb_params), Xte),
        }
        fold = {"test_start": str(s.date()), "test_end_exclusive": str(end.date()),
                "n_train": len(tr), "n_test": len(te),
                "metrics": {k: binary_metrics(yte, v) for k, v in preds.items()}}
        log.info("Backtest %s: %s", fold["test_start"],
                 {k: round(v["roc_auc"], 4) for k, v in fold["metrics"].items()})
        folds.append(fold)
    return folds


def _round(obj: Any, nd: int = 5) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def run_training(paths: Paths | None = None, run_backtest: bool = True) -> dict[str, Any]:
    paths = paths or get_paths()
    paths.ensure()
    snaps = pd.read_parquet(paths.snapshots)
    summary = json.loads((paths.reports / "snapshot_summary.json").read_text())
    split = TemporalSplit.from_last_forecast(snaps["forecast_date"].max())
    masks = split.masks(snaps["forecast_date"])
    tr, va, te = (snaps[masks[k]].reset_index(drop=True) for k in ("train", "val", "test"))
    log.info("Split %s -> train %d / val %d / test %d rows (embargoed rows dropped: %d)",
             split.to_dict(), len(tr), len(va), len(te), len(snaps) - len(tr) - len(va) - len(te))

    Xtr, Xva, Xte = model_matrix(tr), model_matrix(va), model_matrix(te)
    ytr, yva, yte = (d["label"].to_numpy() for d in (tr, va, te))

    # --- Baselines ------------------------------------------------------------------
    hist = HistoricalRateModel().fit(tr, ytr)
    hist_cal, hist_cal_method, _ = _calibrate(hist, va, yva, va["permit_num"])

    # --- Model selection on validation ------------------------------------------------
    log.info("Selecting logistic regression regularization")
    best_C, lr, lr_scores = select_logistic(Xtr, ytr, Xva, yva)
    log.info("Selecting gradient boosting hyperparameters")
    best_hgb, hgb, hgb_scores = select_hgb(Xtr, ytr, Xva, yva)

    lr_cal, lr_cal_method, lr_cal_cv = _calibrate(lr, Xva, yva, va["permit_num"])
    hgb_cal, hgb_cal_method, hgb_cal_cv = _calibrate(hgb, Xva, yva, va["permit_num"])

    candidates = {
        "logistic": (lr_cal, lr_cal_cv[lr_cal_method]),
        "hgb": (hgb_cal, hgb_cal_cv[hgb_cal_method]),
    }
    selected = min(candidates, key=lambda k: candidates[k][1])
    log.info("Selected model family by calibrated validation log loss: %s", selected)

    # --- Test evaluation (scored once) ------------------------------------------------
    preds = {
        "climatology_train_rate": np.full(len(te), ytr.mean()),
        "recent_rate_baseline": recent_rate_forecast(te)[:, 1],
        "historical_rate_baseline": _p(hist, te),
        f"historical_rate_baseline_recalibrated_{hist_cal_method}": _p(hist_cal, te),
        "logistic_raw": _p(lr, Xte),
        f"logistic_calibrated_{lr_cal_method}": _p(lr_cal, Xte),
        "hgb_raw": _p(hgb, Xte),
        f"hgb_calibrated_{hgb_cal_method}": _p(hgb_cal, Xte),
    }
    test_metrics = {k: binary_metrics(yte, p) for k, p in preds.items()}
    # One row per permit: its first snapshot in the test period. Unlike 1/k weighting, this
    # selection does not depend on the outcome (k itself depends on when the permit is issued).
    first = (te["forecast_date"] == te.groupby("permit_num")["forecast_date"].transform("min")).to_numpy()
    test_metrics_first = {k: binary_metrics(yte[first], p[first]) for k, p in preds.items()}
    ref_brier = test_metrics["historical_rate_baseline"]["brier"]
    clim_brier = test_metrics["climatology_train_rate"]["brier"]
    for k in test_metrics:
        test_metrics[k]["brier_skill_vs_historical_rate"] = 1 - test_metrics[k]["brier"] / ref_brier
        test_metrics[k]["brier_skill_vs_climatology"] = 1 - test_metrics[k]["brier"] / clim_brier

    sel_key = f"{selected}_calibrated_{hgb_cal_method if selected == 'hgb' else lr_cal_method}"
    boot_keys = ["historical_rate_baseline", "recent_rate_baseline",
                 f"logistic_calibrated_{lr_cal_method}", f"hgb_calibrated_{hgb_cal_method}"]
    log.info("Permit-clustered bootstrap (%d replicates)", N_BOOTSTRAP)
    bootstrap = cluster_bootstrap(yte, {k: preds[k] for k in boot_keys}, te["permit_num"],
                                  reference="historical_rate_baseline", n_boot=N_BOOTSTRAP)

    slices = {}
    for name, mask in _slices(te).items():
        m = mask.to_numpy()
        if m.sum() < 200:
            continue
        slices[name] = {k: binary_metrics(yte[m], preds[k][m])
                        for k in ("historical_rate_baseline", sel_key)}

    reliability = {k: reliability_table(yte, preds[k]).to_dict(orient="list")
                   for k in ("historical_rate_baseline", f"logistic_calibrated_{lr_cal_method}",
                             "hgb_raw", f"hgb_calibrated_{hgb_cal_method}")}
    plot_reliability({k: pd.DataFrame(v) for k, v in reliability.items()},
                     paths.reports / "figures" / "reliability_test.png",
                     f"Test-period reliability ({split.test_start.date()} to {split.last_forecast.date()})")

    # --- Post-hoc analyses (not used for any selection) -------------------------------
    sel_model = candidates[selected][0]
    sample = te.sample(min(20_000, len(te)), random_state=0)
    imp = permutation_importance(lambda X: _p(sel_model, X), model_matrix(sample),
                                 sample["label"].to_numpy(), n_repeats=3)

    ablations = {}
    for name, drop in (("no_description_keywords", set(KEYWORD_FEATURES) | {"desc_missing"}),
                       ("no_context_features", set(CONTEXT_FEATURES)),
                       ("no_cost", {"log_est_cost", "est_cost_missing"})):
        feats = tuple(c for c in FEATURE_COLUMNS if c not in drop)
        est = _FeatureSubset(_fit_hgb(Xtr, ytr, best_hgb, feats), feats)
        cal_est, method, _ = _calibrate(est, Xva, yva, va["permit_num"])
        ablations[name] = binary_metrics(yte, _p(cal_est, Xte))
        log.info("Ablation %s: test ROC-AUC %.4f, Brier %.4f", name, ablations[name]["roc_auc"],
                 ablations[name]["brier"])
    # Sensitivity to the training window (older cohorts may suffer record survivorship).
    recent = (tr["forecast_date"] >= pd.Timestamp(RECENT_WINDOW_START)).to_numpy()
    est = _fit_hgb(Xtr[recent], ytr[recent], best_hgb)
    cal_est, _, _ = _calibrate(est, Xva, yva, va["permit_num"])
    ablations[f"train_since_{RECENT_WINDOW_START[:4]}"] = {
        **binary_metrics(yte, _p(cal_est, Xte)), "n_train": int(recent.sum())}
    log.info("Sensitivity train_since_%s: %s", RECENT_WINDOW_START[:4],
             {k: round(v, 4) for k, v in ablations[f"train_since_{RECENT_WINDOW_START[:4]}"].items()})

    folds = rolling_backtest(snaps, best_C, best_hgb) if run_backtest else []

    # --- Persist evaluation artifacts -------------------------------------------------
    common_meta = {
        "package_version": __version__,
        "horizon_days": HORIZON_DAYS,
        "max_application_age_days": MAX_APPLICATION_AGE_DAYS,
        "data_as_of": summary["data_as_of"],
        "features": list(FEATURE_COLUMNS),
        "categories_seen_in_training": {
            c: sorted(tr[c].dropna().astype(str).unique().tolist())
            for c in ("permit_type_mapped", "permit_class", "permit_type_desc", "permit_suffix")
        },
        "target": f"P(issued within [t, t+{HORIZON_DAYS}d) | active application at start of day t)",
        "disclaimer": "Predicts permit issuance timing only; not construction start or noise.",
    }
    eval_dir = paths.model_dir / "evaluation"
    for name, model, meta in (
        ("historical_rate_baseline", hist, {}),
        ("logistic", lr_cal, {"C": best_C, "calibration": lr_cal_method}),
        ("hgb", hgb_cal, {"params": best_hgb, "calibration": hgb_cal_method}),
    ):
        save_bundle(eval_dir / name, model, FEATURE_COLUMNS, {
            **common_meta, **meta, "model": name, "role": "evaluation",
            "trained_on": f"forecast dates with t + {HORIZON_DAYS}d <= {split.val_start.date()}",
            "calibrated_on": f"{split.val_start.date()} <= t, t + {HORIZON_DAYS}d <= {split.test_start.date()}",
        })

    # --- Production refit: same procedure, shifted forward one period ------------------
    log.info("Refitting production model (%s) on all data before the test period", selected)
    prod_train = snaps[snaps["forecast_date"] + pd.Timedelta(days=HORIZON_DAYS) <= split.test_start]
    Xpt, ypt = model_matrix(prod_train), prod_train["label"].to_numpy()
    prod_est = _fit_hgb(Xpt, ypt, best_hgb) if selected == "hgb" else _fit_logistic(Xpt, ypt, best_C)
    # Same procedure as evaluation: the calibration method is re-selected on the
    # (shifted) calibration window by permit-grouped CV, then fit on all of it.
    prod_model, prod_method, prod_cal_cv = _calibrate(prod_est, Xte, yte, te["permit_num"])
    save_bundle(paths.model_dir / "production", prod_model, FEATURE_COLUMNS, {
        **common_meta, "model": selected, "role": "production", "calibration": prod_method,
        "hyperparameters": best_hgb if selected == "hgb" else {"C": best_C},
        "trained_on": f"forecast dates with t + {HORIZON_DAYS}d <= {split.test_start.date()}",
        "calibrated_on": f"{split.test_start.date()} <= t <= {split.last_forecast.date()}",
        "n_train_snapshots": len(prod_train), "n_calibration_snapshots": len(te),
        "expected_performance": f"see reports/metrics.json -> test.{sel_key}",
    })

    metrics = {
        "split": split.to_dict(),
        "sizes": {
            "train": {"rows": len(tr), "permits": int(tr["permit_num"].nunique()), "positive_rate": float(ytr.mean())},
            "val": {"rows": len(va), "permits": int(va["permit_num"].nunique()), "positive_rate": float(yva.mean())},
            "test": {"rows": len(te), "permits": int(te["permit_num"].nunique()), "positive_rate": float(yte.mean())},
            "test_permits_also_observed_in_train": int(te.loc[te["permit_num"].isin(tr["permit_num"]),
                                                              "permit_num"].nunique()),
            "test_rows_whose_permit_observed_in_train": int(te["permit_num"].isin(tr["permit_num"]).sum()),
        },
        "selection": {
            "logistic_val_log_loss_by_C": lr_scores, "logistic_best_C": best_C,
            "hgb_grid": hgb_scores, "hgb_best_params": best_hgb,
            "calibration_cv_log_loss": {"logistic": lr_cal_cv, "hgb": hgb_cal_cv},
            "calibration_method": {"logistic": lr_cal_method, "hgb": hgb_cal_method,
                                   "historical_rate": hist_cal_method},
            "selected_model": selected, "selected_test_key": sel_key,
        },
        "test": test_metrics,
        "test_first_snapshot_per_permit": test_metrics_first,
        "test_bootstrap_95ci": bootstrap,
        "test_slices": slices,
        "test_reliability": reliability,
        "permutation_importance_test_sample": imp.head(25).to_dict(orient="records"),
        "ablations_test": ablations,
        "backtest": folds,
    }
    metrics = _round(metrics)
    (paths.reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (paths.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _write_markdown_report(metrics, paths)
    log.info("Test metrics:\n%s", pd.DataFrame(test_metrics).T[
        ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "brier_skill_vs_historical_rate"]].round(4))
    return metrics


def _write_markdown_report(m: dict[str, Any], paths: Paths) -> None:
    cols = ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "calibration_slope",
            "brier_skill_vs_historical_rate", "brier_skill_vs_climatology", "mean_predicted", "prevalence"]
    test = pd.DataFrame(m["test"]).T[cols]
    first = pd.DataFrame(m["test_first_snapshot_per_permit"]).T[["n", "roc_auc", "pr_auc", "brier", "ece",
                                                                 "mean_predicted", "prevalence"]]
    lines = [
        "# HuskyQuiet model report (auto-generated by `huskyquiet train`)",
        "",
        f"Split: `{m['split']}`",
        "",
        "## Sample sizes",
        "",
        "```json",
        json.dumps(m["sizes"], indent=2),
        "```",
        "",
        "## Test-period metrics (each snapshot weighted equally)",
        "",
        test.round(4).to_string(),
        "",
        "## Test-period metrics (first test-period snapshot of each permit)",
        "",
        first.round(4).to_string(),
        "",
        "## Test slices (selected model vs historical-rate baseline)",
        "",
        pd.DataFrame({
            name: {"n": v[m["selection"]["selected_test_key"]]["n"],
                   "prevalence": v[m["selection"]["selected_test_key"]]["prevalence"],
                   "mean_predicted": v[m["selection"]["selected_test_key"]]["mean_predicted"],
                   "auc_model": v[m["selection"]["selected_test_key"]]["roc_auc"],
                   "auc_baseline": v["historical_rate_baseline"]["roc_auc"],
                   "brier_model": v[m["selection"]["selected_test_key"]]["brier"],
                   "brier_baseline": v["historical_rate_baseline"]["brier"]}
            for name, v in m["test_slices"].items()
        }).T.round(4).to_string(),
        "",
        "## Permit-clustered bootstrap 95% intervals",
        "",
        "```json",
        json.dumps(m["test_bootstrap_95ci"], indent=2),
        "```",
        "",
        "## Ablations (test)",
        "",
        pd.DataFrame(m["ablations_test"]).T[["roc_auc", "pr_auc", "brier", "ece"]].round(4).to_string(),
        "",
        "## Rolling-origin backtest (ROC-AUC / Brier, uncalibrated)",
        "",
    ]
    for f in m["backtest"]:
        row = ", ".join(f"{k}: {v['roc_auc']:.4f} / {v['brier']:.4f}" for k, v in f["metrics"].items())
        lines.append(f"- {f['test_start']} (n={f['n_test']}): {row}")
    lines += ["", "## Top permutation importances (selected model, test sample)", ""]
    lines.append(pd.DataFrame(m["permutation_importance_test_sample"]).round(5).to_string(index=False))
    (paths.reports / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
