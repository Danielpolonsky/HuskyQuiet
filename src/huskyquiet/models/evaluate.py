"""Evaluation metrics, calibration diagnostics, and permit-clustered bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, weights: np.ndarray | None = None
) -> float:
    """Weighted mean |observed - predicted| over equal-mass probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, float)
    order = np.argsort(p, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    bins = np.minimum((cum * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        idx = order[bins == b]
        if idx.size == 0:
            continue
        wb = w[idx]
        ece += wb.sum() / w.sum() * abs(np.average(y[idx], weights=wb) - np.average(p[idx], weights=wb))
    return float(ece)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration ``logit P(y=1) = a + b * logit(p)``; ideal a=0, b=1."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = np.log(p / (1 - p))[:, None]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y, int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def binary_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    out: dict[str, float] = {
        "n": int(len(y)),
        "prevalence": float(np.average(y, weights=weights)),
        "mean_predicted": float(np.average(p, weights=weights)),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, weights=weights),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p, sample_weight=weights))
        out["pr_auc"] = float(average_precision_score(y, p, sample_weight=weights))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    if weights is None:
        slope, intercept = calibration_slope_intercept(y, p)
        out["calibration_slope"], out["calibration_intercept"] = slope, intercept
    return out


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
    )


def cluster_bootstrap(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    clusters: pd.Series,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] | None = None,
    reference: str | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Percentile CIs resampling whole permits (clusters) with replacement.

    Implemented with multiplicity weights: each bootstrap replicate draws permits from a
    multinomial and weights every snapshot by its permit's draw count. When ``reference``
    is given, CIs for paired differences (model - reference) are also returned.
    """
    metric_fns = metric_fns or {
        "roc_auc": lambda yy, pp, ww: roc_auc_score(yy, pp, sample_weight=ww),
        "pr_auc": lambda yy, pp, ww: average_precision_score(yy, pp, sample_weight=ww),
        "brier": lambda yy, pp, ww: brier_score_loss(yy, pp, sample_weight=ww),
    }
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(clusters)
    n_clusters = len(uniques)
    draws: dict[str, dict[str, list[float]]] = {m: {k: [] for k in metric_fns} for m in preds}
    diffs: dict[str, dict[str, list[float]]] = {m: {k: [] for k in metric_fns} for m in preds}
    y = np.asarray(y, int)
    for _ in range(n_boot):
        counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
        w = counts[codes].astype(float)
        mask = w > 0
        values = {
            m: {k: float(fn(y[mask], p[mask], w[mask])) for k, fn in metric_fns.items()}
            for m, p in preds.items()
        }
        for m in preds:
            for k in metric_fns:
                draws[m][k].append(values[m][k])
                if reference is not None:
                    diffs[m][k].append(values[m][k] - values[reference][k])

    def summarize(arr: list[float]) -> dict[str, float]:
        a = np.asarray(arr)
        return {"mean": float(a.mean()), "ci_low": float(np.quantile(a, 0.025)),
                "ci_high": float(np.quantile(a, 0.975))}

    result: dict[str, dict[str, dict[str, float]]] = {}
    for m in preds:
        result[m] = {k: summarize(v) for k, v in draws[m].items()}
        if reference is not None and m != reference:
            result[m].update({f"diff_vs_{reference}_{k}": summarize(v) for k, v in diffs[m].items()})
    return result


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Model-agnostic permutation importance (increase in log loss / drop in ROC-AUC)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    base = np.clip(predict(X), EPS, 1 - EPS)
    base_ll, base_auc = log_loss(y, base, labels=[0, 1]), roc_auc_score(y, base)
    rows = []
    for col in X.columns:
        d_ll, d_auc = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = Xp[col].to_numpy()[rng.permutation(len(Xp))]
            p = np.clip(predict(Xp), EPS, 1 - EPS)
            d_ll.append(log_loss(y, p, labels=[0, 1]) - base_ll)
            d_auc.append(base_auc - roc_auc_score(y, p))
        rows.append({"feature": col, "log_loss_increase": float(np.mean(d_ll)),
                     "roc_auc_drop": float(np.mean(d_auc))})
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False, ignore_index=True)


def plot_reliability(curves: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    for name, tab in curves.items():
        ax.plot(tab["mean_predicted"], tab["observed_rate"], marker="o", label=name)
    ax.set_xlabel("Mean predicted probability (decile bins)")
    ax.set_ylabel("Observed 90-day issuance rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
