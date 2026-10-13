"""Read saved evaluation reports. The web app displays these values; it does not recompute them."""

from __future__ import annotations

import json
from pathlib import Path

from huskyquiet.config import Paths, get_paths


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(paths: Paths | None = None) -> dict:
    paths = paths or get_paths()
    path = paths.reports / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics at {path}. Run `python -m huskyquiet train`.")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a metrics object")
    return payload


def load_model_metadata(paths: Paths | None = None) -> dict:
    paths = paths or get_paths()
    path = paths.model_dir / "production" / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No production metadata at {path}.")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a metadata object")
    return payload


def evaluation_headline(metrics: dict) -> dict[str, float | str | int]:
    """Pull the locked test-set comparison out of ``reports/metrics.json``."""
    key = str(metrics["selection"]["selected_test_key"])
    chosen = metrics["test"][key]
    baseline = metrics["test"]["historical_rate_baseline"]
    return {
        "model_key": key,
        "roc_auc": float(chosen["roc_auc"]),
        "pr_auc": float(chosen["pr_auc"]),
        "brier": float(chosen["brier"]),
        "ece": float(chosen["ece"]),
        "baseline_roc_auc": float(baseline["roc_auc"]),
        "baseline_pr_auc": float(baseline["pr_auc"]),
        "baseline_brier": float(baseline["brier"]),
        "baseline_ece": float(baseline["ece"]),
        "n": int(chosen["n"]),
        "prevalence": float(chosen["prevalence"]),
    }
