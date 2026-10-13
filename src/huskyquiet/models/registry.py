"""Persist and load model bundles (estimator + calibrator + metadata).

Bundles are joblib pickles: only load artifacts produced by this pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

BUNDLE_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def save_bundle(directory: Path, model: Any, features: tuple[str, ...], metadata: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "features": tuple(features), "metadata": metadata}
    joblib.dump(bundle, directory / BUNDLE_FILE, compress=3)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return directory / BUNDLE_FILE


def load_bundle(directory: Path) -> dict[str, Any]:
    path = directory / BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"No model bundle at {path}; run `huskyquiet train` first")
    bundle = joblib.load(path)
    if not {"model", "features", "metadata"} <= set(bundle):
        raise ValueError(f"Malformed model bundle at {path}")
    return bundle
