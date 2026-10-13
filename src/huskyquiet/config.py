"""Central configuration: paths, dataset identifiers, and modeling constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DATASET_ID = "76t5-zqzr"
SOCRATA_DOMAIN = "data.seattle.gov"
RESOURCE_URL = f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.json"
METADATA_URL = f"https://{SOCRATA_DOMAIN}/api/views/{DATASET_ID}.json"

# Prediction problem
HORIZON_DAYS = 90
# Applications older than this at the forecast date are outside the modeled population.
# Chosen from the empirical applied->issued duration distribution (see docs/data_audit.md).
MAX_APPLICATION_AGE_DAYS = 730
# First forecast date: the permit system's records become dense in 2005-2006, so a 2008
# start guarantees the 730-day risk set is fully populated.
FIRST_FORECAST_DATE = "2008-01-01"

# Dates before this are treated as sentinel / placeholder values (e.g. 1900-01-01).
MIN_VALID_DATE = "1990-01-01"

# Loose bounding box around the City of Seattle used to reject invalid coordinates.
SEATTLE_BBOX = {"lat_min": 47.45, "lat_max": 47.80, "lon_min": -122.48, "lon_max": -122.22}
# University of Washington (Red Square) reference point.
UW_LAT, UW_LON = 47.6560, -122.3095

RANDOM_SEED = 20261012


def _default_root() -> Path:
    env = os.environ.get("HUSKYQUIET_HOME")
    if env:
        return Path(env).resolve()
    # src/huskyquiet/config.py -> project root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=_default_root)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def raw_permits(self) -> Path:
        return self.raw / "building_permits_raw.parquet"

    @property
    def raw_metadata(self) -> Path:
        return self.raw / "download_metadata.json"

    @property
    def clean_permits(self) -> Path:
        return self.interim / "permits_clean.parquet"

    @property
    def snapshots(self) -> Path:
        return self.processed / "snapshots_h90.parquet"

    @property
    def model_dir(self) -> Path:
        return self.artifacts / "models"

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.artifacts, self.reports):
            p.mkdir(parents=True, exist_ok=True)


def get_paths() -> Paths:
    return Paths()
