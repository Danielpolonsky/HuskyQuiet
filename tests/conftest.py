from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TYPES = [("Building", "Single Family/Duplex", "-CN"), ("Building", "Commercial", "-CN"),
         ("Building", "Multifamily", "-CN"), ("Demolition", "Single Family/Duplex", "-DM")]
DESCRIPTIONS = [
    "Construct new single family residence per plan.",
    "Construct addition and alterations to existing single family residence, per plan.",
    "Tenant improvement to existing office, subject to field inspection (STFI).",
    "Demolish existing single family residence.",
    "Establish use as townhouses and construct 4-unit townhouse per plan.",
]


def make_history(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic permit timeline with the columns produced by ``data.clean``."""
    rng = np.random.default_rng(seed)
    applied = pd.Timestamp("2008-06-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D")
    dur = rng.gamma(1.2, 70, n).astype(int)
    issued = applied + pd.to_timedelta(dur, unit="D")
    never = rng.random(n) < 0.12
    issued = pd.Series(issued).where(~never, pd.NaT)
    kinds = rng.integers(0, len(TYPES), n)
    df = pd.DataFrame({
        "permit_num": [f"{6000000 + i}{TYPES[k][2]}" for i, k in enumerate(kinds)],
        "applied_date": pd.Series(applied).astype("datetime64[ns]"),
        "issued_date": issued.astype("datetime64[ns]"),
        "permit_type_mapped": pd.array([TYPES[k][0] for k in kinds], dtype="string"),
        "permit_class": pd.array([TYPES[k][1] for k in kinds], dtype="string"),
        "permit_type_desc": pd.array(rng.choice(["New", "Addition/Alteration", None], n), dtype="string"),
        "permit_suffix": pd.array([TYPES[k][2][1:] for k in kinds], dtype="string"),
        "description": pd.array(rng.choice(DESCRIPTIONS, n), dtype="string"),
        "est_project_cost": rng.lognormal(11, 1.5, n).round(0),
        "housing_units_added": rng.integers(0, 4, n).astype(float),
        "housing_units_removed": rng.integers(0, 2, n).astype(float),
        "related_mup": pd.array(np.where(rng.random(n) < 0.2, "3000000-LU", None), dtype="string"),
        "latitude": rng.uniform(47.55, 47.70, n),
        "longitude": rng.uniform(-122.40, -122.28, n),
        "zip": pd.array(rng.choice(["98105", "98103", "98115", "98122"], n), dtype="string"),
        "address_key": pd.array([f"{rng.integers(1, 60) * 100} NE 45TH ST" for _ in range(n)],
                                dtype="string"),
        "is_model_eligible": True,
    })
    return df


@pytest.fixture
def history() -> pd.DataFrame:
    return make_history()
