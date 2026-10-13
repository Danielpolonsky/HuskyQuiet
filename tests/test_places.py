import numpy as np
import pandas as pd

from huskyquiet.analysis import (
    ISSUANCE_NOT_NOISE,
    comparison_text,
    explanatory_rows,
    nearby_summary,
    probability_color,
    resolve_pin,
    summarize_permits,
)
from huskyquiet.areas import area_by_name, load_areas, permits_in_area
from huskyquiet.geo import geometry_bounds, haversine_m, in_seattle, point_in_geometry, with_distance
from huskyquiet.mapdeck import mappable_permits
from huskyquiet.reporting import evaluation_headline


def _permits():
    return pd.DataFrame({
        "permit_num": ["1-CN", "2-CN", "3-CN", "4-CN"],
        "address_key": ["A ST", "B ST", "C ST", None],
        "permit_class": ["Multifamily", "Single Family/Duplex", "Multifamily", "Commercial"],
        "permit_type_mapped": ["Building"] * 4,
        "permit_type_desc": ["New", "Addition/Alteration", "New", "Demolition"],
        "description": ["Apartment building", "Deck", "Tower", "No map"],
        "est_project_cost": [1_000_000, None, 2_000, 5],
        "age_days": [10, 20, 30, 40],
        "latitude": [47.6560, 47.6562, 47.70, np.nan],
        "longitude": [-122.3095, -122.3096, -122.30, np.nan],
        "probability_issued_90d": [0.2, 0.5, 0.9, 0.1],
        "dist_uw_km": [0.01, 0.02, 5.0, np.nan],
        "kw_multifamily": [1, 0, 1, 0],
        "desc_missing": [0, 0, 0, 0],
    })


def test_haversine_is_symmetric_and_seattle_scaled():
    uw = (47.6560, -122.3095)
    needle = (47.6205, -122.3493)
    forward = float(haversine_m(*uw, *needle))
    backward = float(haversine_m(*needle, *uw))
    assert abs(forward - backward) < 1e-6
    assert float(haversine_m(*uw, *uw)) == 0
    assert 4800 < forward < 5200


def test_radius_filter_uses_real_coordinates_only():
    nearby = with_distance(_permits(), 47.6560, -122.3095, 500)
    assert nearby["permit_num"].tolist() == ["1-CN", "2-CN"]
    assert nearby["distance_m"].is_monotonic_increasing
    assert "4-CN" not in set(nearby["permit_num"])


def test_point_in_square_respects_holes():
    square = {"type": "Polygon", "coordinates": [[
        [-122.32, 47.65], [-122.30, 47.65], [-122.30, 47.67], [-122.32, 47.67], [-122.32, 47.65],
    ]]}
    assert point_in_geometry(-122.31, 47.66, square)
    assert not point_in_geometry(-122.40, 47.66, square)
    holed = {"type": "Polygon", "coordinates": [
        square["coordinates"][0],
        [[-122.315, 47.655], [-122.305, 47.655], [-122.305, 47.665], [-122.315, 47.665], [-122.315, 47.655]],
    ]}
    assert not point_in_geometry(-122.31, 47.66, holed)
    assert point_in_geometry(-122.318, 47.668, holed)


def test_named_areas_are_real_polygons_and_separate_downtown():
    areas = load_areas()
    assert {area.name for area in areas} >= {
        "University District", "University Village", "Roosevelt", "Wallingford", "Ravenna",
    }
    district = area_by_name("University District")
    min_lon, min_lat, max_lon, max_lat = geometry_bounds(district.geometry)
    assert 47.64 < min_lat < max_lat < 47.70
    assert min_lon < -122.31 < max_lon
    downtown = pd.DataFrame({
        "latitude": [47.6062], "longitude": [-122.3321], "permit_num": ["DOWNTOWN-CN"],
    })
    assert permits_in_area(downtown, district).empty
    assert not in_seattle(40.0, -122.3)


def test_expected_issuances_are_the_sum_of_probabilities():
    frame = _permits().iloc[:3]
    summary = summarize_permits(frame, "Sample", "within 500 m")
    assert summary.n_permits == 3
    assert summary.expected_issuances_90d == 1.6
    assert summary.top[0]["permit_num"] == "3-CN"
    text = comparison_text(summary, summarize_permits(frame.iloc[0:0], "Empty", "within 500 m"))
    assert "1.6" in text and "0.0" in text
    assert "do not measure construction starts" in text
    assert text.endswith(ISSUANCE_NOT_NOISE)


def test_nearby_summary_matches_radius():
    summary = nearby_summary(_permits(), 47.6560, -122.3095, 500, "Near UW")
    assert summary.n_permits == 2
    assert abs(summary.expected_issuances_90d - 0.7) < 1e-9


def test_probability_color_darkens_as_probability_rises():
    low = probability_color(0.0)
    high = probability_color(1.0)
    assert low[2] > high[2]
    assert len(high) == 4


def test_explanatory_rows_name_fired_keywords():
    rows = dict(explanatory_rows(_permits().iloc[0]))
    assert "Apartment or multifamily" in rows["Scope keywords used by the model"]
    assert rows["Estimated project cost"] == "$1,000,000"


def test_resolve_pin_accepts_coordinates_and_rejects_a_blank_address():
    pin = resolve_pin("  ", 47.66, -122.31, True, geocoder=None)
    assert pin.source == "coordinates" and pin.latitude == 47.66
    try:
        resolve_pin("   ", None, None, False, geocoder=None)
    except ValueError as exc:
        assert "address" in str(exc).lower()
    else:
        raise AssertionError("blank address should fail")


def test_marker_circle_stays_on_the_recorded_coordinate():
    from huskyquiet.mapdeck import _circle

    ring = _circle(-122.313, 47.6615, 80.0, steps=12)
    assert ring[0] == ring[-1]
    assert len(ring) == 13
    for lon, lat in ring[:-1]:
        distance = float(haversine_m(47.6615, -122.313, lat, lon))
        assert 70 < distance < 90


def test_map_drops_missing_coordinates():
    mapped = mappable_permits(_permits())
    assert set(mapped["permit_num"]) == {"1-CN", "2-CN", "3-CN"}
    assert all(len(color) == 4 for color in mapped["color"])


def test_headline_reads_the_saved_report():
    import json

    from huskyquiet.config import get_paths

    path = get_paths().reports / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    headline = evaluation_headline(metrics)
    assert headline["model_key"] == "hgb_calibrated_platt"
    assert headline["roc_auc"] == metrics["test"]["hgb_calibrated_platt"]["roc_auc"]
    assert headline["baseline_roc_auc"] == metrics["test"]["historical_rate_baseline"]["roc_auc"]
