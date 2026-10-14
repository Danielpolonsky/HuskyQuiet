import io

import numpy as np
import pandas as pd

from huskyquiet.analysis import (
    EXPECTED_HELP,
    PROBABILITY_COLUMN,
    apply_filters,
    difference_points,
    explanatory_rows,
    format_probability,
    nearby_summary,
    permit_rows,
    permits_csv,
    probability_color,
    summarize_permits,
)
from huskyquiet.areas import area_by_label, area_by_name, load_areas, permits_in_area
from huskyquiet.geo import geometry_bounds, haversine_m, in_seattle, point_in_geometry, with_distance
from huskyquiet.mapdeck import area_deck, location_deck, mappable_permits
from huskyquiet.reporting import evaluation_headline, model_timeline


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
    assert summary.n_likely == 2
    assert EXPECTED_HELP == (
        "Adds the issuance probabilities of nearby applications. "
        "This counts permits, not unique construction projects."
    )


def test_differences_are_factual_and_never_rank_places():
    frame = with_distance(_permits(), 47.6560, -122.3095, 500)
    busy = summarize_permits(frame, "A", "within 500 m")
    empty = summarize_permits(frame.iloc[0:0], "B", "within 500 m")
    points = difference_points(busy, empty, "Location A", "Location B")
    text = " ".join(points)
    assert "Location A has 2 more pending permits within 500 m (2 compared with 0)." in text
    assert "0.7 near Location A and 0.0 near Location B" in text
    assert "Location B has none within 500 m" in text
    for word in ("quiet", "safer", "better", "worse", "risk"):
        assert word not in text.lower()


def test_identical_locations_get_one_plain_statement():
    frame = with_distance(_permits(), 47.6560, -122.3095, 500)
    summary = summarize_permits(frame, "A", "within 500 m")
    points = difference_points(summary, summary, identical=True)
    assert len(points) == 1
    assert points[0].startswith("Both entries resolve to the same point")
    assert "2 pending permits within 500 m" in points[0]


def test_filters_drive_every_figure_and_the_csv():
    frame = with_distance(_permits(), 47.6560, -122.3095, 500)
    filtered = apply_filters(frame, ["Multifamily"], 0.0)
    summary = summarize_permits(filtered, "A", "within 500 m")
    assert filtered["permit_num"].tolist() == ["1-CN"]
    assert summary.n_permits == 1 and abs(summary.expected_issuances_90d - 0.2) < 1e-9
    exported = pd.read_csv(io.BytesIO(permits_csv(filtered)))
    assert exported["Permit number"].tolist() == ["1-CN"]
    assert exported[PROBABILITY_COLUMN].tolist() == [20.0]
    assert list(exported.columns[:3]) == ["Permit number", "Address", "Permit type"]
    assert "Distance (m)" in exported.columns
    assert apply_filters(frame, None, 0.3)["permit_num"].tolist() == ["2-CN"]


def test_empty_filter_result_is_handled():
    frame = with_distance(_permits(), 47.6560, -122.3095, 500)
    none_left = apply_filters(frame, [], 0.0)
    summary = summarize_permits(none_left, "A", "within 500 m")
    assert none_left.empty
    assert (summary.n_permits, summary.expected_issuances_90d, summary.nearest_m) == (0, 0.0, None)
    assert permits_csv(none_left).decode("utf-8").startswith("Permit number,Address")
    assert location_deck(none_left, 47.6560, -122.3095, 500).layers


def test_permit_rows_use_readable_values():
    rows = permit_rows(_permits())
    assert rows.loc[3, "Address"] == "Address not recorded"
    assert rows.loc[0, "Estimated cost (USD)"] == 1_000_000
    assert "Distance (m)" not in rows.columns
    assert format_probability(0.756) == "76%"


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


def test_university_village_is_labeled_as_a_shopping_center_inside_ravenna():
    village = area_by_name("University Village")
    ravenna = area_by_name("Ravenna")
    assert village.label == "University Village (shopping center)"
    assert area_by_label(village.label) is village
    assert "not a neighborhood" in village.scope_note
    assert "unofficial" in ravenna.scope_note
    ring = village.geometry["coordinates"][0]
    assert all(point_in_geometry(lon, lat, ravenna.geometry) for lon, lat in ring)


def test_area_map_outlines_the_extent():
    district = area_by_name("University District")
    deck = area_deck(_permits(), outline=district.geometry)
    assert [layer.id for layer in deck.layers][0] == "extent"


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


def test_model_timeline_states_each_date_range():
    metrics = {"split": {"val_start": "2024-08-01", "test_start": "2025-08-01",
                         "last_forecast": "2026-07-01", "horizon_days": 90}}
    meta = {"data_as_of": "2026-10-10", "calibration": "isotonic"}
    rows = [(p.model, p.role, p.start, p.end) for p in model_timeline(metrics, meta, "2008-01-01")]
    assert rows == [
        ("Evaluation copy", "Training", "Jan 2008", "May 2024"),
        ("Evaluation copy", "Calibration", "Aug 2024", "May 2025"),
        ("Evaluation copy", "Held-out test", "Aug 2025", "Jul 2026"),
        ("Deployed model", "Training", "Jan 2008", "May 2025"),
        ("Deployed model", "Calibration", "Aug 2025", "Jul 2026"),
        ("Deployed model", "Scoring", "Oct 11, 2026", ""),
    ]
