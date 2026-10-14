import pytest

from huskyquiet.geocode import GeocodeResult, RateLimited
from huskyquiet.search import (
    InputError,
    LocationInput,
    Pin,
    current,
    locate,
    parse_coordinates,
    same_location,
    submit,
)


class _Geocoder:
    def __init__(self, error=None):
        self.error = error
        self.queries = []

    def geocode(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return GeocodeResult(query, "4500 UNIVERSITY WAY NE, SEATTLE, WA, 98105", 47.661325, -122.313045, "Test")


def test_valid_coordinates_parse_including_a_typographic_minus():
    assert parse_coordinates(" 47.6615 ", "\u2212122.313") == (47.6615, -122.313)


@pytest.mark.parametrize(
    ("lat", "lon", "fragment"),
    [
        ("47.66", "", "both latitude and longitude"),
        ("north", "-122.31", "decimal numbers"),
        ("nan", "-122.31", "finite"),
        ("-122.31", "47.66", "swapped"),
        ("95", "-122.31", "between -90 and 90"),
        ("40.71", "-74.00", "outside Seattle"),
        ("47.66", "122.31", "longitudes are negative"),
    ],
)
def test_bad_coordinates_are_rejected_with_a_specific_reason(lat, lon, fragment):
    with pytest.raises(InputError) as caught:
        parse_coordinates(lat, lon)
    assert fragment in str(caught.value)


def test_filled_coordinates_win_and_skip_geocoding():
    coder = _Geocoder()
    pin = locate(LocationInput("4500 University Way NE", "47.6600", "-122.3100"), coder)
    assert pin.source == "Entered coordinates"
    assert (pin.latitude, pin.longitude) == (47.66, -122.31)
    assert coder.queries == []


def test_blank_address_asks_for_an_address_or_coordinates():
    with pytest.raises(InputError) as caught:
        locate(LocationInput("   "), _Geocoder())
    assert "Enter coordinates" in str(caught.value)


def test_failed_submission_replaces_the_previous_result():
    state = {}
    good = submit(state, "r", [LocationInput("4500 University Way NE")], 500, "2026-10-10", _Geocoder())
    assert good.ok and state["r"].pins[0].latitude == 47.661325

    limited = _Geocoder(RateLimited("Limited. Try again in about 60 seconds.", retry_after_s=60))
    failed = submit(state, "r", [LocationInput("1200 NE 45th St")], 500, "2026-10-10", limited)
    assert not failed.ok
    assert state["r"] is failed
    assert state["r"].pins == ()
    assert state["r"].error_kind == "rate_limited"
    assert state["r"].inputs[0].address == "1200 NE 45th St"


def test_unexpected_lookup_errors_become_a_message_not_a_crash():
    outcome = submit({}, "r", [LocationInput("4500 University Way NE")], 500, "2026-10-10", _Geocoder(KeyError("x")))
    assert outcome.error_kind == "unavailable"
    assert "enter coordinates" in outcome.error


def test_invalid_coordinates_fail_the_submission_without_geocoding():
    state, coder = {}, _Geocoder()
    outcome = submit(state, "r", [LocationInput("", "47.66", "122.31")], 250, "2026-10-10", coder)
    assert outcome.error_kind == "input"
    assert coder.queries == []


def test_result_knows_when_inputs_or_radius_changed():
    state = {}
    first = LocationInput("4500 University Way NE")
    outcome = submit(state, "r", [first], 500, "2026-10-10", _Geocoder())
    assert outcome.matches([LocationInput(" 4500  university way ne ")], 500)
    assert not outcome.matches([LocationInput("1200 NE 45th St")], 500)
    assert not outcome.matches([first], 1000)
    assert not outcome.matches([LocationInput("4500 University Way NE", "47.66", "-122.31")], 500)


def test_result_from_older_permit_data_is_dropped():
    state = {}
    submit(state, "r", [LocationInput("4500 University Way NE")], 500, "2026-10-10", _Geocoder())
    assert current(state, "r", "2026-10-10") is not None
    assert current(state, "r", "2026-11-01") is None
    assert "r" not in state


def test_comparison_names_the_entry_that_failed():
    state = {}
    outcome = submit(
        state,
        "c",
        [LocationInput("4500 University Way NE"), LocationInput("", "1", "2")],
        500,
        "2026-10-10",
        _Geocoder(),
        names=("Location A", "Location B"),
    )
    assert outcome.error.startswith("Location B:")
    assert outcome.pins == ()


def test_same_location_tolerance():
    a = Pin("A", 47.661325, -122.313045, "x")
    assert same_location(a, Pin("B", 47.66133, -122.31305, "y"))
    assert not same_location(a, Pin("C", 47.6650, -122.3130, "y"))
