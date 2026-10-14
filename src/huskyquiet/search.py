"""Resolve submitted locations and remember exactly which inputs produced a result.

A ``Submission`` stores the inputs, radius, and permit-data cutoff it was computed
for. A failed submission replaces the previous one, so an old result is never shown
under a new error. Pages compare the current widget values with the stored inputs
to tell the user when the results on screen no longer match the form.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass

from huskyquiet.geo import haversine_m, in_seattle
from huskyquiet.geocode import GeocodeError

RADII_M: dict[str, int] = {"250 m": 250, "500 m": 500, "1 km": 1000}


class InputError(ValueError):
    """The form inputs cannot be searched as entered."""


@dataclass(frozen=True)
class LocationInput:
    address: str = ""
    latitude: str = ""
    longitude: str = ""

    @property
    def uses_coordinates(self) -> bool:
        return bool(self.latitude.strip() or self.longitude.strip())

    def key(self) -> tuple[str, str, str]:
        return (" ".join(self.address.split()).casefold(), self.latitude.strip(), self.longitude.strip())


@dataclass(frozen=True)
class Pin:
    label: str
    latitude: float
    longitude: float
    source: str
    matched: str = ""


@dataclass(frozen=True)
class Submission:
    inputs: tuple[LocationInput, ...]
    radius_m: int
    data_as_of: str
    pins: tuple[Pin, ...] = ()
    error: str | None = None
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def matches(self, inputs: Sequence[LocationInput], radius_m: int) -> bool:
        return (
            tuple(entry.key() for entry in inputs) == tuple(entry.key() for entry in self.inputs)
            and int(radius_m) == self.radius_m
        )


def parse_coordinates(latitude_text: str, longitude_text: str) -> tuple[float, float]:
    lat_text = (latitude_text or "").strip().replace("\u2212", "-")
    lon_text = (longitude_text or "").strip().replace("\u2212", "-")
    if not lat_text or not lon_text:
        raise InputError("Enter both latitude and longitude, or clear both to look up the address.")
    try:
        latitude, longitude = float(lat_text), float(lon_text)
    except ValueError:
        raise InputError(
            "Latitude and longitude must be decimal numbers, such as 47.6615 and -122.3130."
        ) from None
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise InputError("Latitude and longitude must be finite decimal numbers.")
    if in_seattle(longitude, latitude):
        raise InputError("Latitude and longitude look swapped. Seattle is near 47.6 latitude, -122.3 longitude.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise InputError("Latitude must be between -90 and 90, and longitude between -180 and 180.")
    if not in_seattle(latitude, longitude):
        hint = " Seattle longitudes are negative." if in_seattle(latitude, -longitude) else ""
        raise InputError(f"{latitude:.5f}, {longitude:.5f} is outside Seattle, the only area HuskyQuiet covers.{hint}")
    return latitude, longitude


def locate(entry: LocationInput, geocoder) -> Pin:
    """Coordinates win when either coordinate field is filled; otherwise the address is looked up."""
    if entry.uses_coordinates:
        latitude, longitude = parse_coordinates(entry.latitude, entry.longitude)
        return Pin(f"{latitude:.5f}, {longitude:.5f}", latitude, longitude, "Entered coordinates")
    address = " ".join(entry.address.split())
    if not address:
        raise InputError("Enter a Seattle address, or open “Enter coordinates”.")
    try:
        hit = geocoder.geocode(address)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    return Pin(address, hit.latitude, hit.longitude, hit.source, matched=hit.display_name)


def submit(
    state: MutableMapping,
    key: str,
    inputs: Sequence[LocationInput],
    radius_m: int,
    data_as_of: str,
    geocoder,
    names: Sequence[str] | None = None,
) -> Submission:
    """Resolve every input and store the outcome under ``key``, replacing any earlier one."""
    entries = tuple(inputs)
    pins: list[Pin] = []
    error: str | None = None
    kind: str | None = None
    for index, entry in enumerate(entries):
        try:
            pins.append(locate(entry, geocoder))
        except InputError as exc:
            error, kind = str(exc), "input"
        except GeocodeError as exc:
            error, kind = str(exc), exc.kind
        except Exception as exc:
            error = (
                f"The address lookup failed unexpectedly ({exc.__class__.__name__}). "
                "Try again, or enter coordinates."
            )
            kind = getattr(exc, "kind", "unavailable")
        if error is not None:
            if names and len(entries) > 1:
                error = f"{names[index]}: {error}"
            break
    if error is None:
        outcome = Submission(entries, int(radius_m), data_as_of, tuple(pins))
    else:
        outcome = Submission(entries, int(radius_m), data_as_of, error=error, error_kind=kind)
    state[key] = outcome
    return outcome


def current(state: MutableMapping, key: str, data_as_of: str) -> Submission | None:
    """The stored submission, or ``None`` if there is none or the permit data has changed since."""
    outcome = state.get(key)
    if not isinstance(outcome, Submission):
        return None
    if outcome.data_as_of != data_as_of:
        state.pop(key, None)
        return None
    return outcome


def same_location(left: Pin, right: Pin, tolerance_m: float = 5.0) -> bool:
    return float(haversine_m(left.latitude, left.longitude, right.latitude, right.longitude)) <= tolerance_m
