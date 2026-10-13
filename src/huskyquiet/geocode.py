"""Geocode a Seattle address with the public Nominatim service.

The public instance asks for at most one request per second, an identifying User-Agent,
and cached results (https://operations.osmfoundation.org/policies/nominatim/). This
client does those three things. A failure raises ``GeocodeError``; coordinates are never
invented. Callers can skip the network by passing latitude and longitude directly.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from huskyquiet.config import SEATTLE_BBOX, get_paths
from huskyquiet.geo import in_seattle

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "HuskyQuiet/0.2 (educational permit-issuance demo; local use)"
# Nominatim viewbox is left, top, right, bottom.
_VIEWBOX = (
    f"{SEATTLE_BBOX['lon_min']},{SEATTLE_BBOX['lat_max']},"
    f"{SEATTLE_BBOX['lon_max']},{SEATTLE_BBOX['lat_min']}"
)


class GeocodeError(RuntimeError):
    """The address could not be resolved to a point inside Seattle."""


@dataclass(frozen=True)
class GeocodeResult:
    query: str
    display_name: str
    latitude: float
    longitude: float
    source: str = "nominatim"


class NominatimGeocoder:
    def __init__(
        self,
        cache_path: Path | None = None,
        session: requests.Session | None = None,
        min_interval_s: float = 1.1,
        timeout_s: float = 15.0,
        sleeper=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.cache_path = cache_path or (get_paths().data / "cache" / "geocode.json")
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._sleep = sleeper
        self._clock = clock
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._cache = self._read_cache()

    def geocode(self, query: str) -> GeocodeResult:
        normalized = " ".join(query.split())
        if not normalized:
            raise GeocodeError("Enter an address.")
        if "seattle" not in normalized.lower():
            normalized = f"{normalized}, Seattle, WA"
        key = normalized.casefold()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return GeocodeResult(query=normalized, **cached)

        hits = self._search(normalized, bounded=True)
        if not hits:
            hits = self._search(normalized, bounded=False)
        match = next((hit for hit in hits if _inside_seattle(hit)), None)
        if match is None:
            raise GeocodeError(
                f"No Seattle location found for {normalized!r}. "
                "Try a fuller street address, or enter coordinates."
            )
        result = GeocodeResult(
            query=normalized,
            display_name=str(match.get("display_name") or normalized),
            latitude=float(match["lat"]),
            longitude=float(match["lon"]),
        )
        with self._lock:
            self._cache[key] = {
                "display_name": result.display_name,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "source": result.source,
            }
            self._write_cache()
        return result

    def _search(self, query: str, bounded: bool) -> list[dict]:
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 5,
            "addressdetails": 0,
            "countrycodes": "us",
        }
        if bounded:
            params["viewbox"] = _VIEWBOX
            params["bounded"] = 1
        self._respect_rate_limit()
        try:
            response = self.session.get(NOMINATIM_URL, params=params, timeout=self.timeout_s)
        except requests.Timeout as exc:
            raise GeocodeError("The geocoder timed out. Enter coordinates, or try again shortly.") from exc
        except requests.RequestException as exc:
            raise GeocodeError(f"The geocoder could not be reached ({exc.__class__.__name__}).") from exc
        if response.status_code == 429:
            raise GeocodeError("The geocoder is rate-limiting requests. Wait a moment, or enter coordinates.")
        if response.status_code >= 400:
            raise GeocodeError(f"The geocoder returned HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeocodeError("The geocoder returned a response that was not JSON.") from exc
        if not isinstance(payload, list):
            raise GeocodeError("The geocoder returned an unexpected response.")
        return payload

    def _respect_rate_limit(self) -> None:
        with self._lock:
            wait = self.min_interval_s - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = self._clock()

    def _read_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._cache), encoding="utf-8")
        temporary.replace(self.cache_path)


def _inside_seattle(hit: dict) -> bool:
    try:
        return in_seattle(float(hit["lat"]), float(hit["lon"]))
    except (KeyError, TypeError, ValueError):
        return False
