"""Turn a Seattle street address into coordinates with free public geocoders.

Providers are tried in the order given by ``HUSKYQUIET_GEOCODERS`` (default
``census,nominatim``):

* ``census``: U.S. Census Bureau geocoder. No key. Matches street addresses, not
  place names.
* ``nominatim``: OpenStreetMap Nominatim. No key. The public server allows at most one
  request per second per application, asks for an identifying User-Agent and for
  cached results (https://operations.osmfoundation.org/policies/nominatim/), and
  throttles busy shared networks. On Streamlit Community Cloud it can answer HTTP 429
  to the first request a session makes.

Requests to one provider share a process-wide limiter, so every session served by
the same Streamlit process waits its turn and honors ``Retry-After``. Retries are
bounded. A failure raises a ``GeocodeError`` subclass naming what went wrong, and
coordinates are never guessed.
"""

from __future__ import annotations

import email.utils
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol

import requests

from huskyquiet import __version__
from huskyquiet.config import SEATTLE_BBOX, get_paths
from huskyquiet.geo import in_seattle

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REPO_URL = "https://github.com/Danielpolonsky/HuskyQuiet"
USER_AGENT = f"HuskyQuiet/{__version__} (+{REPO_URL})"
DEFAULT_PROVIDERS = "census,nominatim"
# The Census geocoder rejects addresses longer than 100 characters.
MAX_QUERY_CHARS = 100
_MAX_RETRY_AFTER_S = 3600.0
# Cache files written before 0.3 stored provider names instead of labels.
_LEGACY_SOURCES = {"nominatim": "OpenStreetMap Nominatim", "census": "U.S. Census Geocoder"}
_MAX_CACHE_ENTRIES = 5000
# Nominatim viewbox is left, top, right, bottom.
_VIEWBOX = (
    f"{SEATTLE_BBOX['lon_min']},{SEATTLE_BBOX['lat_max']},"
    f"{SEATTLE_BBOX['lon_max']},{SEATTLE_BBOX['lat_min']}"
)


class GeocodeError(RuntimeError):
    """The address could not be resolved to a point inside Seattle."""

    kind: ClassVar[str] = "error"

    def __init__(self, message: str = "", *, detail: str = "", retry_after_s: float | None = None) -> None:
        super().__init__(message or detail)
        self.detail = detail or message
        self.retry_after_s = retry_after_s


class AddressNotFound(GeocodeError):
    kind = "not_found"


class RateLimited(GeocodeError):
    kind = "rate_limited"


class ServiceTimeout(GeocodeError):
    kind = "timeout"


class ServiceUnavailable(GeocodeError):
    kind = "unavailable"


class InvalidResponse(GeocodeError):
    kind = "invalid_response"


@dataclass(frozen=True)
class Candidate:
    display_name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class GeocodeResult:
    query: str
    display_name: str
    latitude: float
    longitude: float
    source: str


class Provider(Protocol):
    name: str
    label: str
    url: str
    min_interval_s: float

    def params(self, query: str) -> dict[str, str | int]: ...

    def candidates(self, payload: Any) -> list[Candidate]: ...


@dataclass(frozen=True)
class CensusProvider:
    url: str = CENSUS_URL
    min_interval_s: float = 1.0
    name: str = "census"
    label: str = "U.S. Census Geocoder"

    def params(self, query: str) -> dict[str, str | int]:
        return {"address": query, "benchmark": "Public_AR_Current", "format": "json"}

    def candidates(self, payload: Any) -> list[Candidate]:
        rows = payload["result"]["addressMatches"]
        if not isinstance(rows, list):
            raise ValueError("addressMatches is not a list")
        return [
            Candidate(
                str(row.get("matchedAddress") or ""),
                float(row["coordinates"]["y"]),
                float(row["coordinates"]["x"]),
            )
            for row in rows
        ]


@dataclass(frozen=True)
class NominatimProvider:
    url: str = NOMINATIM_URL
    email: str | None = None
    min_interval_s: float = 1.1
    name: str = "nominatim"
    label: str = "OpenStreetMap Nominatim"

    def params(self, query: str) -> dict[str, str | int]:
        params: dict[str, str | int] = {
            "q": query,
            "format": "jsonv2",
            "limit": 5,
            "addressdetails": 0,
            "countrycodes": "us",
            "viewbox": _VIEWBOX,
            "bounded": 1,
        }
        if self.email:
            params["email"] = self.email
        return params

    def candidates(self, payload: Any) -> list[Candidate]:
        if isinstance(payload, dict) and "error" in payload:
            raise ValueError(str(payload["error"]))
        if not isinstance(payload, list):
            raise ValueError("expected a JSON list")
        return [
            Candidate(str(row.get("display_name") or ""), float(row["lat"]), float(row["lon"]))
            for row in payload
        ]


def providers_from_settings(get: Callable[[str], str | None]) -> tuple[Provider, ...]:
    """Build the provider chain from settings such as environment variables or Streamlit secrets."""
    names = [part.strip().lower() for part in (get("HUSKYQUIET_GEOCODERS") or DEFAULT_PROVIDERS).split(",")]
    nominatim_url = (get("HUSKYQUIET_NOMINATIM_URL") or NOMINATIM_URL).strip()
    if not nominatim_url.startswith("https://"):
        raise ValueError("HUSKYQUIET_NOMINATIM_URL must be an https:// URL.")
    email_setting = (get("HUSKYQUIET_GEOCODER_EMAIL") or "").strip() or None
    providers: list[Provider] = []
    for name in names:
        if not name:
            continue
        if name == "census":
            providers.append(CensusProvider())
        elif name == "nominatim":
            providers.append(NominatimProvider(url=nominatim_url, email=email_setting))
        else:
            raise ValueError(f"Unknown geocoder {name!r} in HUSKYQUIET_GEOCODERS. Use census, nominatim, or both.")
    if not providers:
        raise ValueError("HUSKYQUIET_GEOCODERS does not name any geocoder.")
    return tuple(providers)


def normalize_address(text: str) -> str:
    """Collapse whitespace, drop trailing punctuation, and anchor the query to Seattle."""
    cleaned = " ".join((text or "").replace("\u00a0", " ").split()).strip(" ,.;")
    if not cleaned:
        raise ValueError("Enter an address.")
    if "seattle" not in cleaned.casefold():
        cleaned = f"{cleaned}, Seattle, WA"
    if len(cleaned) > MAX_QUERY_CHARS:
        raise ValueError(f"Shorten the address to {MAX_QUERY_CHARS} characters or fewer.")
    return cleaned


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP date), capped at an hour."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return min(float(text), _MAX_RETRY_AFTER_S)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - (now or datetime.now(UTC))).total_seconds()
    return max(0.0, min(delta, _MAX_RETRY_AFTER_S))


class LimiterBusy(Exception):
    def __init__(self, wait_s: float) -> None:
        super().__init__(f"next request slot in {wait_s:.1f} s")
        self.wait_s = wait_s


class RateLimiter:
    """Spaces requests to one provider for every session in this process."""

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.Lock()
        self._next_slot = float("-inf")
        self._blocked_until = float("-inf")

    def acquire(self, max_wait_s: float) -> None:
        """Reserve the next request slot, sleeping until it opens. Refuse waits above ``max_wait_s``."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next_slot, self._blocked_until)
            wait = start - now
            if wait > max_wait_s:
                raise LimiterBusy(wait)
            self._next_slot = start + self.min_interval_s
        if wait > 0:
            self._sleep(wait)

    def back_off(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + seconds)


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(name: str, min_interval_s: float) -> RateLimiter:
    """The single limiter for ``name`` in this process."""
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(name)
        if limiter is None:
            limiter = _LIMITERS[name] = RateLimiter(min_interval_s)
        return limiter


class Geocoder:
    """Cached, rate-limited lookups over an ordered list of providers."""

    def __init__(
        self,
        providers: Sequence[Provider] | None = None,
        *,
        cache_path: Path | None = None,
        session: requests.Session | None = None,
        limiters: dict[str, RateLimiter] | None = None,
        timeout_s: float = 8.0,
        max_attempts: int = 2,
        max_queue_wait_s: float = 5.0,
        max_retry_wait_s: float = 3.0,
        default_backoff_s: float = 60.0,
        retry_backoff_s: float = 1.0,
        not_found_ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = tuple(providers) if providers else (CensusProvider(), NominatimProvider())
        self.cache_path = cache_path or (get_paths().data / "cache" / "geocode.json")
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._limiters = limiters
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.max_queue_wait_s = max_queue_wait_s
        self.max_retry_wait_s = max_retry_wait_s
        self.default_backoff_s = default_backoff_s
        self.retry_backoff_s = retry_backoff_s
        self.not_found_ttl_s = not_found_ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._flights: dict[str, threading.Lock] = {}
        self._not_found: dict[str, float] = {}
        self._cache = self._read_cache()
        self.network_requests = 0

    def geocode(self, query: str) -> GeocodeResult:
        normalized = normalize_address(query)
        key = normalized.casefold()
        cached = self._cached(key, normalized)
        if cached is not None:
            return cached
        # One lookup per address at a time: concurrent sessions asking for the same
        # address wait here and then read the cached answer.
        with self._flight(key):
            cached = self._cached(key, normalized)
            if cached is not None:
                return cached
            with self._lock:
                expires = self._not_found.get(key)
            if expires is not None and expires > self._clock():
                raise AddressNotFound(_not_found_message(normalized), detail="no Seattle match (recent lookup)")
            return self._lookup(normalized, key)

    def _lookup(self, normalized: str, key: str) -> GeocodeResult:
        attempts: list[tuple[Provider, GeocodeError]] = []
        for provider in self.providers:
            try:
                candidates = self._request(provider, normalized)
            except GeocodeError as exc:
                attempts.append((provider, exc))
                continue
            match = next((c for c in candidates if in_seattle(c.latitude, c.longitude)), None)
            if match is None:
                attempts.append((provider, AddressNotFound(detail="no Seattle match")))
                continue
            result = GeocodeResult(
                query=normalized,
                display_name=match.display_name or normalized,
                latitude=match.latitude,
                longitude=match.longitude,
                source=provider.label,
            )
            self._store(key, result)
            return result
        raise self._failure(normalized, key, attempts)

    def _request(self, provider: Provider, query: str) -> list[Candidate]:
        limiter = self._limiter(provider)
        last: GeocodeError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                limiter.acquire(self.max_queue_wait_s)
            except LimiterBusy as busy:
                raise RateLimited(
                    detail=f"paused, next request allowed in {busy.wait_s:.0f} s",
                    retry_after_s=busy.wait_s,
                ) from None
            self.network_requests += 1
            try:
                response = self.session.get(provider.url, params=provider.params(query), timeout=self.timeout_s)
            except requests.Timeout:
                raise ServiceTimeout(detail=f"no answer within {self.timeout_s:.0f} s") from None
            except requests.RequestException as exc:
                last = ServiceUnavailable(detail=f"could not connect ({exc.__class__.__name__})")
                limiter.back_off(self.retry_backoff_s)
                continue
            status = int(response.status_code)
            retry_after = parse_retry_after((getattr(response, "headers", None) or {}).get("Retry-After"))
            if status == 429:
                wait = retry_after if retry_after is not None else self.default_backoff_s
                limiter.back_off(wait)
                detail = "HTTP 429" + (f", asked to wait {retry_after:.0f} s" if retry_after is not None else "")
                last = RateLimited(detail=detail, retry_after_s=wait)
                if retry_after is not None and retry_after <= self.max_retry_wait_s and attempt < self.max_attempts:
                    continue
                raise last
            if status == 403:
                raise ServiceUnavailable(detail="HTTP 403, request refused")
            if status >= 500:
                wait = retry_after if retry_after is not None else self.retry_backoff_s
                limiter.back_off(wait)
                last = ServiceUnavailable(detail=f"HTTP {status}", retry_after_s=retry_after)
                if wait <= self.max_retry_wait_s and attempt < self.max_attempts:
                    continue
                raise last
            if status >= 400:
                raise InvalidResponse(detail=f"HTTP {status}")
            try:
                payload = response.json()
            except ValueError:
                raise InvalidResponse(detail="response was not JSON") from None
            try:
                return provider.candidates(payload)
            except (KeyError, TypeError, ValueError):
                raise InvalidResponse(detail="unexpected response layout") from None
        assert last is not None
        raise last

    def _failure(self, normalized: str, key: str, attempts: list[tuple[Provider, GeocodeError]]) -> GeocodeError:
        tried = "; ".join(f"{provider.label}: {error.detail}" for provider, error in attempts)
        service = [error for _, error in attempts if not isinstance(error, AddressNotFound)]
        if not service:
            with self._lock:
                self._not_found[key] = self._clock() + self.not_found_ttl_s
            return AddressNotFound(f"{_not_found_message(normalized)} Tried {tried}.", detail=tried)
        first = service[0]
        waits = [error.retry_after_s for error in service if error.retry_after_s is not None]
        message = _service_message(first, max(waits) if waits else None)
        return type(first)(f"{message} Tried {tried}.", detail=tried, retry_after_s=first.retry_after_s)

    def _limiter(self, provider: Provider) -> RateLimiter:
        if self._limiters is not None:
            limiter = self._limiters.get(provider.name)
            if limiter is None:
                limiter = self._limiters[provider.name] = RateLimiter(provider.min_interval_s)
            return limiter
        return shared_limiter(provider.name, provider.min_interval_s)

    def _flight(self, key: str) -> threading.Lock:
        with self._lock:
            lock = self._flights.get(key)
            if lock is None:
                lock = self._flights[key] = threading.Lock()
            return lock

    def _cached(self, key: str, normalized: str) -> GeocodeResult | None:
        with self._lock:
            entry = self._cache.get(key)
        if entry is None:
            return None
        try:
            return GeocodeResult(
                query=normalized,
                display_name=str(entry["display_name"]),
                latitude=float(entry["latitude"]),
                longitude=float(entry["longitude"]),
                source=_LEGACY_SOURCES.get(str(entry.get("source")), str(entry.get("source") or "Saved lookup")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _store(self, key: str, result: GeocodeResult) -> None:
        with self._lock:
            self._cache[key] = {
                "display_name": result.display_name,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "source": result.source,
            }
            while len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
            self._not_found.pop(key, None)
            snapshot = dict(self._cache)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(snapshot), encoding="utf-8")
            temporary.replace(self.cache_path)
        except OSError:
            # The in-memory cache still works when the disk is read-only.
            pass

    def _read_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}


def _not_found_message(normalized: str) -> str:
    return (
        f"No Seattle match for “{normalized}”. Check the house number and street name, "
        "or enter coordinates."
    )


def _service_message(error: GeocodeError, wait_s: float | None) -> str:
    if isinstance(error, RateLimited):
        when = f" in about {max(1, round(wait_s))} seconds" if wait_s else " shortly"
        return f"The address lookup service is limiting requests. Try again{when}, or enter coordinates."
    if isinstance(error, ServiceTimeout):
        return "The address lookup service did not answer in time. Try again, or enter coordinates."
    if isinstance(error, InvalidResponse):
        return "The address lookup service sent a response HuskyQuiet could not read. Enter coordinates instead."
    return "The address lookup service is unavailable right now. Try again later, or enter coordinates."
