import threading
import time
from datetime import UTC, datetime

import pytest
import requests

from huskyquiet.geocode import (
    USER_AGENT,
    AddressNotFound,
    CensusProvider,
    Geocoder,
    InvalidResponse,
    LimiterBusy,
    NominatimProvider,
    RateLimited,
    RateLimiter,
    ServiceTimeout,
    ServiceUnavailable,
    normalize_address,
    parse_retry_after,
    providers_from_settings,
    shared_limiter,
)

_NOT_JSON = object()


class _Response:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is _NOT_JSON:
            raise ValueError("not json")
        return self._payload


class _Session:
    """Replays queued responses or exceptions and records every request."""

    def __init__(self, responses, delay_s=0.0):
        self.responses = list(responses)
        self.requests = []
        self.headers = {}
        self.delay_s = delay_s
        self._lock = threading.Lock()

    def get(self, url, params=None, timeout=None):
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            self.requests.append((url, dict(params or {}), timeout))
            item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def census(lat=47.661325, lon=-122.313045, address="4500 UNIVERSITY WAY NE, SEATTLE, WA, 98105"):
    matches = [] if lat is None else [{"matchedAddress": address, "coordinates": {"x": lon, "y": lat}}]
    return _Response(200, {"result": {"addressMatches": matches}})


def nominatim(lat="47.6613", lon="-122.3130", name="4500, University Way NE, Seattle"):
    return _Response(200, [] if lat is None else [{"lat": lat, "lon": lon, "display_name": name}])


def build(tmp_path, responses, providers=None, clock=None, **kwargs):
    providers = providers or (CensusProvider(),)
    clock = clock or _Clock()
    session = _Session(responses)
    limiters = {p.name: RateLimiter(p.min_interval_s, clock=clock, sleeper=clock.sleep) for p in providers}
    coder = Geocoder(
        providers,
        cache_path=tmp_path / "geocode.json",
        session=session,
        limiters=limiters,
        clock=clock,
        **kwargs,
    )
    return coder, session, clock


def test_success_is_cached_by_normalized_address_and_identifies_the_app(tmp_path):
    coder, session, _clock = build(tmp_path, [census()])
    first = coder.geocode("4500 University Way NE")
    second = coder.geocode("  4500   university way ne,  ")
    assert len(session.requests) == 1
    assert session.headers["User-Agent"] == USER_AGENT
    assert "github.com" in USER_AGENT
    assert session.requests[0][1]["address"] == "4500 University Way NE, Seattle, WA"
    assert (first.latitude, first.longitude) == (second.latitude, second.longitude) == (47.661325, -122.313045)
    assert first.source == "U.S. Census Geocoder"

    reopened, reopened_session, _ = build(tmp_path, [])
    assert reopened.geocode("4500 University Way NE").latitude == 47.661325
    assert reopened_session.requests == []


def test_concurrent_sessions_asking_for_one_address_make_one_request(tmp_path):
    session = _Session([census()], delay_s=0.2)
    coder = Geocoder(
        (CensusProvider(),),
        cache_path=tmp_path / "geocode.json",
        session=session,
        limiters={"census": RateLimiter(0.0)},
    )
    results, errors = [], []

    def lookup():
        try:
            results.append(coder.geocode("4500 University Way NE"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=lookup) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 6
    assert len(session.requests) == 1


def test_census_miss_falls_back_to_nominatim_with_a_bounded_query(tmp_path):
    coder, session, _ = build(
        tmp_path,
        [census(lat=None), nominatim()],
        providers=(CensusProvider(), NominatimProvider(email="maintainer@example.org")),
    )
    hit = coder.geocode("4500 University Way NE")
    assert hit.source == "OpenStreetMap Nominatim"
    assert len(session.requests) == 2
    params = session.requests[1][1]
    assert params["bounded"] == 1 and "viewbox" in params
    assert params["email"] == "maintainer@example.org"


def test_not_found_is_distinct_and_briefly_remembered(tmp_path):
    coder, session, clock = build(
        tmp_path, [census(lat=None), nominatim(lat=None)], providers=(CensusProvider(), NominatimProvider())
    )
    with pytest.raises(AddressNotFound) as caught:
        coder.geocode("99999 Nowhere Ave")
    assert caught.value.kind == "not_found"
    assert "No Seattle match" in str(caught.value)
    with pytest.raises(AddressNotFound):
        coder.geocode("99999 Nowhere Ave")
    assert len(session.requests) == 2
    clock.now += 601
    session.responses += [census()]
    assert coder.geocode("99999 Nowhere Ave").latitude == 47.661325


def test_results_outside_seattle_are_not_accepted(tmp_path):
    coder, _session, _ = build(tmp_path, [census(lat=40.0, lon=-122.3)])
    with pytest.raises(AddressNotFound):
        coder.geocode("1 Main St")


def test_short_retry_after_is_honored_then_retried(tmp_path):
    coder, session, clock = build(tmp_path, [_Response(429, [], {"Retry-After": "2"}), census()])
    hit = coder.geocode("4500 University Way NE")
    assert hit.latitude == 47.661325
    assert len(session.requests) == 2
    assert 2.0 in clock.sleeps


def test_long_retry_after_stops_requests_until_it_passes(tmp_path):
    coder, session, clock = build(tmp_path, [_Response(429, [], {"Retry-After": "120"})])
    with pytest.raises(RateLimited) as caught:
        coder.geocode("4500 University Way NE")
    assert caught.value.kind == "rate_limited"
    assert caught.value.retry_after_s == 120
    assert "about 120 seconds" in str(caught.value)
    assert "HTTP 429" in str(caught.value)

    with pytest.raises(RateLimited):
        coder.geocode("1200 NE 45th St")
    assert len(session.requests) == 1

    clock.now += 121
    session.responses.append(census())
    assert coder.geocode("4500 University Way NE").latitude == 47.661325
    assert len(session.requests) == 2


def test_rate_limit_without_retry_after_backs_off_by_default(tmp_path):
    coder, session, _ = build(tmp_path, [_Response(429, [])], default_backoff_s=60)
    with pytest.raises(RateLimited) as caught:
        coder.geocode("4500 University Way NE")
    assert caught.value.retry_after_s == 60
    assert len(session.requests) == 1


def test_timeout_is_reported_without_retrying(tmp_path):
    coder, session, _ = build(tmp_path, [requests.Timeout("slow")])
    with pytest.raises(ServiceTimeout) as caught:
        coder.geocode("4500 University Way NE")
    assert caught.value.kind == "timeout"
    assert "did not answer in time" in str(caught.value)
    assert len(session.requests) == 1


def test_connection_error_recovers_on_the_bounded_retry(tmp_path):
    coder, session, _ = build(tmp_path, [requests.ConnectionError("reset"), census()])
    assert coder.geocode("4500 University Way NE").latitude == 47.661325
    assert len(session.requests) == 2


def test_repeated_server_errors_are_unavailable_after_two_attempts(tmp_path):
    coder, session, _ = build(tmp_path, [_Response(503), _Response(503)])
    with pytest.raises(ServiceUnavailable) as caught:
        coder.geocode("4500 University Way NE")
    assert caught.value.kind == "unavailable"
    assert len(session.requests) == 2


def test_forbidden_and_unreadable_responses_are_classified(tmp_path):
    coder, _, _ = build(tmp_path, [_Response(403)])
    with pytest.raises(ServiceUnavailable):
        coder.geocode("4500 University Way NE")
    coder, _, _ = build(tmp_path, [_Response(200, _NOT_JSON)])
    with pytest.raises(InvalidResponse):
        coder.geocode("4500 University Way NE")
    coder, _, _ = build(tmp_path, [_Response(200, {"unexpected": True})])
    with pytest.raises(InvalidResponse) as caught:
        coder.geocode("4500 University Way NE")
    assert caught.value.kind == "invalid_response"


def test_service_failure_on_one_provider_and_miss_on_the_other_reports_the_service(tmp_path):
    coder, _, _ = build(
        tmp_path,
        [census(lat=None), _Response(429, [], {"Retry-After": "30"})],
        providers=(CensusProvider(), NominatimProvider()),
    )
    with pytest.raises(RateLimited) as caught:
        coder.geocode("Red Square")
    message = str(caught.value)
    assert "U.S. Census Geocoder: no Seattle match" in message
    assert "OpenStreetMap Nominatim: HTTP 429" in message


def test_limiter_spaces_requests_and_refuses_long_waits():
    clock = _Clock()
    limiter = RateLimiter(1.0, clock=clock, sleeper=clock.sleep)
    for _ in range(3):
        limiter.acquire(max_wait_s=5)
    assert clock.sleeps == [1.0, 1.0]
    limiter.back_off(30)
    with pytest.raises(LimiterBusy):
        limiter.acquire(max_wait_s=5)


def test_every_geocoder_in_the_process_shares_one_limiter_per_provider(tmp_path):
    assert shared_limiter("census", 1.0) is shared_limiter("census", 1.0)
    first = Geocoder(cache_path=tmp_path / "a.json", session=_Session([]))
    second = Geocoder(cache_path=tmp_path / "b.json", session=_Session([]))
    provider = CensusProvider()
    assert first._limiter(provider) is second._limiter(provider)


def test_retry_after_parsing():
    assert parse_retry_after("5") == 5
    assert parse_retry_after("999999") == 3600
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("Wed, 14 Oct 2026 12:00:30 GMT", now=now) == 30


def test_provider_settings():
    names = [p.name for p in providers_from_settings(lambda _name: None)]
    assert names == ["census", "nominatim"]
    only = providers_from_settings({"HUSKYQUIET_GEOCODERS": "nominatim"}.get)
    assert [p.name for p in only] == ["nominatim"]
    with pytest.raises(ValueError):
        providers_from_settings({"HUSKYQUIET_GEOCODERS": "google"}.get)
    with pytest.raises(ValueError):
        providers_from_settings({"HUSKYQUIET_NOMINATIM_URL": "http://example.org/search"}.get)


def test_normalize_address():
    assert normalize_address(" 4500  University Way NE. ") == "4500 University Way NE, Seattle, WA"
    assert normalize_address("1200 NE 45th St, Seattle") == "1200 NE 45th St, Seattle"
    with pytest.raises(ValueError):
        normalize_address("   ")
    with pytest.raises(ValueError):
        normalize_address("1" * 120)
