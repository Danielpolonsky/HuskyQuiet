import requests

from huskyquiet.geocode import USER_AGENT, GeocodeError, NominatimGeocoder


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def _geocoder(tmp_path, responses, sleeps=None):
    session = _Session(responses)
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    return NominatimGeocoder(
        cache_path=tmp_path / "geocode.json",
        session=session,
        sleeper=sleep if sleeps is not None else (lambda _s: None),
        clock=now,
    ), session


def test_geocode_caches_and_sets_a_user_agent(tmp_path):
    hit = _Response(200, [{"lat": "47.6612", "lon": "-122.3134", "display_name": "4500 University Way NE, Seattle"}])
    coder, session = _geocoder(tmp_path, [hit])
    first = coder.geocode("4500 University Way NE")
    second = coder.geocode("4500 university way ne")
    assert session.calls == 1
    assert first.latitude == second.latitude == 47.6612
    assert session.headers["User-Agent"] == USER_AGENT
    assert "Seattle" in first.query


def test_geocode_rejects_results_outside_seattle(tmp_path):
    coder, _session = _geocoder(tmp_path, [
        _Response(200, []),
        _Response(200, [{"lat": "40.0", "lon": "-122.3", "display_name": "Somewhere else"}]),
    ])
    try:
        coder.geocode("1 Main St")
    except GeocodeError as exc:
        assert "No Seattle location" in str(exc)
    else:
        raise AssertionError("expected GeocodeError")


def test_geocode_reports_rate_limits(tmp_path):
    coder, _session = _geocoder(tmp_path, [_Response(429, [])])
    try:
        coder.geocode("1 Main St, Seattle")
    except GeocodeError as exc:
        assert "rate-limiting" in str(exc)
    else:
        raise AssertionError("expected GeocodeError")


def test_geocode_reports_timeouts(tmp_path):
    class TimingOut:
        headers = {}

        def get(self, url, params=None, timeout=None):
            raise requests.Timeout("slow")

    coder = NominatimGeocoder(cache_path=tmp_path / "geocode.json", session=TimingOut(), sleeper=lambda _s: None)
    try:
        coder.geocode("1 Main St, Seattle")
    except GeocodeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected GeocodeError")
