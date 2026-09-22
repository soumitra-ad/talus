"""Download manager tests: streaming, limits, retries, redirects, verification (offline)."""

from __future__ import annotations

import hashlib
import time

import httpx
import pytest

from terrain_agent.acquisition.download import DownloadManager
from terrain_agent.acquisition.errors import (
    DownloadChecksumError,
    DownloadError,
    DownloadIncompleteError,
    DownloadTimeoutError,
    DownloadTooLargeError,
    HostPolicyError,
)
from terrain_agent.acquisition.http import HttpSettings
from terrain_agent.acquisition.net_policy import NASA_DOWNLOAD_HOSTS, HostPolicy
from tests.acquisition_fixtures import MockNasa, public_resolver, stream_response

URL = "https://pds-geosciences.wustl.edu/lro/data/ldem_test.img"
BODY = bytes(range(256)) * 10  # 2560 bytes


def make_manager(mock: MockNasa, *, max_bytes=10_000_000, retries=2, clock=None, resolver=public_resolver, total=900.0):
    sleeps: list[float] = []
    manager = DownloadManager(
        policy=HostPolicy(NASA_DOWNLOAD_HOSTS, resolver),
        settings=HttpSettings(max_retries=retries, backoff_base_s=1.0, total_timeout_s=total),
        max_bytes=max_bytes,
        client_factory=lambda: httpx.Client(transport=mock.transport),
        sleep=sleeps.append,
        clock=clock or time.monotonic,
        chunk_size=512,
    )
    return manager, sleeps


def script(mock: MockNasa, url: str, *items):
    """The next requests for *url* get these responses or exceptions, then normal behaviour resumes."""
    queue = list(items)

    def hook(_request):
        if not queue:
            return None
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    mock.data_script.setdefault(url, []).append(hook)


# ---------------------------------------------------------------------------
# Success and verification
# ---------------------------------------------------------------------------


def test_download_writes_verifies_and_hashes(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    manager, _ = make_manager(mock_nasa)
    result = manager.fetch(URL, tmp_path, "ldem_test.img")

    assert result.path == (tmp_path / "ldem_test.img").resolve()
    assert result.path.read_bytes() == BODY
    assert result.size_bytes == len(BODY)
    assert result.sha256 == hashlib.sha256(BODY).hexdigest()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ldem_test.img"]


def test_request_headers_are_minimal_and_carry_no_credentials(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    manager, _ = make_manager(mock_nasa)
    manager.fetch(URL, tmp_path, "ldem_test.img")
    (request,) = mock_nasa.requests
    assert request.method == "GET" and request.headers["accept-encoding"] == "identity"
    assert not any(k in request.headers for k in ("authorization", "cookie", "x-api-key"))


def test_metadata_size_is_checked_within_the_kilobyte_rounding(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY  # 2560 bytes is listed by ODE as 3 kB
    manager, _ = make_manager(mock_nasa)
    manager.fetch(URL, tmp_path, "a.img", expected_max_bytes=3000)  # inside (2000, 3000]
    for wrong in (2000, 2559, 4000, 9000):
        with pytest.raises(DownloadIncompleteError):
            manager.fetch(URL, tmp_path, "b.img", expected_max_bytes=wrong)
        assert not (tmp_path / "b.img").exists()


def test_metadata_mismatch_is_not_retried(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    manager, sleeps = make_manager(mock_nasa)
    with pytest.raises(DownloadIncompleteError):
        manager.fetch(URL, tmp_path, "a.img", expected_max_bytes=9000)
    assert len(mock_nasa.requests) == 1 and sleeps == []


def test_a_checksum_is_verified_when_supplied(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    manager, _ = make_manager(mock_nasa)
    good = hashlib.sha256(BODY).hexdigest()
    assert manager.fetch(URL, tmp_path, "ok.img", expected_sha256=good.upper()).sha256 == good
    with pytest.raises(DownloadChecksumError):
        manager.fetch(URL, tmp_path, "bad.img", expected_sha256="0" * 64)
    assert not (tmp_path / "bad.img").exists()


def test_an_empty_file_is_rejected(mock_nasa, tmp_path):
    mock_nasa.files[URL] = b""
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(DownloadIncompleteError):
        manager.fetch(URL, tmp_path, "empty.img")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Size limits and streaming
# ---------------------------------------------------------------------------


def test_a_declared_size_over_the_limit_is_refused_before_any_body_is_read(mock_nasa, tmp_path):
    mock_nasa.files[URL] = b"x" * 100_000
    manager, sleeps = make_manager(mock_nasa, max_bytes=50_000)
    with pytest.raises(DownloadTooLargeError):
        manager.fetch(URL, tmp_path, "big.img")
    assert mock_nasa.streams[URL].bytes_served == 0
    assert list(tmp_path.iterdir()) == [] and sleeps == []


def test_a_stream_that_lies_about_its_size_is_cut_off_early(mock_nasa, tmp_path):
    body = b"x" * 5_000_000
    script(mock_nasa, URL, stream_response(body, 200, {"content-length": "1000"}, chunk=1024))
    manager, _ = make_manager(mock_nasa, max_bytes=100_000)
    stream_holder = {}

    original = mock_nasa.data_script[URL][0]

    def capture(request):
        response = original(request)
        stream_holder["stream"] = response.stream
        return response

    mock_nasa.data_script[URL][0] = capture
    with pytest.raises(DownloadTooLargeError):
        manager.fetch(URL, tmp_path, "liar.img")
    assert stream_holder["stream"].bytes_served < 200_000  # nowhere near the 5 MB it offered
    assert list(tmp_path.iterdir()) == []


def test_a_per_call_limit_can_only_lower_the_configured_limit(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    manager, _ = make_manager(mock_nasa, max_bytes=10_000)
    with pytest.raises(DownloadTooLargeError):
        manager.fetch(URL, tmp_path, "a.img", max_bytes=1000)
    assert manager.fetch(URL, tmp_path, "b.img", max_bytes=10**9).size_bytes == len(BODY)  # cannot raise it


# ---------------------------------------------------------------------------
# Retries, timeouts
# ---------------------------------------------------------------------------


def test_transient_errors_are_retried_then_succeed(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    script(mock_nasa, URL, httpx.Response(500), httpx.ReadTimeout("slow"))
    manager, sleeps = make_manager(mock_nasa, retries=3)
    assert manager.fetch(URL, tmp_path, "a.img").size_bytes == len(BODY)
    assert sleeps == [1.0, 2.0] and len(mock_nasa.requests) == 3


def test_retries_are_bounded_and_backoff_is_exponential(mock_nasa, tmp_path):
    script(mock_nasa, URL, *[httpx.Response(503)] * 10)
    manager, sleeps = make_manager(mock_nasa, retries=3)
    with pytest.raises(DownloadError, match="4 attempts"):
        manager.fetch(URL, tmp_path, "a.img")
    assert len(mock_nasa.requests) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_permanent_errors_are_not_retried(mock_nasa, tmp_path):
    manager, sleeps = make_manager(mock_nasa)
    with pytest.raises(DownloadError, match="404"):
        manager.fetch(URL, tmp_path, "missing.img")
    assert len(mock_nasa.requests) == 1 and sleeps == []


def test_a_truncated_transfer_is_retried_and_then_reported_as_incomplete(mock_nasa, tmp_path):
    truncated = lambda: stream_response(BODY[:1000], 200, {"content-length": str(len(BODY))})  # noqa: E731
    script(mock_nasa, URL, truncated(), truncated(), truncated())
    manager, sleeps = make_manager(mock_nasa, retries=2)
    with pytest.raises(DownloadIncompleteError, match="every attempt"):
        manager.fetch(URL, tmp_path, "a.img")
    assert len(mock_nasa.requests) == 3 and sleeps == [1.0, 2.0]
    assert list(tmp_path.iterdir()) == []


def test_a_truncated_transfer_can_recover_on_retry(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY
    script(mock_nasa, URL, stream_response(BODY[:1000], 200, {"content-length": str(len(BODY))}))
    manager, _ = make_manager(mock_nasa)
    assert manager.fetch(URL, tmp_path, "a.img").path.read_bytes() == BODY


def test_the_total_time_budget_stops_a_slow_download_without_retrying(mock_nasa, tmp_path):
    mock_nasa.files[URL] = BODY * 20
    ticks = iter(range(0, 100_000, 400))  # every clock reading is 400 s later
    manager, sleeps = make_manager(mock_nasa, clock=lambda: float(next(ticks)), total=900.0)
    with pytest.raises(DownloadTimeoutError):
        manager.fetch(URL, tmp_path, "slow.img")
    assert len(mock_nasa.requests) == 1 and sleeps == []
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


def test_a_redirect_within_the_allowlist_is_followed_and_rechecked(mock_nasa, tmp_path):
    mock_nasa.files["https://pds-geosciences.wustl.edu/moved/ldem_test.img"] = BODY
    script(mock_nasa, URL, httpx.Response(301, headers={"location": "/moved/ldem_test.img"}))
    manager, _ = make_manager(mock_nasa)
    result = manager.fetch(URL, tmp_path, "a.img")
    assert result.url.endswith("/moved/ldem_test.img") and result.path.read_bytes() == BODY


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example.com/ldem_test.img",
        "http://pds-geosciences.wustl.edu/ldem_test.img",
        "https://127.0.0.1/ldem_test.img",
        "https://pds-geosciences.wustl.edu@evil.example.com/x.img",
        "file:///etc/passwd",
    ],
)
def test_redirects_to_forbidden_places_are_refused(mock_nasa, tmp_path, target):
    script(mock_nasa, URL, httpx.Response(302, headers={"location": target}))
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(HostPolicyError):
        manager.fetch(URL, tmp_path, "a.img")
    assert len(mock_nasa.requests) == 1  # the forbidden target was never contacted
    assert list(tmp_path.iterdir()) == []


def test_a_redirect_hop_is_checked_for_private_addresses(mock_nasa, tmp_path):
    calls = {"n": 0}

    def resolver(_host):
        calls["n"] += 1
        return ["128.252.120.58"] if calls["n"] <= 2 else ["192.168.1.10"]  # public, then a rebound answer

    script(mock_nasa, URL, httpx.Response(302, headers={"location": "/other.img"}))
    manager, _ = make_manager(mock_nasa, resolver=resolver)
    with pytest.raises(HostPolicyError):
        manager.fetch(URL, tmp_path, "a.img")


def test_redirect_loops_are_cut_off(mock_nasa, tmp_path):
    script(mock_nasa, URL, *[httpx.Response(302, headers={"location": "/loop.img"})] * 20)
    mock_nasa.data_script["https://pds-geosciences.wustl.edu/loop.img"] = [
        lambda r: httpx.Response(302, headers={"location": "/loop.img"})
    ]
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(DownloadError, match="redirect"):
        manager.fetch(URL, tmp_path, "a.img")
    assert len(mock_nasa.requests) <= 5


def test_a_redirect_without_a_location_is_refused(mock_nasa, tmp_path):
    script(mock_nasa, URL, httpx.Response(302))
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(DownloadError, match="redirect"):
        manager.fetch(URL, tmp_path, "a.img")


# ---------------------------------------------------------------------------
# Refused before any request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["../evil.img", "..\\evil.img", "/abs.img", "C:\\x.img", ".hidden", "a b.img", "a/b.img", "", "x" * 200, "nul\x00.img"]
)
def test_unsafe_target_names_are_refused(mock_nasa, tmp_path, name):
    mock_nasa.files[URL] = BODY
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(DownloadError):
        manager.fetch(URL, tmp_path, name)
    assert mock_nasa.requests == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://pds-geosciences.wustl.edu/a.img",
        "https://example.com/a.img",
        "https://imbrium.mit.edu/DATA/LOLA_GDR/a.img",
        "https://oderest.rsl.wustl.edu/live2?query=product",
        "ftp://pds-geosciences.wustl.edu/a.img",
        "https://pds-geosciences.wustl.edu:444/a.img",
        "https://user@pds-geosciences.wustl.edu/a.img",
        "https://10.0.0.1/a.img",
        "not a url",
        "",
    ],
)
def test_forbidden_urls_are_refused_before_any_request(mock_nasa, tmp_path, url):
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(HostPolicyError):
        manager.fetch(url, tmp_path, "a.img")
    assert mock_nasa.requests == []


def test_encoded_bodies_are_refused(mock_nasa, tmp_path):
    script(mock_nasa, URL, stream_response(BODY, 200, {"content-encoding": "gzip"}))
    manager, _ = make_manager(mock_nasa)
    with pytest.raises(DownloadError, match="encoded"):
        manager.fetch(URL, tmp_path, "a.img")
    assert list(tmp_path.iterdir()) == []


def test_the_limit_must_be_positive():
    with pytest.raises(ValueError):
        DownloadManager(
            policy=HostPolicy(NASA_DOWNLOAD_HOSTS),
            settings=HttpSettings(),
            max_bytes=0,
        )
