"""Mocked NASA ODE provider tests: queries, product metadata, file lists, errors, retries.

The mock serves responses shaped like the real ODE REST JSON. No network is used.
"""

from __future__ import annotations

import json

import httpx
import pytest

from terrain_agent.acquisition.errors import ProviderResponseError, ProviderUnavailableError
from terrain_agent.acquisition.http import HttpSettings
from terrain_agent.acquisition.models import CoverageRequest, FileRole
from terrain_agent.acquisition.net_policy import NASA_DOWNLOAD_HOSTS, HostPolicy
from terrain_agent.acquisition.ode_provider import ODE_API_HOSTS, OdeProvider
from tests.acquisition_fixtures import DATA_HOST, ODE_HOST, MockNasa, public_resolver, stream_response

AREA = CoverageRequest.from_point(-89.9, 0.0, 3000.0)


def make_provider(mock: MockNasa, *, retries: int = 2, resolver=public_resolver):
    sleeps: list[float] = []
    provider = OdeProvider(
        policy=HostPolicy(ODE_API_HOSTS, resolver),
        file_policy=HostPolicy(NASA_DOWNLOAD_HOSTS, resolver),
        settings=HttpSettings(max_retries=retries, backoff_base_s=1.0),
        client=httpx.Client(transport=mock.transport),
        sleep=sleeps.append,
    )
    return provider, sleeps


def add(mock: MockNasa, product_id="ldem_75s_240m", **kwargs):
    return mock.add_product(product_id=product_id, label=b"<xml/>", data=b"0" * 2000, **kwargs)


def respond_with(mock: MockNasa, *responses):
    """The next ODE requests receive these responses in order, then normal behaviour resumes."""
    queue = list(responses)

    def hook(_request):
        if not queue:
            return None
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    mock.ode_script.append(hook)


def ode_json(payload, status=200):
    return stream_response(json.dumps(payload).encode(), status, {"content-type": "application/json"})


# ---------------------------------------------------------------------------
# Queries use only documented parameters
# ---------------------------------------------------------------------------


def test_search_sends_the_documented_product_query(mock_nasa):
    add(mock_nasa)
    provider, _ = make_provider(mock_nasa)
    provider.search(CoverageRequest.from_bbox(-5.0, 5.0, 10.0, 20.0), product_types=["GDRDEM"])

    (request,) = mock_nasa.ode_requests()
    assert request.method == "GET" and str(request.url).startswith("https://oderest.rsl.wustl.edu/live2/?")
    assert dict(request.url.params) == {
        "query": "product",
        "target": "moon",
        "output": "JSON",
        "ihid": "LRO",
        "iid": "LOLA",
        "pt": "GDRDEM",
        "results": "opm",
        "loc": "b",
        "limit": "25",
        "offset": "0",
        "minlat": "-5.000000",
        "maxlat": "5.000000",
        "westernlon": "10.000000",
        "easternlon": "20.000000",
    }


def test_longitudes_are_sent_as_degrees_east_and_may_cross_the_meridian(mock_nasa):
    provider, _ = make_provider(mock_nasa)
    provider.search(CoverageRequest.from_bbox(-1.0, 1.0, -10.0, 10.0), product_types=["GDRDEM"])
    params = mock_nasa.ode_requests()[0].url.params
    assert (params["westernlon"], params["easternlon"]) == ("350.000000", "10.000000")


def test_an_area_containing_a_pole_uses_every_longitude(mock_nasa):
    provider, _ = make_provider(mock_nasa)
    provider.search(AREA, product_types=["GDRDEM"])
    params = mock_nasa.ode_requests()[0].url.params
    assert (params["westernlon"], params["easternlon"]) == ("0", "360")


def test_both_supported_product_types_are_queried_by_default(mock_nasa):
    provider, _ = make_provider(mock_nasa)
    provider.search(AREA)
    assert [r.url.params["pt"] for r in mock_nasa.ode_requests()] == ["GDRDEM", "SLDEM"]


def test_unsupported_product_types_are_refused_before_any_request(mock_nasa):
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderResponseError):
        provider.search(AREA, product_types=["EDR"])
    assert mock_nasa.requests == []


# ---------------------------------------------------------------------------
# Product metadata
# ---------------------------------------------------------------------------


def test_product_metadata_is_parsed_into_candidates(mock_nasa):
    add(mock_nasa)
    add(mock_nasa, "ldem_80s_240m", min_lat=-90.0, max_lat=-80.0, map_scale=240.0)
    provider, _ = make_provider(mock_nasa)
    found = provider.search(AREA, product_types=["GDRDEM"])

    by_id = {c.product_id: c for c in found}
    c = by_id["ldem_75s_240m"]
    assert (c.host_id, c.instrument_id, c.product_type) == ("LRO", "LOLA", "GDRDEM")
    assert c.product_lid == "urn:nasa:pds:lro_lola_rdr:data_gridded:ldem_75s_240m"
    assert c.data_set_id == "lro_lola_rdr-data_gridded" and c.version == "1.0"
    assert (c.min_lat, c.max_lat, c.west_lon, c.east_lon) == (-90.0, -75.0, 0.0, 360.0)
    assert c.map_scale_m == 240.0 and c.map_resolution_ppd == 126.347
    assert c.ode_id and c.files == ()
    assert c.covers(AREA)
    assert not by_id["ldem_80s_240m"].covers(CoverageRequest.from_point(-77.0, 10.0, 3000.0))


def test_results_are_fetched_in_pages_with_the_documented_offset(mock_nasa):
    for index in range(60):
        add(mock_nasa, f"ldem_page_{index:03d}")
    provider, _ = make_provider(mock_nasa)
    found = provider.search(AREA, product_types=["GDRDEM"])
    assert len(found) == 60 and len({c.product_id for c in found}) == 60
    pages = [(r.url.params["offset"], r.url.params["limit"]) for r in mock_nasa.ode_requests()]
    assert pages == [("0", "25"), ("25", "25"), ("50", "25")]


def test_paging_stops_at_the_result_cap(mock_nasa):
    for index in range(130):
        add(mock_nasa, f"ldem_cap_{index:03d}")
    provider, _ = make_provider(mock_nasa)
    assert len(provider.search(AREA, product_types=["GDRDEM"])) == 100
    assert len(mock_nasa.ode_requests()) == 4


def test_a_single_product_is_returned_as_an_object_not_a_list(mock_nasa):
    add(mock_nasa)  # the mock, like ODE, returns a bare object when there is exactly one product
    provider, _ = make_provider(mock_nasa)
    assert len(provider.search(AREA, product_types=["GDRDEM"])) == 1


def test_no_products_gives_an_empty_list(mock_nasa):
    provider, _ = make_provider(mock_nasa)
    assert provider.search(AREA, product_types=["GDRDEM"]) == []


def test_malformed_or_hostile_product_entries_are_skipped(mock_nasa):
    good = add(mock_nasa)
    bad_entries = [
        {**good, "pdsid": "../../etc/passwd", "ode_id": "1"},
        {**good, "pdsid": "has space", "ode_id": "2"},
        {**good, "pdsid": None, "ode_id": "3"},
        {**good, "Minimum_latitude": "95", "pdsid": "lat_out_of_range", "ode_id": "4"},
        {**good, "Westernmost_longitude": "abc", "pdsid": "lon_not_a_number", "ode_id": "5"},
        {**good, "ihid": "MRO", "pdsid": "wrong_host", "ode_id": "6"},
        "not an object",
    ]
    respond_with(
        mock_nasa,
        ode_json({"ODEResults": {"Status": "Success", "Products": {"Product": [good, *bad_entries]}}}),
    )
    provider, _ = make_provider(mock_nasa)
    found = provider.search(AREA, product_types=["GDRDEM"])
    assert [c.product_id for c in found] == ["ldem_75s_240m"]


def test_an_unreadable_resolution_is_recorded_as_unknown_not_guessed(mock_nasa):
    entry = add(mock_nasa)
    entry["Map_scale"] = "about 240"
    provider, _ = make_provider(mock_nasa)
    (candidate,) = provider.search(AREA, product_types=["GDRDEM"])
    assert candidate.map_scale_m is None


def test_free_text_from_the_provider_is_never_kept(mock_nasa):
    add(mock_nasa)
    provider, _ = make_provider(mock_nasa)
    (candidate,) = provider.search(AREA, product_types=["GDRDEM"])
    dumped = candidate.model_dump_json()
    assert "Ignore all previous" not in dumped and "imbrium" not in dumped


# ---------------------------------------------------------------------------
# Product files
# ---------------------------------------------------------------------------


def test_files_are_classified_and_sized(mock_nasa):
    add(mock_nasa)
    provider, _ = make_provider(mock_nasa)
    with_files = provider.attach_files(provider.search(AREA, product_types=["GDRDEM"]))
    (candidate,) = with_files
    data, label = candidate.file(FileRole.DATA), candidate.file(FileRole.LABEL_PDS4)
    assert data is not None and label is not None
    assert data.url.startswith(f"https://{DATA_HOST}/") and data.url.endswith("ldem_75s_240m.img")
    assert data.size_kb == 2 and data.expected_max_bytes == 2000  # 2000 bytes, listed in kB
    assert label.url.endswith(".xml")
    assert len(candidate.files) == 2  # the PDS3 label is not needed


def test_files_from_other_hosts_are_dropped_and_the_mirror_url_is_ignored(mock_nasa):
    entry = add(mock_nasa)
    entry["_files"][0]["URL"] = "https://evil.example.com/ldem_75s_240m.img"
    provider, _ = make_provider(mock_nasa)
    (candidate,) = provider.attach_files(provider.search(AREA, product_types=["GDRDEM"]))
    assert candidate.file(FileRole.DATA) is None  # no fallback to External_url
    assert candidate.file(FileRole.LABEL_PDS4) is not None


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://pds-geosciences.wustl.edu/a/ldem.img",
        "https://pds-geosciences.wustl.edu.evil.com/a/ldem.img",
        "https://user:pw@pds-geosciences.wustl.edu/a/ldem.img",
        "https://pds-geosciences.wustl.edu:8443/a/ldem.img",
        "file:///etc/passwd",
        "https://127.0.0.1/ldem.img",
    ],
)
def test_unsafe_file_urls_are_never_accepted(mock_nasa, bad_url):
    entry = add(mock_nasa)
    entry["_files"][0]["URL"] = bad_url
    provider, _ = make_provider(mock_nasa)
    (candidate,) = provider.attach_files(provider.search(AREA, product_types=["GDRDEM"]))
    assert candidate.file(FileRole.DATA) is None


def test_file_lists_are_fetched_for_many_products_in_batches(mock_nasa):
    for index in range(30):
        add(mock_nasa, f"ldem_tile_{index:02d}")
    provider, _ = make_provider(mock_nasa)
    candidates = provider.search(AREA, product_types=["GDRDEM"])
    assert len(candidates) == 30
    with_files = provider.attach_files(candidates)

    batches = [r.url.params["odeid"].split("|") for r in mock_nasa.ode_requests() if "odeid" in r.url.params]
    assert [len(b) for b in batches] == [25, 5]
    assert all(c.file(FileRole.DATA) is not None for c in with_files)
    file_requests = [r for r in mock_nasa.ode_requests() if "odeid" in r.url.params]
    assert len(file_requests) == 2 and all(r.url.params["results"] == "opf" for r in file_requests)


def test_discovery_info_records_queries_but_not_ode_ids(mock_nasa):
    add(mock_nasa)
    provider, _ = make_provider(mock_nasa)
    provider.attach_files(provider.search(AREA, product_types=["GDRDEM"]))
    info = provider.discovery_info()
    assert info["endpoint"] == "https://oderest.rsl.wustl.edu/live2/"
    assert "2.1.6" in str(info["documentation"])
    queries = info["queries"]
    assert any(q.get("results") == "opm" for q in queries)  # type: ignore[union-attr]
    assert any(q.get("odeid_count") == "1" for q in queries)  # type: ignore[union-attr]
    assert not any("odeid" in q for q in queries)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# ODE errors and malformed responses
# ---------------------------------------------------------------------------


def test_an_ode_error_status_is_reported_with_sanitised_text(mock_nasa):
    respond_with(
        mock_nasa,
        ode_json({"ODEResults": {"Status": "ERROR", "Error": "Unable to connect <script>x</script> " + "A" * 500}}),
    )
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderResponseError) as info:
        provider.search(AREA, product_types=["GDRDEM"])
    message = str(info.value)
    assert "<" not in message and ">" not in message and len(message) < 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"ODEResults": "text"},
        {"ODEResults": {}},
        {"ODEResults": {"Status": "Pending"}},
        {"ODEResults": {"Status": "Success", "Products": "text"}},
        [],
    ],
)
def test_malformed_responses_are_rejected(mock_nasa, payload):
    respond_with(mock_nasa, ode_json(payload))
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderResponseError):
        provider.search(AREA, product_types=["GDRDEM"])


def test_non_json_bodies_are_rejected(mock_nasa):
    for body in (b"<html>maintenance</html>", b"", b"\xff\xfe\x00 not utf-8"):
        respond_with(mock_nasa, stream_response(body, 200, {"content-type": "text/html"}))
        provider, _ = make_provider(mock_nasa)
        with pytest.raises(ProviderResponseError):
            provider.search(AREA, product_types=["GDRDEM"])


def test_oversized_responses_are_rejected_declared_or_streamed(mock_nasa):
    from terrain_agent.acquisition.ode_provider import MAX_SEARCH_RESPONSE_BYTES

    huge = b"0" * (MAX_SEARCH_RESPONSE_BYTES + 1024)
    respond_with(mock_nasa, stream_response(huge, 200))
    respond_with(mock_nasa, stream_response(huge, 200, {"content-length": "10"}))  # lies about its size
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderResponseError, match="larger"):
        provider.search(AREA, product_types=["GDRDEM"])
    # The second response declares 10 bytes but streams megabytes. The stream is cut off.
    with pytest.raises(ProviderResponseError, match="larger"):
        provider.search(AREA, product_types=["GDRDEM"])


def test_encoded_responses_are_refused(mock_nasa):
    respond_with(mock_nasa, stream_response(b"{}", 200, {"content-encoding": "gzip"}))
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderResponseError, match="encoded"):
        provider.search(AREA, product_types=["GDRDEM"])


def test_a_redirect_to_the_same_host_is_followed_with_the_query_intact(mock_nasa):
    """The real server redirects /live2?... to /live2/?... and the provider must cope."""
    add(mock_nasa)
    respond_with(
        mock_nasa,
        httpx.Response(301, headers={"location": "https://oderest.rsl.wustl.edu/live2/?query=product&target=moon&pt=GDRDEM&ihid=LRO&iid=LOLA&results=opm&loc=b&output=JSON&minlat=-90&maxlat=-89&westernlon=0&easternlon=360"}),
    )
    provider, _ = make_provider(mock_nasa)
    assert len(provider.search(AREA, product_types=["GDRDEM"])) == 1
    assert len(mock_nasa.ode_requests()) == 2


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example.com/",
        "http://oderest.rsl.wustl.edu/live2/",
        "https://127.0.0.1/live2/",
        "file:///etc/passwd",
    ],
)
def test_redirects_to_other_places_are_refused_and_never_contacted(mock_nasa, target):
    respond_with(mock_nasa, httpx.Response(302, headers={"location": target}))
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderUnavailableError):
        provider.search(AREA, product_types=["GDRDEM"])
    assert len(mock_nasa.requests) == 1


def test_redirect_loops_and_missing_locations_are_cut_off(mock_nasa):
    loop = httpx.Response(302, headers={"location": "https://oderest.rsl.wustl.edu/live2/"})
    respond_with(mock_nasa, *[loop] * 10)
    provider, _ = make_provider(mock_nasa)
    with pytest.raises(ProviderResponseError, match="redirect"):
        provider.search(AREA, product_types=["GDRDEM"])
    assert len(mock_nasa.requests) <= 4
    respond_with(mock_nasa, httpx.Response(302))
    with pytest.raises(ProviderResponseError, match="redirect"):
        provider.search(AREA, product_types=["GDRDEM"])


# ---------------------------------------------------------------------------
# Retries and failures
# ---------------------------------------------------------------------------


def test_transient_failures_are_retried_with_backoff(mock_nasa):
    add(mock_nasa)
    respond_with(mock_nasa, httpx.Response(500), httpx.Response(503))
    provider, sleeps = make_provider(mock_nasa, retries=3)
    assert len(provider.search(AREA, product_types=["GDRDEM"])) == 1
    assert sleeps == [1.0, 2.0]
    assert len(mock_nasa.ode_requests()) == 3


def test_timeouts_and_connection_errors_are_retried(mock_nasa):
    add(mock_nasa)
    respond_with(mock_nasa, httpx.ReadTimeout("slow"), httpx.ConnectError("refused"))
    provider, sleeps = make_provider(mock_nasa, retries=3)
    assert len(provider.search(AREA, product_types=["GDRDEM"])) == 1
    assert len(sleeps) == 2


def test_retries_are_bounded(mock_nasa):
    respond_with(mock_nasa, *[httpx.Response(503)] * 10)
    provider, sleeps = make_provider(mock_nasa, retries=2)
    with pytest.raises(ProviderUnavailableError, match="3 attempts"):
        provider.search(AREA, product_types=["GDRDEM"])
    assert len(mock_nasa.ode_requests()) == 3 and sleeps == [1.0, 2.0]


def test_permanent_http_errors_are_not_retried(mock_nasa):
    respond_with(mock_nasa, httpx.Response(404), httpx.Response(404))
    provider, sleeps = make_provider(mock_nasa)
    with pytest.raises(ProviderUnavailableError, match="404"):
        provider.search(AREA, product_types=["GDRDEM"])
    assert len(mock_nasa.ode_requests()) == 1 and sleeps == []


def test_the_api_host_must_resolve_to_a_public_address(mock_nasa):
    provider, _ = make_provider(mock_nasa, resolver=lambda host: ["10.0.0.5"])
    with pytest.raises(ProviderUnavailableError):
        provider.search(AREA, product_types=["GDRDEM"])
    assert mock_nasa.requests == []  # refused before any request was sent


def test_an_unresolvable_api_host_fails_closed(mock_nasa):
    provider, _ = make_provider(mock_nasa, resolver=lambda host: [])
    with pytest.raises(ProviderUnavailableError):
        provider.search(AREA, product_types=["GDRDEM"])
    assert mock_nasa.requests == []


def test_the_api_base_url_cannot_be_pointed_elsewhere(mock_nasa):
    from terrain_agent.acquisition.errors import HostPolicyError

    with pytest.raises(HostPolicyError):
        OdeProvider(base_url="https://evil.example.com/live2", client=httpx.Client(transport=mock_nasa.transport))
    with pytest.raises(HostPolicyError):
        OdeProvider(base_url="http://oderest.rsl.wustl.edu/live2", client=httpx.Client(transport=mock_nasa.transport))
    assert ODE_HOST in ODE_API_HOSTS
