"""The fetch_nasa_dem agent tool and its hand-off to the Phase 5 tools (offline)."""

from __future__ import annotations

import json

import pytest

from terrain_agent.agent import agent as agent_module
from terrain_agent.agent.agent import TALUS_TOOL_DECLARATIONS, dispatch_tool_call
from tests.acquisition_fixtures import write_product


@pytest.fixture
def wired(mock_nasa, make_service, tmp_path, monkeypatch):
    """A working service behind the tool, with one synthetic polar product available."""
    label, data = write_product(tmp_path / "srv", "ldem_75s_240m", kind="polar", lines=200, samples=200)
    mock_nasa.add_product(product_id="ldem_75s_240m", label=label.read_bytes(), data=data.read_bytes())
    service = make_service(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    return service, str(tmp_path / "cache")


def call(args, cache_dir):
    return dispatch_tool_call("fetch_nasa_dem", args, dem_cache_dir=cache_dir)


def test_the_tool_takes_a_location_and_never_a_url():
    declaration = next(t for t in TALUS_TOOL_DECLARATIONS if t["name"] == "fetch_nasa_dem")
    properties = declaration["parameters"]["properties"]
    assert set(properties) == {"lat", "lon", "radius_km", "preferred_dataset", "max_pixel_size_m"}
    assert declaration["parameters"]["required"] == ["lat", "lon"]
    assert properties["preferred_dataset"]["enum"] == ["lola", "sldem"]


def test_a_dem_is_fetched_and_usable_by_the_analysis_tools(wired):
    _service, cache_dir = wired
    fetched = call({"lat": -89.9, "lon": 0.0, "radius_km": 3.0}, cache_dir)

    assert fetched["status"] == "ok" and not fetched["from_cache"]
    assert fetched["dem_path"].startswith("nasa/") and fetched["dem_path"].endswith(".tif")
    provenance = fetched["provenance"]
    assert provenance["product_id"] == "ldem_75s_240m" and provenance["product_type"] == "GDRDEM"
    assert provenance["projection"] == "stere" and provenance["native_pixel_size"] == [240.0, 240.0]
    assert "not a geoid" in provenance["elevation_reference"]
    assert provenance["checksum_status"] == "computed_locally_not_verified_against_nasa"
    assert "NOT certified" in fetched["disclaimer"]
    json.dumps(fetched)  # serialisable for the model

    # The returned file name is what the existing Phase 5 tools accept as dem_path.
    stats = dispatch_tool_call(
        "get_slope_stats",
        {"dem_path": fetched["dem_path"], "min_lat": -89.95, "max_lat": -89.85, "min_lon": -10.0, "max_lon": 10.0},
        dem_cache_dir=cache_dir,
    )
    assert stats["status"] == "ok"
    assert stats["analysis"]["dataset"]["product_type"] == "GDRDEM"
    assert stats["analysis"]["dataset"]["product_lid"].endswith("ldem_75s_240m")

    route = dispatch_tool_call(
        "evaluate_traverse_route",
        {"waypoints": [[-89.95, 10.0], [-89.93, 100.0]], "dem_path": fetched["dem_path"]},
        dem_cache_dir=cache_dir,
    )
    assert route["status"] == "ok" and route["analysis"]["dataset"]["cache_id"] == fetched["provenance"]["cache_id"]


def test_a_repeat_request_is_served_from_the_cache(wired, mock_nasa):
    _service, cache_dir = wired
    call({"lat": -89.9, "lon": 0.0, "radius_km": 3.0}, cache_dir)
    seen = len(mock_nasa.requests)
    again = call({"lat": -89.8, "lon": 40.0, "radius_km": 2.0}, cache_dir)
    assert again["status"] == "ok" and again["from_cache"] and len(mock_nasa.requests) == seen


def test_no_suitable_product_explains_why(wired, mock_nasa):
    _service, cache_dir = wired
    result = call({"lat": -89.9, "lon": 0.0, "radius_km": 3.0, "max_pixel_size_m": 50.0}, cache_dir)
    assert result["status"] == "no_product"
    assert result["excluded_products"][0]["reason"] == "too_coarse"


def test_an_area_without_products_is_reported(wired):
    _service, cache_dir = wired
    result = call({"lat": 40.0, "lon": 10.0, "radius_km": 3.0}, cache_dir)
    assert result["status"] == "no_product" and "no supported DEM product" in result["error"]


def test_disabled_downloads_are_reported_as_disabled(mock_nasa, make_service, tmp_path, monkeypatch):
    service = make_service(enabled=False, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    result = call({"lat": -89.9, "lon": 0.0}, str(tmp_path / "cache"))
    assert result["status"] == "disabled" and mock_nasa.requests == []


@pytest.mark.parametrize(
    "args, error_type",
    [
        ({"lat": 95.0, "lon": 0.0}, "InvalidCoordinateError"),
        ({"lat": 120.0, "lon": 15.0}, "InvalidCoordinateError"),
        ({"lat": -89.9, "lon": 0.0, "radius_km": 500.0}, "OversizedRequestError"),
        ({"lat": -89.9, "lon": 0.0, "radius_km": -1}, "InvalidCoordinateError"),
        ({"lat": -89.9, "lon": 0.0, "max_pixel_size_m": -5}, "InvalidThresholdError"),
        ({"lat": "north", "lon": 0.0}, "InvalidCoordinateError"),
    ],
)
def test_invalid_requests_are_rejected_before_any_network_use(wired, mock_nasa, args, error_type):
    _service, cache_dir = wired
    result = call(args, cache_dir)
    assert result["status"] == "error" and result["error_type"] == error_type
    assert mock_nasa.requests == []


@pytest.mark.parametrize(
    "extra",
    [{"preferred_dataset": "gdrdem&limit=99999"}, {"preferred_dataset": "https://evil.example.com/x.img"}, {"radius_km": True}, {"radius_km": "5"}],
)
def test_arguments_that_are_not_valid_choices_are_rejected(wired, mock_nasa, extra):
    _service, cache_dir = wired
    result = call({"lat": -89.9, "lon": 0.0, **extra}, cache_dir)
    assert result["status"] == "error"
    assert mock_nasa.requests == []


def test_extra_arguments_such_as_urls_are_ignored_not_followed(wired, mock_nasa):
    _service, cache_dir = wired
    result = call({"lat": -89.9, "lon": 0.0, "radius_km": 3.0, "url": "https://evil.example.com/x.img"}, cache_dir)
    assert result["status"] == "ok"
    assert not any("evil" in str(r.url) for r in mock_nasa.requests)


def test_a_missing_argument_is_reported(wired):
    _service, cache_dir = wired
    result = call({"lat": -89.9}, cache_dir)
    assert result["status"] == "error" and "Missing required argument" in result["error"]


def test_provider_failures_do_not_leak_internal_details(mock_nasa, make_service, tmp_path, monkeypatch):
    import httpx

    service = make_service(cache_dir=tmp_path / "cache", retries=1)
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    mock_nasa.ode_script.append(lambda request: httpx.Response(503))
    result = call({"lat": -89.9, "lon": 0.0}, str(tmp_path / "cache"))
    assert result["status"] == "error" and result["error_type"] == "ProviderUnavailableError"
    assert str(tmp_path) not in json.dumps(result)
