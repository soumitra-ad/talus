"""Cache, coverage request and product selection tests (offline)."""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from terrain_agent.acquisition.cache import CACHE_SUBDIR, DemCache, make_cache_id
from terrain_agent.acquisition.errors import CacheError
from terrain_agent.acquisition.models import (
    CoverageRequest,
    FileRole,
    ProductCandidate,
    ProductFile,
    arc_contains,
)
from terrain_agent.acquisition.selection import choose_downloadable, rank_for_request
from terrain_agent.terrain import (
    InvalidCoordinateError,
    InvalidThresholdError,
    InvalidWaypointError,
    OversizedRequestError,
)

STEREO = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"


# ---------------------------------------------------------------------------
# Cache identifiers and paths
# ---------------------------------------------------------------------------


def test_cache_ids_are_deterministic_safe_and_source_specific():
    a = make_cache_id("nasa_ode", "ldem_75s_240m", "https://pds-geosciences.wustl.edu/a/ldem_75s_240m.img")
    assert a == make_cache_id("nasa_ode", "ldem_75s_240m", "https://pds-geosciences.wustl.edu/a/ldem_75s_240m.img")
    assert a != make_cache_id("nasa_ode", "ldem_75s_240m", "https://pds-geosciences.wustl.edu/b/ldem_75s_240m.img")
    assert a != make_cache_id("nasa_ode", "ldem_80s_240m", "https://pds-geosciences.wustl.edu/a/ldem_75s_240m.img")
    assert a.startswith("ldem_75s_240m-") and len(a.rsplit("-", 1)[1]) == 12


def test_hostile_product_ids_cannot_produce_hostile_cache_ids(tmp_path):
    cache = DemCache(tmp_path, max_bytes=10**9)
    for evil in ("../../etc/passwd", "..\\..\\x", "a/b", "C:\\x", "x" * 400, "\x00null", "", "...", "UPPER CASE"):
        cache_id = make_cache_id("nasa_ode", evil, "https://pds-geosciences.wustl.edu/x")
        tif, js, label = cache._paths(cache_id)  # must not raise, and must stay inside the cache
        assert tif.parent == js.parent == label.parent == cache.root


@pytest.mark.parametrize(
    "bad",
    ["../x", "..\\x", "/abs-0123456789ab", "a/b-0123456789ab", "", "UPPER-0123456789ab", "x", "x-short", None, 5,
     "a" * 200 + "-0123456789ab", "ok-0123456789ab\n", "ok-0123456789ab/../../etc"],
)
def test_invalid_cache_identifiers_are_refused(tmp_path, bad):
    cache = DemCache(tmp_path, max_bytes=10**9)
    with pytest.raises(CacheError):
        cache.lookup(bad)
    with pytest.raises(CacheError):
        cache.remove(bad)


def test_a_symlink_that_leaves_the_cache_is_refused(tmp_path):
    cache = DemCache(tmp_path / "cache", max_bytes=10**9)
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"secret")
    link = cache.root / "evil-0123456789ab.tif"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")
    with pytest.raises(CacheError):
        cache.lookup("evil-0123456789ab")
    assert outside.read_bytes() == b"secret"


def test_the_cache_location_is_configurable_and_lives_in_a_fixed_subdirectory(tmp_path):
    cache = DemCache(tmp_path / "custom" / "place", max_bytes=1000)
    assert cache.root == (tmp_path / "custom" / "place" / CACHE_SUBDIR).resolve()
    assert cache.root.is_dir() and (cache.root / ".tmp").is_dir()


def test_the_default_service_uses_the_configured_cache_directory(tmp_path):
    from terrain_agent.acquisition.service import build_default_service

    service = build_default_service(cache_dir=tmp_path / "configured", max_cache_bytes=5_000_000)
    assert service.cache.root == (tmp_path / "configured" / CACHE_SUBDIR).resolve()
    assert service.cache.max_bytes == 5_000_000


def test_cache_limits_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        DemCache(tmp_path, max_bytes=0)


# ---------------------------------------------------------------------------
# Entries: commit, verification, damage
# ---------------------------------------------------------------------------


def write_tif(path, value=1500.0, pixel=240.0, size=40):
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float32",
        crs=CRS.from_string(STEREO), transform=from_origin(-size / 2 * pixel, size / 2 * pixel, pixel, pixel), nodata=-9999.0,
    ) as dst:
        dst.write(np.full((size, size), value, dtype="float32"), 1)
    return path


def sidecar(cache_id, tif, *, product_type="GDRDEM", pixel=240.0):
    digest = hashlib.sha256(tif.read_bytes()).hexdigest()
    return {
        "product_id": "ldem_test",
        "nasa_provenance": {
            "cache_id": cache_id,
            "product_type": product_type,
            "raster": {"native_pixel_size": [pixel, pixel], "native_units": "metre"},
            "normalization": {"output_bytes": tif.stat().st_size, "output_sha256": digest},
        },
    }


def commit(cache, tmp_path, cache_id="ldem_test-0123456789ab", **kwargs):
    tif = write_tif(tmp_path / f"{cache_id}.build.tif")
    return cache.commit(cache_id, tif, sidecar(cache_id, tif, **kwargs))


def test_commit_lookup_and_relative_name(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=10**9)
    entry = commit(cache, tmp_path)
    assert entry.relative_name == "nasa/ldem_test-0123456789ab.tif"
    assert cache.lookup("ldem_test-0123456789ab").tif_path == entry.tif_path
    assert cache.list_ids() == ["ldem_test-0123456789ab"]
    assert cache.lookup("other-0123456789ab") is None


def test_a_modified_entry_is_removed_on_use(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=10**9)
    entry = commit(cache, tmp_path)
    with open(entry.tif_path, "r+b") as handle:
        handle.seek(1500)
        handle.write(b"\xff\xff\xff\xff")
    assert cache.lookup(entry.cache_id) is None
    assert cache.list_ids() == [] and not entry.tif_path.exists()


def test_integrity_checking_can_be_relaxed_to_a_size_check(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=10**9, verify_on_use=False)
    entry = commit(cache, tmp_path)
    with open(entry.tif_path, "r+b") as handle:
        handle.seek(1500)
        handle.write(b"\xff\xff")
    assert cache.lookup(entry.cache_id) is not None  # same size, not re-hashed
    entry.tif_path.write_bytes(b"short")
    assert cache.lookup(entry.cache_id) is None  # a size change is still caught


@pytest.mark.parametrize("damage", ["invalid_json", "wrong_id", "no_block", "tif_missing"])
def test_damaged_or_incomplete_entries_are_ignored_and_cleaned(tmp_path, damage):
    cache = DemCache(tmp_path / "c", max_bytes=10**9)
    entry = commit(cache, tmp_path)
    if damage == "invalid_json":
        entry.json_path.write_text("{not json", encoding="utf-8")
    elif damage == "wrong_id":
        doc = json.loads(entry.json_path.read_text(encoding="utf-8"))
        doc["nasa_provenance"]["cache_id"] = "other-0123456789ab"
        entry.json_path.write_text(json.dumps(doc), encoding="utf-8")
    elif damage == "no_block":
        entry.json_path.write_text(json.dumps({"product_id": "x"}), encoding="utf-8")
    else:
        entry.tif_path.unlink()
    assert cache.lookup(entry.cache_id) is None


def test_an_entry_without_provenance_is_never_listed(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=10**9)
    write_tif(cache.root / "orphan-0123456789ab.tif")  # no provenance file, as after an interrupted commit
    assert cache.list_ids() == [] and cache.lookup("orphan-0123456789ab") is None


def test_stale_work_directories_are_purged_and_fresh_ones_kept(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=10**9)
    old, fresh = cache.new_work_dir(), cache.new_work_dir()
    (old / "leftover.img").write_bytes(b"x" * 100)
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(old, (two_days_ago, two_days_ago))
    DemCache(tmp_path / "c", max_bytes=10**9)  # a new process starts
    assert not old.exists() and fresh.exists()


# ---------------------------------------------------------------------------
# Quota and coverage lookup
# ---------------------------------------------------------------------------


def test_the_quota_evicts_the_least_recently_used_entry(tmp_path):
    probe = DemCache(tmp_path / "probe", max_bytes=10**9)
    size = commit(probe, tmp_path, "aaa-0123456789ab").size_bytes
    cache = DemCache(tmp_path / "c", max_bytes=int(size * 2.6))
    commit(cache, tmp_path, "aaa-0123456789ab")
    commit(cache, tmp_path, "bbb-0123456789ab")
    cache.lookup("aaa-0123456789ab")  # aaa is now the most recently used
    commit(cache, tmp_path, "ccc-0123456789ab")  # needs room: bbb is the oldest
    assert cache.list_ids() == ["aaa-0123456789ab", "ccc-0123456789ab"]
    assert cache.total_bytes() <= cache.max_bytes


def test_an_entry_larger_than_the_quota_is_refused_and_leaves_nothing(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=500)
    tif = write_tif(tmp_path / "big.tif")
    with pytest.raises(CacheError):
        cache.commit("big-0123456789ab", tif, sidecar("big-0123456789ab", tif))
    assert cache.list_ids() == [] and cache.total_bytes() == 0


def test_find_covering_uses_only_local_files_and_orders_by_resolution(tmp_path):
    cache = DemCache(tmp_path / "c", max_bytes=10**9)
    coarse = write_tif(tmp_path / "coarse.tif", pixel=240.0, size=40)  # about 4.8 km across
    fine = write_tif(tmp_path / "fine.tif", pixel=60.0, size=100)  # 3 km across
    cache.commit("coarse-0123456789ab", coarse, sidecar("coarse-0123456789ab", coarse, pixel=240.0))
    cache.commit("fine-0123456789ab", fine, sidecar("fine-0123456789ab", fine, product_type="SLDEM", pixel=60.0))

    near_pole = CoverageRequest.from_point(-89.98, 0.0, 300.0)
    assert [e.cache_id for e in cache.find_covering(near_pole)] == ["fine-0123456789ab", "coarse-0123456789ab"]
    assert [e.cache_id for e in cache.find_covering(near_pole, ["GDRDEM"])] == ["coarse-0123456789ab"]
    assert [e.cache_id for e in cache.find_covering(CoverageRequest.from_point(-89.98, 0.0, 300.0, max_pixel_size_m=100.0))] == [
        "fine-0123456789ab"
    ]
    assert cache.find_covering(CoverageRequest.from_point(-80.0, 0.0, 300.0)) == []  # outside every raster


# ---------------------------------------------------------------------------
# Coverage requests
# ---------------------------------------------------------------------------


def test_point_bbox_and_route_requests():
    point = CoverageRequest.from_point(0.5, 10.0, 5000.0)
    assert point.min_lat < 0.5 < point.max_lat and not point.full_longitude
    assert point.west_lon < 10.0 < point.east_lon and (point.center_lat, point.center_lon) == (0.5, 10.0)

    box = CoverageRequest.from_bbox(-10.0, 10.0, -5.0, 5.0)  # crosses the 0/360 meridian
    assert (box.west_lon, box.east_lon, box.lon_width) == (355.0, 5.0, 10.0)

    route = CoverageRequest.from_waypoints([(0.1, 359.95), (0.1, 0.05)], margin_m=500.0)
    assert route.lon_width < 1.0 and route.west_lon > 359.0 and route.east_lon < 1.0


def test_areas_containing_a_pole_span_every_longitude():
    pole = CoverageRequest.from_point(-89.9, 30.0, 5000.0)
    assert pole.full_longitude and pole.min_lat == -90.0 and pole.lon_width == 360.0
    lons = {p[1] for p in pole.sample_points()}
    assert {0.0, 90.0, 180.0, -90.0} <= lons


@pytest.mark.parametrize(
    "call",
    [
        lambda: CoverageRequest.from_point(95.0, 0.0, 1000.0),
        lambda: CoverageRequest.from_point(float("nan"), 0.0, 1000.0),
        lambda: CoverageRequest.from_point(120.0, 15.0, 1000.0),  # probable swap
        lambda: CoverageRequest.from_bbox(10.0, 5.0, 0.0, 1.0),
        lambda: CoverageRequest.from_bbox(0.0, 1.0, 5.0, 5.0),
    ],
)
def test_invalid_areas_are_rejected(call):
    with pytest.raises(InvalidCoordinateError):
        call()


def test_invalid_sizes_are_rejected():
    for radius in (0, -1, float("nan"), True, "5000"):
        with pytest.raises((InvalidThresholdError, InvalidCoordinateError)):
            CoverageRequest.from_point(0.0, 0.0, radius)
    with pytest.raises(OversizedRequestError):
        CoverageRequest.from_point(0.0, 0.0, 60_000.0)
    for bad in (0, -5.0, float("nan"), float("inf"), True, "10"):
        with pytest.raises(InvalidThresholdError):
            CoverageRequest.from_point(0.0, 0.0, 1000.0, max_pixel_size_m=bad)
    with pytest.raises(InvalidWaypointError):
        CoverageRequest.from_waypoints([(0.0, 0.0)])


def test_longitude_arc_containment():
    assert arc_contains(0.0, 360.0, 123.0, 50.0)  # a full circle contains anything
    assert arc_contains(350.0, 30.0, 355.0, 10.0)  # both wrap the meridian
    assert arc_contains(350.0, 30.0, 0.0, 10.0)
    assert not arc_contains(350.0, 30.0, 10.0, 20.0)  # sticks out the far end
    assert not arc_contains(10.0, 20.0, 5.0, 10.0)  # sticks out the near end
    assert not arc_contains(10.0, 20.0, 12.0, 30.0)  # wider than the outer arc


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def cand(pid, scale, *, min_lat=-90.0, max_lat=-75.0, west=0.0, east=360.0, kb=None, files=True, pt="GDRDEM"):
    items = ()
    if files:
        base = f"https://pds-geosciences.wustl.edu/lro/data/{pid}"
        items = (
            ProductFile(role=FileRole.DATA, file_name=f"{pid}.img", url=f"{base}.img", size_kb=kb),
            ProductFile(role=FileRole.LABEL_PDS4, file_name=f"{pid}.xml", url=f"{base}.xml", size_kb=12),
        )
    return ProductCandidate(
        provider_id="nasa_ode", host_id="LRO", instrument_id="LOLA", product_type=pt, product_id=pid,
        min_lat=min_lat, max_lat=max_lat, west_lon=west, east_lon=east, map_scale_m=scale, files=items,
    )


REQUEST = CoverageRequest.from_point(-89.9, 0.0, 3000.0)


def test_ranking_orders_by_resolution_then_id_and_records_exclusions():
    candidates = [
        cand("c_240", 240.0), cand("b_120", 120.0), cand("a_120", 120.0),
        cand("no_scale", None),  # type: ignore[arg-type]
        cand("elsewhere", 20.0, min_lat=-70.0, max_lat=-60.0),
        cand("half", 20.0, min_lat=-90.0, max_lat=-89.95),
    ]
    ranked, excluded = rank_for_request(candidates, REQUEST)
    assert [c.product_id for c in ranked] == ["a_120", "b_120", "c_240"]
    assert {e.product_id: e.reason for e in excluded} == {
        "no_scale": "resolution_unknown", "elsewhere": "does_not_cover", "half": "does_not_cover",
    }
    assert [c.product_id for c in rank_for_request(list(reversed(candidates)), REQUEST)[0]] == ["a_120", "b_120", "c_240"]


def test_a_required_resolution_excludes_coarser_products():
    request = CoverageRequest.from_point(-89.9, 0.0, 3000.0, max_pixel_size_m=150.0)
    ranked, excluded = rank_for_request([cand("fine", 120.0), cand("coarse", 240.0)], request)
    assert [c.product_id for c in ranked] == ["fine"]
    assert excluded[0].reason == "too_coarse" and "240" in excluded[0].detail


def test_choice_takes_the_finest_product_within_the_limit_and_the_smaller_of_equals():
    ranked = [
        cand("fine_big", 20.0, kb=150_000),  # too big
        cand("mid_float", 120.0, kb=226_000),  # too big
        cand("mid_int", 120.0, kb=113_000),
        cand("mid_int_small", 120.0, kb=60_000),
        cand("coarse", 240.0, kb=29_000),
    ]
    chosen, excluded = choose_downloadable(ranked, 120_000_000)
    assert chosen is not None and chosen.product_id == "mid_int_small"
    reasons = {e.product_id: e.reason for e in excluded}
    assert reasons["fine_big"] == "exceeds_download_limit" and reasons["mid_float"] == "exceeds_download_limit"
    assert reasons["mid_int"] == "not_selected" and reasons["coarse"] == "not_selected"
    assert "20 m per pixel" in next(e.detail for e in excluded if e.product_id == "fine_big")


def test_products_with_incomplete_or_unsized_files_are_never_chosen():
    chosen, excluded = choose_downloadable(
        [cand("no_files", 10.0, files=False), cand("no_size", 20.0, kb=None), cand("ok", 240.0, kb=1000)], 10**9
    )
    assert chosen is not None and chosen.product_id == "ok"
    assert {e.product_id: e.reason for e in excluded if e.product_id != "ok"} == {
        "no_files": "files_incomplete", "no_size": "size_unknown",
    }


def test_a_label_that_does_not_belong_to_the_data_file_is_not_chosen():
    good = cand("prod", 240.0, kb=1000)
    label = good.file(FileRole.LABEL_PDS4)
    swapped = good.model_copy(
        update={"files": (good.file(FileRole.DATA), label.model_copy(update={"url": "https://pds-geosciences.wustl.edu/elsewhere/prod.xml"}))}
    )
    chosen, excluded = choose_downloadable([swapped], 10**9)
    assert chosen is None and excluded[0].reason == "label_mismatch"


def test_nothing_chosen_when_everything_is_too_large():
    chosen, excluded = choose_downloadable([cand("big", 20.0, kb=500_000)], 100_000_000)
    assert chosen is None and excluded[0].reason == "exceeds_download_limit"


def test_provider_sizes_are_kilobytes_of_1000_bytes():
    assert ProductFile(role=FileRole.DATA, file_name="a.img", url="https://x/a.img", size_kb=29063).expected_max_bytes == 29_063_000
    assert ProductFile(role=FileRole.DATA, file_name="a.img", url="https://x/a.img").expected_max_bytes is None
