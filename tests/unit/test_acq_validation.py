"""Validation and normalisation of downloaded PDS4 products (offline, synthetic).

Covers valid products, corrupt or incomplete files, hostile labels, CRS rules, resolution,
coverage, elevation plausibility, and correctness of the conversion the terrain engine relies on.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
import rasterio

from terrain_agent.acquisition.errors import DemValidationError
from terrain_agent.acquisition.models import CoverageRequest, ProductCandidate
from terrain_agent.acquisition.normalize import NODATA_M, normalize_to_geotiff, verify_normalized
from terrain_agent.acquisition.pds_validation import HeightConvention, parse_label, validate_pds4_product
from terrain_agent.safety import SafetyStatus, check_rover_safety
from terrain_agent.terrain import open_dem_context
from tests.acquisition_fixtures import R_M, pds4_label, synthetic_terrain, write_product

POLE = CoverageRequest.from_point(-89.9, 0.0, 3000.0)
EQC_AREA = CoverageRequest.from_bbox(0.0, 1.0, 12.0, 14.0)


def candidate(scale=240.0):
    return ProductCandidate(
        provider_id="nasa_ode", host_id="LRO", instrument_id="LOLA", product_type="GDRDEM",
        product_id="x", min_lat=-90.0, max_lat=-75.0, west_lon=0.0, east_lon=360.0, map_scale_m=scale,
    )


def fails(check, label, data, **kwargs):
    with pytest.raises(DemValidationError) as info:
        validate_pds4_product(label, data, **kwargs)
    assert info.value.check == check, str(info.value)
    return info.value


# ---------------------------------------------------------------------------
# Valid products
# ---------------------------------------------------------------------------


def test_a_valid_polar_stereographic_integer_product_passes(pds_product):
    label, data = pds_product()
    report, convention = validate_pds4_product(label, data, candidate=candidate(), request=POLE)

    assert report.passed and report.source_driver == "PDS4" and report.source_dtype == "int16"
    assert (report.width, report.height) == (200, 200)
    assert report.crs_kind == "projected" and report.projection == "stere"
    assert "R=1737400" in report.crs_proj and "lat_0=-90" in report.crs_proj
    assert report.native_pixel_size == (240.0, 240.0) and report.pixel_size_m[0] == pytest.approx(240.0, rel=1e-3)
    assert report.request_coverage_checked and report.finite_fraction == 1.0
    assert [c.name for c in report.checks][-1] == "elevation_values"
    assert convention.factor_to_metres == 0.5 and convention.reference_radius_m == 1737400.0


def test_a_valid_equirectangular_float_kilometre_product_passes(pds_product):
    label, data = pds_product("ldem_16_float", kind="eqc", lines=100, samples=200)
    report, convention = validate_pds4_product(label, data, candidate=candidate(1895.21), request=EQC_AREA)
    assert report.projection == "eqc" and report.source_dtype == "float32" and report.unit == "KILOMETER"
    assert convention.factor_to_metres == 1000.0 and convention.reference_radius_m == pytest.approx(1737400.0)
    assert report.pixel_size_m[1] == pytest.approx(1895.21, rel=1e-3)


# ---------------------------------------------------------------------------
# Corrupt or incomplete data
# ---------------------------------------------------------------------------


def test_a_truncated_data_file_is_rejected(pds_product):
    label, data = pds_product(truncate=100)
    fails("file_size", label, data)


def test_a_data_file_with_extra_bytes_is_rejected(pds_product):
    label, data = pds_product()
    data.write_bytes(data.read_bytes() + b"\x00" * 10)
    fails("file_size", label, data)


def test_a_zero_filled_placeholder_is_rejected(pds_product):
    label, data = pds_product(elevation_m=np.zeros((200, 200)))
    fails("elevation_values", label, data)


def test_mostly_non_finite_values_are_rejected(pds_product):
    elevation = synthetic_terrain(100, 200)
    elevation[:, :140] = np.nan
    label, data = pds_product("nan_eqc", kind="eqc", lines=100, samples=200, elevation_m=elevation)
    fails("elevation_values", label, data)


def test_implausible_elevations_are_rejected(pds_product):
    elevation = synthetic_terrain(100, 200)
    elevation[10, 10] = 5.0e7  # 50,000 km above the reference sphere
    label, data = pds_product("wild_eqc", kind="eqc", lines=100, samples=200, elevation_m=elevation)
    fails("elevation_range", label, data)


def test_a_missing_data_file_is_rejected(pds_product):
    label, data = pds_product()
    data.unlink()
    fails("files", label, data)


def test_label_and_data_in_different_directories_are_rejected(pds_product, tmp_path):
    label, data = pds_product()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    moved = shutil.copy(data, elsewhere / data.name)
    fails("files", label, __import__("pathlib").Path(moved))


# ---------------------------------------------------------------------------
# Hostile or malformed labels
# ---------------------------------------------------------------------------


def write_raw_label(directory, name, text):
    label, data = write_product(directory, name)
    label.write_text(text, encoding="utf-8") if isinstance(text, str) else label.write_bytes(text)
    return label, data


def test_labels_with_entity_declarations_are_rejected(tmp_path):
    bomb = '<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>\n'
    label = pds4_label(name="ldem_75s_240m", lines=200, samples=200, doctype=bomb)
    label_path, data = write_raw_label(tmp_path, "ldem_75s_240m", label)
    fails("label_safety", label_path, data)


def test_an_external_entity_is_rejected(tmp_path):
    xxe = '<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///etc/passwd">]>\n'
    label_path, data = write_raw_label(
        tmp_path, "ldem_75s_240m", pds4_label(name="ldem_75s_240m", lines=200, samples=200, doctype=xxe)
    )
    fails("label_safety", label_path, data)


def test_oversized_labels_are_rejected(tmp_path):
    label = pds4_label(name="ldem_75s_240m", lines=200, samples=200).replace(
        "</Product_Observational>", "<!--" + "x" * 300_000 + "--></Product_Observational>"
    )
    label_path, data = write_raw_label(tmp_path, "ldem_75s_240m", label)
    fails("label_size", label_path, data)


@pytest.mark.parametrize(
    "payload, check",
    [
        ("this is not xml", "label_parse"),
        ("<a><b></a>", "label_parse"),
        (b"\xe9\xe8 not utf-8", "label_encoding"),
        ("<Other_Root/>", "label_structure"),
        ("<Product_Observational xmlns='http://pds.nasa.gov/pds4/pds/v1'/>", "label_structure"),
    ],
)
def test_malformed_labels_are_rejected(tmp_path, payload, check):
    label_path, data = write_raw_label(tmp_path, "ldem_75s_240m", payload)
    fails(check, label_path, data)


@pytest.mark.parametrize(
    "declared",
    ["other.img", "../../etc/passwd", "sub/dir.img", "..\\..\\windows\\system32", "C:\\x.img", "/etc/passwd", ".hidden.img", ""],
)
def test_the_label_may_only_name_the_file_that_was_downloaded(pds_product, declared):
    """A label naming another file could make the raster driver read an arbitrary local file."""
    label, data = pds_product(file_name=declared)
    fails("label_file", label, data)


def test_two_arrays_are_rejected(pds_product):
    label, data = pds_product(extra_array=True)
    fails("label_structure", label, data)


@pytest.mark.parametrize(
    "kwargs, check",
    [
        ({"data_type": "UnsignedByte"}, "data_type"),
        ({"data_type": "IEEE754LSBDouble"}, "data_type"),
        ({"unit": "CENTIMETER"}, "unit"),
        ({"unit": "FEET"}, "unit"),
        ({"scaling": -1.0}, "label_structure"),
        ({"scaling": 0.0}, "label_structure"),
        ({"byte_offset": 16}, "label_structure"),
        ({"lines_text": "abc"}, "label_structure"),
        ({"lines_text": "0"}, "label_structure"),
        ({"lines_text": "-5"}, "label_structure"),
        ({"lines_text": "999999999"}, "label_structure"),
    ],
)
def test_unsupported_or_invalid_label_values_are_rejected(pds_product, kwargs, check):
    label, data = pds_product(**kwargs)
    fails(check, label, data)


def test_an_unrecognised_height_convention_is_rejected_not_interpreted(pds_product):
    label, data = pds_product(offset=0.0)  # offset is not the reference radius
    fails("height_reference", label, data)


# ---------------------------------------------------------------------------
# CRS, resolution, coverage
# ---------------------------------------------------------------------------


def test_an_earth_sized_reference_sphere_is_rejected(pds_product):
    label, data = pds_product(radius_km=6378.137, offset=6378137.0)
    error = fails("crs", label, data)
    assert "not lunar" in str(error)


def test_a_product_without_georeferencing_is_rejected(tmp_path):
    label = pds4_label(name="ldem_75s_240m", lines=200, samples=200)
    start, end = label.index("<cart:Cartography>"), label.index("</cart:Cartography>") + len("</cart:Cartography>")
    label_path, data = write_raw_label(tmp_path, "ldem_75s_240m", label[:start] + label[end:])
    with pytest.raises(DemValidationError) as info:
        validate_pds4_product(label_path, data)
    assert info.value.check in ("crs", "raster_open", "georeferencing")


def test_a_pixel_size_that_contradicts_the_metadata_is_rejected(pds_product):
    label, data = pds_product()  # the label says 240 m
    error = fails("resolution", label, data, candidate=candidate(scale=100.0))
    assert "differs" in str(error)


def test_matching_metadata_resolution_is_accepted_within_tolerance(pds_product):
    label, data = pds_product()
    validate_pds4_product(label, data, candidate=candidate(scale=241.0))


def test_an_area_outside_the_raster_is_rejected(pds_product):
    label, data = pds_product()  # covers about 24 km around the pole
    error = fails("coverage", label, data, request=CoverageRequest.from_point(-80.0, 0.0, 3000.0))
    assert "outside the raster" in str(error)


def test_an_area_partly_outside_the_raster_is_rejected(pds_product):
    lat = -90.0 + math.degrees(2 * math.atan(20_000.0 / (2 * R_M)))  # 20 km from the pole, toward lon 90
    label, data = pds_product()
    fails("coverage", label, data, request=CoverageRequest.from_point(lat, 90.0, 8000.0))


def test_an_area_near_the_pole_on_a_cylindrical_product_is_rejected(pds_product):
    label, data = pds_product("ldem_16_float", kind="eqc", lines=100, samples=200)
    fails("projection", label, data, request=CoverageRequest.from_bbox(88.0, 89.0, 12.0, 14.0))


# ---------------------------------------------------------------------------
# Label parsing details
# ---------------------------------------------------------------------------


def test_the_parsed_label_reports_the_conventions(pds_product):
    label, data = pds_product()
    parsed = parse_label(label, data.name)
    assert (parsed.lines, parsed.samples, parsed.dtype, parsed.itemsize) == (200, 200, "int16", 2)
    assert parsed.convention.unit == "METER" and parsed.convention.scale_factor == 0.5
    assert parsed.data_file_name == data.name


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalisation_applies_the_scale_for_integers_and_kilometres_for_floats(pds_product, tmp_path):
    label, data = pds_product()
    _report, convention = validate_pds4_product(label, data, request=POLE)
    result = normalize_to_geotiff(label, tmp_path / "polar.tif", convention)
    with rasterio.open(result.path) as tif:
        np.testing.assert_allclose(tif.read(1), np.round(synthetic_terrain(200, 200) / 0.5) * 0.5, atol=1e-3)
    assert result.masked_cells == 0 and result.total_cells == 40_000

    label, data = pds_product("eqc", kind="eqc", lines=100, samples=200)
    _report, convention = validate_pds4_product(label, data, request=EQC_AREA)
    result = normalize_to_geotiff(label, tmp_path / "eqc.tif", convention)
    with rasterio.open(result.path) as tif:
        np.testing.assert_allclose(tif.read(1), synthetic_terrain(100, 200), atol=0.05)


def test_invalid_values_become_nodata_and_are_counted(pds_product, tmp_path):
    elevation = synthetic_terrain(100, 200)
    elevation[5, 5] = 50_000.0  # beyond the plausible range
    elevation[6, 6] = np.nan
    label, _data = pds_product("bad_cells", kind="eqc", lines=100, samples=200, elevation_m=elevation)
    convention = HeightConvention("KILOMETER", 1000.0, 1.0, 1737.4, 1_737_400.0, "float32")
    result = normalize_to_geotiff(label, tmp_path / "out.tif", convention)
    assert result.masked_cells == 2
    with rasterio.open(result.path) as tif:
        values = tif.read(1)
        assert values[5, 5] == NODATA_M and values[6, 6] == NODATA_M and tif.nodata == NODATA_M


def test_normalisation_does_not_depend_on_the_block_size(pds_product, tmp_path):
    label, data = pds_product()
    _report, convention = validate_pds4_product(label, data)
    a = normalize_to_geotiff(label, tmp_path / "a.tif", convention)
    b = normalize_to_geotiff(label, tmp_path / "b.tif", convention, block_rows=7)
    with rasterio.open(a.path) as fa, rasterio.open(b.path) as fb:
        np.testing.assert_array_equal(fa.read(1), fb.read(1))


def test_normalisation_keeps_the_source_georeferencing(pds_product, tmp_path):
    label, data = pds_product()
    _report, convention = validate_pds4_product(label, data)
    result = normalize_to_geotiff(label, tmp_path / "a.tif", convention, tags={"TALUS_SOURCE_PRODUCT_ID": "ldem_test"})
    with rasterio.open(label) as src, rasterio.open(result.path) as out:
        assert src.crs == out.crs and tuple(src.transform)[:6] == tuple(out.transform)[:6]
        assert out.tags()["TALUS_SOURCE_PRODUCT_ID"] == "ldem_test"
        assert out.tags()["TALUS_HEIGHT_REFERENCE_RADIUS_M"] == "1.7374e+06"


def test_a_product_with_no_valid_cell_is_refused_and_leaves_no_file(pds_product, tmp_path):
    label, _data = pds_product("all_bad", kind="eqc", lines=100, samples=200, elevation_m=np.full((100, 200), np.nan))
    convention = HeightConvention("KILOMETER", 1000.0, 1.0, 1737.4, 1_737_400.0, "float32")
    with pytest.raises(DemValidationError):
        normalize_to_geotiff(label, tmp_path / "out.tif", convention)
    assert not (tmp_path / "out.tif").exists()


def test_insufficient_disk_space_is_reported(pds_product, tmp_path, monkeypatch):
    label, data = pds_product()
    _report, convention = validate_pds4_product(label, data)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(DemValidationError) as info:
        normalize_to_geotiff(label, tmp_path / "out.tif", convention)
    assert info.value.check == "disk_space"


def test_tampering_with_the_converted_file_is_detected(pds_product, tmp_path):
    label, data = pds_product()
    _report, convention = validate_pds4_product(label, data)
    result = normalize_to_geotiff(label, tmp_path / "a.tif", convention)
    verify_normalized(label, result.path, convention)
    with rasterio.open(result.path, "r+") as tif:
        block = tif.read(1)
        block[::3, ::3] += 500.0
        tif.write(block, 1)
    with pytest.raises(DemValidationError) as info:
        verify_normalized(label, result.path, convention)
    assert info.value.check == "normalized_output"


# ---------------------------------------------------------------------------
# The engine reads the converted values correctly (guards against ignoring scale and unit)
# ---------------------------------------------------------------------------


def test_a_ten_degree_slope_in_a_polar_product_is_measured_as_ten_degrees(pds_product, tmp_path):
    """Raw integers are half-metres. Ignoring the 0.5 scale would read 5 degrees, not 10."""
    columns = np.arange(200) * 240.0 * math.tan(math.radians(10.0))
    elevation = np.tile(columns, (200, 1)) - 4000.0
    label, data = pds_product(elevation_m=elevation)
    _report, convention = validate_pds4_product(label, data)
    dem = normalize_to_geotiff(label, tmp_path / "ramp.tif", convention).path

    ctx = open_dem_context(dem)
    lats, lons = ctx.to_latlon([-8000.0, 8000.0], [1000.0, 1000.0])
    result = check_rover_safety(list(zip(lats.tolist(), lons.tolist())), 15.0, dem_path=dem)
    assert result.coverage_fraction == 1.0
    assert result.max_slope_deg == pytest.approx(10.0, abs=0.3)
    assert result.status is SafetyStatus.PASS


def test_a_three_degree_slope_in_a_kilometre_product_is_measured_as_three_degrees(pds_product, tmp_path):
    """Raw floats are kilometres. Ignoring the unit would read a slope 1000 times too small."""
    dx = 1895.21 * math.cos(math.radians(0.5))
    elevation = np.tile(np.arange(200) * dx * math.tan(math.radians(3.0)), (100, 1))
    label, data = pds_product("ramp_eqc", kind="eqc", lines=100, samples=200, elevation_m=elevation)
    _report, convention = validate_pds4_product(label, data)
    dem = normalize_to_geotiff(label, tmp_path / "ramp_eqc.tif", convention).path

    result = check_rover_safety([(0.5, 13.0), (0.5, 15.0)], 15.0, dem_path=dem)
    assert result.coverage_fraction == 1.0
    assert result.max_slope_deg == pytest.approx(3.0, abs=0.15)
