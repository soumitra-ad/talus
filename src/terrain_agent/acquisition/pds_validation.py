"""Validation of a downloaded PDS4 gridded product before it may enter the cache.

The label and the data file are untrusted. Validation runs in a fixed order and stops at the
first failure with a :class:`DemValidationError` that names the failed check.

What is accepted
----------------
A PDS4 label with one ``Array_2D_Image`` and a raw binary data file, as used by the NASA LOLA
gridded products (GDRDEM) and SLDEM2015:

* data types ``SignedLSB2`` (16-bit integer) and ``IEEE754LSBSingle`` (32-bit float)
* height units METER or KILOMETER
* heights defined by the label as ``DN * scaling_factor`` relative to a reference sphere whose
  radius is the label ``value_offset``. This is the convention stated in the product labels.
  A product using any other convention is rejected, not interpreted.

What is checked
---------------
label safety and structure, declared data file name, exact data file size, GDAL opening the
label with the PDS4 driver, dimensions and data type agreeing with the label, an explicit CRS with
a lunar reference sphere (never assumed to be WGS84), a supported projection, resolution
agreeing with provider metadata, coverage of the requested area, and readable, plausible,
non-constant elevation values including a full-resolution read at the requested location.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from pydantic import BaseModel
from rasterio.windows import Window

from terrain_agent.acquisition.errors import DemValidationError
from terrain_agent.acquisition.models import CoverageRequest, ProductCandidate
from terrain_agent.terrain.coordinates import LUNAR_RADIUS_METERS
from terrain_agent.terrain.georef import DemGeoContext, _ellipsoid_from_crs, open_dem_context
from terrain_agent.terrain.resource_safety import TerrainAnalysisError
from terrain_agent.terrain.terrain_grids import ELEVATION_SENTINEL_LIMIT_M

MAX_LABEL_BYTES = 256 * 1024
MAX_DIMENSION = 200_000
RADIUS_TOLERANCE = 0.01
RESOLUTION_TOLERANCE = 0.02

_DATA_TYPES = {"SignedLSB2": ("int16", 2), "IEEE754LSBSingle": ("float32", 4)}
_UNIT_TO_METRES = {"METER": 1.0, "METERS": 1.0, "KILOMETER": 1000.0, "KILOMETERS": 1000.0}


@dataclass(frozen=True)
class HeightConvention:
    """How raw values become elevation in metres, as stated by the product label."""

    unit: str
    unit_factor_m: float
    scale_factor: float
    value_offset: float
    reference_radius_m: float
    source_dtype: str

    @property
    def factor_to_metres(self) -> float:
        return self.scale_factor * self.unit_factor_m


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ValidationReport(BaseModel):
    """Outcome of validation. Only reports with ``passed`` true are ever produced."""

    passed: bool
    checks: list[ValidationCheck]
    warnings: list[str]
    width: int
    height: int
    source_driver: str
    source_dtype: str
    crs_proj: str
    crs_kind: str
    projection: Optional[str]
    native_pixel_size: tuple[float, float]
    native_units: str
    pixel_size_m: Optional[tuple[float, float]]
    bounds_native: tuple[float, float, float, float]
    data_bytes: int
    unit: str
    scaling_factor: float
    value_offset: float
    reference_radius_m: float
    sample_min_m: float
    sample_max_m: float
    finite_fraction: float
    request_coverage_checked: bool


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    return next((c for c in parent if _local(c.tag) == name), None)


def _text(parent: ET.Element, name: str) -> Optional[str]:
    node = _child(parent, name)
    return node.text.strip() if node is not None and node.text else None


def _number(value: Optional[str], what: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise DemValidationError("label_structure", f"{what} is missing or not a number.") from None
    if not math.isfinite(number):
        raise DemValidationError("label_structure", f"{what} is not finite.")
    return number


def _integer(value: Optional[str], what: str, low: int, high: int) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise DemValidationError("label_structure", f"{what} is missing or not an integer.") from None
    if not low <= number <= high:
        raise DemValidationError("label_structure", f"{what} is outside the supported range.")
    return number


@dataclass(frozen=True)
class ParsedLabel:
    lines: int
    samples: int
    dtype: str
    itemsize: int
    data_offset: int
    convention: HeightConvention
    data_file_name: str


def parse_label(label_path: Path, data_file_name: str) -> ParsedLabel:
    """Parse a PDS4 label defensively.

    Raises
    ------
    DemValidationError
        For an oversized label, XML with a DTD or entities, an unexpected structure, an
        unsupported data type or unit, or a declared data file that is not ``data_file_name``.
    """
    raw = label_path.read_bytes()
    if len(raw) > MAX_LABEL_BYTES:
        raise DemValidationError("label_size", "The label is larger than the allowed size.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise DemValidationError("label_encoding", "The label is not valid UTF-8.") from None
    lowered = text.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise DemValidationError("label_safety", "The label contains a DTD or entity declaration.")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise DemValidationError("label_parse", "The label is not well-formed XML.") from None
    if _local(root.tag) != "Product_Observational":
        raise DemValidationError("label_structure", "The label is not a PDS4 observational product.")

    areas = [e for e in root if _local(e.tag) == "File_Area_Observational"]
    if len(areas) != 1:
        raise DemValidationError(
            "label_structure", "Exactly one observational file area is required."
        )
    area = areas[0]
    file_node = _child(area, "File")
    declared = _text(file_node, "file_name") if file_node is not None else None
    if not declared or "/" in declared or "\\" in declared or declared.startswith("."):
        raise DemValidationError("label_file", "The label declares no usable data file name.")
    if declared.lower() != data_file_name.lower():
        raise DemValidationError(
            "label_file", "The data file named in the label is not the file that was downloaded."
        )

    arrays = [e for e in area if _local(e.tag) == "Array_2D_Image"]
    if len(arrays) != 1:
        raise DemValidationError("label_structure", "Exactly one Array_2D_Image is required.")
    array = arrays[0]
    if _integer(_text(array, "axes"), "axes", 2, 2) != 2:
        raise DemValidationError("label_structure", "The array must have two axes.")
    if _text(array, "axis_index_order") != "Last Index Fastest":
        raise DemValidationError("label_structure", "Unsupported axis index order.")
    offset_node = _child(array, "offset")
    data_offset = _integer(offset_node.text if offset_node is not None else None, "offset", 0, 0)

    element = _child(array, "Element_Array")
    if element is None:
        raise DemValidationError("label_structure", "The array has no Element_Array.")
    data_type = _text(element, "data_type") or ""
    if data_type not in _DATA_TYPES:
        raise DemValidationError("data_type", f"Unsupported data type {data_type[:30]!r}.")
    dtype, itemsize = _DATA_TYPES[data_type]
    unit = (_text(element, "unit") or "").upper()
    if unit not in _UNIT_TO_METRES:
        raise DemValidationError("unit", f"Unsupported height unit {unit[:30]!r}.")
    scale = _number(_text(element, "scaling_factor"), "scaling_factor")
    offset = _number(_text(element, "value_offset"), "value_offset")
    if scale <= 0.0:
        raise DemValidationError("label_structure", "scaling_factor must be positive.")

    axes = {}
    for axis in (e for e in array if _local(e.tag) == "Axis_Array"):
        name = (_text(axis, "axis_name") or "").lower()
        axes[name] = _integer(_text(axis, "elements"), f"{name} elements", 1, MAX_DIMENSION)
    if set(axes) != {"line", "sample"}:
        raise DemValidationError("label_structure", "The array needs one Line and one Sample axis.")

    unit_factor = _UNIT_TO_METRES[unit]
    convention = HeightConvention(
        unit=unit,
        unit_factor_m=unit_factor,
        scale_factor=scale,
        value_offset=offset,
        reference_radius_m=offset * unit_factor,
        source_dtype=dtype,
    )
    return ParsedLabel(axes["line"], axes["sample"], dtype, itemsize, data_offset, convention, declared)


def _proj_text(crs: Any) -> str:
    try:
        params = crs.to_dict()
    except Exception:  # noqa: BLE001
        return "unknown"
    return " ".join(f"{k}={v}" for k, v in sorted(params.items()))[:300]


def validate_pds4_product(
    label_path: Path,
    data_path: Path,
    *,
    candidate: Optional[ProductCandidate] = None,
    request: Optional[CoverageRequest] = None,
) -> tuple[ValidationReport, HeightConvention]:
    """Validate a downloaded label and data file. Returns the report and the height convention."""
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    def ok(name: str, detail: str = "") -> None:
        checks.append(ValidationCheck(name=name, passed=True, detail=detail))

    if not label_path.is_file() or not data_path.is_file():
        raise DemValidationError("files", "The label or data file is missing.")
    if label_path.parent != data_path.parent:
        raise DemValidationError("files", "The label and data file must be in the same directory.")
    ok("files")

    parsed = parse_label(label_path, data_path.name)
    ok("label", f"{parsed.samples} x {parsed.lines}, {parsed.dtype}, unit {parsed.convention.unit}")

    expected_bytes = parsed.data_offset + parsed.lines * parsed.samples * parsed.itemsize
    actual_bytes = data_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise DemValidationError(
            "file_size",
            f"The data file has {actual_bytes} bytes but the label describes {expected_bytes}. "
            "The file is incomplete or corrupted.",
        )
    ok("file_size", f"{actual_bytes} bytes")

    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        try:
            src = rasterio.open(label_path)
        except (rasterio.errors.RasterioError, OSError):
            raise DemValidationError("raster_open", "GDAL could not open the product label.") from None
        with src:
            if src.driver != "PDS4":
                raise DemValidationError("raster_open", f"Unexpected raster driver {src.driver!r}.")
            if src.count != 1 or src.width != parsed.samples or src.height != parsed.lines:
                raise DemValidationError("raster_open", "Raster dimensions disagree with the label.")
            if src.dtypes[0] != parsed.dtype:
                raise DemValidationError("raster_open", "Raster data type disagrees with the label.")
            gdal_scale = src.scales[0] if src.scales else 1.0
            gdal_offset = src.offsets[0] if src.offsets else 0.0
            if not (
                math.isclose(gdal_scale, parsed.convention.scale_factor, rel_tol=1e-9)
                and math.isclose(gdal_offset, parsed.convention.value_offset, rel_tol=1e-9, abs_tol=1e-9)
            ):
                raise DemValidationError("raster_open", "GDAL scale and offset disagree with the label.")
            ok("raster_open", f"driver {src.driver}")

            crs = src.crs
            if crs is None:
                raise DemValidationError("crs", "The product declares no coordinate reference system.")
            semi_major, semi_minor = _ellipsoid_from_crs(crs)
            if semi_major is None:
                raise DemValidationError("crs", "The CRS declares no reference sphere or ellipsoid.")
            if abs(semi_major - LUNAR_RADIUS_METERS) / LUNAR_RADIUS_METERS > RADIUS_TOLERANCE:
                raise DemValidationError(
                    "crs", "The CRS reference radius is not lunar. It is not assumed to be WGS84 or any Earth datum."
                )
            if semi_minor is not None and abs(semi_major - semi_minor) > 1e-6 * semi_major:
                warnings.append("The CRS uses a non-spherical ellipsoid; metric cell size assumes a sphere.")
            if not (crs.is_projected or crs.is_geographic):
                raise DemValidationError("crs", "The CRS is neither projected nor geographic.")
            ok("crs", _proj_text(crs))

            if abs(parsed.convention.reference_radius_m - semi_major) > 1.0:
                raise DemValidationError(
                    "height_reference",
                    "The label height offset is not the CRS reference radius, so the height "
                    "convention is not the supported one.",
                )
            ok("height_convention", "height = value * scaling_factor above the reference sphere")

            try:
                ctx: DemGeoContext = open_dem_context(label_path)
            except TerrainAnalysisError as exc:
                raise DemValidationError("georeferencing", str(exc)) from None
            warnings.extend(ctx.warnings)
            res_x, res_y = abs(ctx.metadata.res_x), abs(ctx.metadata.res_y)
            if not (math.isfinite(res_x) and math.isfinite(res_y) and res_x > 0 and res_y > 0):
                raise DemValidationError("georeferencing", "The pixel size is invalid.")
            left, bottom, right, top = ctx.metadata.bounds
            if not all(math.isfinite(v) for v in (left, bottom, right, top)) or right <= left or top <= bottom:
                raise DemValidationError("georeferencing", "The raster bounds are invalid.")
            native_units = "degree" if ctx.is_geographic else "metre"
            if ctx.is_geographic:
                warnings.append("The product is in geographic coordinates.")
            elif abs(ctx.unit_factor - 1.0) > 1e-9:
                warnings.append("The CRS length unit is not the metre.")
            ok("georeferencing", f"pixel {res_x:g} x {res_y:g} {native_units}")

            if candidate is not None and candidate.map_scale_m and not ctx.is_geographic:
                stated = res_y * ctx.unit_factor
                if abs(stated - candidate.map_scale_m) / candidate.map_scale_m > RESOLUTION_TOLERANCE:
                    raise DemValidationError(
                        "resolution",
                        "The pixel size in the label differs from the resolution in the product metadata.",
                    )
                ok("resolution", f"{stated:g} m per pixel matches metadata")

            centre_lat, centre_lon = (
                (request.center_lat, request.center_lon) if request else _raster_centre(ctx)
            )
            pixel_size_m: Optional[tuple[float, float]] = None
            try:
                pixel_size_m = ctx.pixel_size_m(centre_lat, centre_lon)
            except TerrainAnalysisError as exc:
                raise DemValidationError("projection", str(exc)) from None
            ok("projection", f"metric pixel size {pixel_size_m[0]:.2f} x {pixel_size_m[1]:.2f} m")

            coverage_checked = False
            if request is not None:
                points = request.sample_points()
                lats = [p[0] for p in points]
                lons = [p[1] for p in points]
                row_f, col_f, finite = ctx.latlon_to_rowcol_float(lats, lons)
                inside = (
                    finite
                    & (row_f >= 0)
                    & (row_f < ctx.height)
                    & (col_f >= 0)
                    & (col_f < ctx.width)
                )
                if not bool(inside.all()):
                    raise DemValidationError(
                        "coverage",
                        f"{int((~inside).sum())} of {len(points)} points of the requested area "
                        "lie outside the raster.",
                    )
                coverage_checked = True
                ok("coverage", f"{len(points)} points of the requested area are inside the raster")

            stats = _check_values(src, parsed.convention, ctx, request, warnings)
            ok("elevation_values", f"{stats['min']:.1f} to {stats['max']:.1f} m")

            report = ValidationReport(
                passed=True,
                checks=checks,
                warnings=warnings,
                width=src.width,
                height=src.height,
                source_driver=src.driver,
                source_dtype=parsed.dtype,
                crs_proj=_proj_text(crs),
                crs_kind="geographic" if ctx.is_geographic else "projected",
                projection=ctx.projection,
                native_pixel_size=(res_x, res_y),
                native_units=native_units,
                pixel_size_m=pixel_size_m,
                bounds_native=(left, bottom, right, top),
                data_bytes=actual_bytes,
                unit=parsed.convention.unit,
                scaling_factor=parsed.convention.scale_factor,
                value_offset=parsed.convention.value_offset,
                reference_radius_m=parsed.convention.reference_radius_m,
                sample_min_m=stats["min"],
                sample_max_m=stats["max"],
                finite_fraction=stats["finite_fraction"],
                request_coverage_checked=coverage_checked,
            )
    return report, parsed.convention


def _raster_centre(ctx: DemGeoContext) -> tuple[float, float]:
    left, bottom, right, top = ctx.metadata.bounds
    lat, lon = ctx.to_latlon([0.5 * (left + right)], [0.5 * (bottom + top)])
    return float(lat[0]), float(lon[0])


def _check_values(
    src: Any,
    convention: HeightConvention,
    ctx: DemGeoContext,
    request: Optional[CoverageRequest],
    warnings: list[str],
) -> dict[str, float]:
    """Read a decimated overview, the last rows, and a full-resolution window; judge them."""
    height, width = src.height, src.width
    try:
        overview = src.read(1, out_shape=(min(height, 512), min(width, 512)))
        tail = src.read(1, window=Window(0, max(0, height - 4), width, min(4, height)))
        if request is not None:
            row_f, col_f, finite = ctx.latlon_to_rowcol_float([request.center_lat], [request.center_lon])
            row = int(np.clip(row_f[0] if finite[0] else height / 2, 0, height - 1))
            col = int(np.clip(col_f[0] if finite[0] else width / 2, 0, width - 1))
            half = 32
            window = Window(max(0, col - half), max(0, row - half), min(width, 2 * half), min(height, 2 * half))
            src.read(1, window=window)
    except (rasterio.errors.RasterioError, OSError):
        raise DemValidationError("elevation_values", "Elevation values could not be read.") from None

    factor = convention.factor_to_metres
    metres = overview.astype(np.float64) * factor
    finite_mask = np.isfinite(metres)
    finite_fraction = float(finite_mask.mean())
    if finite_fraction < 0.5:
        raise DemValidationError("elevation_values", "Most elevation values are not finite.")
    if finite_fraction < 0.999:
        warnings.append(f"{100 * (1 - finite_fraction):.2f}% of sampled elevation values are not finite.")
    values = metres[finite_mask]
    if float(np.max(np.abs(values))) >= ELEVATION_SENTINEL_LIMIT_M:
        raise DemValidationError(
            "elevation_range", "Elevation values fall outside the plausible range for the Moon."
        )
    if height > 16 and width > 16 and float(np.ptp(values)) == 0.0:
        raise DemValidationError(
            "elevation_values",
            "All sampled elevation values are identical, which indicates an empty or corrupted file.",
        )
    if height > 16 and width > 16 and tail.size and float(np.ptp(tail.astype(np.float64))) == 0.0 and float(np.ptp(values)) < 1e-9:
        raise DemValidationError("elevation_values", "The end of the file contains no data.")
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "finite_fraction": finite_fraction,
    }
