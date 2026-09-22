"""Dataset provenance and header information for a DEM file.

Provenance comes from the JSON sidecar written by the secure downloader, when present.
The sidecar is treated as untrusted text. Only a fixed set of fields is read, each is
checked against a strict pattern, and anything that does not match is dropped with a
warning. Nothing is inferred or invented: a DEM without a sidecar is reported as having
unknown provenance.

The recorded SHA-256 is reported as recorded. It is not re-verified here.

Text that passes validation is still untrusted data. A pattern cannot judge intent, so
free-text fields are short and limited to plain characters, and they can never change a
safety status: statuses are computed only from measured terrain and configured thresholds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel

from terrain_agent.terrain.georef import DemGeoContext
from terrain_agent.terrain.resource_safety import UnsupportedCRSError

_MAX_SIDECAR_BYTES = 64 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,100}$")
_TEXT_RE = re.compile(r"^[A-Za-z0-9 ._,:()/+\-]{1,60}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TIME_RE = re.compile(r"^[0-9T:.+\-Z]{10,40}$")
_CACHE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]{0,100}-[0-9a-f]{12}$")
_LID_RE = re.compile(r"^urn:[A-Za-z0-9._:\-]{1,200}$")
_TYPE_RE = re.compile(r"^[A-Z0-9]{1,20}$")


class DatasetInfo(BaseModel):
    """Facts about the DEM used for an analysis."""

    file_name: str
    provenance: Literal["sidecar", "none"]
    product_id: Optional[str] = None
    dataset: Optional[str] = None
    instrument: Optional[str] = None
    mission: Optional[str] = None
    source_url: Optional[str] = None
    sha256_recorded: Optional[str] = None
    downloaded_at: Optional[str] = None
    driver: str
    width: int
    height: int
    crs_kind: Literal["projected", "geographic"]
    crs_epsg: Optional[int] = None
    native_pixel_size: tuple[float, float]
    pixel_size_m: Optional[tuple[float, float]] = None
    resolution_m: Optional[float] = None
    bounds_native: tuple[float, float, float, float]
    nodata_value: Optional[float] = None
    provider_id: Optional[str] = None
    product_type: Optional[str] = None
    product_lid: Optional[str] = None
    data_set_id: Optional[str] = None
    cache_id: Optional[str] = None
    acquired_at: Optional[str] = None
    elevation_reference: Optional[str] = None
    warnings: list[str] = []


def _read_sidecar(dem_path: Path) -> tuple[dict[str, str], list[str]]:
    sidecar = dem_path.with_suffix(".json")
    warnings: list[str] = []
    if not sidecar.is_file():
        return {}, warnings
    try:
        if sidecar.stat().st_size > _MAX_SIDECAR_BYTES:
            return {}, ["Provenance sidecar is larger than the allowed size and was ignored."]
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, ["Provenance sidecar could not be read and was ignored."]
    if not isinstance(raw, dict):
        return {}, ["Provenance sidecar has an unexpected structure and was ignored."]

    from terrain_agent.tools.dem_downloader import ALLOWED_HOSTS

    def accept(key: str, pattern: re.Pattern[str]) -> Optional[str]:
        value = raw.get(key)
        if value is None:
            return None
        if isinstance(value, str) and pattern.fullmatch(value):
            return value
        warnings.append(f"Provenance field {key!r} did not pass validation and was ignored.")
        return None

    clean: dict[str, str] = {}
    for key, pattern in (
        ("product_id", _ID_RE),
        ("dataset", _TEXT_RE),
        ("instrument", _TEXT_RE),
        ("mission", _TEXT_RE),
        ("sha256", _SHA_RE),
        ("downloaded_at", _TIME_RE),
    ):
        accepted = accept(key, pattern)
        if accepted is not None:
            clean[key] = accepted

    url = raw.get("source_url")
    if url is not None:
        parsed = urlparse(url) if isinstance(url, str) and len(url) <= 500 else None
        if parsed and parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS:
            clean["source_url"] = url
        else:
            warnings.append("Provenance field 'source_url' did not pass validation and was ignored.")

    block = raw.get("nasa_provenance")
    if isinstance(block, dict):
        for key, pattern in (
            ("provider_id", _ID_RE),
            ("product_type", _TYPE_RE),
            ("product_lid", _LID_RE),
            ("data_set_id", _ID_RE),
            ("cache_id", _CACHE_ID_RE),
            ("acquired_at", _TIME_RE),
        ):
            value = block.get(key)
            if isinstance(value, str) and pattern.fullmatch(value):
                clean["nasa_" + key] = value
        height = block.get("height")
        radius = height.get("reference_radius_m") if isinstance(height, dict) else None
        if isinstance(radius, (int, float)) and not isinstance(radius, bool) and 1.0e6 < radius < 2.0e6:
            clean["nasa_reference_radius_m"] = f"{float(radius):g}"
    return clean, warnings


def build_dataset_info(ctx: DemGeoContext, ref_lat: float, ref_lon: float) -> DatasetInfo:
    """Describe the DEM, with metric resolution evaluated at a reference location."""
    meta = ctx.metadata
    sidecar, warnings = _read_sidecar(ctx.path)
    warnings = list(warnings) + list(ctx.warnings)
    if not sidecar:
        warnings.append(
            "No provenance sidecar found; dataset name, source and checksum are unknown. "
            "Do not present this DEM as a specific NASA product."
        )

    pixel_size_m: Optional[tuple[float, float]] = None
    resolution_m: Optional[float] = None
    try:
        pixel_size_m = ctx.pixel_size_m(ref_lat, ref_lon)
        resolution_m = 0.5 * (pixel_size_m[0] + pixel_size_m[1])
    except UnsupportedCRSError as exc:
        warnings.append(str(exc))

    return DatasetInfo(
        file_name=ctx.path.name,
        provenance="sidecar" if sidecar else "none",
        product_id=sidecar.get("product_id"),
        dataset=sidecar.get("dataset"),
        instrument=sidecar.get("instrument"),
        mission=sidecar.get("mission"),
        source_url=sidecar.get("source_url"),
        sha256_recorded=sidecar.get("sha256"),
        downloaded_at=sidecar.get("downloaded_at"),
        driver=meta.driver,
        width=meta.width,
        height=meta.height,
        crs_kind="geographic" if ctx.is_geographic else "projected",
        crs_epsg=meta.crs_epsg,
        native_pixel_size=(abs(meta.res_x), abs(meta.res_y)),
        pixel_size_m=pixel_size_m,
        resolution_m=resolution_m,
        bounds_native=meta.bounds,
        nodata_value=meta.nodata_value,
        provider_id=sidecar.get("nasa_provider_id"),
        product_type=sidecar.get("nasa_product_type"),
        product_lid=sidecar.get("nasa_product_lid"),
        data_set_id=sidecar.get("nasa_data_set_id"),
        cache_id=sidecar.get("nasa_cache_id"),
        acquired_at=sidecar.get("nasa_acquired_at"),
        elevation_reference=(
            f"height above a reference sphere of radius {sidecar['nasa_reference_radius_m']} m, not a geoid"
            if "nasa_reference_radius_m" in sidecar
            else None
        ),
        warnings=warnings,
    )
