"""Structured provenance for a NASA DEM in the local cache.

The provenance file sits next to the cached GeoTIFF as ``<cache_id>.json``. Its top level holds
the small set of fields the analysis layer already reads from download sidecars, and the
``nasa_provenance`` block holds the full structured record.

Provenance describes where the data came from and how it was processed. It is display data. It
never changes an analysis result or a safety status: those are computed only from measured
terrain and configured thresholds.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel

from terrain_agent import __version__
from terrain_agent.acquisition.models import CoverageRequest, FileRole, ProductCandidate
from terrain_agent.acquisition.normalize import NORMALIZATION_VERSION, NormalizedDem
from terrain_agent.acquisition.ode_provider import (
    PROVIDER_ID,
    PROVIDER_NAME,
    SUPPORTED_PRODUCTS,
)
from terrain_agent.acquisition.pds_validation import HeightConvention, ValidationReport

SCHEMA_VERSION = 1

DISCLAIMER = (
    "Elevations are heights above a reference sphere, not above a geoid. This DEM is a research "
    "input. Results computed from it are not certified for flight or mission use."
)

CHECKSUM_NOTE = (
    "ODE provides no checksum for product files. The SHA-256 was computed locally on receipt "
    "and is recorded so later corruption can be detected. It was not verified against a NASA value."
)


class DiscoveryInfo(BaseModel):
    provider_endpoint: str
    documentation: str
    queries: list[dict[str, str]]
    queried_at: str
    coverage_request: dict[str, Any]


class SourceInfo(BaseModel):
    data_url: str
    label_url: str
    data_bytes_expected_max: Optional[int]
    data_bytes_received: int
    data_sha256: str
    label_sha256: str
    checksum_status: Literal["computed_locally_not_verified_against_nasa", "verified_against_expected"]
    checksum_note: str = CHECKSUM_NOTE


class RasterInfo(BaseModel):
    source_driver: str
    source_dtype: str
    crs_proj: str
    crs_kind: str
    projection: Optional[str]
    native_pixel_size: tuple[float, float]
    native_units: str
    pixel_size_m: Optional[tuple[float, float]]
    width: int
    height: int
    bounds_native: tuple[float, float, float, float]
    product_bounds_lat: tuple[float, float]
    product_bounds_lon_east: tuple[float, float]


class HeightInfo(BaseModel):
    unit: str
    scaling_factor: float
    value_offset: float
    reference_radius_m: float
    statement: str = "elevation = raw value * scaling factor above a reference sphere, not a geoid"


class NormalizationInfo(BaseModel):
    version: str
    output_dtype: str = "float32"
    elevation_units: str = "metre"
    output_bytes: int
    output_sha256: str
    total_cells: int
    masked_cells: int
    elevation_min_m: float
    elevation_max_m: float


class ValidationSummary(BaseModel):
    passed: bool
    checks: list[str]
    warnings: list[str]


class NasaDemProvenance(BaseModel):
    """Complete record of a cached NASA DEM."""

    schema_version: int = SCHEMA_VERSION
    provider_id: str
    provider: str
    mission: str
    instrument: str
    product_type: str
    product_type_name: str
    data_set_id: Optional[str]
    product_id: str
    product_lid: Optional[str]
    product_version: Optional[str]
    acquired_at: str
    cache_id: str
    discovery: DiscoveryInfo
    source: SourceInfo
    raster: RasterInfo
    height: HeightInfo
    normalization: NormalizationInfo
    validation: ValidationSummary
    talus_version: str = __version__
    disclaimer: str = DISCLAIMER


def utc_now_text(now: Optional[datetime] = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_provenance(
    *,
    candidate: ProductCandidate,
    request: CoverageRequest,
    discovery: dict[str, Any],
    cache_id: str,
    acquired_at: str,
    data_sha256: str,
    data_bytes: int,
    label_sha256: str,
    report: ValidationReport,
    convention: HeightConvention,
    normalized: NormalizedDem,
) -> NasaDemProvenance:
    data_file = candidate.file(FileRole.DATA)
    label_file = candidate.file(FileRole.LABEL_PDS4)
    assert data_file is not None and label_file is not None
    name = SUPPORTED_PRODUCTS[(candidate.host_id, candidate.instrument_id, candidate.product_type)]
    return NasaDemProvenance(
        provider_id=PROVIDER_ID,
        provider=PROVIDER_NAME,
        mission=candidate.host_id,
        instrument=candidate.instrument_id,
        product_type=candidate.product_type,
        product_type_name=name,
        data_set_id=candidate.data_set_id,
        product_id=candidate.product_id,
        product_lid=candidate.product_lid,
        product_version=candidate.version,
        acquired_at=acquired_at,
        cache_id=cache_id,
        discovery=DiscoveryInfo(
            provider_endpoint=str(discovery.get("endpoint", "")),
            documentation=str(discovery.get("documentation", "")),
            queries=[dict(q) for q in discovery.get("queries", [])],  # type: ignore[union-attr]
            queried_at=acquired_at,
            coverage_request=request.model_dump(),
        ),
        source=SourceInfo(
            data_url=data_file.url,
            label_url=label_file.url,
            data_bytes_expected_max=data_file.expected_max_bytes,
            data_bytes_received=data_bytes,
            data_sha256=data_sha256,
            label_sha256=label_sha256,
            checksum_status="computed_locally_not_verified_against_nasa",
        ),
        raster=RasterInfo(
            source_driver=report.source_driver,
            source_dtype=report.source_dtype,
            crs_proj=report.crs_proj,
            crs_kind=report.crs_kind,
            projection=report.projection,
            native_pixel_size=report.native_pixel_size,
            native_units=report.native_units,
            pixel_size_m=report.pixel_size_m,
            width=report.width,
            height=report.height,
            bounds_native=report.bounds_native,
            product_bounds_lat=(candidate.min_lat, candidate.max_lat),
            product_bounds_lon_east=(candidate.west_lon, candidate.east_lon),
        ),
        height=HeightInfo(
            unit=convention.unit,
            scaling_factor=convention.scale_factor,
            value_offset=convention.value_offset,
            reference_radius_m=convention.reference_radius_m,
        ),
        normalization=NormalizationInfo(
            version=NORMALIZATION_VERSION,
            output_bytes=normalized.size_bytes,
            output_sha256=normalized.sha256,
            total_cells=normalized.total_cells,
            masked_cells=normalized.masked_cells,
            elevation_min_m=normalized.min_m,
            elevation_max_m=normalized.max_m,
        ),
        validation=ValidationSummary(
            passed=report.passed,
            checks=[c.name for c in report.checks if c.passed],
            warnings=list(report.warnings),
        ),
    )


_SAFE_TEXT = re.compile(r"[^A-Za-z0-9 ._,:()/+\-]")


def sidecar_document(provenance: NasaDemProvenance) -> dict[str, Any]:
    """The JSON stored next to the cached DEM.

    Top-level keys follow the format the analysis layer reads from download sidecars. The full
    structured record is in ``nasa_provenance``.
    """
    return {
        "product_id": provenance.product_id,
        "dataset": _SAFE_TEXT.sub("", provenance.product_type_name)[:60],
        "instrument": provenance.instrument,
        "mission": provenance.mission,
        "sha256": provenance.normalization.output_sha256,
        "downloaded_at": provenance.acquired_at,
        "source_url": provenance.source.data_url,
        "nasa_provenance": provenance.model_dump(mode="json"),
    }
