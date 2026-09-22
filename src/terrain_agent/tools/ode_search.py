"""
NASA Lunar ODE REST API product search.

All query parameters are derived from the official ODE REST API V2.1.x.
Endpoint: https://oderest.rsl.wustl.edu/live2/
Documentation: https://oderest.rsl.wustl.edu/ODE_REST_V2.1.6.pdf

Product types verified against live ODE IIPT endpoint on 2026-09-19
for target=moon, ihid=LRO.

Host access is restricted to the configured allowlist defined in
:mod:`terrain_agent.tools.dem_downloader`.  This module never
initiates downloads — it only returns structured metadata.

Research/Demo Disclaimer
------------------------
TALUS is a research and demonstration system.  Search results must
NOT be used for certified flight safety, operational landing approval,
autonomous spacecraft control, or guaranteed rover safety.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants - derived from live ODE IIPT query (target=moon, ihid=LRO)
# ---------------------------------------------------------------------------

#: Base URL for the ODE V2 REST endpoint (live2 path, no trailing slash).
ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2"

#: ODE target identifier for the Moon.
ODE_TARGET_MOON = "moon"

#: Dataset strategies: list of (ihid, iid, pt, human_label) in priority order.
#: Product types verified against live ODE IIPT endpoint (2026-09-19).
DATASET_STRATEGIES: list[tuple[str, str, str, str]] = [
    # LOLA Global Gridded DEM Shape Map (0.236 km/pix at 128 ppd)
    ("LRO", "LOLA", "GDRDEM", "LOLA Gridded DEM Shape Map"),
    # LOLA SLDEM2015 - co-registered with SELENE/Kaguya TC
    ("LRO", "LOLA", "SLDEM", "LOLA SLDEM2015 (SELENE co-registered)"),
    # LROC NAC Digital Terrain Model (highest resolution, regional)
    ("LRO", "LROC", "SDNDTM", "LROC NAC Digital Terrain Model"),
    # LROC WAC Digital Terrain Model (medium resolution, regional)
    ("LRO", "LROC", "SDWDTM", "LROC WAC Digital Terrain Model"),
]

#: Mapping from user-facing dataset preference string to IIPT codes.
PREFERRED_DATASET_MAP: dict[str, tuple[str, str, str]] = {
    "lola": ("LRO", "LOLA", "GDRDEM"),
    "lola_gdrdem": ("LRO", "LOLA", "GDRDEM"),
    "sldem": ("LRO", "LOLA", "SLDEM"),
    "sldem2015": ("LRO", "LOLA", "SLDEM"),
    "lroc_nac": ("LRO", "LROC", "SDNDTM"),
    "lroc_wac": ("LRO", "LROC", "SDWDTM"),
}

#: Maximum results returned per ODE query.
ODE_MAX_RESULTS = 20

#: HTTP timeout in seconds for ODE metadata queries (not downloads).
ODE_SEARCH_TIMEOUT_S = 30.0

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DEMProductRecord:
    """Structured metadata for a single DEM product from ODE."""

    product_id: str
    """ODE product identifier."""

    mission: str
    """Instrument host name (e.g. 'Lunar Reconnaissance Orbiter')."""

    instrument: str
    """Instrument identifier (e.g. 'LOLA', 'LROC')."""

    dataset: str
    """Human-readable dataset label."""

    product_type: str
    """ODE product type code (e.g. 'GDRDEM', 'SLDEM')."""

    file_url: str
    """Primary download URL for the raster file (may be empty)."""

    files_page_url: str
    """URL to the ODE product files listing page."""

    min_lat: float
    """Southern boundary latitude in decimal degrees."""

    max_lat: float
    """Northern boundary latitude in decimal degrees."""

    min_lon: float
    """Western boundary longitude in decimal degrees (0-360)."""

    max_lon: float
    """Eastern boundary longitude in decimal degrees (0-360)."""

    center_lat: float
    """Center latitude in decimal degrees."""

    center_lon: float
    """Center longitude in decimal degrees."""

    description: str = ""
    """Short product description from ODE metadata."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Any additional ODE metadata fields preserved for reference."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_float(value: Any, default: float = 0.0) -> float:
    """Safely convert an ODE field value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_primary_url(product: dict[str, Any]) -> str:
    """
    Extract the most relevant raster download URL from an ODE product record.

    ODE products carry the raster URL in different fields depending on
    PDS version (PDS3 External_url vs PDS4 Product_file_url).
    We prefer the direct file URL; fall back to empty string.
    """
    for key in ("External_url", "Product_file_url", "Label_product_url"):
        val = product.get(key, "")
        if val and isinstance(val, str) and val.startswith("https://"):
            return val
    return ""


def _record_from_ode_product(
    product: dict[str, Any],
    ihid: str,
    iid: str,
    pt: str,
    dataset_label: str,
) -> DEMProductRecord:
    """Build a DEMProductRecord from a raw ODE product dict."""
    product_id = product.get("Product_Id", product.get("PDSId", "unknown"))
    files_url = product.get("FilesURL", "")
    file_url = _extract_primary_url(product)

    # Bounding box - ODE uses 0..360 longitude convention
    min_lat = _parse_float(product.get("Minimum_latitude", product.get("BB_MinLat", 0)))
    max_lat = _parse_float(product.get("Maximum_latitude", product.get("BB_MaxLat", 0)))
    min_lon = _parse_float(
        product.get("Westernmost_longitude", product.get("BB_WestLon", 0))
    )
    max_lon = _parse_float(
        product.get("Easternmost_longitude", product.get("BB_EastLon", 360))
    )
    center_lat = _parse_float(product.get("Center_latitude", 0))
    center_lon = _parse_float(product.get("Center_longitude", 180))

    description_raw = product.get("Description", "")
    description = " ".join(description_raw.split())[:200] if description_raw else ""

    ihname = product.get("IHName", ihid)

    _SKIP_KEYS = frozenset({
        "Product_Id", "PDSId", "FilesURL", "External_url",
        "Product_file_url", "Label_product_url",
        "Minimum_latitude", "Maximum_latitude",
        "Westernmost_longitude", "Easternmost_longitude",
        "Center_latitude", "Center_longitude",
        "Description", "IHName",
        # Large footprint geometry strings
        "Footprint_geometry", "Footprint_C0_geometry",
        "Footprint_NP_geometry", "Footprint_GL_geometry",
    })

    return DEMProductRecord(
        product_id=product_id,
        mission=ihname,
        instrument=iid,
        dataset=dataset_label,
        product_type=pt,
        file_url=file_url,
        files_page_url=files_url,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        center_lat=center_lat,
        center_lon=center_lon,
        description=description,
        extra={k: v for k, v in product.items() if k not in _SKIP_KEYS},
    )


def _query_ode(
    client: httpx.Client,
    ihid: str,
    iid: str,
    pt: str,
    dataset_label: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit: int,
) -> list[DEMProductRecord]:
    """
    Issue a single ODE product metadata query and return parsed records.

    Uses query=products, results=cm (count + metadata),
    loc=b (bounding box intersection), output=JSON.
    """
    params: dict[str, str] = {
        "query": "products",
        "target": ODE_TARGET_MOON,
        "results": "cm",
        "output": "JSON",
        "ihid": ihid,
        "iid": iid,
        "pt": pt,
        "minlat": str(min_lat),
        "maxlat": str(max_lat),
        "westernlon": str(min_lon),
        "easternlon": str(max_lon),
        "loc": "b",
        "limit": str(limit),
    }

    url = f"{ODE_BASE_URL}/"
    log.debug("ODE search: %s params=%s", url, params)

    try:
        response = client.get(url, params=params, timeout=ODE_SEARCH_TIMEOUT_S)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        log.warning("ODE search timeout for %s/%s/%s: %s", ihid, iid, pt, exc)
        return []
    except httpx.HTTPStatusError as exc:
        log.warning(
            "ODE HTTP error %s for %s/%s/%s", exc.response.status_code, ihid, iid, pt
        )
        return []
    except httpx.RequestError as exc:
        log.warning("ODE request error for %s/%s/%s: %s", ihid, iid, pt, exc)
        return []

    try:
        data: dict[str, Any] = response.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("ODE JSON parse error for %s/%s/%s: %s", ihid, iid, pt, exc)
        return []

    ode_results = data.get("ODEResults", {})
    status = ode_results.get("Status", "")
    if status == "ERROR":
        log.warning(
            "ODE returned error for %s/%s/%s: %s",
            ihid,
            iid,
            pt,
            ode_results.get("Error", "unknown"),
        )
        return []

    count_str = ode_results.get("Count", "0")
    try:
        count = int(count_str)
    except (TypeError, ValueError):
        count = 0

    if count == 0:
        log.debug("ODE: 0 products for %s/%s/%s in bbox", ihid, iid, pt)
        return []

    products_raw = ode_results.get("Products", {}).get("Product", [])
    if isinstance(products_raw, dict):
        # ODE returns a single dict when count == 1
        products_raw = [products_raw]

    records: list[DEMProductRecord] = []
    for prod in products_raw:
        if not isinstance(prod, dict):
            continue
        rec = _record_from_ode_product(prod, ihid, iid, pt, dataset_label)
        records.append(rec)

    log.debug("ODE: %d product(s) for %s/%s/%s in bbox", len(records), ihid, iid, pt)
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_lunar_dem(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    preferred_dataset: str = "lola",
    max_results: int = ODE_MAX_RESULTS,
) -> list[DEMProductRecord]:
    """
    Search NASA ODE for lunar DEM products intersecting a bounding box.

    Products are returned in priority order:
    1. Preferred dataset (if specified and available).
    2. Remaining dataset strategies in DATASET_STRATEGIES order.

    Parameters
    ----------
    min_lat:
        Southern boundary latitude (decimal degrees, -90 to +90).
    max_lat:
        Northern boundary latitude (decimal degrees, -90 to +90).
    min_lon:
        Western boundary longitude (decimal degrees, 0 to 360 or -180 to +180).
        Negative values are normalised to 0-360 automatically.
    max_lon:
        Eastern boundary longitude (decimal degrees, same convention as min_lon).
    preferred_dataset:
        One of "lola" (default), "sldem", "lroc_nac", "lroc_wac"
        or any key in PREFERRED_DATASET_MAP.
    max_results:
        Upper bound on the number of products returned across all strategies.

    Returns
    -------
    list[DEMProductRecord]
        Structured metadata records, ordered by priority.
        Returns an empty list when no products are found.

    Raises
    ------
    ValueError
        If the bounding box coordinates are not finite or are logically
        inconsistent.
    """
    for name, val in [
        ("min_lat", min_lat),
        ("max_lat", max_lat),
        ("min_lon", min_lon),
        ("max_lon", max_lon),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")

    if min_lat > max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be <= max_lat ({max_lat})")
    if min_lat < -90 or max_lat > 90:
        raise ValueError(
            f"Latitude out of range: [{min_lat}, {max_lat}] (valid: -90 to +90)"
        )

    def _norm_lon(lon: float) -> float:
        """Normalise longitude to 0..360 (ODE convention)."""
        if lon < 0:
            return lon + 360.0
        return lon

    min_lon_n = _norm_lon(min_lon)
    max_lon_n = _norm_lon(max_lon)

    # Build strategy order: preferred first, then others
    preferred_key = preferred_dataset.strip().lower()
    preferred_iipt: tuple[str, str, str] | None = PREFERRED_DATASET_MAP.get(preferred_key)

    ordered_strategies: list[tuple[str, str, str, str]] = []
    if preferred_iipt is not None:
        pref_ihid, pref_iid, pref_pt = preferred_iipt
        for ihid, iid, pt, label in DATASET_STRATEGIES:
            if ihid == pref_ihid and iid == pref_iid and pt == pref_pt:
                ordered_strategies.append((ihid, iid, pt, label))
                break

    for entry in DATASET_STRATEGIES:
        if entry not in ordered_strategies:
            ordered_strategies.append(entry)

    all_records: list[DEMProductRecord] = []

    with httpx.Client(
        headers={"User-Agent": "TALUS/1.0 (research; contact: talus-dev@example.com)"},
        follow_redirects=False,
    ) as client:
        for ihid, iid, pt, label in ordered_strategies:
            if len(all_records) >= max_results:
                break

            remaining = max_results - len(all_records)
            records = _query_ode(
                client=client,
                ihid=ihid,
                iid=iid,
                pt=pt,
                dataset_label=label,
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon_n,
                max_lon=max_lon_n,
                limit=min(remaining, ODE_MAX_RESULTS),
            )
            all_records.extend(records)

    log.info(
        "ODE search complete: %d product(s) for bbox [%.2f,%.2f]x[%.2f,%.2f]",
        len(all_records),
        min_lat,
        max_lat,
        min_lon_n,
        max_lon_n,
    )
    return all_records[:max_results]
