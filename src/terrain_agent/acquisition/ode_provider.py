"""NASA PDS Geosciences Node Orbital Data Explorer (ODE) provider.

Only capabilities documented in the ODE REST Interface Manual, version 2.1.6, are used:

* Address: ``https://oderest.rsl.wustl.edu/live2/``. GET only.
* ``query=product`` with ``target``, ``ihid``, ``iid``, ``pt``, ``output=JSON``, a bounding box
  (``minlat``, ``maxlat``, ``westernlon``, ``easternlon``) with ``loc=b``, and ``limit``.
  Results letters ``o`` (ODE id), ``p`` (PDS identifiers), ``m`` (metadata) and ``f`` (files).
* ``odeid=id1|id2|...`` to fetch the files of several products in one request. The manual
  notes ODE ids can change when a data set is rebuilt, so they are used immediately after
  they are obtained and never stored as identity. The stable identifiers are the PDS product
  id and the PDS4 logical identifier.

Latitudes are planetocentric and longitudes are degrees east from 0 to 360, as documented.

Everything ODE returns is untrusted. Each field used is checked against a strict pattern, file
URLs must be https URLs to an allowlisted NASA host, and free-text descriptions are never kept.
File URLs come only from the ``Product_files`` list. The ``External_url`` field is ignored
because it can point at a non-NASA mirror.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urljoin

import httpx

from terrain_agent.acquisition.errors import (
    HostPolicyError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from terrain_agent.acquisition.http import (
    RETRYABLE_STATUS,
    HttpSettings,
    backoff_delay,
    build_client,
)
from terrain_agent.acquisition.models import CoverageRequest, FileRole, ProductCandidate, ProductFile
from terrain_agent.acquisition.net_policy import NASA_DOWNLOAD_HOSTS, HostPolicy

log = logging.getLogger(__name__)

# The live server answers "/live2?..." with a permanent redirect to "/live2/?...". The manual
# uses both forms in its examples. The slash form avoids the extra round trip.
ODE_LIVE2_URL = "https://oderest.rsl.wustl.edu/live2/"
ODE_API_HOSTS = frozenset({"oderest.rsl.wustl.edu"})

PROVIDER_ID = "nasa_ode"
PROVIDER_NAME = "NASA PDS Geosciences Node, Orbital Data Explorer (ODE)"

#: Supported product types, keyed by (instrument host id, instrument id, product type).
#: Names are fixed text owned by this code, not taken from provider responses.
SUPPORTED_PRODUCTS: dict[tuple[str, str, str], str] = {
    ("LRO", "LOLA", "GDRDEM"): "LOLA Gridded DEM Shape Map",
    ("LRO", "LOLA", "SLDEM"): "LOLA SLDEM (SELENE co-registered)",
}
DEFAULT_PRODUCT_TYPES = ("GDRDEM", "SLDEM")

# Real ODE metadata is large: about 145 kB per product, mostly footprint geometry. Results are
# fetched in pages using the documented limit and offset parameters.
SEARCH_PAGE_SIZE = 25
MAX_SEARCH_RESULTS = 100
FILES_BATCH_SIZE = 25
MAX_SEARCH_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_FILES_RESPONSE_BYTES = 4 * 1024 * 1024

_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,120}$")
_LID_RE = re.compile(r"^urn:[A-Za-z0-9._:\-]{1,200}$")
_SHORT_RE = re.compile(r"^[A-Za-z0-9._\-]{1,60}$")
_TIME_RE = re.compile(r"^[0-9T:.\-Z]{4,32}$")
_ODE_ID_RE = re.compile(r"^[0-9]{1,12}$")
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]{1,150}$")
_ERROR_TEXT_RE = re.compile(r"[^A-Za-z0-9 .,:;()_/\-]")


def _text(value: Any, pattern: re.Pattern[str]) -> Optional[str]:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


#: Confirmed against the live ODE REST endpoint (oderest.rsl.wustl.edu/live2/): a query that
#: matches zero products answers with this literal string in the "Products" slot, not a JSON
#: null, an empty object, or an empty list. Matched case-insensitively since ODE's casing for
#: this text is not documented and has not been observed to be stable.
_NO_PRODUCTS_TEXT = "no products found"


def _parse_products_field(products: Any) -> Optional[list[dict[str, Any]]]:
    """Normalise the ``ODEResults.Products`` field to a list of product dicts.

    Handles every shape observed from ODE or documented by its REST manual: absent/``null``
    (no products), the ``"No Products Found"`` zero-match sentinel, a bare list of products
    with no ``{"Product": ...}`` wrapper, and the usual wrapped dict (a single product object,
    or a list of them, under ``"Product"``). Returns ``None`` only for a type/value this field
    has never been observed to take, which the caller reports as a real structure error --
    including any other, unrecognised string, which is not assumed to mean "zero results".
    """
    if products is None:
        return []
    if isinstance(products, str):
        return [] if products.strip().lower() == _NO_PRODUCTS_TEXT else None
    if isinstance(products, list):
        return [p for p in products if isinstance(p, dict)]
    if isinstance(products, dict):
        return [p for p in _as_list(products.get("Product")) if isinstance(p, dict)]
    return None


MAX_REDIRECTS = 3
_REDIRECT_STATUS = frozenset({301, 302, 307, 308})


class _RetryableError(Exception):
    """Internal marker for failures worth another attempt."""


class OdeProvider:
    """Discovers LOLA and SLDEM products through the documented ODE REST interface."""

    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        policy: Optional[HostPolicy] = None,
        file_policy: Optional[HostPolicy] = None,
        settings: Optional[HttpSettings] = None,
        client: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = ODE_LIVE2_URL,
    ) -> None:
        self._settings = settings or HttpSettings()
        self._policy = policy or HostPolicy(allowed_hosts=ODE_API_HOSTS)
        self._file_policy = file_policy or HostPolicy(allowed_hosts=NASA_DOWNLOAD_HOSTS)
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep
        self._base_url = base_url
        self._policy.check_url(base_url)
        self._last_queries: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        request: CoverageRequest,
        *,
        product_types: Optional[Sequence[str]] = None,
    ) -> list[ProductCandidate]:
        types = tuple(product_types) if product_types else DEFAULT_PRODUCT_TYPES
        supported = {pt for (_h, _i, pt) in SUPPORTED_PRODUCTS}
        unknown = [t for t in types if t not in supported]
        if unknown:
            raise ProviderResponseError(f"Unsupported product type requested: {unknown[0]!r}.")

        candidates: list[ProductCandidate] = []
        for host_id, instrument_id, product_type in SUPPORTED_PRODUCTS:
            if product_type not in types:
                continue
            params = {
                "query": "product",
                "target": "moon",
                "output": "JSON",
                "ihid": host_id,
                "iid": instrument_id,
                "pt": product_type,
                "results": "opm",
                "loc": "b",
                "limit": str(SEARCH_PAGE_SIZE),
                "minlat": f"{request.min_lat:.6f}",
                "maxlat": f"{request.max_lat:.6f}",
                "westernlon": "0" if request.full_longitude else f"{request.west_lon:.6f}",
                "easternlon": "360" if request.full_longitude else f"{request.east_lon:.6f}",
            }
            seen = 0
            while seen < MAX_SEARCH_RESULTS:
                params["offset"] = str(seen)
                products = self._products(self._get_json(params, MAX_SEARCH_RESPONSE_BYTES))
                for raw in products:
                    candidate = self._parse_candidate(raw, host_id, instrument_id, product_type)
                    if candidate is not None:
                        candidates.append(candidate)
                seen += len(products)
                if len(products) < SEARCH_PAGE_SIZE:
                    break
            else:
                log.warning("Stopped after %d %s products; more may exist.", seen, product_type)
        return candidates

    def attach_files(self, candidates: Sequence[ProductCandidate]) -> list[ProductCandidate]:
        by_ode_id = {c.ode_id: c for c in candidates if c.ode_id}
        files_by_key: dict[tuple[str, str, str, str], tuple[ProductFile, ...]] = {}
        ids = list(by_ode_id)
        for start in range(0, len(ids), FILES_BATCH_SIZE):
            batch = ids[start : start + FILES_BATCH_SIZE]
            params = {
                "query": "product",
                "target": "moon",
                "output": "JSON",
                "results": "opf",
                "odeid": "|".join(batch),
            }
            payload = self._get_json(params, MAX_FILES_RESPONSE_BYTES)
            for raw in self._products(payload):
                key = self._identity(raw)
                if key is not None:
                    files_by_key[key] = self._parse_files(raw)

        result: list[ProductCandidate] = []
        for candidate in candidates:
            key = (candidate.host_id, candidate.instrument_id, candidate.product_type, candidate.product_id)
            files = files_by_key.get(key, ())
            result.append(candidate.model_copy(update={"files": files}))
        return result

    def discovery_info(self) -> dict[str, object]:
        return {
            "provider": PROVIDER_NAME,
            "endpoint": self._base_url,
            "documentation": "ODE REST Interface Manual v2.1.6",
            "queries": list(self._last_queries),
        }

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, params: dict[str, str], max_bytes: int) -> dict[str, Any]:
        self._last_queries.append(
            {k: v for k, v in params.items() if k not in ("odeid",)}
            | ({"odeid_count": str(len(params["odeid"].split("|")))} if "odeid" in params else {})
        )
        client = self._client or build_client(self._settings)
        if self._client is None:
            self._client = client
        last_error: Optional[BaseException] = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                self._policy.check(self._base_url)
                return self._fetch_once(client, params, max_bytes)
            except HostPolicyError as exc:
                raise ProviderUnavailableError(str(exc)) from exc
            except (httpx.TimeoutException, httpx.TransportError, _RetryableError) as exc:
                last_error = exc
                if attempt < self._settings.max_retries:
                    self._sleep(backoff_delay(attempt, self._settings))
        raise ProviderUnavailableError(
            "The NASA ODE service could not be reached after "
            f"{self._settings.max_retries + 1} attempts ({type(last_error).__name__})."
        )

    def _fetch_once(
        self, client: httpx.Client, params: dict[str, str], max_bytes: int
    ) -> dict[str, Any]:
        url = self._base_url
        query: Optional[dict[str, str]] = params
        body = bytearray()
        for hop in range(MAX_REDIRECTS + 1):
            with client.stream(
                "GET",
                url,
                params=query,
                headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            ) as response:
                status = response.status_code
                if status in _REDIRECT_STATUS:
                    location = response.headers.get("location")
                    if not location or hop >= MAX_REDIRECTS:
                        raise ProviderResponseError("ODE sent too many or malformed redirects.")
                    url = urljoin(str(response.url), location)
                    self._policy.check(url)  # the new location must pass the same policy
                    query = None  # the redirect target already carries the query string
                    continue
                if status in RETRYABLE_STATUS:
                    raise _RetryableError(f"HTTP {status}")
                if status != 200:
                    raise ProviderUnavailableError(f"ODE answered with HTTP status {status}.")
                log.info(
                    "ODE HTTP response: status=%d, content_type=%s",
                    status, response.headers.get("content-type", "?"),
                )
                encoding = response.headers.get("content-encoding", "identity").strip().lower()
                if encoding not in ("", "identity"):
                    raise ProviderResponseError("ODE sent an encoded response that was not requested.")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise ProviderResponseError("The ODE response is larger than the allowed size.")
                for chunk in response.iter_raw(65536):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise ProviderResponseError("The ODE response is larger than the allowed size.")
                break
        try:
            payload = json.loads(bytes(body).decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProviderResponseError("ODE did not return valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ProviderResponseError("ODE returned an unexpected JSON structure.")
        log.info(
            "ODE response parsed: format=JSON, bytes=%d, top_level_keys=%s",
            len(body), sorted(payload.keys()),
        )
        return payload

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _results(payload: dict[str, Any]) -> dict[str, Any]:
        results = payload.get("ODEResults")
        if not isinstance(results, dict):
            raise ProviderResponseError("The ODE response has no ODEResults section.")
        status = str(results.get("Status", results.get("status", ""))).strip().upper()
        if status == "ERROR":
            message = _ERROR_TEXT_RE.sub("", str(results.get("Error", results.get("error", ""))))[:100]
            raise ProviderResponseError(f"ODE reported an error: {message or 'no detail given'}")
        if status != "SUCCESS":
            raise ProviderResponseError("The ODE response does not report success.")
        return results

    def _products(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        results = self._results(payload)
        products = results.get("Products")
        parsed = _parse_products_field(products)
        if parsed is None:
            raise ProviderResponseError(
                "The ODE Products section has an unexpected structure "
                f"(type={type(products).__name__})."
            )
        # Diagnostic only: no full response bodies, no secrets, just shape and identity.
        first_keys = sorted(parsed[0].keys())[:12] if parsed else []
        log.info(
            "ODE Products section parsed: type=%s, product_count=%d, first_product_keys=%s",
            type(products).__name__, len(parsed), first_keys,
        )
        return parsed

    @staticmethod
    def _identity(raw: dict[str, Any]) -> Optional[tuple[str, str, str, str]]:
        ihid = _text(raw.get("ihid"), _SHORT_RE)
        iid = _text(raw.get("iid"), _SHORT_RE)
        pt = _text(raw.get("pt"), _SHORT_RE)
        pid = _text(raw.get("pdsid"), _ID_RE)
        if not (ihid and iid and pt and pid):
            return None
        return ihid.upper(), iid.upper(), pt.upper(), pid

    def _parse_candidate(
        self, raw: dict[str, Any], host_id: str, instrument_id: str, product_type: str
    ) -> Optional[ProductCandidate]:
        identity = self._identity(raw)
        if identity is None or identity[:3] != (host_id, instrument_id, product_type):
            return None
        min_lat = _float(raw.get("Minimum_latitude"))
        max_lat = _float(raw.get("Maximum_latitude"))
        west = _float(raw.get("Westernmost_longitude"))
        east = _float(raw.get("Easternmost_longitude"))
        if None in (min_lat, max_lat, west, east):
            return None
        assert min_lat is not None and max_lat is not None and west is not None and east is not None
        if not (-90.0 <= min_lat <= max_lat <= 90.0) or not (0.0 <= west <= 360.0) or not (0.0 <= east <= 360.0):
            return None
        scale = _float(raw.get("Map_scale"))
        ppd = _float(raw.get("Map_resolution"))
        return ProductCandidate(
            provider_id=PROVIDER_ID,
            host_id=host_id,
            instrument_id=instrument_id,
            product_type=product_type,
            product_id=identity[3],
            product_lid=_text(raw.get("Product_lid"), _LID_RE),
            data_set_id=_text(raw.get("Data_Set_Id"), _ID_RE),
            version=_text(raw.get("Product_version_id"), _SHORT_RE),
            min_lat=min_lat,
            max_lat=max_lat,
            west_lon=west,
            east_lon=east,
            map_scale_m=scale if scale and scale > 0 else None,
            map_resolution_ppd=ppd if ppd and ppd > 0 else None,
            creation_time=_text(raw.get("Product_creation_time"), _TIME_RE),
            ode_id=_text(str(raw.get("ode_id", "")), _ODE_ID_RE),
        )

    def _parse_files(self, raw: dict[str, Any]) -> tuple[ProductFile, ...]:
        container = raw.get("Product_files")
        entries = _as_list(container.get("Product_file")) if isinstance(container, dict) else []
        files: list[ProductFile] = []
        for entry in entries[:60]:
            if not isinstance(entry, dict) or entry.get("Type") != "Product":
                continue
            name = _text(entry.get("FileName"), _FILE_NAME_RE)
            url = entry.get("URL")
            if not name or not isinstance(url, str):
                continue
            upper = name.upper()
            description = str(entry.get("Description", "")).upper()
            if upper.endswith(".IMG"):
                role = FileRole.DATA
            elif upper.endswith(".XML") and "PDS4" in description:
                role = FileRole.LABEL_PDS4
            else:
                continue
            try:
                self._file_policy.check_url(url)
            except HostPolicyError:
                log.info("Ignoring a product file whose host is not permitted.")
                continue
            kb = entry.get("KBytes")
            size_kb = int(kb) if isinstance(kb, str) and kb.isdigit() else None
            files.append(ProductFile(role=role, file_name=name, url=url, size_kb=size_kb))
        return tuple(files)
