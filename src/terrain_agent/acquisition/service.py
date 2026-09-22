"""NASA DEM acquisition service: the whole flow from request to a cached, analysis-ready DEM.

    request
      -> local cache lookup (no network)
      -> provider search and product selection
      -> controlled download of the PDS4 label and data file into a private work directory
      -> validation (format, size, CRS, resolution, coverage, elevation values)
      -> normalisation to a GeoTIFF of elevation in metres
      -> provenance record and cache commit
      -> path handed to the existing terrain engine

The service does not analyse terrain. It provides a validated DEM to the existing Phase 5
functions. The returned path is inside the managed cache directory.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlsplit

from terrain_agent.acquisition.cache import CacheEntry, DemCache, make_cache_id
from terrain_agent.acquisition.download import DownloadManager
from terrain_agent.acquisition.errors import (
    AcquisitionDisabledError,
    DownloadError,
    NoCoverageError,
    NoSuitableProductError,
)
from terrain_agent.acquisition.http import HttpSettings
from terrain_agent.acquisition.models import CoverageRequest, FileRole, ProductCandidate
from terrain_agent.acquisition.net_policy import NASA_DOWNLOAD_HOSTS, HostPolicy
from terrain_agent.acquisition.normalize import normalize_to_geotiff, verify_normalized
from terrain_agent.acquisition.ode_provider import PROVIDER_ID, OdeProvider
from terrain_agent.acquisition.provenance import build_provenance, sidecar_document, utc_now_text
from terrain_agent.acquisition.provider import DemProvider
from terrain_agent.acquisition.pds_validation import MAX_LABEL_BYTES, validate_pds4_product
from terrain_agent.acquisition.selection import choose_downloadable, rank_for_request

log = logging.getLogger(__name__)

_URL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,149}$")

#: One acquisition at a time per process, so concurrent requests cannot multiply downloads.
_ACQUISITION_LOCK = threading.Lock()


@dataclass(frozen=True)
class AcquiredDem:
    """A validated DEM ready for the terrain engine."""

    cache_id: str
    dem_path: Path
    relative_name: str
    provenance: dict[str, Any]
    from_cache: bool
    notes: list[str]


def _file_name_from_url(url: str, declared: str) -> str:
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    if not _URL_NAME_RE.fullmatch(name) or name.lower() != declared.lower():
        raise DownloadError("The file name in the URL does not match the product metadata.")
    return name


class NasaDemService:
    """Acquire NASA DEMs on demand, with a cache, and hand them to the terrain engine."""

    def __init__(
        self,
        provider: DemProvider,
        downloader: DownloadManager,
        cache: DemCache,
        *,
        enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider = provider
        self._downloader = downloader
        self._cache = cache
        self._enabled = enabled
        self._clock = clock

    @property
    def cache(self) -> DemCache:
        return self._cache

    def acquire(
        self,
        request: CoverageRequest,
        *,
        product_types: Optional[Sequence[str]] = None,
        refresh: bool = False,
    ) -> AcquiredDem:
        """Return a DEM covering *request*, downloading it only if it is not cached.

        Raises
        ------
        AcquisitionDisabledError
            Network acquisition is switched off. Cached DEMs can still be used through
            :meth:`find_cached`.
        NoCoverageError, NoSuitableProductError
            No supported product can be used. The second lists every rejected product and why.
        DownloadError, DemValidationError, CacheError
            A stage failed. Nothing partial is left in the cache.
        """
        types = tuple(product_types) if product_types else None
        if not refresh:
            cached = self.find_cached(request, types)
            if cached is not None:
                return cached
        if not self._enabled:
            raise AcquisitionDisabledError("NASA data downloads are disabled in this deployment.")
        with _ACQUISITION_LOCK:
            return self._acquire_locked(request, types, refresh)

    def find_cached(
        self, request: CoverageRequest, product_types: Optional[Sequence[str]] = None
    ) -> Optional[AcquiredDem]:
        """Best cached DEM covering the request. Local only, never touches the network."""
        entries = self._cache.find_covering(request, product_types)
        if not entries:
            return None
        return self._from_entry(entries[0], from_cache=True, notes=["Served from the local cache."])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _acquire_locked(
        self, request: CoverageRequest, types: Optional[tuple[str, ...]], refresh: bool
    ) -> AcquiredDem:
        candidates = self._provider.search(request, product_types=types)
        log.info("NASA ODE search: candidate_count=%d", len(candidates))
        if not candidates:
            raise NoCoverageError("NASA ODE lists no supported DEM product for this area.")
        ranked, excluded = rank_for_request(candidates, request)
        if not ranked:
            raise NoSuitableProductError(
                "No supported product fully covers the requested area at the requested resolution.",
                [e.model_dump() for e in excluded],
            )
        with_files = self._provider.attach_files(ranked)
        chosen, more_excluded = choose_downloadable(with_files, self._downloader.max_bytes)
        excluded.extend(more_excluded)
        if chosen is None:
            raise NoSuitableProductError(
                "Products cover the area, but none can be downloaded under the configured "
                "size limit. Raise the download limit to use a larger, finer product.",
                [e.model_dump() for e in excluded],
            )
        data = chosen.file(FileRole.DATA)
        label = chosen.file(FileRole.LABEL_PDS4)
        assert data is not None and label is not None
        log.info(
            "NASA DEM product selected: product_id=%s, data_url=%s, label_url=%s",
            chosen.product_id, data.url, label.url,
        )
        cache_id = make_cache_id(PROVIDER_ID, chosen.product_id, data.url)

        if not refresh:
            hit = self._cache.lookup(cache_id)
            if hit is not None:
                return self._from_entry(hit, from_cache=True, notes=["Served from the local cache."])

        notes = [
            f"Selected {chosen.product_id} ({chosen.map_scale_m:g} m per pixel).",
        ] + [
            f"Not used: {e.product_id} ({e.reason}{': ' + e.detail if e.detail else ''})"
            for e in excluded
            if e.reason == "exceeds_download_limit"
        ][:5]

        work = self._cache.new_work_dir()
        try:
            return self._download_and_store(work, chosen, request, cache_id, notes)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _download_and_store(
        self,
        work: Path,
        chosen: ProductCandidate,
        request: CoverageRequest,
        cache_id: str,
        notes: list[str],
    ) -> AcquiredDem:
        data = chosen.file(FileRole.DATA)
        label = chosen.file(FileRole.LABEL_PDS4)
        assert data is not None and label is not None

        label_name = _file_name_from_url(label.url, label.file_name)
        data_name = _file_name_from_url(data.url, data.file_name)
        label_file = self._downloader.fetch(
            label.url,
            work,
            label_name,
            expected_max_bytes=label.expected_max_bytes,
            max_bytes=MAX_LABEL_BYTES,
        )
        data_file = self._downloader.fetch(
            data.url, work, data_name, expected_max_bytes=data.expected_max_bytes
        )

        report, convention = validate_pds4_product(
            label_file.path, data_file.path, candidate=chosen, request=request
        )
        normalized = normalize_to_geotiff(
            label_file.path,
            work / f"{cache_id}.tif",
            convention,
            tags={"TALUS_SOURCE_PRODUCT_ID": chosen.product_id},
        )
        verify_normalized(label_file.path, normalized.path, convention)

        provenance = build_provenance(
            candidate=chosen,
            request=request,
            discovery=self._provider.discovery_info(),
            cache_id=cache_id,
            acquired_at=utc_now_text(self._clock() if self._clock else None),
            data_sha256=data_file.sha256,
            data_bytes=data_file.size_bytes,
            label_sha256=label_file.sha256,
            report=report,
            convention=convention,
            normalized=normalized,
        )
        entry = self._cache.commit(
            cache_id, normalized.path, sidecar_document(provenance), label_source=label_file.path
        )
        log.info("Cached NASA product %s as %s.", chosen.product_id, cache_id)
        return self._from_entry(entry, from_cache=False, notes=notes + report.warnings)

    def _from_entry(self, entry: CacheEntry, *, from_cache: bool, notes: list[str]) -> AcquiredDem:
        return AcquiredDem(
            cache_id=entry.cache_id,
            dem_path=entry.tif_path,
            relative_name=entry.relative_name,
            provenance=entry.provenance,
            from_cache=from_cache,
            notes=notes,
        )


def build_default_service(
    *,
    cache_dir: Optional[Path] = None,
    http_settings: Optional[HttpSettings] = None,
    max_download_bytes: Optional[int] = None,
    max_cache_bytes: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> NasaDemService:
    """Build the service with the configured limits and the real ODE provider."""
    from terrain_agent.config import settings

    nasa = settings.nasa
    http = http_settings or HttpSettings(
        connect_timeout_s=nasa.connect_timeout_s,
        read_timeout_s=nasa.read_timeout_s,
        total_timeout_s=nasa.total_timeout_s,
        max_retries=nasa.max_retries,
        backoff_base_s=nasa.backoff_base_s,
    )
    download_policy = HostPolicy(allowed_hosts=NASA_DOWNLOAD_HOSTS)
    provider = OdeProvider(settings=http, file_policy=download_policy)
    downloader = DownloadManager(
        policy=download_policy,
        settings=http,
        max_bytes=max_download_bytes or settings.resources.max_download_size_bytes,
    )
    cache = DemCache(
        cache_dir or settings.paths.cache_dir,
        max_bytes=max_cache_bytes or settings.resources.max_cache_size_mb * 1024 * 1024,
    )
    return NasaDemService(
        provider, downloader, cache, enabled=nasa.downloads_enabled if enabled is None else enabled
    )


__all__ = ["AcquiredDem", "NasaDemService", "build_default_service"]
