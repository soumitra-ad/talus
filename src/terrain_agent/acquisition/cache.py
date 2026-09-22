"""Local cache of validated, normalised NASA DEMs.

Layout under the configured cache directory::

    nasa/<cache_id>.tif          elevation in metres (GeoTIFF)
    nasa/<cache_id>.json         provenance (read by the analysis layer)
    nasa/<cache_id>.label.xml    the product label as received, for audit only
    nasa/.tmp/acq-*/             working directories for acquisitions in progress

Cache identifiers are derived only from the validated product id, the data URL and the
normalisation version. They match a strict pattern and no caller supplied text becomes a path.
Every resolved path is checked to lie directly inside the cache directory.

Behaviour: a total size quota is enforced with least-recently-used eviction. Entries are
integrity checked on use by recomputing the SHA-256 recorded in their provenance. An entry that
fails is removed. A commit writes the GeoTIFF first and the provenance last, so an entry without
provenance is incomplete and is ignored and cleaned up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from terrain_agent.acquisition.errors import CacheError
from terrain_agent.acquisition.models import CoverageRequest
from terrain_agent.acquisition.normalize import NORMALIZATION_VERSION
from terrain_agent.terrain.georef import open_dem_context
from terrain_agent.terrain.resource_safety import TerrainAnalysisError

log = logging.getLogger(__name__)

CACHE_SUBDIR = "nasa"
_CACHE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]{0,100}-[0-9a-f]{12}$")
MAX_ENTRIES_SCANNED = 500
STALE_WORK_DIR_SECONDS = 24 * 3600


def make_cache_id(provider_id: str, product_id: str, data_url: str) -> str:
    """Deterministic identifier for one product from one source in one normalisation format."""
    readable = re.sub(r"[^a-z0-9_.\-]", "_", product_id.lower()).strip("._-")[:80] or "product"
    digest = hashlib.sha256(
        f"{provider_id}|{product_id}|{data_url}|{NORMALIZATION_VERSION}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{readable}-{digest}"


@dataclass(frozen=True)
class CacheEntry:
    """A complete entry in the cache."""

    cache_id: str
    tif_path: Path
    json_path: Path
    size_bytes: int
    provenance: dict[str, Any]

    @property
    def relative_name(self) -> str:
        return f"{CACHE_SUBDIR}/{self.cache_id}.tif"


class DemCache:
    """Bounded on-disk cache of normalised DEMs."""

    def __init__(self, cache_dir: Path, *, max_bytes: int, verify_on_use: bool = True) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._root = (Path(cache_dir).resolve() / CACHE_SUBDIR)
        self._root.mkdir(parents=True, exist_ok=True)
        self._tmp = self._root / ".tmp"
        self._tmp.mkdir(exist_ok=True)
        self._max_bytes = max_bytes
        self._verify = verify_on_use
        self._lock = threading.RLock()
        self.purge_stale_work_dirs()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _paths(self, cache_id: str) -> tuple[Path, Path, Path]:
        if not isinstance(cache_id, str) or not _CACHE_ID_RE.fullmatch(cache_id):
            raise CacheError("The cache identifier is not valid.")
        tif = (self._root / f"{cache_id}.tif").resolve()
        js = (self._root / f"{cache_id}.json").resolve()
        label = (self._root / f"{cache_id}.label.xml").resolve()
        for path in (tif, js, label):
            if path.parent != self._root:
                raise CacheError("The cache path is outside the cache directory.")
        return tif, js, label

    def new_work_dir(self) -> Path:
        """A fresh private directory for an acquisition in progress."""
        return Path(tempfile.mkdtemp(dir=self._tmp, prefix="acq-"))

    def purge_stale_work_dirs(self) -> None:
        cutoff = time.time() - STALE_WORK_DIR_SECONDS
        for child in self._tmp.iterdir():
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _load_entry(self, cache_id: str, *, verify: bool) -> Optional[CacheEntry]:
        tif, js, _label = self._paths(cache_id)
        if not tif.is_file() or not js.is_file():
            return None
        try:
            document = json.loads(js.read_text(encoding="utf-8"))
            block = document["nasa_provenance"]
            if block["cache_id"] != cache_id:
                raise ValueError("identifier mismatch")
            expected_size = int(block["normalization"]["output_bytes"])
            expected_sha = str(block["normalization"]["output_sha256"])
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("Removing a cache entry with unreadable provenance.")
            self.remove(cache_id)
            return None
        if tif.stat().st_size != expected_size or (verify and _sha256(tif) != expected_sha):
            log.warning("Removing a cache entry that failed its integrity check.")
            self.remove(cache_id)
            return None
        return CacheEntry(cache_id, tif, js, expected_size, block)

    def lookup(self, cache_id: str) -> Optional[CacheEntry]:
        """Return a verified entry and mark it recently used, or ``None``."""
        with self._lock:
            entry = self._load_entry(cache_id, verify=self._verify)
            if entry is not None:
                os.utime(entry.json_path, None)
            return entry

    def list_ids(self) -> list[str]:
        ids = []
        for path in sorted(self._root.glob("*.json"))[:MAX_ENTRIES_SCANNED]:
            stem = path.name[: -len(".json")]
            if _CACHE_ID_RE.fullmatch(stem):
                ids.append(stem)
        return ids

    def total_bytes(self) -> int:
        total = 0
        for path in self._root.iterdir():
            if path.is_file():
                total += path.stat().st_size
        return total

    def find_covering(
        self, request: CoverageRequest, product_types: Optional[Sequence[str]] = None
    ) -> list[CacheEntry]:
        """Cached entries whose raster contains the requested area, finest resolution first.

        Uses only local files. No network access.
        """
        with self._lock:
            found: list[tuple[float, str]] = []
            for cache_id in self.list_ids():
                entry = self._load_entry(cache_id, verify=False)
                if entry is None:
                    continue
                block = entry.provenance
                if product_types and block.get("product_type") not in product_types:
                    continue
                pixel = (block.get("raster", {}).get("native_pixel_size") or [None, None])[1]
                unit_scale = 1.0 if block.get("raster", {}).get("native_units") == "metre" else None
                if pixel is None or unit_scale is None:
                    continue
                if request.max_pixel_size_m is not None and pixel > request.max_pixel_size_m:
                    continue
                if self._raster_covers(entry.tif_path, request):
                    found.append((float(pixel), cache_id))
            found.sort()
            entries = []
            for _pixel, cache_id in found:
                verified = self._load_entry(cache_id, verify=self._verify)
                if verified is not None:
                    os.utime(verified.json_path, None)
                    entries.append(verified)
            return entries

    @staticmethod
    def _raster_covers(tif: Path, request: CoverageRequest) -> bool:
        try:
            ctx = open_dem_context(tif)
            points = request.sample_points()
            row_f, col_f, finite = ctx.latlon_to_rowcol_float(
                [p[0] for p in points], [p[1] for p in points]
            )
        except TerrainAnalysisError:
            return False
        return bool(
            (finite & (row_f >= 0) & (row_f < ctx.height) & (col_f >= 0) & (col_f < ctx.width)).all()
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def commit(
        self,
        cache_id: str,
        tif_source: Path,
        sidecar: dict[str, Any],
        label_source: Optional[Path] = None,
    ) -> CacheEntry:
        """Move a finished GeoTIFF into the cache, evicting old entries if needed."""
        tif, js, label = self._paths(cache_id)
        payload = json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8")
        new_bytes = tif_source.stat().st_size + len(payload)
        if label_source is not None:
            new_bytes += label_source.stat().st_size
        with self._lock:
            if new_bytes > self._max_bytes:
                raise CacheError("The product is larger than the whole cache quota.")
            self._evict_for(new_bytes, keep=cache_id)
            try:
                os.replace(tif_source, tif)
                if label_source is not None:
                    shutil.copyfile(label_source, label)
                temp_json = js.with_suffix(".json.part")
                temp_json.write_bytes(payload)
                os.replace(temp_json, js)
            except OSError as exc:
                for path in (tif, js, label):
                    path.unlink(missing_ok=True)
                raise CacheError("The product could not be written to the cache.") from exc
            entry = self._load_entry(cache_id, verify=False)
            if entry is None:
                raise CacheError("The cache entry could not be read back after writing.")
            return entry

    def _evict_for(self, needed: int, *, keep: str) -> None:
        used = self.total_bytes()
        # An existing entry with the same identifier is about to be overwritten.
        for path in self._paths(keep):
            if path.is_file():
                used -= path.stat().st_size
        if used + needed <= self._max_bytes:
            return
        candidates = []
        for cache_id in self.list_ids():
            if cache_id == keep:
                continue
            _tif, js, _label = self._paths(cache_id)
            candidates.append((js.stat().st_mtime, cache_id))
        for _mtime, cache_id in sorted(candidates):
            if used + needed <= self._max_bytes:
                break
            before = self.total_bytes()
            self.remove(cache_id)
            used -= max(0, before - self.total_bytes())
            log.info("Evicted a cache entry to stay within the quota.")
        if used + needed > self._max_bytes:
            raise CacheError("The cache quota cannot be met even after eviction.")

    def remove(self, cache_id: str) -> None:
        for path in self._paths(cache_id):
            path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
