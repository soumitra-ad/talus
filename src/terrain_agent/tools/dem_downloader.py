"""
Secure DEM downloader for TALUS.

Downloads raster files from NASA/LROC allowlisted hosts only.
Enforces SSRF protection, redirect validation, streaming with size caps,
SHA-256 checksum computation, and JSON sidecar writing.

Security invariants (from .agents/rules/security.md):
- ONLY downloads from ALLOWED_HOSTS.
- Rejects http://, ftp://, file://, and any non-https scheme.
- Rejects localhost, loopback, RFC-1918, and link-local addresses.
- Never follows redirects to unapproved hosts.
- Max download size: 100 MB.
- Writes to data/cache/ only; blocks path traversal.

Research/Demo Disclaimer
------------------------
TALUS is a research and demonstration system. Downloaded data must
NOT be used for certified flight safety or operational mission approval.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

#: The only hostnames permitted for remote connections.
ALLOWED_HOSTS: frozenset[str] = frozenset({
    "ode.rsl.wustl.edu",
    "oderest.rsl.wustl.edu",
    "pds-geosciences.wustl.edu",
    "wac.lroc.asu.edu",
    "lroc.sese.asu.edu",
    "imbrium.mit.edu",        # MIT mirror of LOLA GDR data
})

#: Maximum permitted download size in bytes (100 MB).
MAX_DOWNLOAD_BYTES: int = 100 * 1024 * 1024

#: Streaming chunk size in bytes.
CHUNK_SIZE: int = 65_536

#: HTTP connect + read timeout in seconds for downloads.
DOWNLOAD_TIMEOUT_S: float = 120.0

#: Maximum number of redirects we will manually follow (0 = no redirects).
MAX_REDIRECTS: int = 3

#: Private/loopback IP networks that must be rejected (SSRF protection).
_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.IPv4Network("127.0.0.0/8"),      # loopback
    ipaddress.IPv4Network("10.0.0.0/8"),       # RFC-1918
    ipaddress.IPv4Network("172.16.0.0/12"),    # RFC-1918
    ipaddress.IPv4Network("192.168.0.0/16"),   # RFC-1918
    ipaddress.IPv4Network("169.254.0.0/16"),   # link-local / AWS metadata
    ipaddress.IPv6Network("::1/128"),          # IPv6 loopback
    ipaddress.IPv6Network("fc00::/7"),         # IPv6 ULA
    ipaddress.IPv6Network("fe80::/10"),        # IPv6 link-local
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class HostNotAllowedError(PermissionError):
    """Raised when a URL points to a host not in ALLOWED_HOSTS."""


class SSRFBlockedError(PermissionError):
    """Raised when a hostname resolves to a private/loopback IP address."""


class DownloadSizeLimitError(ValueError):
    """Raised when a download would exceed MAX_DOWNLOAD_BYTES."""


class ChecksumMismatchError(ValueError):
    """Raised when the downloaded file SHA-256 does not match the expected value."""


class DownloadTimeoutError(TimeoutError):
    """Raised when a download exceeds its total wall-clock budget.

    Distinct from httpx's per-read timeout: a connection that keeps sending small chunks just
    under the read timeout could otherwise stay open indefinitely (a slow-drip resource
    exhaustion), so the total elapsed time is checked on every chunk as well.
    """


#: Wall-clock budget for one complete download attempt, independent of per-read timeouts.
MAX_DOWNLOAD_WALL_CLOCK_S: float = 300.0


# ---------------------------------------------------------------------------
# URL / host validation
# ---------------------------------------------------------------------------


def validate_url(url: str) -> None:
    """
    Validate that *url* is safe to connect to.

    Checks:
    - Scheme must be ``https``.
    - Hostname must be in :data:`ALLOWED_HOSTS`.
    - Hostname must not resolve to a private/loopback IP.

    Raises
    ------
    ValueError
        If the scheme is not https or the URL cannot be parsed.
    HostNotAllowedError
        If the hostname is not in the allowlist.
    SSRFBlockedError
        If the hostname resolves to a blocked IP range.
    """
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError(
            f"Insecure or unsupported URL scheme rejected: {parsed.scheme!r}. "
            "Only https:// is permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"URL has no resolvable hostname: {url!r}")

    if hostname not in ALLOWED_HOSTS:
        raise HostNotAllowedError(
            f"Host {hostname!r} is not in the NASA/LROC allowlist. "
            f"Allowed: {sorted(ALLOWED_HOSTS)}"
        )

    _block_private_ip(hostname)


def _is_global_unicast(ip_addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for a genuine global unicast address.

    Unwraps IPv4-mapped and 6to4 IPv6 addresses first, so a blocked IPv4 address smuggled
    inside one of those forms (e.g. ``::ffff:169.254.169.254``) is still caught -- a plain
    per-network blocklist checked against the un-unwrapped address would miss it, because it
    would never match either the IPv4Network or IPv6Network entries in ``_BLOCKED_NETWORKS``.
    """
    if isinstance(ip_addr, ipaddress.IPv6Address) and ip_addr.ipv4_mapped is not None:
        ip_addr = ip_addr.ipv4_mapped
    if isinstance(ip_addr, ipaddress.IPv6Address) and ip_addr.sixtofour is not None:
        ip_addr = ip_addr.sixtofour
    if any(ip_addr in net for net in _BLOCKED_NETWORKS):
        return False
    return bool(ip_addr.is_global and not ip_addr.is_multicast and not ip_addr.is_unspecified)


def _block_private_ip(hostname: str) -> None:
    """
    Resolve *hostname* to IP addresses and block unless every address is a genuine global
    unicast address (fails closed).

    Raises
    ------
    SSRFBlockedError
        If the hostname cannot be resolved, or resolves to any address that is not a global
        unicast address (private, loopback, link-local, multicast, reserved, or a private
        address smuggled inside an IPv4-mapped/6to4 IPv6 address).
    """
    try:
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        # Fail closed: an unresolvable allowlisted host is refused rather than let the
        # subsequent httpx call attempt its own (unvalidated) resolution.
        raise SSRFBlockedError(f"Host {hostname!r} could not be resolved.") from exc

    if not results:
        raise SSRFBlockedError(f"Host {hostname!r} did not resolve to any address.")

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            ip_addr = ipaddress.ip_address(ip_str.split("%", 1)[0])
        except ValueError:
            continue
        if not _is_global_unicast(ip_addr):
            raise SSRFBlockedError(
                f"Host {hostname!r} resolves to blocked private/loopback "
                f"address {ip_str!r}. Connection refused."
            )


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


def get_cache_dir(base_dir: Path | None = None) -> Path:
    """
    Return the DEM tile cache directory, creating it if needed.

    The cache directory is ``<project_root>/data/cache/`` by default.
    """
    if base_dir is None:
        # Resolve relative to this file: src/terrain_agent/tools/ -> project root
        base_dir = Path(__file__).resolve().parents[3]

    cache_dir = base_dir / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _safe_cache_path(cache_dir: Path, product_id: str, suffix: str) -> Path:
    """
    Build a cache path for *product_id* and validate against path traversal.

    Raises
    ------
    ValueError
        If the resolved path would escape *cache_dir*.
    """
    # Prevent absolute paths which might ignore cache_dir
    if Path(product_id).is_absolute() or product_id.startswith("/") or product_id.startswith("\\"):
        raise ValueError("Path traversal detected: absolute paths are not allowed")

    # Check for path traversal before sanitisation
    try:
        raw_candidate = (cache_dir / product_id).resolve()
    except Exception:
        raw_candidate = Path("/") # Fallback if resolution fails
        
    if not str(raw_candidate).startswith(str(cache_dir.resolve())):
        raise ValueError(
            f"Path traversal detected: resolved path {raw_candidate!r} "
            f"is outside cache dir {cache_dir!r}"
        )

    # Sanitise product_id: keep only safe characters
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in product_id)
    return (cache_dir / (safe_id + suffix)).resolve()


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of *file_path*."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Sidecar writing
# ---------------------------------------------------------------------------


def write_sidecar(
    dest_path: Path,
    *,
    product_id: str,
    source_url: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    sha256: str,
    dataset: str,
    instrument: str,
    mission: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Write a JSON metadata sidecar alongside *dest_path*.

    The sidecar file has the same stem with a ``.json`` extension.
    """
    sidecar_path = dest_path.with_suffix(".json")
    metadata: dict[str, Any] = {
        "product_id": product_id,
        "source_url": source_url,
        "dataset": dataset,
        "instrument": instrument,
        "mission": mission,
        "bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        "sha256": sha256,
        "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
        "talus_version": "1.0.0",
        "disclaimer": (
            "TALUS research/demo system. "
            "NOT certified for flight safety or operational use."
        ),
    }
    if extra:
        metadata["extra"] = extra

    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    log.debug("Sidecar written: %s", sidecar_path)
    return sidecar_path


# ---------------------------------------------------------------------------
# Core download
# ---------------------------------------------------------------------------


def download_dem(
    url: str,
    product_id: str,
    *,
    min_lat: float = 0.0,
    max_lat: float = 0.0,
    min_lon: float = 0.0,
    max_lon: float = 0.0,
    dataset: str = "unknown",
    instrument: str = "unknown",
    mission: str = "unknown",
    expected_sha256: str | None = None,
    cache_dir: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    max_wall_clock_s: float = MAX_DOWNLOAD_WALL_CLOCK_S,
) -> Path:
    """
    Securely download a DEM raster from an allowlisted NASA/LROC host.

    The file is streamed to a temporary path first (``.tmp``), then renamed
    to the final destination only after all checks pass.

    If the file already exists in cache and its SHA-256 matches, the cached
    copy is returned immediately without a new download.

    Parameters
    ----------
    url:
        Download URL. Must be https:// and host must be in ALLOWED_HOSTS.
    product_id:
        ODE product identifier used to name the cached file.
    min_lat, max_lat, min_lon, max_lon:
        Spatial bounds for sidecar metadata.
    dataset, instrument, mission:
        Dataset identifiers for sidecar metadata.
    expected_sha256:
        If provided, the downloaded file must match this SHA-256 hex digest.
    cache_dir:
        Override for the cache directory (default: data/cache/).
    extra_metadata:
        Additional key-value pairs written to the JSON sidecar.

    Returns
    -------
    Path
        Absolute path to the cached raster file.

    Raises
    ------
    ValueError / HostNotAllowedError / SSRFBlockedError:
        On URL/host security violations.
    DownloadSizeLimitError:
        If the file exceeds MAX_DOWNLOAD_BYTES.
    ChecksumMismatchError:
        If expected_sha256 is given and the download does not match.
    httpx.HTTPStatusError / httpx.RequestError:
        On network errors that exhaust redirects.
    """
    validate_url(url)

    resolved_cache_dir = get_cache_dir(cache_dir)
    # Determine file suffix from URL
    url_path = urlparse(url).path
    suffix = Path(url_path).suffix or ".img"
    dest_path = _safe_cache_path(resolved_cache_dir, product_id, suffix)

    # --- Cache hit check ---
    if dest_path.exists():
        cached_sha = compute_sha256(dest_path)
        if expected_sha256 and cached_sha != expected_sha256:
            log.warning(
                "Cache hit for %s but SHA-256 mismatch; re-downloading.", product_id
            )
            dest_path.unlink()
        else:
            log.info("Cache hit for product %s: %s", product_id, dest_path)
            return dest_path

    # --- Streaming download with redirect validation ---
    current_url = url
    redirects_followed = 0
    deadline = time.monotonic() + max_wall_clock_s

    while True:
        validate_url(current_url)
        log.info("Downloading %s -> %s", current_url, dest_path)

        tmp_path = dest_path.with_suffix(".tmp")
        bytes_written = 0

        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=httpx.Timeout(DOWNLOAD_TIMEOUT_S, connect=30.0),
                headers={
                    "User-Agent": (
                        "TALUS/1.0 (research; contact: talus-dev@example.com)"
                    )
                },
            ) as client:
                with client.stream("GET", current_url) as response:
                    # Handle redirects manually so we can validate each hop
                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirects_followed >= MAX_REDIRECTS:
                            raise httpx.TooManyRedirects(
                                f"Exceeded {MAX_REDIRECTS} redirects", request=response.request
                            )
                        redirect_url = response.headers.get("location", "")
                        if not redirect_url:
                            raise ValueError("Redirect with no Location header.")
                        # Resolve relative redirects
                        if redirect_url.startswith("/"):
                            parsed_current = urlparse(current_url)
                            redirect_url = (
                                f"{parsed_current.scheme}://{parsed_current.netloc}"
                                f"{redirect_url}"
                            )
                        log.debug(
                            "Following redirect (%d/%d): %s -> %s",
                            redirects_followed + 1,
                            MAX_REDIRECTS,
                            current_url,
                            redirect_url,
                        )
                        current_url = redirect_url
                        redirects_followed += 1
                        continue

                    response.raise_for_status()

                    # Pre-flight content-length check
                    content_length_str = response.headers.get("content-length", "")
                    if content_length_str:
                        try:
                            content_length = int(content_length_str)
                            if content_length > MAX_DOWNLOAD_BYTES:
                                raise DownloadSizeLimitError(
                                    f"Content-Length {content_length:,} bytes exceeds "
                                    f"limit of {MAX_DOWNLOAD_BYTES:,} bytes."
                                )
                        except ValueError:
                            pass  # Non-integer Content-Length — check during streaming

                    # Stream to temp file
                    with open(tmp_path, "wb") as fh:
                        for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                            bytes_written += len(chunk)
                            if bytes_written > MAX_DOWNLOAD_BYTES:
                                raise DownloadSizeLimitError(
                                    f"Download of {current_url!r} exceeded "
                                    f"{MAX_DOWNLOAD_BYTES:,} bytes limit."
                                )
                            if time.monotonic() > deadline:
                                # A connection that trickles small chunks, each arriving just
                                # under the per-read timeout, would otherwise stay open
                                # indefinitely. This bounds total wall-clock time regardless.
                                raise DownloadTimeoutError(
                                    f"Download of {current_url!r} exceeded its "
                                    f"{max_wall_clock_s:.0f} s wall-clock budget."
                                )
                            fh.write(chunk)

                    # If we reach here, download completed successfully
                    break

        except (DownloadSizeLimitError, HostNotAllowedError, SSRFBlockedError):
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    log.info(
        "Download complete: %d bytes written for product %s", bytes_written, product_id
    )

    # --- Checksum verification ---
    actual_sha256 = compute_sha256(tmp_path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {product_id}: "
            f"expected {expected_sha256!r}, got {actual_sha256!r}"
        )

    # --- Atomic rename to final path ---
    tmp_path.rename(dest_path)
    log.info("File saved: %s (sha256=%s)", dest_path, actual_sha256)

    # --- Write JSON sidecar ---
    write_sidecar(
        dest_path,
        product_id=product_id,
        source_url=url,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        sha256=actual_sha256,
        dataset=dataset,
        instrument=instrument,
        mission=mission,
        extra=extra_metadata,
    )

    return dest_path


# ---------------------------------------------------------------------------
# Convenience: download from DEMProductRecord
# ---------------------------------------------------------------------------


def download_dem_product(
    record: Any,  # DEMProductRecord from ode_search
    expected_sha256: str | None = None,
    cache_dir: Path | None = None,
) -> Path | None:
    """
    Download the raster file for a :class:`~terrain_agent.tools.ode_search.DEMProductRecord`.

    Returns ``None`` if the record has no downloadable file URL.

    Parameters
    ----------
    record:
        A ``DEMProductRecord`` instance from :func:`~terrain_agent.tools.ode_search.search_lunar_dem`.
    expected_sha256:
        Optional expected SHA-256 for integrity checking.
    cache_dir:
        Override for the cache directory.

    Returns
    -------
    Path or None
        Path to the cached raster file, or ``None`` if no file URL is available.
    """
    if not record.file_url:
        log.warning(
            "No direct file URL for product %s; visit %s to download manually.",
            record.product_id,
            record.files_page_url,
        )
        return None

    return download_dem(
        url=record.file_url,
        product_id=record.product_id,
        min_lat=record.min_lat,
        max_lat=record.max_lat,
        min_lon=record.min_lon,
        max_lon=record.max_lon,
        dataset=record.dataset,
        instrument=record.instrument,
        mission=record.mission,
        expected_sha256=expected_sha256,
        cache_dir=cache_dir,
        extra_metadata={"product_type": record.product_type},
    )
