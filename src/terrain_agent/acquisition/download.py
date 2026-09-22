"""Controlled file download.

Properties:

* Streams to a file. The body is never held in memory.
* Every request and every redirect hop passes the host policy, including a fresh address check.
* Redirects are followed manually, at most three.
* Separate connect and read timeouts, plus a wall-clock budget for the whole attempt.
* Bounded retries with deterministic exponential backoff, for timeouts, connection failures
  and retryable HTTP status codes only.
* A maximum size, enforced from the declared length and again while streaming.
* Identity encoding only, so the counted bytes are the wire bytes and compressed responses
  are refused.
* The received size must equal the declared content length and lie in the range implied by the
  provider metadata. A checksum is verified when one is supplied.
* The SHA-256 of what was received is always computed and returned.
* A failed download leaves no partial file behind.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin

import httpx

from terrain_agent.acquisition.errors import (
    DownloadChecksumError,
    DownloadError,
    DownloadIncompleteError,
    DownloadTimeoutError,
    DownloadTooLargeError,
)
from terrain_agent.acquisition.http import (
    RETRYABLE_STATUS,
    HttpSettings,
    backoff_delay,
    build_client,
)
from terrain_agent.acquisition.net_policy import HostPolicy

log = logging.getLogger(__name__)

MAX_REDIRECTS = 3
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,149}$")
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


class _RetryableError(Exception):
    """Internal marker for failures worth another attempt."""


class _TruncatedError(_RetryableError):
    """The connection ended before the declared length arrived. Worth another attempt."""


@dataclass(frozen=True)
class DownloadedFile:
    """A completed, size-verified download."""

    path: Path
    size_bytes: int
    sha256: str
    url: str


class DownloadManager:
    """Downloads one file at a time into a directory the caller controls."""

    def __init__(
        self,
        *,
        policy: HostPolicy,
        settings: HttpSettings,
        max_bytes: int,
        client_factory: Optional[Callable[[], httpx.Client]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        chunk_size: int = 65536,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._policy = policy
        self._settings = settings
        self._max_bytes = max_bytes
        self._client_factory = client_factory or (lambda: build_client(settings))
        self._sleep = sleep
        self._clock = clock
        self._chunk_size = chunk_size

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def fetch(
        self,
        url: str,
        dest_dir: Path,
        file_name: str,
        *,
        expected_max_bytes: Optional[int] = None,
        expected_sha256: Optional[str] = None,
        max_bytes: Optional[int] = None,
    ) -> DownloadedFile:
        """Download *url* to ``dest_dir / file_name`` and verify it.

        ``expected_max_bytes`` is the upper bound of the size implied by provider metadata in
        kilobytes. The received size must be within 1000 bytes below it, because the provider
        rounds sizes up to whole kilobytes.

        Raises
        ------
        HostPolicyError, DownloadTooLargeError, DownloadIncompleteError,
        DownloadTimeoutError, DownloadChecksumError, DownloadError
        """
        if not _FILE_NAME_RE.fullmatch(file_name) or ".." in file_name:
            raise DownloadError("The target file name is not permitted.")
        dest_dir = Path(dest_dir).resolve()
        target = (dest_dir / file_name).resolve()
        if target.parent != dest_dir:
            raise DownloadError("The target path is outside the download directory.")
        limit = min(self._max_bytes, max_bytes) if max_bytes else self._max_bytes

        self._policy.check(url)
        client = self._client_factory()
        last_error: Optional[BaseException] = None
        try:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    size, digest, final_url = self._attempt(client, url, target, limit)
                    self._verify(size, digest, expected_max_bytes, expected_sha256)
                    return DownloadedFile(target, size, digest, final_url)
                except (
                    httpx.TimeoutException,
                    httpx.TransportError,
                    _RetryableError,
                ) as exc:
                    last_error = exc
                    target.unlink(missing_ok=True)
                    if attempt < self._settings.max_retries:
                        log.info("Download attempt %d failed (%s); retrying.", attempt + 1, type(exc).__name__)
                        self._sleep(backoff_delay(attempt, self._settings))
            if isinstance(last_error, _TruncatedError):
                raise DownloadIncompleteError(
                    "The download ended before the declared length arrived, on every attempt."
                )
            raise DownloadError(
                f"The download failed after {self._settings.max_retries + 1} attempts "
                f"({type(last_error).__name__})."
            )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        finally:
            client.close()

    # ------------------------------------------------------------------
    # One attempt
    # ------------------------------------------------------------------

    def _attempt(
        self, client: httpx.Client, url: str, target: Path, limit: int
    ) -> tuple[int, str, str]:
        deadline = self._clock() + self._settings.total_timeout_s
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            self._policy.check(current)
            with client.stream("GET", current, headers={"Accept-Encoding": "identity"}) as response:
                status = response.status_code
                if status in _REDIRECT_STATUS:
                    location = response.headers.get("location")
                    if hop >= MAX_REDIRECTS or not location:
                        raise DownloadError("The server sent too many or malformed redirects.")
                    current = urljoin(current, location)
                    continue
                if status in RETRYABLE_STATUS:
                    raise _RetryableError(f"HTTP {status}")
                if status != 200:
                    raise DownloadError(f"The server answered with HTTP status {status}.")
                encoding = response.headers.get("content-encoding", "identity").strip().lower()
                if encoding not in ("", "identity"):
                    raise DownloadError("The server sent an encoded body that was not requested.")
                declared = response.headers.get("content-length")
                content_length = int(declared) if declared and declared.isdigit() else None
                if content_length is not None and content_length > limit:
                    raise DownloadTooLargeError(
                        f"The file is {content_length / 1e6:.1f} MB, above the limit of "
                        f"{limit / 1e6:.1f} MB."
                    )
                size, digest = self._stream(response, target, limit, deadline)
                if content_length is not None and size != content_length:
                    raise _TruncatedError("size differs from the declared content length")
                return size, digest, current
        raise DownloadError("The server sent too many redirects.")  # pragma: no cover

    def _stream(
        self, response: httpx.Response, target: Path, limit: int, deadline: float
    ) -> tuple[int, str]:
        hasher = hashlib.sha256()
        size = 0
        with open(target, "wb") as handle:
            for chunk in response.iter_raw(self._chunk_size):
                size += len(chunk)
                if size > limit:
                    raise DownloadTooLargeError(
                        f"The download exceeded the limit of {limit / 1e6:.1f} MB."
                    )
                if self._clock() > deadline:
                    raise DownloadTimeoutError(
                        f"The download exceeded its time budget of {self._settings.total_timeout_s:.0f} s."
                    )
                hasher.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        return size, hasher.hexdigest()

    @staticmethod
    def _verify(
        size: int, digest: str, expected_max_bytes: Optional[int], expected_sha256: Optional[str]
    ) -> None:
        if size == 0:
            raise DownloadIncompleteError("The downloaded file is empty.")
        if expected_max_bytes is not None and not (expected_max_bytes - 1000 < size <= expected_max_bytes):
            raise DownloadIncompleteError(
                "The received size does not match the size listed in the product metadata."
            )
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise DownloadChecksumError("The SHA-256 of the file does not match the expected value.")


__all__ = ["DownloadManager", "DownloadedFile"]
