"""Errors raised by the NASA DEM acquisition layer.

Every message is written to be safe to show to a user or a language model. Messages never
contain file system paths, credentials, or raw remote text.
"""

from __future__ import annotations

from typing import Any


class AcquisitionError(Exception):
    """Base class for acquisition failures."""


# ---------------------------------------------------------------------------
# Provider (search and metadata)
# ---------------------------------------------------------------------------


class ProviderError(AcquisitionError):
    """The data provider could not answer a query."""


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached, timed out, or returned an HTTP error."""


class ProviderResponseError(ProviderError):
    """The provider answered with something that is not a valid documented response."""


class NoCoverageError(ProviderError):
    """The provider lists no supported product that intersects the requested area."""


class NoSuitableProductError(ProviderError):
    """Products exist, but none can be used under the configured limits.

    ``excluded`` lists safe summaries of every rejected product so the caller can see why,
    for example that a finer product exists but exceeds the download size limit.
    """

    def __init__(self, message: str, excluded: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.excluded = excluded or []


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class DownloadError(AcquisitionError):
    """A download failed."""


class HostPolicyError(DownloadError):
    """A URL or the address it resolves to is not permitted."""


class DownloadTooLargeError(DownloadError):
    """The file is larger than the configured limit."""


class DownloadIncompleteError(DownloadError):
    """The received size does not match what the server or the metadata declared."""


class DownloadTimeoutError(DownloadError):
    """The download exceeded its time budget."""


class DownloadChecksumError(DownloadError):
    """A checksum supplied with the metadata does not match the downloaded file."""


# ---------------------------------------------------------------------------
# Validation, cache, configuration
# ---------------------------------------------------------------------------


class DemValidationError(AcquisitionError):
    """A downloaded product failed validation and must not be used."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


class CacheError(AcquisitionError):
    """The local cache could not store or retrieve an entry."""


class AcquisitionDisabledError(AcquisitionError):
    """Network acquisition is switched off in this deployment."""
