"""HTTP client construction and retry helpers shared by the provider and the downloader."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

USER_AGENT = "TALUS-terrain-agent/0.1 (research; lunar terrain analysis)"

#: HTTP status codes worth retrying. Other 4xx codes are permanent failures.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HttpSettings:
    """Timeouts and retry bounds for outbound requests.

    ``read_timeout_s`` applies to each individual read, so a slow but steady download is not
    cut off by it. ``total_timeout_s`` bounds the wall-clock time of one download attempt.
    """

    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0
    total_timeout_s: float = 900.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 30.0

    def timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_timeout_s,
            read=self.read_timeout_s,
            write=self.read_timeout_s,
            pool=self.connect_timeout_s,
        )


def backoff_delay(attempt: int, settings: HttpSettings) -> float:
    """Exponential backoff without jitter, so retry timing is deterministic."""
    return float(min(settings.backoff_cap_s, settings.backoff_base_s * (2**attempt)))


def build_client(settings: HttpSettings) -> httpx.Client:
    """Create a client that never follows redirects and never accepts compressed bodies.

    Redirects are followed by the caller so every hop can be checked against the host policy.
    Requesting the identity encoding means the bytes counted against the size limit are the
    bytes on the wire, which removes decompression bombs as a risk.
    """
    return httpx.Client(
        follow_redirects=False,
        timeout=settings.timeout(),
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
