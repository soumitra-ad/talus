"""Provider interface for lunar DEM discovery.

The NASA integration lives behind this interface so the rest of the application never depends on
the shape of one provider responses. A provider only discovers products and describes their
files. Downloading, validation, caching and analysis are handled elsewhere.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from terrain_agent.acquisition.models import CoverageRequest, ProductCandidate


class DemProvider(Protocol):
    """Discovery of DEM products and their downloadable files."""

    provider_id: str

    def search(
        self,
        request: CoverageRequest,
        *,
        product_types: Optional[Sequence[str]] = None,
    ) -> list[ProductCandidate]:
        """Return products whose bounding box intersects the request, without file lists."""
        ...

    def attach_files(self, candidates: Sequence[ProductCandidate]) -> list[ProductCandidate]:
        """Return the same candidates with their downloadable files filled in."""
        ...

    def discovery_info(self) -> dict[str, object]:
        """Safe description of how discovery was performed, for provenance."""
        ...
