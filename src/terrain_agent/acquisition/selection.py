"""Product selection: which discovered product to download for a request.

Selection is deterministic and explainable. Every product that is not chosen is recorded with
a reason, so a caller can see, for example, that a finer product exists but exceeds the
download size limit. Nothing is guessed: a product with an unknown resolution or an unknown
size is excluded, not assumed.
"""

from __future__ import annotations

from typing import Sequence

from terrain_agent.acquisition.models import (
    CoverageRequest,
    Exclusion,
    FileRole,
    ProductCandidate,
)


def rank_for_request(
    candidates: Sequence[ProductCandidate], request: CoverageRequest
) -> tuple[list[ProductCandidate], list[Exclusion]]:
    """Keep products that fully cover the request and are fine enough, finest first.

    Ties on resolution are broken by product id so the order never depends on the order in
    which the provider returned results.
    """
    usable: list[ProductCandidate] = []
    excluded: list[Exclusion] = []
    for candidate in candidates:
        if candidate.map_scale_m is None:
            excluded.append(Exclusion(product_id=candidate.product_id, reason="resolution_unknown"))
        elif not candidate.covers(request):
            excluded.append(Exclusion(product_id=candidate.product_id, reason="does_not_cover"))
        elif request.max_pixel_size_m is not None and candidate.map_scale_m > request.max_pixel_size_m:
            excluded.append(
                Exclusion(
                    product_id=candidate.product_id,
                    reason="too_coarse",
                    detail=f"{candidate.map_scale_m:g} m per pixel",
                )
            )
        else:
            usable.append(candidate)
    usable.sort(key=lambda c: (c.map_scale_m or 0.0, c.product_id))
    return usable, excluded


def choose_downloadable(
    ranked: Sequence[ProductCandidate], max_download_bytes: int
) -> tuple[ProductCandidate | None, list[Exclusion]]:
    """Pick the finest product whose files are complete and fit the download limit.

    Among products of equal resolution the smaller download is preferred, so the integer
    product is chosen over its larger floating point twin.
    """
    excluded: list[Exclusion] = []
    feasible: list[ProductCandidate] = []
    for candidate in ranked:
        data = candidate.file(FileRole.DATA)
        label = candidate.file(FileRole.LABEL_PDS4)
        if data is None or label is None:
            excluded.append(Exclusion(product_id=candidate.product_id, reason="files_incomplete"))
            continue
        if data.expected_max_bytes is None:
            excluded.append(Exclusion(product_id=candidate.product_id, reason="size_unknown"))
            continue
        if data.expected_max_bytes > max_download_bytes:
            excluded.append(
                Exclusion(
                    product_id=candidate.product_id,
                    reason="exceeds_download_limit",
                    detail=(
                        f"{data.expected_max_bytes / 1e6:.1f} MB, limit "
                        f"{max_download_bytes / 1e6:.1f} MB, {candidate.map_scale_m:g} m per pixel"
                    ),
                )
            )
            continue
        if not _same_location(data.url, label.url) or _stem(data.file_name) != _stem(label.file_name):
            excluded.append(Exclusion(product_id=candidate.product_id, reason="label_mismatch"))
            continue
        feasible.append(candidate)

    if not feasible:
        return None, excluded
    best_scale = min(c.map_scale_m or 0.0 for c in feasible)
    finest = [c for c in feasible if (c.map_scale_m or 0.0) == best_scale]
    finest.sort(key=lambda c: (c.file(FileRole.DATA).expected_max_bytes or 0, c.product_id))  # type: ignore[union-attr]
    chosen = finest[0]
    for other in feasible:
        if other is not chosen:
            excluded.append(Exclusion(product_id=other.product_id, reason="not_selected"))
    return chosen, excluded


def _stem(file_name: str) -> str:
    return file_name.rsplit(".", 1)[0].lower()


def _same_location(url_a: str, url_b: str) -> bool:
    return url_a.rsplit("/", 1)[0].lower() == url_b.rsplit("/", 1)[0].lower()
