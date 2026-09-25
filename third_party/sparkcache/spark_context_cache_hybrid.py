"""Pure block-page codec for heterogeneous KV-cache groups.

The codec treats every layer page as opaque bytes. Group block tables define
logical ordering; layer names, dtypes, page shapes, bytes-per-page, and the
semantic reuse window define the layout. This preserves compressed attention
and recurrent compressor state without assigning misleading per-token
geometry to either one.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

_MAGIC = b"SPHP1\x00"
_DELTA_MAGIC = b"SPHD1\x00"
_HEADER_LENGTH = struct.Struct("<I")


class HybridCodecError(ValueError):
    """A block-page snapshot does not match its declared layout."""


@dataclass(frozen=True)
class PageLayer:
    name: str
    dtype: str
    page_shape: tuple[int, ...]
    bytes_per_page: int

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.page_shape:
            raise HybridCodecError("page layer metadata is incomplete")
        if self.bytes_per_page <= 0 or any(size <= 0 for size in self.page_shape):
            raise HybridCodecError("page layer geometry must be positive")


@dataclass(frozen=True)
class PageGroup:
    block_size: int
    layers: tuple[PageLayer, ...]
    reuse_policy: str = "full"
    reuse_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.block_size <= 0 or not self.layers:
            raise HybridCodecError("page group geometry is incomplete")
        if self.reuse_policy not in {"full", "sliding", "recurrent_align"}:
            raise HybridCodecError("page group reuse policy is unsupported")
        if self.reuse_policy in {"full", "recurrent_align"} and (
            self.reuse_window_tokens is not None
        ):
            raise HybridCodecError(
                f"{self.reuse_policy} page group cannot declare a reuse window"
            )
        if self.reuse_policy == "sliding" and (
            self.reuse_window_tokens is None or self.reuse_window_tokens <= 1
        ):
            raise HybridCodecError("sliding page group reuse window is invalid")
        names = [layer.name for layer in self.layers]
        if names != sorted(names) or len(names) != len(set(names)):
            raise HybridCodecError("page group layers must be sorted and unique")


@dataclass(frozen=True)
class PageLayout:
    groups: tuple[PageGroup, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise HybridCodecError("page layout has no groups")
        names = [layer.name for group in self.groups for layer in group.layers]
        if len(names) != len(set(names)):
            raise HybridCodecError("a layer belongs to multiple page groups")

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PagePayloadSpan:
    layer_name: str
    group_index: int
    source_start: int
    source_end: int
    page_count: int
    bytes_per_page: int


@dataclass(frozen=True)
class PageSnapshotPlan:
    header_bytes: int
    total_bytes: int
    block_counts: tuple[int, ...]
    spans: tuple[PagePayloadSpan, ...]


@dataclass(frozen=True)
class PageDeltaTail:
    """One authenticated delta payload extent and its final layer offset."""

    layer_name: str
    group_index: int
    source_start: int
    source_end: int
    destination_byte_offset: int
    sha256: str


@dataclass(frozen=True)
class PageDeltaPlan:
    """Validated delta header geometry without materializing its base."""

    header_bytes: int
    total_bytes: int
    base_block_counts: tuple[int, ...]
    result_block_counts: tuple[int, ...]
    base_boundary_tokens: int
    result_boundary_tokens: int
    base_snapshot_sha256: str
    result_snapshot_sha256: str
    reused_pages_by_group: tuple[int, ...]
    tails: tuple[PageDeltaTail, ...]


def page_snapshot_encoded_size(
    layout: PageLayout,
    block_counts: Sequence[int],
) -> int:
    """Return the exact encoded size without allocating page payload bytes."""

    header = encode_page_snapshot_header(layout, block_counts)
    counts = tuple(int(count) for count in block_counts)
    payload_bytes = sum(
        count * layer.bytes_per_page
        for group, count in zip(layout.groups, counts, strict=True)
        for layer in group.layers
    )
    return len(header) + payload_bytes


def encode_page_snapshot_header(
    layout: PageLayout,
    block_counts: Sequence[int],
) -> bytes:
    """Encode only the deterministic SPHP1 header for one page snapshot."""

    if len(block_counts) != len(layout.groups):
        raise HybridCodecError("block-count vector disagrees with page groups")
    counts = tuple(int(count) for count in block_counts)
    if any(count <= 0 for count in counts):
        raise HybridCodecError("every page group must contribute at least one block")
    header = json.dumps(
        {
            "schema": "sparkcache-hybrid-pages/v1",
            "layout_sha256": layout.digest,
            "block_counts": counts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _MAGIC + _HEADER_LENGTH.pack(len(header)) + header


def plan_page_snapshot(
    layout: PageLayout,
    encoded_prefix: bytes | bytearray | memoryview,
    expected_block_counts: Sequence[int],
    *,
    total_bytes: int | None = None,
) -> PageSnapshotPlan:
    """Validate the small SPHP1 header and describe payload byte extents.

    ``encoded_prefix`` may be the complete snapshot or only the header bytes.
    SparkCache CUDA restore uses the latter after authenticating the containing .spcc
    object, avoiding a Python slice for every layer payload.
    """

    prefix = len(_MAGIC) + _HEADER_LENGTH.size
    view = memoryview(encoded_prefix).cast("B")
    if len(view) < prefix or bytes(view[: len(_MAGIC)]) != _MAGIC:
        raise HybridCodecError("hybrid page snapshot has an invalid prefix")
    (header_length,) = _HEADER_LENGTH.unpack_from(view, len(_MAGIC))
    header_end = prefix + header_length
    if header_length <= 0 or header_end > len(view):
        raise HybridCodecError("hybrid page snapshot header is truncated")
    try:
        header = json.loads(bytes(view[prefix:header_end]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridCodecError("hybrid page snapshot header is invalid") from error
    try:
        counts = tuple(int(count) for count in expected_block_counts)
    except (TypeError, ValueError) as error:
        raise HybridCodecError("hybrid page block counts are invalid") from error
    if any(count <= 0 for count in counts):
        raise HybridCodecError("hybrid page block counts must be positive")
    if (
        not isinstance(header, dict)
        or header.get("schema") != "sparkcache-hybrid-pages/v1"
        or header.get("layout_sha256") != layout.digest
        or tuple(header.get("block_counts", ())) != counts
        or len(counts) != len(layout.groups)
    ):
        raise HybridCodecError("hybrid page snapshot identity or block counts differ")
    if total_bytes is None:
        total = len(view)
    elif isinstance(total_bytes, bool) or not isinstance(total_bytes, int):
        raise HybridCodecError("hybrid page total bytes must be an integer")
    else:
        total = total_bytes
    if total < header_end or len(view) > total:
        raise HybridCodecError("hybrid page snapshot total length is invalid")

    spans = []
    offset = header_end
    for group_index, (group, count) in enumerate(zip(layout.groups, counts)):
        for layer in group.layers:
            size = count * layer.bytes_per_page
            end = offset + size
            if end > total:
                raise HybridCodecError(
                    f"hybrid page snapshot is truncated at {layer.name}"
                )
            spans.append(
                PagePayloadSpan(
                    layer_name=layer.name,
                    group_index=group_index,
                    source_start=offset,
                    source_end=end,
                    page_count=count,
                    bytes_per_page=layer.bytes_per_page,
                )
            )
            offset = end
    if offset != total:
        raise HybridCodecError("hybrid page snapshot has trailing bytes")
    return PageSnapshotPlan(
        header_bytes=header_end,
        total_bytes=total,
        block_counts=counts,
        spans=tuple(spans),
    )


def encode_page_snapshot(
    layout: PageLayout,
    block_counts: Sequence[int],
    layer_payloads: Mapping[str, bytes],
) -> bytes:
    if len(block_counts) != len(layout.groups):
        raise HybridCodecError("block-count vector disagrees with page groups")
    counts = tuple(int(count) for count in block_counts)
    if any(count <= 0 for count in counts):
        raise HybridCodecError("every page group must contribute at least one block")
    expected_names = {layer.name for group in layout.groups for layer in group.layers}
    if set(layer_payloads) != expected_names:
        raise HybridCodecError("page payload names disagree with layout")

    parts: list[bytes] = []
    for group, count in zip(layout.groups, counts):
        for layer in group.layers:
            payload = layer_payloads[layer.name]
            expected = count * layer.bytes_per_page
            if len(payload) != expected:
                raise HybridCodecError(
                    f"layer {layer.name} carries {len(payload)} bytes, expected {expected}"
                )
            parts.append(payload)
    return b"".join([encode_page_snapshot_header(layout, counts), *parts])


def decode_page_snapshot(
    layout: PageLayout,
    encoded: bytes,
    expected_block_counts: Sequence[int],
) -> dict[str, bytes]:
    plan = plan_page_snapshot(layout, encoded, expected_block_counts)
    return {
        span.layer_name: encoded[span.source_start : span.source_end]
        for span in plan.spans
    }


def _validate_delta_boundaries(
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> None:
    if (
        isinstance(base_boundary_tokens, bool)
        or isinstance(result_boundary_tokens, bool)
        or not isinstance(base_boundary_tokens, int)
        or not isinstance(result_boundary_tokens, int)
        or base_boundary_tokens <= 0
        or result_boundary_tokens <= base_boundary_tokens
    ):
        raise HybridCodecError("page delta boundaries are invalid")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def plan_page_delta(
    layout: PageLayout,
    encoded_prefix: bytes | bytearray | memoryview,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
    total_bytes: int | None = None,
) -> PageDeltaPlan:
    """Validate a delta header and describe its ordered layer-tail extents.

    ``encoded_prefix`` may contain only the complete header. Callers that
    authenticate macro objects can therefore plan source ranges without
    assembling the complete delta or reading its base snapshot.
    """

    _validate_delta_boundaries(base_boundary_tokens, result_boundary_tokens)
    prefix_bytes = len(_DELTA_MAGIC) + _HEADER_LENGTH.size
    view = memoryview(encoded_prefix).cast("B")
    if len(view) < prefix_bytes or bytes(view[: len(_DELTA_MAGIC)]) != _DELTA_MAGIC:
        raise HybridCodecError("hybrid page delta has an invalid prefix")
    (header_length,) = _HEADER_LENGTH.unpack_from(view, len(_DELTA_MAGIC))
    header_end = prefix_bytes + header_length
    if header_length <= 0 or header_end > len(view):
        raise HybridCodecError("hybrid page delta header is truncated")
    try:
        header = json.loads(bytes(view[prefix_bytes:header_end]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridCodecError("hybrid page delta header is invalid") from error
    expected_keys = {
        "schema",
        "layout_sha256",
        "base_snapshot_sha256",
        "result_snapshot_sha256",
        "base_block_counts",
        "result_block_counts",
        "base_boundary_tokens",
        "result_boundary_tokens",
        "reused_pages_by_group",
        "layer_tails",
    }
    if not isinstance(header, dict) or set(header) != expected_keys:
        raise HybridCodecError("hybrid page delta header fields differ")
    raw_base_counts = header["base_block_counts"]
    raw_result_counts = header["result_block_counts"]
    if (
        not isinstance(raw_base_counts, list)
        or not isinstance(raw_result_counts, list)
        or any(type(value) is not int for value in raw_base_counts)
        or any(type(value) is not int for value in raw_result_counts)
        or type(header["base_boundary_tokens"]) is not int
        or type(header["result_boundary_tokens"]) is not int
    ):
        raise HybridCodecError("hybrid page delta header geometry is invalid")
    try:
        base_counts = tuple(int(value) for value in base_block_counts)
        result_counts = tuple(int(value) for value in result_block_counts)
    except (TypeError, ValueError) as error:
        raise HybridCodecError("hybrid page delta block counts are invalid") from error
    if (
        len(base_counts) != len(layout.groups)
        or len(result_counts) != len(layout.groups)
        or any(
            base <= 0 or result < base
            for base, result in zip(base_counts, result_counts, strict=True)
        )
    ):
        raise HybridCodecError("page delta block counts do not extend the base")
    if (
        header["schema"] != "sparkcache-hybrid-page-delta/v1"
        or header["layout_sha256"] != layout.digest
        or tuple(raw_base_counts) != base_counts
        or tuple(raw_result_counts) != result_counts
        or header["base_boundary_tokens"] != base_boundary_tokens
        or header["result_boundary_tokens"] != result_boundary_tokens
        or not _is_sha256(header["base_snapshot_sha256"])
        or not _is_sha256(header["result_snapshot_sha256"])
    ):
        raise HybridCodecError("hybrid page delta identity or base differs")
    reused = header["reused_pages_by_group"]
    raw_tails = header["layer_tails"]
    expected_layers = [
        (group_index, layer)
        for group_index, group in enumerate(layout.groups)
        for layer in group.layers
    ]
    if (
        not isinstance(reused, list)
        or len(reused) != len(layout.groups)
        or not isinstance(raw_tails, list)
        or len(raw_tails) != len(expected_layers)
    ):
        raise HybridCodecError("hybrid page delta descriptor count differs")
    if total_bytes is None:
        total = len(view)
    elif isinstance(total_bytes, bool) or not isinstance(total_bytes, int):
        raise HybridCodecError("hybrid page delta total bytes must be an integer")
    else:
        total = total_bytes
    if total < header_end or len(view) > total:
        raise HybridCodecError("hybrid page delta total length is invalid")

    tails: list[PageDeltaTail] = []
    source_offset = header_end
    layer_index = 0
    for group_index, group in enumerate(layout.groups):
        reused_pages = reused[group_index]
        if (
            type(reused_pages) is not int
            or not 0 <= reused_pages <= base_counts[group_index]
        ):
            raise HybridCodecError("hybrid page delta reuse count is invalid")
        for layer in group.layers:
            descriptor = raw_tails[layer_index]
            layer_index += 1
            expected_bytes = (
                result_counts[group_index] - reused_pages
            ) * layer.bytes_per_page
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"name", "bytes", "sha256"}
                or descriptor["name"] != layer.name
                or type(descriptor["bytes"]) is not int
                or descriptor["bytes"] != expected_bytes
                or not _is_sha256(descriptor["sha256"])
            ):
                raise HybridCodecError("hybrid page delta layer descriptor differs")
            source_end = source_offset + expected_bytes
            if source_end > total:
                raise HybridCodecError("hybrid page delta payload is truncated")
            tails.append(
                PageDeltaTail(
                    layer_name=layer.name,
                    group_index=group_index,
                    source_start=source_offset,
                    source_end=source_end,
                    destination_byte_offset=reused_pages * layer.bytes_per_page,
                    sha256=descriptor["sha256"],
                )
            )
            source_offset = source_end
    if source_offset != total:
        raise HybridCodecError("hybrid page delta has trailing bytes")
    return PageDeltaPlan(
        header_bytes=header_end,
        total_bytes=total,
        base_block_counts=base_counts,
        result_block_counts=result_counts,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
        base_snapshot_sha256=header["base_snapshot_sha256"],
        result_snapshot_sha256=header["result_snapshot_sha256"],
        reused_pages_by_group=tuple(reused),
        tails=tuple(tails),
    )


def encode_page_delta(
    layout: PageLayout,
    base_snapshot: bytes,
    result_snapshot: bytes,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> bytes:
    """Encode changed page suffixes relative to one authenticated snapshot.

    Reuse is established page-by-page across every layer in a page group. A
    group reuses a page only when the result carries byte-identical opaque
    state at the same logical page index. When the base boundary lies inside
    a page, changed bytes make that complete terminal page part of the delta;
    preceding byte-identical pages remain reusable. The base snapshot digest
    and both semantic boundaries are bound into the delta header.
    """

    _validate_delta_boundaries(base_boundary_tokens, result_boundary_tokens)
    base_counts = tuple(int(value) for value in base_block_counts)
    result_counts = tuple(int(value) for value in result_block_counts)
    if len(base_counts) != len(layout.groups) or len(result_counts) != len(
        layout.groups
    ):
        raise HybridCodecError("page delta block counts disagree with layout")
    if any(
        base <= 0 or result < base for base, result in zip(base_counts, result_counts)
    ):
        raise HybridCodecError("page delta block counts do not extend the base")
    base_payloads = decode_page_snapshot(layout, base_snapshot, base_counts)
    result_payloads = decode_page_snapshot(layout, result_snapshot, result_counts)

    reused_by_group: list[int] = []
    payload_parts: list[bytes] = []
    layer_tails: list[dict[str, object]] = []
    for group_index, group in enumerate(layout.groups):
        reusable = 0
        for page_index in range(base_counts[group_index]):
            if all(
                base_payloads[layer.name][
                    page_index * layer.bytes_per_page : (page_index + 1)
                    * layer.bytes_per_page
                ]
                == result_payloads[layer.name][
                    page_index * layer.bytes_per_page : (page_index + 1)
                    * layer.bytes_per_page
                ]
                for layer in group.layers
            ):
                reusable += 1
                continue
            break
        reused_by_group.append(reusable)
        for layer in group.layers:
            tail = result_payloads[layer.name][reusable * layer.bytes_per_page :]
            payload_parts.append(tail)
            layer_tails.append(
                {
                    "name": layer.name,
                    "bytes": len(tail),
                    "sha256": hashlib.sha256(tail).hexdigest(),
                }
            )
    header = json.dumps(
        {
            "schema": "sparkcache-hybrid-page-delta/v1",
            "layout_sha256": layout.digest,
            "base_snapshot_sha256": hashlib.sha256(base_snapshot).hexdigest(),
            "result_snapshot_sha256": hashlib.sha256(result_snapshot).hexdigest(),
            "base_block_counts": base_counts,
            "result_block_counts": result_counts,
            "base_boundary_tokens": base_boundary_tokens,
            "result_boundary_tokens": result_boundary_tokens,
            "reused_pages_by_group": reused_by_group,
            "layer_tails": layer_tails,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        _DELTA_MAGIC
        + _HEADER_LENGTH.pack(len(header))
        + header
        + b"".join(payload_parts)
    )


def encode_page_delta_from_capture(
    layout: PageLayout,
    base_snapshot: bytes | bytearray | memoryview,
    captured_tail: object,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    reused_pages_by_group: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> bytes:
    """Encode an authenticated delta directly from selected captured pages.

    The caller supplies reuse counts established before native capture. The
    captured payload must contain each layer's non-reused result pages in
    layout order. The result snapshot checksum is computed incrementally from
    the verified base prefixes and captured page bytes, so no complete result
    snapshot or page-by-page comparison is required.
    """

    _validate_delta_boundaries(base_boundary_tokens, result_boundary_tokens)
    try:
        base_counts = tuple(int(value) for value in base_block_counts)
        result_counts = tuple(int(value) for value in result_block_counts)
        reused = tuple(int(value) for value in reused_pages_by_group)
    except (TypeError, ValueError) as error:
        raise HybridCodecError("captured page extension counts are invalid") from error
    if not (
        len(base_counts)
        == len(result_counts)
        == len(reused)
        == len(layout.groups)
    ):
        raise HybridCodecError("captured page extension geometry differs")
    if any(
        base <= 0
        or result < base
        or not 0 <= shared <= base
        for base, result, shared in zip(
            base_counts, result_counts, reused, strict=True
        )
    ):
        raise HybridCodecError("captured page extension counts are invalid")

    base_view = memoryview(base_snapshot).cast("B")
    base_plan = plan_page_snapshot(
        layout,
        base_view,
        base_counts,
        total_bytes=len(base_view),
    )
    total_bytes = getattr(captured_tail, "total_bytes", None)
    read_range = getattr(captured_tail, "read_range", None)
    if isinstance(captured_tail, (bytes, bytearray, memoryview)):
        captured_view = memoryview(captured_tail).cast("B")
        total_bytes = len(captured_view)

        def read_range(start: int, end: int) -> bytes:
            return captured_view[start:end].tobytes()

    if (
        type(total_bytes) is not int
        or total_bytes <= 0
        or not callable(read_range)
    ):
        raise HybridCodecError("captured page extension payload is invalid")

    result_digest = hashlib.sha256(encode_page_snapshot_header(layout, result_counts))
    payload_parts: list[bytes] = []
    layer_tails: list[dict[str, object]] = []
    cursor = 0
    span_index = 0
    for group_index, group in enumerate(layout.groups):
        shared = reused[group_index]
        tail_pages = result_counts[group_index] - shared
        for layer in group.layers:
            span = base_plan.spans[span_index]
            span_index += 1
            prefix_end = span.source_start + shared * layer.bytes_per_page
            prefix = base_view[span.source_start : prefix_end]
            tail_bytes = tail_pages * layer.bytes_per_page
            end = cursor + tail_bytes
            if end > total_bytes:
                raise HybridCodecError("captured page extension payload is truncated")
            tail = read_range(cursor, end)
            if not isinstance(tail, bytes) or len(tail) != tail_bytes:
                raise HybridCodecError("captured page extension range differs")
            result_digest.update(prefix)
            result_digest.update(tail)
            payload_parts.append(tail)
            layer_tails.append(
                {
                    "name": layer.name,
                    "bytes": tail_bytes,
                    "sha256": hashlib.sha256(tail).hexdigest(),
                }
            )
            cursor = end
    if cursor != total_bytes:
        raise HybridCodecError("captured page extension has trailing bytes")

    header = json.dumps(
        {
            "schema": "sparkcache-hybrid-page-delta/v1",
            "layout_sha256": layout.digest,
            "base_snapshot_sha256": hashlib.sha256(base_view).hexdigest(),
            "result_snapshot_sha256": result_digest.hexdigest(),
            "base_block_counts": base_counts,
            "result_block_counts": result_counts,
            "base_boundary_tokens": base_boundary_tokens,
            "result_boundary_tokens": result_boundary_tokens,
            "reused_pages_by_group": reused,
            "layer_tails": layer_tails,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        _DELTA_MAGIC
        + _HEADER_LENGTH.pack(len(header))
        + header
        + b"".join(payload_parts)
    )


def apply_page_delta(
    layout: PageLayout,
    base_snapshot: bytes,
    encoded_delta: bytes,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> bytes:
    """Verify and apply one page-semantic delta to its exact base snapshot."""

    plan = plan_page_delta(
        layout,
        encoded_delta,
        base_block_counts=base_block_counts,
        result_block_counts=result_block_counts,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
        total_bytes=len(encoded_delta),
    )
    if plan.base_snapshot_sha256 != hashlib.sha256(base_snapshot).hexdigest():
        raise HybridCodecError("hybrid page delta identity or base differs")
    base_payloads = decode_page_snapshot(
        layout,
        base_snapshot,
        plan.base_block_counts,
    )
    encoded_view = memoryview(encoded_delta)
    result_payloads: dict[str, bytes] = {}
    layers = [layer for group in layout.groups for layer in group.layers]
    for tail, layer in zip(plan.tails, layers, strict=True):
        payload_tail = encoded_view[tail.source_start : tail.source_end]
        if hashlib.sha256(payload_tail).hexdigest() != tail.sha256:
            raise HybridCodecError("hybrid page delta payload checksum mismatch")
        prefix = base_payloads[layer.name][: tail.destination_byte_offset]
        result_payloads[layer.name] = prefix + payload_tail.tobytes()
    result = encode_page_snapshot(
        layout,
        plan.result_block_counts,
        result_payloads,
    )
    if hashlib.sha256(result).hexdigest() != plan.result_snapshot_sha256:
        raise HybridCodecError("hybrid page delta result checksum mismatch")
    return result


def materialize_page_extension_capture(
    layout: PageLayout,
    base_snapshot: bytes,
    captured_tail: object,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    reused_pages_by_group: Sequence[int],
) -> bytes:
    """Build a complete snapshot from verified base bytes and captured pages."""

    base_counts = tuple(int(value) for value in base_block_counts)
    result_counts = tuple(int(value) for value in result_block_counts)
    reused = tuple(int(value) for value in reused_pages_by_group)
    if not (
        len(base_counts)
        == len(result_counts)
        == len(reused)
        == len(layout.groups)
    ):
        raise HybridCodecError("captured page extension geometry differs")
    if any(
        base <= 0
        or result < base
        or not 0 <= shared <= base
        for base, result, shared in zip(
            base_counts, result_counts, reused, strict=True
        )
    ):
        raise HybridCodecError("captured page extension counts are invalid")
    total_bytes = getattr(captured_tail, "total_bytes", None)
    read_range = getattr(captured_tail, "read_range", None)
    if isinstance(captured_tail, (bytes, bytearray, memoryview)):
        view = memoryview(captured_tail)
        total_bytes = len(view)

        def read_range(start: int, end: int) -> bytes:
            return view[start:end].tobytes()

    if (
        type(total_bytes) is not int
        or total_bytes <= 0
        or not callable(read_range)
    ):
        raise HybridCodecError("captured page extension payload is invalid")
    base_payloads = decode_page_snapshot(layout, base_snapshot, base_counts)
    result_payloads: dict[str, bytes] = {}
    cursor = 0
    for group_index, group in enumerate(layout.groups):
        shared = reused[group_index]
        tail_pages = result_counts[group_index] - shared
        for layer in group.layers:
            tail_bytes = tail_pages * layer.bytes_per_page
            end = cursor + tail_bytes
            if end > total_bytes:
                raise HybridCodecError("captured page extension payload is truncated")
            tail = read_range(cursor, end)
            if not isinstance(tail, bytes) or len(tail) != tail_bytes:
                raise HybridCodecError("captured page extension range differs")
            prefix = base_payloads[layer.name][
                : shared * layer.bytes_per_page
            ]
            result_payloads[layer.name] = prefix + tail
            cursor = end
    if cursor != total_bytes:
        raise HybridCodecError("captured page extension has trailing bytes")
    return encode_page_snapshot(layout, result_counts, result_payloads)


def split_snapshot(encoded: bytes, part_count: int) -> tuple[bytes, ...]:
    if part_count <= 0 or len(encoded) < part_count:
        raise HybridCodecError("hybrid snapshot cannot cover the declared span")
    width, remainder = divmod(len(encoded), part_count)
    parts = []
    offset = 0
    for index in range(part_count):
        size = width + (1 if index < remainder else 0)
        parts.append(encoded[offset : offset + size])
        offset += size
    return tuple(parts)
