# SPDX-License-Identifier: Apache-2.0
"""Bounded disk staging for SparkCache's plain page-snapshot format.

The manifest construction in :meth:`StreamingManifestStore.commit_page_snapshot`
is adapted from FujitsuPolycom/sparkcache ``cache_manifest.py`` at commit
66057174301a4759ca3a45207ea41016689449cb.  This version preserves its v2 wire
format while streaming each 64 MiB immutable object through 8 MiB buffers.
"""

from __future__ import annotations

import errno as _stream_errno
import hashlib as _stream_hashlib
import os as _stream_os
import stat as _stream_stat
import tempfile as _stream_tempfile
from pathlib import Path as _StreamPath
from typing import Any as _StreamAny, Mapping as _StreamMapping, Sequence as _StreamSequence

from sparkcache.persistent_context_cache.cache_manifest import (
    FORMAT_ABI as _stream_FORMAT_ABI,
    CacheFormatError as _StreamCacheFormatError,
    CacheIdentity as _StreamCacheIdentity,
    CommitConflict as _StreamCommitConflict,
    CommitReceipt as _StreamCommitReceipt,
    ManifestStore as _UnstreamedManifestStore,
    StateRecord as _StreamStateRecord,
    _PAGE_SNAPSHOT_MANIFEST_SCHEMA as _stream_MANIFEST_SCHEMA,
    _PAGE_SNAPSHOT_OBJECT_BYTES as _stream_OBJECT_BYTES,
    _RootGuard as _StreamRootGuard,
    _canonical_json as _stream_canonical_json,
    _ensure_durable_directory as _stream_ensure_directory,
    _fsync_directory as _stream_fsync_directory,
    _is_page_snapshot_root as _stream_is_snapshot_root,
    _publish_immutable as _stream_publish_immutable,
    _sha256 as _stream_sha256,
    _tracked_publication as _stream_tracked_publication,
    _validate_digest as _stream_validate_digest,
    _validate_page_snapshot_root as _stream_validate_snapshot_root,
)
from sparkcache.spark_context_cache_hybrid import (
    page_snapshot_encoded_size as _stream_snapshot_encoded_size,
)


STREAM_CHUNK_BYTES = 8 << 20


def _stream_drop_cache(descriptor: int, start: int, length: int) -> None:
    advice = getattr(_stream_os, "POSIX_FADV_DONTNEED", None)
    advise = getattr(_stream_os, "posix_fadvise", None)
    if advice is None or advise is None or length <= 0:
        return
    try:
        advise(descriptor, start, length, advice)
    except OSError as error:
        if error.errno not in (
            _stream_errno.EINVAL,
            _stream_errno.ENOSYS,
            getattr(_stream_errno, "ENOTSUP", _stream_errno.EINVAL),
        ):
            raise


def _stream_sync_and_drop(descriptor: int, start: int, length: int) -> None:
    sync = getattr(_stream_os, "fdatasync", _stream_os.fsync)
    sync(descriptor)
    _stream_drop_cache(descriptor, start, length)


def _stream_write_all(descriptor: int, payload: memoryview) -> None:
    written = 0
    while written < len(payload):
        count = _stream_os.write(descriptor, payload[written:])
        if count <= 0:
            raise OSError("short write while staging page snapshot")
        written += count


def _stream_open_readonly(path: _StreamPath) -> int:
    flags = _stream_os.O_RDONLY | getattr(_stream_os, "O_CLOEXEC", 0)
    flags |= getattr(_stream_os, "O_NOFOLLOW", 0)
    return _stream_os.open(path, flags)


def _stream_require_regular_size(descriptor: int, size: int) -> None:
    metadata = _stream_os.fstat(descriptor)
    if not _stream_stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
        raise _StreamCacheFormatError("page snapshot object file geometry differs")


def _stream_pread_into(
    descriptor: int,
    target: memoryview,
    offset: int,
) -> None:
    filled = 0
    while filled < len(target):
        view = target[filled:]
        if hasattr(_stream_os, "preadv"):
            count = _stream_os.preadv(descriptor, [view], offset + filled)
        else:  # pragma: no cover - Python/Linux deployment always has preadv.
            payload = _stream_os.pread(descriptor, len(view), offset + filled)
            count = len(payload)
            view[:count] = payload
        if count <= 0:
            raise _StreamCacheFormatError("page snapshot object file was truncated")
        filled += count


class DiskSnapshot:
    """One private, unlinked snapshot staging file with bounded I/O."""

    def __init__(self, root: _StreamPath, expected_bytes: int) -> None:
        if type(expected_bytes) is not int or expected_bytes <= 0:
            raise ValueError("expected_bytes must be a positive integer")
        self._root = _StreamPath(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._stream = _stream_tempfile.TemporaryFile(dir=self._root, mode="w+b")
        _stream_os.fchmod(self._stream.fileno(), 0o600)
        self._expected_bytes = expected_bytes
        self._written = 0
        self._durable = 0
        self._sealed = False
        self._released = False

    @property
    def total_bytes(self) -> int:
        return self._written

    @property
    def sealed(self) -> bool:
        return self._sealed and not self._released

    def append(self, data: _StreamAny) -> None:
        if self._released or self._sealed:
            raise ValueError("page snapshot is not writable")
        try:
            payload = memoryview(data).cast("B")
        except (TypeError, ValueError) as error:
            raise TypeError("page snapshot append requires contiguous bytes-like data") from error
        if len(payload) > STREAM_CHUNK_BYTES:
            raise ValueError("page snapshot append exceeds STREAM_CHUNK_BYTES")
        if self._written + len(payload) > self._expected_bytes:
            raise ValueError("page snapshot exceeds its declared size")
        consumed = 0
        while consumed < len(payload):
            interval = STREAM_CHUNK_BYTES - (self._written - self._durable)
            piece = payload[consumed : consumed + interval]
            _stream_write_all(self._stream.fileno(), piece)
            self._written += len(piece)
            consumed += len(piece)
            if self._written - self._durable == STREAM_CHUNK_BYTES:
                _stream_sync_and_drop(
                    self._stream.fileno(), self._durable, STREAM_CHUNK_BYTES
                )
                self._durable = self._written

    def seal(self) -> None:
        if self._released:
            raise ValueError("page snapshot was released")
        if self._sealed:
            return
        if self._written != self._expected_bytes:
            self.release()
            raise ValueError("page snapshot size differs from its declaration")
        self._stream.flush()
        if self._durable != self._written:
            _stream_sync_and_drop(
                self._stream.fileno(), self._durable, self._written - self._durable
            )
            self._durable = self._written
        self._sealed = True

    def read_range(self, start: int, end: int) -> bytes:
        if not self.sealed:
            raise ValueError("page snapshot must be sealed before reading")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > self._written
            or end - start > STREAM_CHUNK_BYTES
        ):
            raise ValueError("page snapshot read range is invalid")
        result = bytearray(end - start)
        if result:
            view = memoryview(result)
            _stream_pread_into(self._stream.fileno(), view, start)
            view.release()
            _stream_drop_cache(self._stream.fileno(), start, len(result))
        return bytes(result)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._stream.close()

    def __enter__(self) -> "DiskSnapshot":
        return self

    def __exit__(self, exc_type: _StreamAny, exc: _StreamAny, tb: _StreamAny) -> None:
        self.release()


def _stream_existing_matches(path: _StreamPath, staged_fd: int, size: int) -> bool:
    try:
        existing_fd = _stream_open_readonly(path)
    except OSError as error:
        raise _StreamCommitConflict(
            f"cannot verify existing immutable object {path}"
        ) from error
    try:
        try:
            _stream_require_regular_size(existing_fd, size)
        except _StreamCacheFormatError:
            return False
        for start in range(0, size, STREAM_CHUNK_BYTES):
            length = min(STREAM_CHUNK_BYTES, size - start)
            existing = bytearray(length)
            staged = bytearray(length)
            existing_view = memoryview(existing)
            staged_view = memoryview(staged)
            _stream_pread_into(existing_fd, existing_view, start)
            _stream_pread_into(staged_fd, staged_view, start)
            existing_view.release()
            staged_view.release()
            if existing != staged:
                return False
            _stream_drop_cache(existing_fd, start, length)
            _stream_drop_cache(staged_fd, start, length)
        _stream_require_regular_size(existing_fd, size)
        return True
    finally:
        _stream_os.close(existing_fd)


def _stream_publish_object(
    store: "StreamingManifestStore",
    snapshot: DiskSnapshot,
    start: int,
    end: int,
    snapshot_digest: _StreamAny,
) -> dict[str, _StreamAny]:
    object_root = store.root / "chunks"
    _stream_ensure_directory(object_root)
    temporary = None
    temporary_path = None
    try:
        temporary = _stream_tempfile.NamedTemporaryFile(
            dir=object_root, prefix=".snapshot.writing-", delete=False
        )
        temporary_path = _StreamPath(temporary.name)
        temporary_fd = temporary.fileno()
        _stream_os.fchmod(temporary_fd, 0o600)
        publication = store._active_publication()
        publication.record_staged(end - start)
        digest = _stream_hashlib.sha256()
        for offset in range(start, end, STREAM_CHUNK_BYTES):
            piece_end = min(end, offset + STREAM_CHUNK_BYTES)
            encoded = snapshot.read_range(offset, piece_end)
            digest.update(encoded)
            snapshot_digest.update(encoded)
            _stream_write_all(temporary_fd, memoryview(encoded))
            _stream_sync_and_drop(temporary_fd, offset - start, len(encoded))
        object_digest = digest.hexdigest()
        path = object_root / f"{object_digest}.spcc"
        try:
            _stream_os.link(temporary_path, path)
            publication.record_unique(end - start)
        except FileExistsError:
            if _stream_existing_matches(path, temporary_fd, end - start):
                publication.record_deduplicated(end - start)
            else:
                _stream_os.replace(temporary_path, path)
                publication.record_unique(end - start)
        return {
            "sha256": object_digest,
            "bytes": end - start,
            "encoded_start": start,
            "encoded_end": end,
        }
    finally:
        try:
            if temporary is not None:
                temporary.close()
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
            _stream_fsync_directory(object_root)


class SnapshotReader:
    """Sequential authenticated reader over immutable snapshot objects."""

    def __init__(
        self,
        root: _StreamPath,
        descriptors: _StreamSequence[_StreamMapping[str, _StreamAny]],
        encoded_bytes: int,
        encoded_sha256: str,
    ) -> None:
        self._root = root / "chunks"
        self._descriptors = tuple(descriptors)
        self._encoded_bytes = encoded_bytes
        self._encoded_sha256 = encoded_sha256
        self._descriptor_index = 0
        self._object_offset = 0
        self._position = 0
        self._fd: int | None = None
        self._object_digest: _StreamAny = None
        self._snapshot_digest = _stream_hashlib.sha256()
        self._closed = False
        self._finished = False
        self._preflight_files()

    def _preflight_files(self) -> None:
        for descriptor in self._descriptors:
            fd = _stream_open_readonly(
                self._root / f"{descriptor['sha256']}.spcc"
            )
            try:
                _stream_require_regular_size(fd, int(descriptor["bytes"]))
            finally:
                _stream_os.close(fd)

    def _open_object(self) -> None:
        descriptor = self._descriptors[self._descriptor_index]
        self._fd = _stream_open_readonly(
            self._root / f"{descriptor['sha256']}.spcc"
        )
        try:
            _stream_require_regular_size(self._fd, int(descriptor["bytes"]))
        except BaseException:
            _stream_os.close(self._fd)
            self._fd = None
            raise
        self._object_digest = _stream_hashlib.sha256()
        self._object_offset = 0

    def _finish_object(self) -> None:
        assert self._fd is not None and self._object_digest is not None
        descriptor = self._descriptors[self._descriptor_index]
        try:
            _stream_require_regular_size(self._fd, int(descriptor["bytes"]))
            if self._object_digest.hexdigest() != descriptor["sha256"]:
                raise _StreamCacheFormatError(
                    "page snapshot object checksum mismatch"
                )
        finally:
            _stream_os.close(self._fd)
            self._fd = None
            self._object_digest = None
        self._descriptor_index += 1
        self._object_offset = 0

    def read_exact(self, size: int) -> bytes:
        if self._closed or self._finished:
            raise ValueError("page snapshot reader is closed")
        if type(size) is not int or not 0 <= size <= STREAM_CHUNK_BYTES:
            raise ValueError("read_exact size exceeds STREAM_CHUNK_BYTES")
        if self._position + size > self._encoded_bytes:
            raise _StreamCacheFormatError("page snapshot read exceeds payload")
        result = bytearray(size)
        output = memoryview(result)
        copied = 0
        try:
            while copied < size:
                if self._fd is None:
                    self._open_object()
                descriptor = self._descriptors[self._descriptor_index]
                remaining = int(descriptor["bytes"]) - self._object_offset
                count = min(size - copied, remaining)
                target = output[copied : copied + count]
                assert self._fd is not None and self._object_digest is not None
                _stream_pread_into(self._fd, target, self._object_offset)
                self._object_digest.update(target)
                self._snapshot_digest.update(target)
                _stream_drop_cache(self._fd, self._object_offset, count)
                self._object_offset += count
                self._position += count
                copied += count
                if self._object_offset == int(descriptor["bytes"]):
                    self._finish_object()
            return bytes(result)
        except BaseException:
            self.close()
            raise
        finally:
            output.release()

    def finish(self) -> None:
        if self._finished:
            return
        if self._closed:
            raise ValueError("page snapshot reader is closed")
        try:
            if (
                self._position != self._encoded_bytes
                or self._descriptor_index != len(self._descriptors)
                or self._fd is not None
            ):
                raise _StreamCacheFormatError("page snapshot payload is incomplete")
            if self._snapshot_digest.hexdigest() != self._encoded_sha256:
                raise _StreamCacheFormatError("page snapshot payload checksum mismatch")
            self._finished = True
        finally:
            if not self._finished:
                self.close()

    def close(self) -> None:
        if self._fd is not None:
            _stream_os.close(self._fd)
            self._fd = None
        self._closed = True

    def __enter__(self) -> "SnapshotReader":
        return self

    def __exit__(self, exc_type: _StreamAny, exc: _StreamAny, tb: _StreamAny) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.close()


class StreamingManifestStore(_UnstreamedManifestStore):
    """Pinned ManifestStore with bounded plain-snapshot publication and reads."""

    @_stream_tracked_publication("page_snapshot")
    def commit_page_snapshot(
        self,
        *,
        identity: _StreamCacheIdentity,
        context_digest: str,
        span_tokens: int,
        snapshot: _StreamAny,
    ) -> _StreamCommitReceipt:
        try:
            _stream_validate_digest(context_digest, "context_digest")
            if not isinstance(snapshot, DiskSnapshot) or not snapshot.sealed:
                raise ValueError("streaming page snapshot must be a sealed DiskSnapshot")
            if identity.publication_schema not in (
                "",
                "page-tail-cow-v1",
                "page-tail-cow-v2",
            ):
                raise ValueError("page snapshot publication schema differs")
            if identity.required_records != frozenset(
                (_StreamStateRecord.TARGET_CKV, _StreamStateRecord.LOGICAL_POSITIONS)
            ):
                raise ValueError("page snapshot record schema differs")
            if (
                type(span_tokens) is not int
                or span_tokens <= 0
                or span_tokens % identity.chunk_tokens
            ):
                raise ValueError("page snapshot span must cover complete logical chunks")
            snapshot_bytes = snapshot.total_bytes
            with _StreamRootGuard(self.root, shared=True, blocking=True):
                self._active_publication().describe_payload(snapshot_bytes)
                descriptors: list[dict[str, _StreamAny]] = []
                snapshot_digest = _stream_hashlib.sha256()
                for start in range(0, snapshot_bytes, _stream_OBJECT_BYTES):
                    descriptors.append(
                        _stream_publish_object(
                            self,
                            snapshot,
                            start,
                            min(snapshot_bytes, start + _stream_OBJECT_BYTES),
                            snapshot_digest,
                        )
                    )
                root = {
                    "schema": _stream_MANIFEST_SCHEMA,
                    "format_abi": _stream_FORMAT_ABI,
                    "storage_mode": "block_pages_v1",
                    "identity": identity.to_wire(),
                    "context_digest": context_digest,
                    "committed_tokens": span_tokens,
                    "snapshot_encoded_bytes": snapshot_bytes,
                    "snapshot_object_bytes": _stream_OBJECT_BYTES,
                    "snapshot_objects": descriptors,
                    "snapshot_sha256": snapshot_digest.hexdigest(),
                    "logical_chunk_tokens": identity.chunk_tokens,
                    "logical_chunk_count": span_tokens // identity.chunk_tokens,
                }
                root["metadata_sha256"] = _stream_sha256(
                    _stream_canonical_json(root)
                )
                encoded_root = _stream_canonical_json(root)
                _stream_publish_immutable(
                    self._manifest_path(identity, context_digest), encoded_root
                )
                return _StreamCommitReceipt(
                    manifest_digest=_stream_sha256(encoded_root),
                    committed_tokens=span_tokens,
                    encoded_bytes=len(encoded_root)
                    + sum(int(item["bytes"]) for item in descriptors),
                    allocated_bytes_upper_bound=sum(
                        (size + 4095) // 4096 * 4096
                        for size in (
                            len(encoded_root),
                            *(int(item["bytes"]) for item in descriptors),
                        )
                    ),
                )
        finally:
            if isinstance(snapshot, DiskSnapshot):
                snapshot.release()

    def open_snapshot(
        self,
        lookup: _StreamAny,
        *,
        layout: _StreamAny,
        result_block_counts: _StreamSequence[int],
        result_boundary_tokens: int,
    ) -> SnapshotReader:
        if (
            not lookup.is_hit
            or lookup.root_kind != "page_snapshot"
            or lookup._manifest is None
            or not _stream_is_snapshot_root(lookup._manifest)
        ):
            raise ValueError("cannot stream a non-page-snapshot lookup")
        manifest = lookup._manifest
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = _StreamCacheIdentity(**identity_wire)
        descriptors = _stream_validate_snapshot_root(
            manifest,
            identity=identity,
            context_digest=manifest["context_digest"],
        )
        if manifest["committed_tokens"] != result_boundary_tokens:
            raise _StreamCacheFormatError("page snapshot restore boundary differs")
        expected_bytes = _stream_snapshot_encoded_size(layout, result_block_counts)
        if manifest["snapshot_encoded_bytes"] != expected_bytes:
            raise _StreamCacheFormatError("page snapshot restore geometry differs")
        return SnapshotReader(
            self.root,
            descriptors,
            int(manifest["snapshot_encoded_bytes"]),
            manifest["snapshot_sha256"],
        )
