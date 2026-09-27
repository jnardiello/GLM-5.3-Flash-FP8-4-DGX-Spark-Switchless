#!/usr/bin/env python3
"""Verify a flat Hugging Face snapshot directory against a pinned file manifest (stdlib only).

The manifest lists every file of the pinned revision with its size and either the LFS SHA-256
(large files) or the git blob id (small files), as returned by the Hub's model_info with
files_metadata. Every listed file must exist with the exact size and hash.

    python3 verify_hf_snapshot.py MANIFEST.json SNAPSHOT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    root = Path(sys.argv[2])
    bad = 0
    for f in manifest["files"]:
        p = root / f["path"]
        if not p.is_file():
            print(f"MISSING {f['path']}"); bad += 1; continue
        if f.get("size") is not None and p.stat().st_size != f["size"]:
            print(f"SIZE {f['path']}"); bad += 1; continue
        got, want = (sha256(p), f["sha256"]) if f.get("sha256") else (git_blob(p), f["blob_id"])
        if got != want:
            print(f"HASH {f['path']}"); bad += 1
    total = sum(f.get("size") or 0 for f in manifest["files"])
    print(f"{'OK' if not bad else 'FAIL'} {len(manifest['files'])} files {total} bytes, {bad} problems "
          f"({manifest['repo']}@{manifest['revision']})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
