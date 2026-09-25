#!/usr/bin/env python3
"""Exercise encoder mounts through the native launcher dry-run, without nodes."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "scripts/launcher/launch-glm53-tp4.sh"
TARGET = "/usr/local/lib/python3.12/dist-packages/sparkcache/spark_context_cache_hybrid.py"
CONNECTOR_TARGET = "/usr/local/lib/python3.12/dist-packages/sparkcache/spark_context_cache_connector.py"
template = (REPO / "cluster.env.example").read_text(encoding="utf-8")
site_fixture = '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
'''


def docker_argv(stdout: str) -> list[str]:
    # The native dry-run prints every argv item on its own indented line.
    values = [line[2:] for line in stdout.splitlines() if line.startswith("  ")]
    assert values[:2] == ["sudo", "docker"], values[:4]
    return values


def mounts(argv: list[str], target: str) -> list[str]:
    return [argv[i + 1] for i, value in enumerate(argv[:-1])
            if value == "-v" and argv[i + 1].split(":")[1] == target]


with tempfile.TemporaryDirectory(prefix="tp4-encoder-launcher.") as temp:
    root = Path(temp)
    launcher = root / "launch-glm53-tp4.sh"
    shutil.copyfile(SOURCE, launcher)
    config = root / "cluster.env"
    forbidden_log = root / "forbidden-command.log"
    stub_dir = root / "bin"
    stub_dir.mkdir()
    for name in ("sudo", "docker", "ssh", "systemctl", "curl", "ip", "sysctl"):
        path = stub_dir / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$TP4_TEST_COMMAND_LOG"\nexit 97\n',
                        encoding="utf-8")
        path.chmod(0o700)
    env = dict(os.environ)
    env.pop("TP4_ENV", None)
    env.update(TP4_DRY_RUN="1", TP4_TEST_COMMAND_LOG=str(forbidden_log),
               PATH=str(stub_dir) + os.pathsep + env["PATH"])

    def launch(delta: str = "", rank: int = 0):
        config.write_text(template + site_fixture + delta, encoding="utf-8")
        result = subprocess.run(["bash", str(launcher), str(rank)], env=env,
                                capture_output=True, text=True, timeout=15, check=False)
        assert not forbidden_log.exists(), forbidden_log.read_text() if forbidden_log.exists() else ""
        return result

    for rank in range(4):
        result = launch(rank=rank)
        assert result.returncode == 0, result.stderr
        argv = docker_argv(result.stdout)
        assert len(mounts(argv, TARGET)) == len(mounts(argv, CONNECTOR_TARGET)) == 1
        assert mounts(argv, TARGET)[0].endswith(":ro")
        assert mounts(argv, TARGET)[0].startswith(str(Path.home()) + "/tp4/sparkcache/")
        assert argv[argv.index("--entrypoint") + 1] == "/opt/sircl-serving/entrypoint.sh"
        assert argv[argv.index("--node-rank") + 1] == str(rank)
        assert "--kv-cache-memory-bytes=17179869184" in argv
        assert "--kv-transfer-config" in argv

    mount = f'_TEST_ENCODER_MOUNT="-v $SPARKCACHE_ENCODER:{TARGET}:ro"\n'
    remove = 'EXTRA_DOCKER_ENV="${EXTRA_DOCKER_ENV/$_TEST_ENCODER_MOUNT/}"\n'
    cases = {
        "missing": mount + remove,
        "duplicate_same_source": mount + 'EXTRA_DOCKER_ENV="$EXTRA_DOCKER_ENV $_TEST_ENCODER_MOUNT"\n',
        "duplicate_other_source": f'EXTRA_DOCKER_ENV="$EXTRA_DOCKER_ENV -v /unused/other.py:{TARGET}:ro"\n',
        "wrong_source": mount + remove + f'EXTRA_DOCKER_ENV="$EXTRA_DOCKER_ENV -v /unused/other.py:{TARGET}:ro"\n',
        "writable": mount + remove + f'EXTRA_DOCKER_ENV="$EXTRA_DOCKER_ENV -v $SPARKCACHE_ENCODER:{TARGET}:rw"\n',
    }
    for label, delta in cases.items():
        result = launch(delta)
        assert result.returncode != 0, f"accepted {label} encoder mount"
        assert "encoder must be mounted exactly once" in result.stderr, result.stderr

    # The historical recipe deliberately uses the encoder built into the image.
    result = launch(mount + remove + 'SPARKCACHE_ENCODER=""\nSPARKCACHE_ENCODER_SHA256=""\n')
    assert result.returncode == 0, result.stderr
    assert mounts(docker_argv(result.stdout), TARGET) == []
    assert len(mounts(docker_argv(result.stdout), CONNECTOR_TARGET)) == 1

    # Dry-run intentionally skips file hashing. Exercise the launcher's unchanged
    # native hashing function separately on a tiny local payload.
    source = SOURCE.read_text(encoding="utf-8")
    match = re.search(r"^verify_pinned_file\(\) \{.*?^\}", source, re.M | re.S)
    assert match
    payload = root / "encoder.py"
    payload.write_text("# synthetic encoder\n", encoding="utf-8")
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    command = match.group(0) + '\nverify_pinned_file "$1" "$2" "SparkCache encoder"\n'

    def verify(path: Path, expected: str):
        return subprocess.run(["bash", "-c", command, "encoder-pin", str(path), expected],
                              env=env, capture_output=True, text=True, timeout=5, check=False)

    assert verify(payload, sha).returncode == 0
    assert "SHA-256 mismatch" in verify(payload, "0" * 64).stderr
    assert "no SHA-256 pin" in verify(payload, "").stderr
    alias = root / "encoder-alias.py"
    alias.symlink_to(payload)
    assert "non-symlink" in verify(alias, sha).stderr
    assert not forbidden_log.exists()

print("test-sparkcache-launcher: PASS")
