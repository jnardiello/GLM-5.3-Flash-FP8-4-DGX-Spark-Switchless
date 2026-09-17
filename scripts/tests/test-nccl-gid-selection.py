#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import ipaddress
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("nccl_gid_check", REPO / "scripts/nccl_gid_check.py")
assert SPEC and SPEC.loader
gid_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gid_check)


def add_gid(
    root: Path, hca: str, index: int, gid_type: str, gid: str, netdev: str, port_number: str = "1"
) -> None:
    port = root / "infiniband" / hca / "ports" / port_number
    for relative, value in (
        ("state", "4: ACTIVE"),
        ("phys_state", "5: LinkUp"),
        (f"gid_attrs/types/{index}", gid_type),
        (f"gid_attrs/ndevs/{index}", netdev),
        (f"gids/{index}", gid),
    ):
        path = port / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")


def reader(mapping: dict[str, str]):
    return lambda netdev: {ipaddress.IPv4Address(mapping[netdev])} if netdev in mapping else set()


with tempfile.TemporaryDirectory(prefix="tp4-gid-test.") as temp:
    root = Path(temp) / "sys" / "class"
    add_gid(root, "h0", 1, "RoCE v2", "fe80::1", "f0")
    add_gid(root, "h0", 3, "RoCE v2", "::ffff:10.10.1.1", "f0")
    add_gid(root, "h1", 4, "RoCE v2", "::ffff:10.10.2.1", "f1")
    detail = gid_check.validate_selection(
        "h0,h1", "-1", "f0 f1 duplicate0 duplicate1",
        sys_class=root, address_reader=reader({"f0": "10.10.1.1", "f1": "10.10.2.1"}),
    )
    assert "automatic (-1)" in detail and "h0/1=3" in detail and "h1/1=4" in detail

    explicit_root = Path(temp) / "explicit" / "class"
    add_gid(explicit_root, "h0", 3, "RoCE v2", "::ffff:10.20.1.1", "f0")
    add_gid(explicit_root, "h1", 3, "RoCE v2", "::ffff:10.20.2.1", "f1")
    detail = gid_check.validate_selection(
        "h0,h1", "3", "f0 f1",
        sys_class=explicit_root,
        address_reader=reader({"f0": "10.20.1.1", "f1": "10.20.2.1"}),
    )
    assert "explicit index 3" in detail

    (explicit_root / "infiniband/h1/ports/1/state").write_text("1: DOWN\n", encoding="utf-8")
    try:
        gid_check.validate_selection(
            "h0,h1", "3", "f0 f1", sys_class=explicit_root,
            address_reader=reader({"f0": "10.20.1.1", "f1": "10.20.2.1"}),
        )
    except gid_check.SelectionError:
        pass
    else:
        raise AssertionError("accepted a GID on an inactive HCA port")
    (explicit_root / "infiniband/h1/ports/1/state").write_text("4: ACTIVE\n", encoding="utf-8")

    for malformed in ("", "-2", "abc", "+1", "1x"):
        try:
            gid_check.parse_gid_index(malformed)
        except gid_check.SelectionError:
            pass
        else:
            raise AssertionError(f"accepted malformed GID index: {malformed!r}")

    for hcas, fabrics in (("h0,h0", "f0 f1"), ("h0,h1", "f0 f0")):
        try:
            gid_check.validate_selection(
                hcas, "-1", fabrics, sys_class=root,
                address_reader=reader({"f0": "10.10.1.1", "f1": "10.10.2.1"}),
            )
        except gid_check.SelectionError:
            pass
        else:
            raise AssertionError(f"accepted ambiguous HCA/fabric selection: {hcas}, {fabrics}")

    linklocal_root = Path(temp) / "linklocal" / "class"
    add_gid(linklocal_root, "h0", 1, "RoCE v2", "fe80::1", "f0")
    add_gid(linklocal_root, "h1", 1, "RoCE v2", "fe80::2", "f1")
    try:
        gid_check.validate_selection(
            "h0,h1", "-1", "f0 f1", sys_class=linklocal_root,
            address_reader=reader({"f0": "10.30.1.1", "f1": "10.30.2.1"}),
        )
    except gid_check.SelectionError:
        pass
    else:
        raise AssertionError("accepted link-local-only RoCE v2 GIDs")

    try:
        gid_check.validate_selection(
            "h0,h1", "3", "f0 f1", sys_class=explicit_root,
            address_reader=reader({"f0": "10.99.1.1", "f1": "10.20.2.1"}),
        )
    except gid_check.SelectionError:
        pass
    else:
        raise AssertionError("accepted an IPv4-mapped GID that does not match the netdev address")

    all_ports_root = Path(temp) / "all-ports" / "class"
    add_gid(all_ports_root, "h0", 3, "RoCE v2", "::ffff:10.40.1.1", "f0")
    add_gid(all_ports_root, "h0", 1, "RoCE v2", "fe80::2", "f0", "2")
    add_gid(all_ports_root, "h1", 3, "RoCE v2", "::ffff:10.40.2.1", "f1")
    try:
        gid_check.validate_selection(
            "h0,h1", "-1", "f0 f1", sys_class=all_ports_root,
            address_reader=reader({"f0": "10.40.1.1", "f1": "10.40.2.1"}),
        )
    except gid_check.SelectionError:
        pass
    else:
        raise AssertionError("accepted an active selected HCA port with only a link-local GID")

    one_edge_root = Path(temp) / "one-edge" / "class"
    add_gid(one_edge_root, "h0", 3, "RoCE v2", "::ffff:10.50.1.1", "f0")
    add_gid(one_edge_root, "h1", 3, "RoCE v2", "::ffff:10.50.1.1", "f0")
    try:
        gid_check.validate_selection(
            "h0,h1", "3", "f0 f1", sys_class=one_edge_root,
            address_reader=reader({"f0": "10.50.1.1", "f1": "10.50.2.1"}),
        )
    except gid_check.SelectionError:
        pass
    else:
        raise AssertionError("accepted HCA GIDs that leave one addressed fabric edge uncovered")

    launch_dir = Path(temp) / "launcher"
    launch_dir.mkdir()
    shutil.copy(REPO / "scripts/launcher/launch-glm53-tp4.sh", launch_dir / "launch-glm53-tp4.sh")
    config = (REPO / "cluster.env.example").read_text(encoding="utf-8") + """
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=bob@192.0.2.23
NCCL_IB_HCA_BY_RANK=("h0a,h0b" "h1a,h1b" "h2a,h2b" "h3a,h3b")
NCCL_IB_GID_INDEX_BY_RANK=(-1 3 4 -1)
"""
    (launch_dir / "cluster.env").write_text(config, encoding="utf-8")
    launcher = launch_dir / "launch-glm53-tp4.sh"
    launch_env = {
        "HOME": str(Path(temp) / "home"),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TP4_DRY_RUN": "1",
    }
    for rank, expected in enumerate(("-1", "3", "4", "-1")):
        result = subprocess.run(
            ["bash", str(launcher), str(rank)], env=launch_env,
            check=False, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert f"NCCL_IB_GID_INDEX={expected}" in result.stdout
        assert "NCCL_IB_ROCE_VERSION_NUM=2" in result.stdout
        assert "NCCL_IB_ADDR_FAMILY=AF_INET" in result.stdout
        assert "adaptive_k_scheduler.py" in result.stdout and "NVIDIA_GB10" in result.stdout

    env_file = launch_dir / "conflicting.env"
    env_file.write_text("NCCL_IB_ADDR_RANGE=10.0.0.0/8\n", encoding="utf-8")
    conflicts = (
        "-e NCCL_IB_HCA=wrong0,wrong1",
        "-eNCCL_IB_GID_INDEX=99",
        "--env NCCL_IB_ADDR_FAMILY=AF_INET6",
        "--env=NCCL_IB_ROCE_VERSION_NUM=1",
        "-e NCCL_IB_ADDR_RANGE",
        "--env=NCCL_IB_GID_INDEX",
        f"--env-file={env_file}",
    )
    for conflict in conflicts:
        (launch_dir / "cluster.env").write_text(
            config + f'\nEXTRA_DOCKER_ENV="$EXTRA_DOCKER_ENV {conflict}"\n', encoding="utf-8"
        )
        result = subprocess.run(
            ["bash", str(launcher), "0"], env=launch_env,
            check=False, capture_output=True, text=True,
        )
        assert result.returncode != 0 and "EXTRA_DOCKER_ENV" in result.stderr, (
            conflict, result.returncode, result.stderr
        )

    for malformed in ("-2", "abc"):
        (launch_dir / "cluster.env").write_text(
            config + f"\nNCCL_IB_GID_INDEX_BY_RANK=({malformed} 3 4 -1)\n", encoding="utf-8"
        )
        result = subprocess.run(
            ["bash", str(launcher), "0"], env=launch_env,
            check=False, capture_output=True, text=True,
        )
        assert result.returncode != 0 and "must be -1 or a non-negative integer" in result.stderr

print("test-nccl-gid-selection: PASS")
