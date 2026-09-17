#!/usr/bin/env python3
"""Validate the configured NCCL RoCE GID selection against local sysfs."""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable


class SelectionError(ValueError):
    pass


def parse_gid_index(value: str) -> int:
    if value == "-1":
        return -1
    if not re.fullmatch(r"[0-9]+", value):
        raise SelectionError("NCCL_IB_GID_INDEX must be -1 or a non-negative integer")
    return int(value, 10)


def _hca_port(spec: str) -> tuple[str, str | None]:
    device, separator, port = spec.rpartition(":")
    if separator and device and port.isdecimal():
        return device, str(int(port, 10))
    return spec, None


def _read_ipv4(netdev: str) -> set[ipaddress.IPv4Address]:
    try:
        result = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", netdev],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return set()
    addresses: set[ipaddress.IPv4Address] = set()
    if result.returncode != 0:
        return addresses
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[2] != "inet":
            continue
        try:
            addresses.add(ipaddress.ip_interface(fields[3]).ip)
        except ValueError:
            continue
    return addresses


def validate_selection(
    hca_value: str,
    gid_value: str,
    fabric_value: str,
    *,
    sys_class: Path = Path("/sys/class"),
    address_reader: Callable[[str], set[ipaddress.IPv4Address]] = _read_ipv4,
) -> str:
    gid_index = parse_gid_index(gid_value)
    hcas = hca_value.split(",")
    if len(hcas) < 2 or any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", item) for item in hcas):
        raise SelectionError("NCCL_IB_HCA must name at least two comma-separated HCAs")
    normalized_hcas = [_hca_port(item) for item in hcas]
    if len(set(normalized_hcas)) != len(normalized_hcas):
        raise SelectionError("NCCL_IB_HCA contains a duplicate HCA/port selection")
    fabric_list = fabric_value.split()
    if len(fabric_list) < 2:
        raise SelectionError("resolved fabric interface list has fewer than two entries")
    # The repository contract defines the first two entries as the addressed ring ports.
    fabric_ifaces = set(fabric_list[:2])
    if len(fabric_ifaces) != 2:
        raise SelectionError("the two addressed fabric interface entries must be distinct")

    selected: list[str] = []
    failures: list[str] = []
    covered_ifaces: set[str] = set()
    addresses: dict[str, set[ipaddress.IPv4Address]] = {}
    for hca_spec in hcas:
        hca, requested_port = _hca_port(hca_spec)
        ports_root = sys_class / "infiniband" / hca / "ports"
        ports: Iterable[Path]
        if requested_port is None:
            ports = sorted(ports_root.glob("*"), key=lambda path: path.name)
        else:
            ports = [ports_root / requested_port]
        active_ports = 0
        for port in ports:
            if not port.is_dir():
                continue
            try:
                state = (port / "state").read_text(encoding="utf-8").strip().upper()
                physical = (port / "phys_state").read_text(encoding="utf-8").strip().upper()
            except OSError:
                continue
            state_name = state.rsplit(":", 1)[-1].strip()
            physical_name = physical.rsplit(":", 1)[-1].replace(" ", "")
            if state_name != "ACTIVE" or physical_name != "LINKUP":
                continue
            active_ports += 1
            type_root = port / "gid_attrs" / "types"
            if gid_index < 0:
                type_files = sorted(
                    (path for path in type_root.glob("*") if path.name.isdecimal()),
                    key=lambda path: int(path.name, 10),
                )
            else:
                type_files = [type_root / str(gid_index)]
            port_matches: list[str] = []
            for type_file in type_files:
                try:
                    gid_type = type_file.read_text(encoding="utf-8").strip()
                    ndev = (port / "gid_attrs" / "ndevs" / type_file.name).read_text(
                        encoding="utf-8"
                    ).strip()
                    gid = ipaddress.IPv6Address(
                        (port / "gids" / type_file.name).read_text(encoding="utf-8").strip()
                    )
                except (OSError, ValueError):
                    continue
                mapped = gid.ipv4_mapped
                if gid_type != "RoCE v2" or mapped is None or ndev not in fabric_ifaces:
                    continue
                if ndev not in addresses:
                    addresses[ndev] = address_reader(ndev)
                if mapped not in addresses[ndev]:
                    continue
                port_matches.append(f"{hca_spec}/{port.name}={type_file.name}({ndev},{mapped})")
                covered_ifaces.add(ndev)
            if port_matches:
                selected.extend(port_matches)
            else:
                wanted = "automatic" if gid_index < 0 else f"index {gid_index}"
                failures.append(
                    f"{hca_spec}/{port.name}: no usable {wanted} IPv4-mapped RoCE v2 fabric GID"
                )
        if active_ports == 0:
            failures.append(f"{hca_spec}: no active LINK_UP HCA port")
    missing_ifaces = sorted(fabric_ifaces - covered_ifaces)
    if missing_ifaces:
        failures.append("no selected HCA GID covers addressed fabric netdev(s): " + ",".join(missing_ifaces))

    if failures:
        raise SelectionError("; ".join(failures))
    mode = "automatic (-1)" if gid_index < 0 else f"explicit index {gid_index}"
    return f"{mode}, AF_INET/RoCEv2: " + ", ".join(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcas", required=True)
    parser.add_argument("--gid-index", required=True)
    parser.add_argument("--fabric-ifaces", required=True)
    args = parser.parse_args()
    try:
        print(validate_selection(args.hcas, args.gid_index, args.fabric_ifaces))
    except SelectionError as exc:
        print(f"NCCL HCA/GID selection invalid: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
