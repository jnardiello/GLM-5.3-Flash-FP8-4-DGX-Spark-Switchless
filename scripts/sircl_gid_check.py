#!/usr/bin/env python3
"""Check the explicit single-rail SIRCL selectors before importing vLLM.

Run with python3 -S after sourcing the serving runtime. This deliberately does
not select a replacement GID or import sitecustomize, torch, or model code.
"""

from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys


class SelectionError(ValueError):
    pass


def read_addresses(netdev):
    # The pinned serving image has no iproute2. Linux's read-only interface
    # ioctls expose the primary IPv4 and its netmask without external utilities.
    # The reviewed ring has one primary address per selected interface; a GID
    # for a secondary address is intentionally not admitted by this profile.
    name = netdev.encode("ascii")
    if len(name) > 15:
        raise SelectionError("fabric interface name exceeds Linux IFNAMSIZ")
    request = struct.pack("256s", name)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        address = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, request)[20:24])
        netmask = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x891B, request)[20:24])
    return [ipaddress.IPv4Interface(f"{address}/{netmask}")]


def validate_selection(env, fabric_ifaces, *, sys_class=Path("/sys/class"),
                       address_reader=read_addresses):
    if env.get("VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE") != "single":
        raise SelectionError("this checker requires the reviewed single-rail profile")
    if len(fabric_ifaces) != 2 or len(set(fabric_ifaces)) != 2:
        raise SelectionError("exactly two distinct fabric interfaces are required")
    records = []
    devices = set()
    for slot, expected_netdev in enumerate(fabric_ifaces):
        device = env.get(f"SPARK_TP4_DEVICE{slot}", "")
        index = env.get(f"SPARK_TP4_GID{slot}", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", device):
            raise SelectionError(f"slot {slot}: invalid or missing HCA")
        if device in devices:
            raise SelectionError("the two SIRCL slots must use distinct HCAs")
        devices.add(device)
        if not re.fullmatch(r"[0-9]+", index) or not 0 <= int(index) <= 255:
            raise SelectionError(f"slot {slot}: GID must be an explicit index in [0,255]")
        index = str(int(index))
        port = sys_class / "infiniband" / device / "ports" / "1"
        try:
            state = (port / "state").read_text().rsplit(":", 1)[-1].strip()
            physical = (port / "phys_state").read_text().rsplit(":", 1)[-1].strip()
            layer = (port / "link_layer").read_text().strip()
            if (state, physical.replace(" ", ""), layer) != ("ACTIVE", "LinkUp", "Ethernet"):
                raise SelectionError(f"slot {slot}: port 1 is not active Ethernet")
            gid = ipaddress.IPv6Address((port / "gids" / index).read_text().strip())
            kind = (port / "gid_attrs" / "types" / index).read_text().strip()
            netdev = (port / "gid_attrs" / "ndevs" / index).read_text().strip()
            if kind != "RoCE v2" or gid.ipv4_mapped is None:
                raise SelectionError(f"slot {slot}: configured GID is not IPv4 RoCEv2")
            if netdev != expected_netdev:
                raise SelectionError(f"slot {slot}: configured GID belongs to the wrong netdev")
            addresses = address_reader(netdev)
            matches = [address for address in addresses if address.ip == gid.ipv4_mapped]
            if not matches:
                raise SelectionError(f"slot {slot}: GID does not match an active netdev IPv4 address")
            peer = ipaddress.IPv4Address(env.get(f"SPARK_TP4_PEER{slot}", ""))
            if peer == gid.ipv4_mapped or not any(peer in address.network for address in matches):
                raise SelectionError(f"slot {slot}: peer is not a distinct neighbor on this edge")
        except (OSError, ValueError) as error:
            if isinstance(error, SelectionError):
                raise
            # Keep site addresses and raw subprocess output out of public logs.
            raise SelectionError(f"slot {slot}: cannot validate HCA/GID ({type(error).__name__})") from error
        records.append({"slot": slot, "device": device, "port": 1,
                        "gid_index": int(index), "netdev": netdev,
                        "ipv4_rocev2_and_peer_match": True})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fabric-ifaces", nargs=2, required=True)
    args = parser.parse_args()
    try:
        records = validate_selection(os.environ, args.fabric_ifaces)
    except SelectionError as error:
        print(f"SIRCL GID preflight FAIL: {error}", file=sys.stderr)
        return 1
    print("SIRCL GID preflight PASS: " + json.dumps(records, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
