#!/usr/bin/env python3
"""Regression cases for stale SIRCL selectors and wrong-but-present GIDs."""
import importlib.util
import ipaddress
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import socket

SPEC = importlib.util.spec_from_file_location(
    "sircl_gid_check", Path(__file__).resolve().parents[1] / "sircl_gid_check.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {"VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "single"}
        for slot in range(2):
            self.env.update({f"SPARK_TP4_DEVICE{slot}": f"h{slot}",
                             f"SPARK_TP4_GID{slot}": "3",
                             f"SPARK_TP4_PEER{slot}": f"192.0.{slot + 2}.2"})
            for name, value in {
                "state": "4: ACTIVE", "phys_state": "5: LinkUp", "link_layer": "Ethernet",
                "gids/3": f"::ffff:192.0.{slot + 2}.1", "gid_attrs/types/3": "RoCE v2",
                "gid_attrs/ndevs/3": f"f{slot}",
            }.items():
                self.write(slot, name, value)

    def write(self, slot, name, value):
        p = self.root / "infiniband" / f"h{slot}" / "ports/1" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def validate(self):
        return check.validate_selection(
            self.env, ["f0", "f1"], sys_class=self.root,
            address_reader=lambda n: [ipaddress.IPv4Interface(f"192.0.{int(n[1]) + 2}.1/24")])

    def test_valid(self):
        self.assertEqual([r["gid_index"] for r in self.validate()], [3, 3])

    def test_primary_address_probe_needs_no_ip_utility(self):
        replies = [b'\0' * 20 + socket.inet_aton(value) + b'\0' * 232
                   for value in ['192.0.2.1', '255.255.255.0']]
        with patch.object(check.fcntl, 'ioctl', side_effect=replies) as probe:
            self.assertEqual(check.read_addresses('f0'), [ipaddress.IPv4Interface('192.0.2.1/24')])
        self.assertEqual([call.args[1] for call in probe.call_args_list], [0x8915, 0x891B])

    def test_stale_index_is_not_replaced_automatically(self):
        self.env["SPARK_TP4_GID0"] = "4"
        with self.assertRaises(check.SelectionError): self.validate()

    def test_wrong_but_present_gid(self):
        for value in ["::", "fe80::1", "::ffff:192.0.2.99"]:
            with self.subTest(value=value):
                self.write(0, "gids/3", value)
                with self.assertRaises(check.SelectionError): self.validate()

    def test_roce_v1_rejected(self):
        self.write(0, "gid_attrs/types/3", "IB/RoCE v1")
        with self.assertRaises(check.SelectionError): self.validate()

    def test_wrong_netdev_rejected(self):
        self.write(0, "gid_attrs/ndevs/3", "duplicate0")
        with self.assertRaises(check.SelectionError): self.validate()

    def test_inactive_port_rejected(self):
        self.write(0, "state", "1: DOWN")
        with self.assertRaises(check.SelectionError): self.validate()

    def test_wrong_peer_rejected(self):
        for value in ["192.0.2.1", "192.0.3.2", ""]:
            with self.subTest(value=value):
                self.env["SPARK_TP4_PEER0"] = value
                with self.assertRaises(check.SelectionError): self.validate()

    def test_duplicate_hca_rejected(self):
        self.env["SPARK_TP4_DEVICE1"] = "h0"
        with self.assertRaises(check.SelectionError): self.validate()

    def test_unsupported_mode_and_indices_rejected(self):
        for value in ["-1", "256", "", "3.0"]:
            with self.subTest(value=value):
                self.env["SPARK_TP4_GID0"] = value
                with self.assertRaises(check.SelectionError): self.validate()
        self.env["SPARK_TP4_GID0"] = "3"
        self.env["VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE"] = "dual"
        with self.assertRaises(check.SelectionError): self.validate()


if __name__ == "__main__":
    unittest.main()
