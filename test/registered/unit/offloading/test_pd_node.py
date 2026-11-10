"""PD role selection and libc IPv4 enumeration, without serving dependencies."""

import ctypes
import errno
import importlib.util
import io
import socket
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

_PATH = Path(__file__).resolve().parents[4] / "scripts/moe_offload/pd_node.py"
_SPEC = importlib.util.spec_from_file_location("pd_node", _PATH)
pd_node = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pd_node
_SPEC.loader.exec_module(pd_node)


class TestLocalIpv4Interfaces(unittest.TestCase):
    def _libc(self, records, platform):
        entries = [pd_node._Ifaddrs() for _ in records]
        buffers = []
        for index, (name, flags, family, address) in enumerate(records):
            entry = entries[index]
            entry.ifa_name = name.encode() if name else None
            entry.ifa_flags = flags
            if address is not None:
                header = (
                    bytes((16, family))
                    if platform == "darwin"
                    else family.to_bytes(2, sys.byteorder)
                )
                raw = header + b"\x00\x00" + socket.inet_aton(address) + b"\x00" * 8
                buffers.append(ctypes.create_string_buffer(raw))
                entry.ifa_addr = ctypes.addressof(buffers[-1])
            if index + 1 < len(entries):
                entry.ifa_next = ctypes.pointer(entries[index + 1])

        def getifaddrs(out):
            destination = ctypes.cast(
                out, ctypes.POINTER(ctypes.POINTER(pd_node._Ifaddrs))
            )
            destination[0] = ctypes.pointer(entries[0])
            return 0

        libc = Mock()
        libc.getifaddrs.side_effect = getifaddrs
        # Retain test-owned memory until freeifaddrs has been observed.
        libc.keepalive = entries, buffers
        return libc

    def test_all_ipv4_addresses_up_flags_and_aliases(self):
        records = [
            ("lo", 0x9, socket.AF_INET, "127.0.0.1"),
            ("management0", 0x1, socket.AF_INET, "192.168.99.20"),
            ("bond0", 0x1, socket.AF_INET, "10.20.30.10"),
            ("bond0:1", 0x1, socket.AF_INET, "10.20.30.11"),
            ("bond0", 0x1, socket.AF_INET, "10.20.30.10"),
            ("other0", 0x1, socket.AF_INET, "10.20.30.10"),
            ("down0", 0x0, socket.AF_INET, "10.20.30.12"),
            ("ipv6", 0x1, socket.AF_INET6, "10.20.30.13"),
            ("no-address", 0x1, socket.AF_INET, None),
            (None, 0x1, socket.AF_INET, "10.20.30.14"),
        ]
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                libc = self._libc(records, platform)
                with (
                    patch.object(pd_node.sys, "platform", platform),
                    patch.object(pd_node.ctypes, "CDLL", return_value=libc),
                ):
                    self.assertEqual(
                        pd_node.local_ipv4_interfaces(),
                        {
                            "192.168.99.20": {"management0"},
                            "10.20.30.10": {"bond0", "other0"},
                            "10.20.30.11": {"bond0"},
                        },
                    )
                libc.freeifaddrs.assert_called_once()

    def test_getifaddrs_failure_preserves_errno(self):
        libc = Mock()
        libc.getifaddrs.return_value = -1
        with (
            patch.object(pd_node.ctypes, "CDLL", return_value=libc),
            patch.object(pd_node.ctypes, "get_errno", return_value=errno.EPERM),
        ):
            with self.assertRaises(OSError) as error:
                pd_node.local_ipv4_interfaces()
        self.assertEqual(error.exception.errno, errno.EPERM)
        libc.freeifaddrs.assert_not_called()

    def test_iteration_error_still_frees_native_list(self):
        libc = self._libc([("eth0", 0x1, socket.AF_INET, "10.20.30.10")], sys.platform)
        with (
            patch.object(pd_node.ctypes, "CDLL", return_value=libc),
            patch.object(
                pd_node.socket, "inet_ntoa", side_effect=ValueError("invalid address")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "invalid address"):
                pd_node.local_ipv4_interfaces()
        libc.freeifaddrs.assert_called_once()


class TestPdNodeSelection(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            pd_node.Node(name, f"10.20.30.{10 + index}", role, rank)
            for index, (name, role, rank) in enumerate(
                (
                    ("P0_IP", "prefill", "0"),
                    ("P1_IP", "prefill", "1"),
                    ("D0_IP", "decode", "0"),
                    ("D1_IP", "decode", "1"),
                )
            )
        ]

    def test_selects_each_role_and_rank(self):
        for node in self.nodes:
            with (
                self.subTest(name=node.name),
                patch.object(
                    pd_node,
                    "local_ipv4_interfaces",
                    return_value={
                        "192.168.99.20": {"management0"},
                        node.address: {"bond0"},
                    },
                ),
            ):
                self.assertEqual(pd_node.identify_node(self.nodes), (node, "bond0"))

    def test_rejects_missing_ambiguous_or_conflicting_identity(self):
        cases = (
            ({}, {}, "exactly one local node IP match"),
            (
                {node.address: {"eth0"} for node in self.nodes[:2]},
                {},
                "exactly one local node IP match",
            ),
            (
                {self.nodes[1].address: {"eth0"}},
                {"node_rank": "0"},
                "conflicts with detected P1_IP",
            ),
            (
                {self.nodes[0].address: {"eth0"}},
                {"comm_iface": "eth1"},
                "does not own detected IP",
            ),
            ({self.nodes[0].address: {"eth0", "eth1"}}, {}, "multiple interfaces"),
        )
        for interfaces, overrides, message in cases:
            with (
                self.subTest(message=message),
                patch.object(pd_node, "local_ipv4_interfaces", return_value=interfaces),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    pd_node.identify_node(self.nodes, **overrides)

    def test_explicit_interface_resolves_same_ip_on_two_interfaces(self):
        node = self.nodes[1]
        with patch.object(
            pd_node,
            "local_ipv4_interfaces",
            return_value={node.address: {"eth0", "eth1"}},
        ):
            self.assertEqual(
                pd_node.identify_node(self.nodes, node_rank="1", comm_iface="eth1"),
                (node, "eth1"),
            )

    def test_invalid_config_is_rejected_before_enumeration(self):
        for address in ("", "hostname", "::1", "0.0.0.0", "127.0.0.1", "224.0.0.1"):
            with (
                self.subTest(address=address),
                patch.object(pd_node, "local_ipv4_interfaces") as enumerate_interfaces,
            ):
                with self.assertRaises(ValueError):
                    pd_node.identify_node(
                        [pd_node.Node("P_IP", address, "prefill", "0")]
                    )
                enumerate_interfaces.assert_not_called()
        with self.assertRaisesRegex(ValueError, "distinct node addresses"):
            pd_node.identify_node([self.nodes[0], self.nodes[0]])

    def test_cli_has_only_four_output_lines_and_clear_enumeration_errors(self):
        args = ["--node", "P_IP", self.nodes[0].address, "prefill", "0"]
        output = io.StringIO()
        with (
            patch.object(
                pd_node,
                "local_ipv4_interfaces",
                return_value={self.nodes[0].address: {"bond0"}},
            ),
            redirect_stdout(output),
        ):
            pd_node.main(args)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["prefill", "0", self.nodes[0].address, "bond0"],
        )
        error = io.StringIO()
        with (
            patch.object(
                pd_node, "local_ipv4_interfaces", side_effect=OSError("denied")
            ),
            redirect_stderr(error),
        ):
            with self.assertRaises(SystemExit) as exit_status:
                pd_node.main(args)
        self.assertEqual(exit_status.exception.code, 1)
        self.assertIn("Cannot identify local PD node", error.getvalue())


if __name__ == "__main__":
    unittest.main()
