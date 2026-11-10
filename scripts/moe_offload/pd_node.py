"""Identify a PD worker using stdlib and libc, without network CLI tools."""

import argparse
import ctypes
import errno
import ipaddress
import os
import socket
import sys
from dataclasses import dataclass


class _Ifaddrs(ctypes.Structure):
    pass


_Ifaddrs._fields_ = [
    ("ifa_next", ctypes.POINTER(_Ifaddrs)),
    ("ifa_name", ctypes.c_char_p),
    ("ifa_flags", ctypes.c_uint),
    ("ifa_addr", ctypes.c_void_p),
    ("ifa_netmask", ctypes.c_void_p),
    ("ifa_ifu", ctypes.c_void_p),
    ("ifa_data", ctypes.c_void_p),
]


def local_ipv4_interfaces():
    """Return every IPv4 on an UP, non-loopback interface, including aliases.

    getifaddrs returns one entry per address, unlike SIOCGIFADDR, which only
    returns the primary IPv4. The linked list belongs to libc and must be freed.
    https://man7.org/linux/man-pages/man3/getifaddrs.3.html
    """
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise OSError(errno.ENOTSUP, "PD address enumeration requires Linux or macOS")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(_Ifaddrs))]
    libc.getifaddrs.restype = ctypes.c_int
    libc.freeifaddrs.argtypes = [ctypes.POINTER(_Ifaddrs)]
    libc.freeifaddrs.restype = None
    head = ctypes.POINTER(_Ifaddrs)()
    if libc.getifaddrs(ctypes.byref(head)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"getifaddrs: {os.strerror(code)}")
    local = {}
    try:
        cursor = head
        while cursor:
            entry = cursor.contents
            cursor = entry.ifa_next
            # IFF_UP=0x1, IFF_LOOPBACK=0x8 (Linux and Darwin).
            if not entry.ifa_flags & 0x1 or entry.ifa_flags & 0x8:
                continue
            if not entry.ifa_name or not entry.ifa_addr:
                continue
            header = ctypes.string_at(entry.ifa_addr, 2)
            # Darwin has sa_len + uint8 sa_family; Linux uses native uint16.
            family = (
                header[1]
                if sys.platform == "darwin"
                else int.from_bytes(header, sys.byteorder)
            )
            if family == socket.AF_INET:
                address = socket.inet_ntoa(ctypes.string_at(entry.ifa_addr, 8)[4:8])
                name = os.fsdecode(entry.ifa_name).split(":", 1)[0]
                local.setdefault(address, set()).add(name)
    finally:
        libc.freeifaddrs(head)
    return local


@dataclass(frozen=True)
class Node:
    name: str
    address: str
    role: str
    rank: str


def identify_node(nodes, *, node_rank="", comm_iface=""):
    for node in nodes:
        try:
            address = ipaddress.IPv4Address(node.address)
        except ipaddress.AddressValueError as exc:
            raise ValueError(
                f"Set {node.name} to a valid node IPv4 address (got {node.address!r})"
            ) from exc
        if (
            address.is_loopback
            or address.is_unspecified
            or address.is_multicast
            or node.address == "255.255.255.255"
        ):
            raise ValueError(
                f"{node.name} must be a reachable, non-loopback unicast IPv4 address"
            )
    if len({node.address for node in nodes}) != len(nodes):
        raise ValueError("Node IPs must be distinct node addresses")
    local = local_ipv4_interfaces()
    matches = [node for node in nodes if node.address in local]
    if len(matches) != 1:
        found = ", ".join(f"{node.name}={node.address}" for node in matches) or "none"
        raise ValueError(
            f"Expected exactly one local node IP match, found {found}; "
            "check configuration and container host networking"
        )
    node = matches[0]
    if node_rank and node_rank != node.rank:
        raise ValueError(
            f"NODE_RANK={node_rank} conflicts with detected {node.name} "
            f"(rank {node.rank}); unset NODE_RANK"
        )
    interfaces = local[node.address]
    if comm_iface:
        if comm_iface not in interfaces:
            raise ValueError(
                f"COMM_IFACE={comm_iface} does not own detected IP {node.address}"
            )
        return node, comm_iface
    if len(interfaces) != 1:
        raise ValueError(
            f"IP {node.address} belongs to multiple interfaces; "
            f"set COMM_IFACE to one of {sorted(interfaces)}"
        )
    return node, next(iter(interfaces))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "IP", "ROLE", "RANK"),
    )
    parser.add_argument("--node-rank", default="")
    parser.add_argument("--comm-iface", default="")
    args = parser.parse_args(argv)
    try:
        node, iface = identify_node(
            [Node(*values) for values in args.node],
            node_rank=args.node_rank,
            comm_iface=args.comm_iface,
        )
    except (ValueError, OSError, AttributeError) as exc:
        parser.exit(1, f"ERROR: Cannot identify local PD node: {exc}\n")
    print(node.role, node.rank, node.address, iface, sep="\n")


if __name__ == "__main__":
    main()
