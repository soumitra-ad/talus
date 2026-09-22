"""Network host policy for acquisition: which hosts may be contacted and which addresses are safe.

The policy is stricter than the earlier downloader helper:

* https only, no user information, no explicit port other than 443, no IP-literal hosts.
* The host must be on an explicit allowlist. There is no wildcard and no subdomain matching.
* The host name is resolved and every address must be a global unicast address. Loopback,
  private, link-local, carrier-grade NAT, reserved, multicast, unspecified and IPv6 addresses
  that wrap a private IPv4 address are all refused.
* If the host name cannot be resolved the request is refused. The policy fails closed.

The resolver is injectable so tests never need the network. A residual limitation is that the
HTTP client resolves the name again when it connects. An attacker would need to control DNS for
an allowlisted NASA host name to exploit that gap.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Callable, Iterable
from urllib.parse import urlsplit

from terrain_agent.acquisition.errors import HostPolicyError

Resolver = Callable[[str], Iterable[str]]

#: Hosts that product files may be downloaded from. This is the PDS Geosciences Node data
#: server. Mirrors such as imbrium.mit.edu are deliberately not included.
NASA_DOWNLOAD_HOSTS = frozenset({"pds-geosciences.wustl.edu"})


def system_resolver(hostname: str) -> list[str]:
    """Resolve a host name to IP address strings using the operating system."""
    try:
        infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HostPolicyError(f"Host {hostname!r} could not be resolved.") from exc
    return [str(info[4][0]) for info in infos]


def is_safe_address(address: str) -> bool:
    """True only for a global unicast address."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv6Address) and ip.sixtofour is not None:
        ip = ip.sixtofour
    return bool(ip.is_global and not ip.is_multicast and not ip.is_unspecified)


@dataclass(frozen=True)
class HostPolicy:
    """Allowlist plus address checks for outbound requests."""

    allowed_hosts: frozenset[str]
    resolver: Resolver = system_resolver

    def check_url(self, url: str) -> str:
        """Validate the syntax and host of *url* and return the host name.

        Raises
        ------
        HostPolicyError
            If the URL is not an https URL to an allowlisted host name.
        """
        if not isinstance(url, str) or not url or len(url) > 2048:
            raise HostPolicyError("The URL is missing or too long.")
        if any(ord(ch) < 33 or ord(ch) > 126 for ch in url):
            raise HostPolicyError("The URL contains characters that are not permitted.")
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError as exc:
            raise HostPolicyError("The URL is malformed.") from exc
        if parts.scheme != "https":
            raise HostPolicyError("Only https URLs are permitted.")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise HostPolicyError("URLs with user information are not permitted.")
        if port not in (None, 443):
            raise HostPolicyError("Only the default https port is permitted.")
        host = (parts.hostname or "").lower()
        if not host:
            raise HostPolicyError("The URL has no host.")
        if host not in self.allowed_hosts:
            raise HostPolicyError(f"Host {host!r} is not on the acquisition allowlist.")
        return host

    def check_resolution(self, host: str) -> None:
        """Refuse the host if any address it resolves to is not a global unicast address."""
        addresses = list(self.resolver(host))
        if not addresses:
            raise HostPolicyError(f"Host {host!r} did not resolve to any address.")
        for address in addresses:
            if not is_safe_address(str(address)):
                raise HostPolicyError(
                    f"Host {host!r} resolves to an address that is not permitted."
                )

    def check(self, url: str) -> str:
        """Full check performed before every request and every redirect hop."""
        host = self.check_url(url)
        self.check_resolution(host)
        return host
