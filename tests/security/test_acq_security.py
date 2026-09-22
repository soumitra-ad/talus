"""Security tests for the NASA acquisition layer: SSRF, host policy, secrets, unrestricted fetching."""

from __future__ import annotations

import inspect
import re
import socket
from pathlib import Path

import pytest

import terrain_agent.acquisition as acquisition_package
from terrain_agent.acquisition import download, ode_provider, service
from terrain_agent.acquisition.errors import HostPolicyError
from terrain_agent.acquisition.net_policy import (
    NASA_DOWNLOAD_HOSTS,
    HostPolicy,
    is_safe_address,
    system_resolver,
)

POLICY = HostPolicy(NASA_DOWNLOAD_HOSTS, lambda host: ["128.252.120.58"])


# ---------------------------------------------------------------------------
# URL syntax and allowlist
# ---------------------------------------------------------------------------


def test_the_download_allowlist_is_only_the_pds_geosciences_data_server():
    assert NASA_DOWNLOAD_HOSTS == frozenset({"pds-geosciences.wustl.edu"})


def test_a_good_url_passes():
    assert POLICY.check("https://pds-geosciences.wustl.edu/lro/x/ldem.img") == "pds-geosciences.wustl.edu"
    assert POLICY.check("https://PDS-Geosciences.WUSTL.edu/lro/x/ldem.img") == "pds-geosciences.wustl.edu"


@pytest.mark.parametrize(
    "url",
    [
        "http://pds-geosciences.wustl.edu/a.img",  # not https
        "ftp://pds-geosciences.wustl.edu/a.img",
        "file:///etc/passwd",
        "//pds-geosciences.wustl.edu/a.img",
        "https://pds-geosciences.wustl.edu.evil.com/a.img",  # look-alike suffix
        "https://evilpds-geosciences.wustl.edu/a.img",
        "https://sub.pds-geosciences.wustl.edu/a.img",  # no subdomain matching
        "https://evil.com/?u=pds-geosciences.wustl.edu",
        "https://evil.com/pds-geosciences.wustl.edu/a.img",
        "https://pds-geosciences.wustl.edu@evil.com/a.img",  # userinfo trick
        "https://user:pw@pds-geosciences.wustl.edu/a.img",
        "https://pds-geosciences.wustl.edu:8443/a.img",
        "https://pds-geosciences.wustl.edu:80/a.img",
        "https://127.0.0.1/a.img",
        "https://[::1]/a.img",
        "https://2130706433/a.img",  # decimal form of 127.0.0.1
        "https://0x7f000001/a.img",
        "https://localhost/a.img",
        "https://169.254.169.254/latest/meta-data",
        "https://pds-geosciences.wustl.edu/a b.img",  # whitespace
        "https://pds-geosciences.wustl.edu/a\n.img",  # control character
        "https://pds-geosciences.wustl.edu/ä.img",  # non-ASCII
        "https://pds-geosciences.wustl.edu/" + "a" * 3000,
        "",
        None,
        123,
    ],
)
def test_forbidden_urls_are_refused(url):
    with pytest.raises(HostPolicyError):
        POLICY.check(url)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Address checks (SSRF)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1", "127.255.255.254", "0.0.0.0", "10.0.0.1", "172.16.5.4", "172.31.255.255",
        "192.168.0.1", "169.254.169.254", "100.64.0.1", "100.127.255.255",  # carrier-grade NAT
        "192.0.0.1", "198.18.0.1", "224.0.0.1", "240.0.0.1", "255.255.255.255",
        "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "ff02::1",
        "::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:10.0.0.1",  # IPv4 wrapped in IPv6
        "2002:7f00:0001::1",  # 6to4 wrapping 127.0.0.1
        "not-an-ip", "", "999.1.1.1",
    ],
)
def test_non_global_addresses_are_refused(address):
    assert not is_safe_address(address)
    with pytest.raises(HostPolicyError):
        HostPolicy(NASA_DOWNLOAD_HOSTS, lambda host: [address]).check("https://pds-geosciences.wustl.edu/a.img")


@pytest.mark.parametrize("address", ["128.252.120.58", "8.8.8.8", "2606:4700:4700::1111"])
def test_global_addresses_are_accepted(address):
    assert is_safe_address(address)


def test_every_resolved_address_must_be_safe():
    mixed = HostPolicy(NASA_DOWNLOAD_HOSTS, lambda host: ["128.252.120.58", "10.0.0.7"])
    with pytest.raises(HostPolicyError):
        mixed.check("https://pds-geosciences.wustl.edu/a.img")


def test_resolution_failure_fails_closed():
    with pytest.raises(HostPolicyError):
        HostPolicy(NASA_DOWNLOAD_HOSTS, lambda host: []).check("https://pds-geosciences.wustl.edu/a.img")

    def boom(host):
        raise HostPolicyError("no dns")

    with pytest.raises(HostPolicyError):
        HostPolicy(NASA_DOWNLOAD_HOSTS, boom).check("https://pds-geosciences.wustl.edu/a.img")


def test_the_system_resolver_turns_dns_failure_into_a_policy_error(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("temporary failure")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(HostPolicyError):
        system_resolver("pds-geosciences.wustl.edu")


# ---------------------------------------------------------------------------
# There is no way to fetch an arbitrary URL, and no credentials are involved
# ---------------------------------------------------------------------------


def test_the_service_accepts_areas_not_urls():
    parameters = inspect.signature(service.NasaDemService.acquire).parameters
    assert set(parameters) == {"self", "request", "product_types", "refresh"}
    assert not any("url" in name for name in parameters)


def test_product_types_cannot_be_used_to_smuggle_query_parameters(mock_nasa):
    import httpx

    from terrain_agent.acquisition.errors import ProviderResponseError
    from terrain_agent.acquisition.models import CoverageRequest
    from terrain_agent.acquisition.ode_provider import OdeProvider

    provider = OdeProvider(
        policy=HostPolicy(frozenset({"oderest.rsl.wustl.edu"}), lambda host: ["128.252.120.58"]),
        client=httpx.Client(transport=mock_nasa.transport),
    )
    for evil in ("GDRDEM&limit=100000", "GDRDEM|SLDEM", "gdrdem", "../x", "GDRDEM\n"):
        with pytest.raises(ProviderResponseError):
            provider.search(CoverageRequest.from_point(-89.9, 0.0, 3000.0), product_types=[evil])
    assert mock_nasa.requests == []


def test_the_acquisition_source_has_no_credential_handling():
    root = Path(inspect.getfile(acquisition_package)).parent
    forbidden = re.compile(r"api[_-]?key|authorization|bearer|password|secret|token|getenv|environ", re.I)
    offenders = []
    for path in root.glob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "parts.username" in line:  # the check that rejects URLs carrying a password
                continue
            if forbidden.search(line) and not line.lstrip().startswith(("#", '"', "*")):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == []


def test_the_user_agent_contains_no_personal_or_placeholder_contact():
    from terrain_agent.acquisition.http import USER_AGENT

    assert "@" not in USER_AGENT and "example" not in USER_AGENT.lower()


def test_hosts_are_constants_in_code_and_not_configurable():
    from terrain_agent.config import NasaConfig

    assert not any("host" in name or "url" in name for name in NasaConfig.model_fields)
    assert "imbrium" not in inspect.getsource(download) + inspect.getsource(ode_provider).split("External_url")[0]


# ---------------------------------------------------------------------------
# A trailing newline must not slip past an anchored pattern
# ---------------------------------------------------------------------------


def test_a_trailing_newline_is_not_accepted_in_provider_fields(mock_nasa):
    import httpx

    from terrain_agent.acquisition.models import CoverageRequest
    from terrain_agent.acquisition.ode_provider import ODE_API_HOSTS, OdeProvider

    entry = mock_nasa.add_product(product_id="ldem_ok", label=b"<xml/>", data=b"0" * 100)
    for field in ("pdsid", "Product_lid", "Data_Set_Id"):
        broken = dict(entry)
        broken[field] = entry[field] + "\n"
        mock_nasa.catalog[:] = [broken]
        provider = OdeProvider(
            policy=HostPolicy(ODE_API_HOSTS, lambda host: ["128.252.120.58"]),
            client=httpx.Client(transport=mock_nasa.transport),
        )
        found = provider.search(CoverageRequest.from_point(-89.9, 0.0, 3000.0), product_types=["GDRDEM"])
        if field == "pdsid":
            assert found == []
        else:
            assert found[0].product_lid != broken["Product_lid"] or field != "Product_lid"
            assert found[0].data_set_id != broken["Data_Set_Id"] or field != "Data_Set_Id"


def test_a_trailing_newline_is_not_accepted_in_download_file_names(tmp_path):
    import httpx

    from terrain_agent.acquisition.download import DownloadManager
    from terrain_agent.acquisition.errors import DownloadError
    from terrain_agent.acquisition.http import HttpSettings

    manager = DownloadManager(policy=POLICY, settings=HttpSettings(), max_bytes=1000, client_factory=lambda: httpx.Client())
    with pytest.raises(DownloadError):
        manager.fetch("https://pds-geosciences.wustl.edu/a.img", tmp_path, "a.img\n")
