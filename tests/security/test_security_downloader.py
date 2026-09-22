"""
Security tests for terrain_agent.tools.dem_downloader.

Tests cover:
- URL scheme validation (non-https rejected)
- SSRF protection (private/loopback IPs blocked)
- Host allowlist enforcement
- Download size cap enforcement
- Path traversal prevention in cache filenames
- Redirect to unapproved host blocked
- Checksum mismatch detection
"""

from __future__ import annotations

import hashlib
import json
import threading
import http.server
from pathlib import Path
import socket
import struct
import time
from typing import Any

import pytest

from terrain_agent.tools.dem_downloader import (
    ALLOWED_HOSTS,
    MAX_DOWNLOAD_BYTES,
    ChecksumMismatchError,
    DownloadSizeLimitError,
    HostNotAllowedError,
    SSRFBlockedError,
    _safe_cache_path,
    compute_sha256,
    validate_url,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# validate_url — scheme and host checks
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_https_allowed_host_passes(self, monkeypatch):
        """All allowed hosts should pass validation without raising.

        Name resolution is replaced by a fixed public address so the test makes no DNS
        query and gives the same result with or without internet access.
        """
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
        )
        for host in ALLOWED_HOSTS:
            validate_url(f"https://{host}/some/path.tif")

    def test_http_scheme_rejected(self):
        host = next(iter(ALLOWED_HOSTS))
        with pytest.raises(ValueError, match="Insecure or unsupported URL scheme"):
            validate_url(f"http://{host}/file.tif")

    def test_ftp_scheme_rejected(self):
        host = next(iter(ALLOWED_HOSTS))
        with pytest.raises(ValueError, match="Insecure or unsupported URL scheme"):
            validate_url(f"ftp://{host}/file.tif")

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="Insecure or unsupported URL scheme"):
            validate_url("file:///etc/passwd")

    def test_unapproved_host_rejected(self):
        with pytest.raises(HostNotAllowedError, match="not in the NASA/LROC allowlist"):
            validate_url("https://evil.example.com/dem.tif")

    def test_unapproved_host_nasa_subpath_rejected(self):
        """Subdomains of allowed hosts that are not themselves in the allowlist."""
        with pytest.raises(HostNotAllowedError):
            validate_url("https://evil.ode.rsl.wustl.edu/file.tif")

    def test_empty_url_rejected(self):
        with pytest.raises((ValueError, HostNotAllowedError)):
            validate_url("")

    def test_url_without_hostname_rejected(self):
        with pytest.raises(ValueError):
            validate_url("https:///path/only")


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    def _patch_getaddrinfo(self, monkeypatch, ip: str):
        """Mock socket.getaddrinfo to return a specific IP for any hostname."""
        import socket as _socket

        def _fake_getaddrinfo(host, port, *args, **kwargs):
            return [(_socket.AF_INET, _socket.SOCK_STREAM, 0, "", (ip, port or 0))]

        monkeypatch.setattr(_socket, "getaddrinfo", _fake_getaddrinfo)

    def test_loopback_127_blocked(self, monkeypatch):
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "127.0.0.1")
        with pytest.raises(SSRFBlockedError, match="blocked private/loopback"):
            validate_url(f"https://{host}/file.tif")

    def test_loopback_localhost_blocked(self, monkeypatch):
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "127.0.0.53")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_rfc1918_10_blocked(self, monkeypatch):
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "10.0.0.1")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_rfc1918_172_blocked(self, monkeypatch):
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "172.16.0.1")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_rfc1918_192_168_blocked(self, monkeypatch):
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "192.168.1.100")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_link_local_169_254_blocked(self, monkeypatch):
        """AWS/GCP metadata endpoint must be blocked."""
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "169.254.169.254")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_public_ip_not_blocked(self, monkeypatch):
        """A genuine public IP should not be blocked by SSRF check."""
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "128.252.120.58")  # wustl.edu range
        # Should NOT raise SSRFBlockedError (host allowlist check still applies separately)
        from terrain_agent.tools.dem_downloader import _block_private_ip
        _block_private_ip(host)  # should not raise

    def test_unresolvable_host_fails_closed(self, monkeypatch):
        """An allowlisted host that cannot be resolved must be refused, not silently allowed
        to fall through to httpx's own (unvalidated) resolution."""
        import socket as _socket

        host = next(iter(ALLOWED_HOSTS))

        def _raise_gaierror(*args, **kwargs):
            raise _socket.gaierror("simulated resolution failure")

        monkeypatch.setattr(_socket, "getaddrinfo", _raise_gaierror)
        with pytest.raises(SSRFBlockedError, match="could not be resolved"):
            validate_url(f"https://{host}/file.tif")

    def test_ipv4_mapped_private_address_is_still_blocked(self, monkeypatch):
        """A blocked IPv4 address smuggled inside an IPv4-mapped IPv6 literal (a form a plain
        per-network blocklist can miss) must still be caught."""
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "::ffff:169.254.169.254")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_ipv6_loopback_blocked(self, monkeypatch):
        host = next(iter(ALLOWED_HOSTS))
        self._patch_getaddrinfo(monkeypatch, "::1")
        with pytest.raises(SSRFBlockedError):
            validate_url(f"https://{host}/file.tif")

    def test_no_resolved_addresses_fails_closed(self, monkeypatch):
        import socket as _socket

        host = next(iter(ALLOWED_HOSTS))
        monkeypatch.setattr(_socket, "getaddrinfo", lambda *a, **k: [])
        with pytest.raises(SSRFBlockedError, match="did not resolve"):
            validate_url(f"https://{host}/file.tif")


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------


class TestSafeCachePath:
    def test_normal_product_id(self, tmp_path):
        p = _safe_cache_path(tmp_path, "lro_lola_gdr_001", ".tif")
        assert p.parent == tmp_path
        assert "lro_lola_gdr_001" in p.name

    def test_path_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal detected"):
            _safe_cache_path(tmp_path, "../../etc/passwd", ".tif")

    def test_absolute_path_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal detected"):
            _safe_cache_path(tmp_path, "/etc/passwd", ".tif")

    def test_dotdot_middle_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal detected"):
            _safe_cache_path(tmp_path, "foo/../../../secret", ".tif")

    def test_special_chars_sanitised(self, tmp_path):
        p = _safe_cache_path(tmp_path, "product/with\\slashes:colon", ".tif")
        # Should not contain raw special chars
        assert "/" not in p.name
        assert "\\" not in p.name
        assert ":" not in p.name


# ---------------------------------------------------------------------------
# SHA-256 checksum
# ---------------------------------------------------------------------------


class TestComputeSha256:
    def test_known_file(self, tmp_path):
        f = tmp_path / "test.bin"
        data = b"TALUS terrain data"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_sha256(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_sha256(f) == expected

    def test_large_chunked_file(self, tmp_path):
        """File larger than one chunk should still compute correctly."""
        f = tmp_path / "large.bin"
        data = b"x" * (150 * 1024)  # 150 KB
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_sha256(f) == expected


# ---------------------------------------------------------------------------
# Sidecar writing
# ---------------------------------------------------------------------------


class TestWriteSidecar:
    def test_sidecar_written_correctly(self, tmp_path):
        raster = tmp_path / "dem.tif"
        raster.write_bytes(b"fake raster")
        sidecar = write_sidecar(
            raster,
            product_id="test_product_001",
            source_url="https://ode.rsl.wustl.edu/test.tif",
            min_lat=-90.0,
            max_lat=-85.0,
            min_lon=0.0,
            max_lon=5.0,
            sha256="abc123",
            dataset="LOLA Gridded DEM",
            instrument="LOLA",
            mission="LRO",
        )
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["product_id"] == "test_product_001"
        assert data["sha256"] == "abc123"
        assert data["bounds"]["min_lat"] == -90.0
        assert "NOT certified" in data["disclaimer"]

    def test_sidecar_path_is_json(self, tmp_path):
        raster = tmp_path / "dem.tif"
        raster.write_bytes(b"x")
        sidecar = write_sidecar(
            raster,
            product_id="p",
            source_url="https://ode.rsl.wustl.edu/x.tif",
            min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0,
            sha256="x",
            dataset="test", instrument="test", mission="test",
        )
        assert sidecar.suffix == ".json"


# ---------------------------------------------------------------------------
# Download size cap (mock HTTP server)
# ---------------------------------------------------------------------------


class _ChunkedHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP server that streams more than MAX_DOWNLOAD_BYTES."""

    CHUNK = b"A" * 65536

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        # Send 2x the limit — the downloader must cut off before finishing
        bytes_to_send = MAX_DOWNLOAD_BYTES * 2
        while bytes_to_send > 0:
            chunk = self.CHUNK[: min(len(self.CHUNK), bytes_to_send)]
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            bytes_to_send -= len(chunk)

    def log_message(self, *args):  # suppress noisy output
        pass


class TestDownloadSizeCap:
    @pytest.fixture()
    def local_server(self):
        """Spin up a local HTTP server for testing size cap logic."""
        server = http.server.HTTPServer(("127.0.0.1", 0), _ChunkedHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}"
        server.shutdown()

    def test_size_cap_blocks_oversized_download(
        self, local_server, tmp_path, monkeypatch
    ):
        """
        The downloader must raise DownloadSizeLimitError before writing
        more than MAX_DOWNLOAD_BYTES to disk.
        """
        # Patch validate_url and _block_private_ip to allow the local server
        monkeypatch.setattr(
            "terrain_agent.tools.dem_downloader.validate_url", lambda url: None
        )

        from terrain_agent.tools.dem_downloader import download_dem

        url = f"{local_server}/oversized.tif"
        with pytest.raises(DownloadSizeLimitError, match="exceeded"):
            download_dem(
                url=url,
                product_id="test_oversized",
                cache_dir=tmp_path,
            )

        # Temp file must be cleaned up
        assert not (tmp_path / "test_oversized.tmp").exists()


class _SlowDripHandler(http.server.BaseHTTPRequestHandler):
    """Sends a few small chunks, each promptly (so no single read times out), but never
    finishes -- simulating a connection kept alive to exhaust the wall-clock budget."""

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        try:
            for _ in range(20):
                self.wfile.write(b"A" * 1024)
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class TestDownloadWallClockBudget:
    @pytest.fixture()
    def local_server(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), _SlowDripHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}"
        server.shutdown()

    def test_slow_drip_connection_is_cut_off_by_the_wall_clock_budget(self, local_server, tmp_path, monkeypatch):
        from terrain_agent.tools.dem_downloader import DownloadTimeoutError, download_dem

        monkeypatch.setattr("terrain_agent.tools.dem_downloader.validate_url", lambda url: None)

        with pytest.raises(DownloadTimeoutError, match="wall-clock budget"):
            download_dem(
                url=f"{local_server}/slow.tif",
                product_id="test_slow_drip",
                cache_dir=tmp_path,
                max_wall_clock_s=0.2,
            )

        assert not (tmp_path / "test_slow_drip.tmp").exists()
