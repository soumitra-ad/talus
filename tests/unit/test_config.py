"""Unit tests for the TALUS configuration layer."""

import os
import sys
import unittest
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from terrain_agent.config import settings, TerrainSettings


class TestConfig(unittest.TestCase):
    def test_default_settings_without_credentials(self):
        """Verify that default settings instantiate cleanly with zero secrets required."""
        s = TerrainSettings()
        self.assertEqual(s.app_name, "Terrain Analysis for Landing and Uncrewed Systems (TALUS)")
        self.assertEqual(s.safety.default_max_slope_deg, 15.0)
        self.assertEqual(s.safety.default_roughness_limit, 0.5)
        self.assertEqual(s.safety.default_rover_clearance_m, 0.35)
        self.assertEqual(s.resources.max_raster_window_size, 2048)
        self.assertEqual(s.resources.max_download_size_bytes, 100 * 1024 * 1024)
        self.assertEqual(s.resources.request_timeout_seconds, 30.0)
        self.assertEqual(s.resources.max_waypoints, 100)
        self.assertIn("ode.rsl.wustl.edu", s.approved_domains)
        self.assertIn("pds-geosciences.wustl.edu", s.approved_domains)

    def test_offline_mode_by_default(self):
        """Verify system defaults to offline demo mode when credentials are absent."""
        orig_key = os.environ.pop("GEMINI_API_KEY", None)
        orig_project = os.environ.pop("VERTEX_PROJECT_ID", None)
        try:
            s = TerrainSettings()
            self.assertFalse(s.is_gemini_available)
        finally:
            if orig_key:
                os.environ["GEMINI_API_KEY"] = orig_key
            if orig_project:
                os.environ["VERTEX_PROJECT_ID"] = orig_project

    def test_environment_variable_override(self):
        """Verify configuration parameters can be overridden via environment variables."""
        os.environ["TALUS_MAX_SLOPE_DEG"] = "18.5"
        os.environ["GEMINI_MODEL"] = "gemini-1.5-pro"
        try:
            s = TerrainSettings()
            self.assertEqual(s.safety.default_max_slope_deg, 18.5)
            self.assertEqual(s.model.model_name, "gemini-1.5-pro")
        finally:
            os.environ.pop("TALUS_MAX_SLOPE_DEG", None)
            os.environ.pop("GEMINI_MODEL", None)


if __name__ == "__main__":
    unittest.main()
