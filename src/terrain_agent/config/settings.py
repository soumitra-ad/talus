"""Configuration layer for TALUS.

Provides typed, environment-configurable application settings with secure-by-default
fallbacks. No secrets or credentials are required to instantiate or use the system in demo mode.
"""

from pathlib import Path
from typing import Optional, Set
import os

from pydantic import BaseModel, Field

# Load a root-level .env file, if one exists, into the process environment before any
# os.getenv() call below reads it. python-dotenv never overrides a variable that is already
# set in the real environment (override=False, the default), so explicit environment
# variables (e.g. set by a deployment platform or Secret Manager) always win over .env. This
# never prints, logs, or otherwise reveals the values it loads. The .env file itself is
# git-ignored (see .gitignore) and is never required -- every setting has a safe default.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    pass


class SafetyConfig(BaseModel):
    """Configurable mission safety parameters."""
    default_max_slope_deg: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_MAX_SLOPE_DEG", "15.0")),
        description="Configured maximum allowable slope threshold in degrees (mission specific)",
        ge=0.0,
        le=90.0,
    )
    default_roughness_limit: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_ROUGHNESS_LIMIT", "0.5")),
        description="Configured terrain roughness index (TRI) upper bound",
        ge=0.0,
    )
    default_rover_clearance_m: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_ROVER_CLEARANCE_M", "0.35")),
        description="Nominal rover belly clearance in meters",
        gt=0.0,
    )


class ResourceLimitsConfig(BaseModel):
    """Resource bounds to protect against memory exhaustion and oversized downloads."""
    max_raster_window_size: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_MAX_WINDOW_SIZE", "2048")),
        description="Maximum bounding box dimension (cells) for windowed raster reads. "
        "Capped at 2048 -- AGENTS.md §5 fixes this as a hard architectural ceiling, not an "
        "operator-adjustable tuning knob, so this field cannot be configured above it even "
        "though the actual enforcement in terrain.resource_safety.MAX_RASTER_DIM is a "
        "separate, always-2048 constant that does not read this setting.",
        ge=64,
        le=2048,
    )
    max_download_size_bytes: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_MAX_DOWNLOAD_BYTES", str(100 * 1024 * 1024))),
        description="Maximum single tile download size in bytes (100 MB default)",
        gt=0,
    )
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_REQUEST_TIMEOUT_SEC", "30.0")),
        description="Network request and tool timeout limit in seconds",
        gt=0.0,
        le=120.0,
    )
    max_waypoints: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_MAX_WAYPOINTS", "100")),
        description="Maximum allowable waypoints in a single traverse route. Capped at 100 to "
        "match the fixed ceiling actually enforced by "
        "terrain.resource_safety.MAX_WAYPOINTS, which does not read this setting.",
        ge=2,
        le=100,
    )
    max_cache_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_MAX_CACHE_MB", "2048")),
        description="Total tile cache size quota in megabytes",
        gt=0,
    )


class AgentLimitsConfig(BaseModel):
    """Bounds on the conversational agent loop itself, independent of the deterministic
    terrain tools' own resource limits (``ResourceLimitsConfig``)."""
    max_message_chars: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_AGENT_MAX_MESSAGE_CHARS", "4000")),
        description="Maximum length of a single user chat message, in characters.",
        ge=1,
    )
    max_tool_iterations: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_AGENT_MAX_ITERATIONS", "10")),
        description="Maximum number of tool-call rounds in one agent turn before the loop "
        "is terminated with a bounded-iteration message.",
        ge=1,
        le=50,
    )
    max_requests_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_AGENT_RATE_LIMIT_PER_MIN", "20")),
        description="Maximum agent chat turns per minute, per process, before requests are "
        "rejected with a rate-limited response instead of reaching the model.",
        ge=1,
    )
    max_tool_calls_per_turn: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_AGENT_MAX_TOOL_CALLS_PER_TURN", "30")),
        description="Maximum total individual tool invocations across one chat turn, on top "
        "of max_tool_iterations -- bounds a single model turn that requests an unusually "
        "large batch of parallel tool calls in one round.",
        ge=1,
    )
    max_dispatch_calls_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_DISPATCH_RATE_LIMIT_PER_MIN", "120")),
        description="Process-wide backstop on deterministic tool invocations per minute, "
        "applied inside dispatch_tool_call itself so it covers every caller -- the "
        "conversational agent and any UI that calls tools directly -- not just chat turns.",
        ge=1,
    )
    max_history_messages: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_AGENT_MAX_HISTORY_MESSAGES", "40")),
        description="Maximum prior conversation turns sent to the model as history. Older "
        "turns are dropped (the most recent context is kept) rather than sent unbounded, "
        "which would grow the request size and cost with every turn.",
        ge=0,
    )
    max_network_tool_calls_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_NETWORK_TOOL_RATE_LIMIT_PER_MIN", "10")),
        description="Stricter process-wide limit specifically on tools that contact NASA "
        "(search_dem_products, fetch_nasa_dem), to protect the upstream service and the "
        "local download budget regardless of which caller is making the requests.",
        ge=1,
    )


class ModelConfig(BaseModel):
    """Google Gemini & AI orchestration settings."""
    model_name: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        description="Configurable LLM model identifier",
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="Optional Gemini API key. If omitted, system runs in offline deterministic demo mode.",
    )
    vertex_project_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("VERTEX_PROJECT_ID"),
        description="Optional Google Cloud Project ID for Vertex AI Application Default Credentials (ADC).",
    )
    vertex_location: str = Field(
        default_factory=lambda: os.getenv("VERTEX_LOCATION", "us-central1"),
        description="Google Cloud Vertex AI region",
    )
    temperature: float = Field(
        default=0.1,
        description="Low temperature for reproducible, deterministic reasoning",
        ge=0.0,
        le=1.0,
    )


class StoragePathsConfig(BaseModel):
    """Local filesystem paths for cache, sample data, and metadata."""
    cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TALUS_CACHE_DIR", "data/cache")).resolve()
    )
    sample_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TALUS_SAMPLE_DIR", "data/sample")).resolve()
    )
    metadata_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TALUS_METADATA_DIR", "data/metadata")).resolve()
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TALUS_OUTPUT_DIR", "outputs")).resolve()
    )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class NasaConfig(BaseModel):
    """NASA DEM acquisition settings. The download size limit and cache quota are in
    ``ResourceLimitsConfig``. The NASA hosts themselves are fixed in code and are not
    configurable, so configuration cannot redirect downloads elsewhere."""
    downloads_enabled: bool = Field(
        default_factory=lambda: _env_flag(
            "TALUS_NASA_DOWNLOADS", os.getenv("TALUS_ENV", "development") != "production"
        ),
        description="Allow the application to download NASA DEMs. Off by default in production.",
    )
    connect_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_NASA_CONNECT_TIMEOUT_SEC", "10")),
        gt=0.0,
        le=120.0,
    )
    read_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_NASA_READ_TIMEOUT_SEC", "60")),
        gt=0.0,
        le=600.0,
        description="Timeout for each individual network read",
    )
    total_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_NASA_TOTAL_TIMEOUT_SEC", "900")),
        gt=0.0,
        le=7200.0,
        description="Wall-clock budget for one file download attempt",
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("TALUS_NASA_MAX_RETRIES", "3")), ge=0, le=5
    )
    backoff_base_s: float = Field(
        default_factory=lambda: float(os.getenv("TALUS_NASA_BACKOFF_SEC", "1.0")), gt=0.0, le=30.0
    )


class TerrainSettings(BaseModel):
    """Root configuration for TALUS."""
    app_name: str = "Terrain Analysis for Landing and Uncrewed Systems (TALUS)"
    app_version: str = "0.1.0"
    environment: str = Field(
        default_factory=lambda: os.getenv("TALUS_ENV", "development"),
        description="Execution environment (development, test, production)",
    )
    
    # Nested configurations
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    agent: AgentLimitsConfig = Field(default_factory=AgentLimitsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: StoragePathsConfig = Field(default_factory=StoragePathsConfig)
    nasa: NasaConfig = Field(default_factory=NasaConfig)
    
    # Security domain allowlist for external network queries
    approved_domains: Set[str] = {
        "ode.rsl.wustl.edu",
        "pds-geosciences.wustl.edu",
        "wac.lroc.asu.edu",
        "lroc.sese.asu.edu",
    }

    @property
    def is_gemini_available(self) -> bool:
        """Check if Gemini credentials (API key or ADC) are provided."""
        return bool(self.model.api_key or self.model.vertex_project_id)


# Global settings instance
settings = TerrainSettings()
