# Terrain Analysis for Landing and Uncrewed Systems (TALUS)

TALUS is an agentic AI system that ingests lunar Digital Elevation Model (DEM) / Digital Terrain Model (DTM) data, answers lunar-terrain questions in natural language, performs deterministic terrain analysis, and evaluates rover traverse or landing-site candidates using configurable safety rules.

> [!CAUTION]
> **RESEARCH & DEMONSTRATION SYSTEM DISCLAIMER**
> TALUS is strictly a research and technology demonstration system. It must **NEVER** claim to provide certified flight safety, operational landing approval, autonomous spacecraft control, or guaranteed rover safety. All outputs are for research exploration and mission-planning prototyping only.

---

## Suggested Mission Questions

Users can interact with the agent using natural language or select from the clickable questions in the UI:

1. **Elevation Statistics**: *What is the average elevation around Shackleton Crater?*
2. **Slope Characterization**: *What is the mean slope around this location?*
3. **Safe Region Search**: *Find regions within 10 km of Shackleton with mean slope below 12 degrees.*
4. **Traverse Safety**: *Is this rover path safe?*
5. **Segment Hazard Identification**: *Which route segments exceed the configured slope limit?*
6. **Terrain Risk Scoring**: *Calculate the terrain risk score for this route.*
7. **Rerouting Assistance**: *Find alternative waypoints around an unsafe route segment.*
8. **Landing Site Comparison**: *Compare three candidate landing regions.*
9. **Provenance & Dataset Tracking**: *Which DEM is being used?*
10. **Resolution Queries**: *What is the resolution of the terrain dataset?*
11. **Scientific Limitations**: *What are the limitations of this terrain dataset?*
12. **Hazard Inspection**: *Check this traverse for terrain-related hazards.*

---

## System Architecture: Agentic vs. Deterministic Boundaries

TALUS enforces a strict division of responsibility between natural-language reasoning and scientific computation:

```
                  USER REQUEST
                       │
                       ▼
             ┌──────────────────┐
             │  Interpret Intent│
             └─────────┬────────┘
                       │
             ┌─────────▼────────┐
             │ Missing Params?  ├──[Yes]──> Ask User for Clarification
             └─────────┬────────┘
                       │ [No]
             ┌─────────▼────────┐
             │   Select Tool    │
             └─────────┬────────┘
                       │
                       ▼
 ═══════════════════════════════════════════════════════════════
   DETERMINISTIC PYTHON COMPUTATION LAYER (Strictly non-LLM)
   • DEM Window Loading (Rasterio)
   • Horn's Slope & Aspect Calculations
   • Surface Roughness & Terrain Ruggedness (TRI)
   • Path Segmentation & Haversine Distance
   • Threshold Comparisons (e.g., slope > 15°)
   • Coordinate Validation & Bounds Checking
 ═══════════════════════════════════════════════════════════════
                       │
                       ▼
             ┌──────────────────┐
             │Validate Tool Res │
             └─────────┬────────┘
                       │
             ┌─────────▼────────┐
             │ Interpret Result │
             └─────────┬────────┘
                       │
             ┌─────────▼────────┐
             │ Synthesize Answer│
             │ + Attach Evidence│
             │ + Cite Limits    │
             └──────────────────┘
```

The LLM does **NOT** calculate terrain values itself. All numerical terrain values originate from verified, deterministic Python functions.

---

## Data Sources

TALUS discovers and retrieves lunar terrain products exclusively through official NASA and planetary data archives:
- **Discovery**: NASA Lunar Orbital Data Explorer (ODE) REST API (`ode.rsl.wustl.edu`).
- **Global / Polar Basemaps**: NASA Lunar Reconnaissance Orbiter (LRO) LOLA GDR / Polar DEMs.
- **Topographic Products**: NASA SLDEM2015 (multi-sensor LOLA + Kaguya TC merger, ~60m/px near equator).
- **High-Resolution Regional Products**: LROC NAC stereo DEMs (~2–5m/px) for selected regions of interest.
- **Allowed Network Domains**: `ode.rsl.wustl.edu`, `pds-geosciences.wustl.edu`, `wac.lroc.asu.edu`, `lroc.sese.asu.edu`.

### Real NASA DEMs (Phase 4)

TALUS can obtain real NASA lunar DEMs on demand. It searches NASA ODE, downloads the product from
the PDS Geosciences Node under strict limits, validates it, converts it to a GeoTIFF of elevation
in metres, caches it with provenance, and hands it to the same terrain and safety analysis used
for every other DEM.

- Supported: LOLA gridded DEMs (`GDRDEM`) and SLDEM2015 (`SLDEM`), in PDS4 format.
- The agent tool `fetch_nasa_dem` takes a location and radius. There is no URL parameter.
- Downloads are capped at 100 MiB by default (`TALUS_MAX_DOWNLOAD_BYTES`). Finer products,
  including the 20 m polar DEMs, are larger and need a higher limit.
- Elevations are heights above a reference sphere of radius 1737.4 km, not above a geoid.

See [docs/nasa_data_acquisition.md](docs/nasa_data_acquisition.md) for the workflow, limits, and
how to run the optional live NASA test.

---

## Rover Safety Evaluation

Safety evaluations are parameterized by mission-specific constraints.
For example, when evaluating maximum slope:
- The UI explicitly states: **"Configured analysis threshold: 15°"**
- Thresholds are never presented as absolute or universal scientific limits.
- Evaluated statuses:
  - `PASS`: All segments comply with configured thresholds.
  - `REVIEW_REQUIRED`: High uncertainty, boundary conditions, or data resolution limits.
  - `FAIL`: One or more segments exceed configured thresholds.
- Every result details the number of segments analyzed, maximum observed slope, failing segment indices, terrain data source, spatial resolution, and scientific limitations.

---

## Resource Bounds & Security

- **Windowed Raster Reads**: Operates on target sub-windows ($\le 2048 \times 2048$ cells) without loading massive global DEMs into memory.
- **Safe by Default**: Operates out-of-the-box without requiring any mandatory secrets or external API keys.
- **Local Fallback**: Includes bundled sample DEMs and mock data for offline demonstration.
- **Hardened**: Protects against SSRF, arbitrary file paths, prompt injection, and command execution.

---

## Deployment

### Streamlit Community Cloud (public URL, no infrastructure to manage)

1. On [share.streamlit.io](https://share.streamlit.io), click **Create app** and pick this
   GitHub repository.
2. **Branch**: `main` (or whichever branch you want live). **Main file path**:
   `app/streamlit_app.py`.
3. Dependencies are read from the root [`requirements.txt`](requirements.txt) (which installs
   this project, pulling in everything declared in `pyproject.toml`) — no extra setup needed.
4. Under **Advanced settings → Secrets**, paste the keys you want from
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) (TOML format). Every
   key is optional; with none set, TALUS runs in offline deterministic demo mode. To enable
   live Gemini chat and NASA DEM fetch on the public deployment, at minimum set:
   ```toml
   GEMINI_API_KEY = "..."
   TALUS_NASA_DOWNLOADS = "true"
   ```
5. Click **Deploy**. Never commit a real `secrets.toml` — it's git-ignored, and secrets only
   ever live in the Streamlit Cloud Secrets box.

For deploying TALUS to Google Cloud Run instead (container build, Secret Manager, dedicated
least-privilege service identity, resource limits, health checks, and a post-deploy smoke
test), see [deploy/README.md](deploy/README.md).

---

## Quickstart

### 1. Installation

```bash
# Clone and enter directory
git clone <repository-url> talus
cd talus

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -e .
```

### 2. Run the Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

### 3. Optional Google AI / Vertex AI Configuration

To enable Gemini-powered natural-language reasoning, set the environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
# or configure Google Cloud Application Default Credentials (ADC) for Vertex AI
```
If no key is configured, TALUS automatically runs in **Deterministic Demonstration Mode**, providing full access to all terrain tools, sample datasets, and interactive visualizations.
