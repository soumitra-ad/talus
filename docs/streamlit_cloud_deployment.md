# TALUS on Streamlit Community Cloud

Exact settings, failure diagnosis, and a release checklist for the public deployment.
TALUS is a research and demonstration system; nothing here makes it operationally certified.

## 1. App settings (share.streamlit.io → your app → Settings)

| Setting | Value |
|---|---|
| Repository | `soumitra-ad/talus` |
| Branch | **`master`** (this repository has no `main` branch) |
| Main file path | `app/streamlit_app.py` |
| Python version (Advanced settings) | **3.12** (tested; 3.13 also resolves) |
| Dependencies | `requirements.txt` → `-e .` → `pyproject.toml` (no `packages.txt`) |

The Python version can only be chosen when the app is created. If the app was created with a
different version and it causes errors, Streamlit's documented fix is to delete and redeploy
the app with 3.12 selected. Try the logs first; the version is rarely the cause.

## 2. Secrets (Settings → Secrets), TOML

```toml
GEMINI_API_KEY = "YOUR_KEY"
TALUS_ENV = "production"
TALUS_NASA_DOWNLOADS = "true"
# Optional: require an access token before the app renders
# TALUS_ACCESS_TOKEN = "choose-a-long-random-value"
```

Rules that matter:

* Keys must be at the **top level** (not under a `[section]`), or TALUS will not see them.
* `TALUS_NASA_DOWNLOADS` is **required** with `TALUS_ENV = "production"`: production turns
  NASA downloads off by default, and the Cloud DEM cache starts empty, so without it every
  place question ends with "no DEM available". (`true` without quotes also works.)
* Save secrets, then **Reboot app** so the running process picks them up.

## 3. Reading the real error behind "Oh no. Error running app"

That page hides the cause. To see it:

1. Open the app, then **Manage app** (bottom right) → the log panel opens.
2. Scroll to the first `Traceback` or `ERROR` after the most recent `Starting up` line.
3. Match it:

| Log shows | Cause | Fix |
|---|---|---|
| `ERROR: ... Could not find a version` / `ResolutionImpossible` during install | dependency install failed | Reboot; if persistent, check the Python version (3.12) |
| `ModuleNotFoundError: terrain_agent` | editable install skipped | Fixed in code: the app falls back to `./src` |
| `Your app is having trouble loading` + memory warnings | over the Community Cloud memory limit | Reboot; the app idles at ~80 MB and a DEM analysis window is at most 2048 × 2048 cells |
| `branch ... does not exist` | app points at `main` | Set branch to `master` |
| Traceback inside `app/streamlit_app.py` | code error | Send the traceback |

TALUS makes no required network call at startup: the NASA reachability probe has short
timeouts and a 10-minute cache, and never raises, so Gemini or NASA outages cannot stop the UI
from loading.

## 4. Known operating limits

* **Gemini free tier: 20 requests per day per model.** One place question uses about four.
  When the quota is gone, TALUS shows "Gemini daily quota reached. Terrain tools still
  available." and answers questions about named places (Shackleton, Haworth, Malapert,
  Connecting Ridge, Aristarchus) with the deterministic NASA DEM pipeline. It then skips
  Gemini for 30 minutes. For regular use, enable billing on the Google AI project, or set
  `GEMINI_MODEL` to a model with a larger free quota.
* **The DEM cache is empty after every Cloud restart.** The first south-pole question
  downloads LOLA `ldem_75s_240m` (about 29 MB) from PDS Geosciences, which took about
  6 minutes in testing. Later questions are served from the cache.

## 5. In-app health check

Sidebar → **🩺 System health** lists:

* required secrets
* Gemini
* NASA ODE
* internet access to the NASA download host
* DEM cache contents
* cache-folder writability

Each is shown as 🟢 working, 🟡 warning or 🔴 failed. **Run full health check** adds one live
Gemini request (it counts against the daily quota). The status bar under the title shows
Gemini, NASA and DEM Cache at a glance.

## 6. Release checklist

- [ ] `git push origin master`, with tests passing locally (`python -m pytest -q`)
- [ ] Streamlit app settings: branch `master`, main file `app/streamlit_app.py`, Python 3.12
- [ ] Secrets added exactly as in section 2 (top level), then **Reboot app**
- [ ] App loads; no "Oh no" (if it appears, read the log as in section 3)
- [ ] Sidebar → System health: Required secrets 🟢, NASA ODE 🟢, Internet 🟢, Cache folder writable 🟢
- [ ] **Run full health check** → Gemini 🟢 (or 🔴 "daily quota reached": wait for the reset)
- [ ] Ask "What is the average elevation around Shackleton Crater?" and wait for the first
      DEM download (several minutes). The Elevation card shows a value, and the Dataset card
      shows LRO / LOLA
- [ ] Ask "Check rover safety for a terrain region near Shackleton Crater." and check that
      the Safety score card and the configured-threshold statement appear
