# Build and runtime container for TALUS (Terrain Analysis for Landing and Uncrewed Systems)
#
# Multi-stage build: build tools (compilers, -dev headers) live only in the builder stage.
# The runtime stage ships just the installed Python packages and app code, which keeps the
# final image smaller and reduces the attack surface (no compilers/headers at runtime).

# ---- Builder stage ------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install into an isolated prefix so it can be copied verbatim into the runtime stage.
# README.md is required because pyproject.toml's [project] table references it as `readme`.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage --------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    PATH="/usr/local/bin:$PATH"

# Runtime-only shared libraries (no compilers or -dev headers) plus curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    libproj25 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in the packages installed in the builder stage.
COPY --from=builder /install /usr/local

# Copy source tree and assets. No .env, secrets, or credentials are ever copied into the
# image -- see .dockerignore. Configuration is supplied at runtime via environment variables.
COPY src/ ./src/
COPY app/ ./app/
COPY data/metadata/ ./data/metadata/
COPY prompts/ ./prompts/

# Create runtime directories
RUN mkdir -p data/cache data/sample outputs logs

# Non-root user for security compliance
RUN useradd -m -u 1001 talususer && \
    chown -R talususer:talususer /app
USER talususer

EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
