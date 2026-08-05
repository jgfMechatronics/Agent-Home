# Agent Home Dockerfile
#
# Build with: ./build.sh
# Or manually: docker build -t agent-home:experimental .

FROM python:3.12-slim

# Build args for traceability (set by build.sh)
ARG GIT_COMMIT=unknown
ARG GIT_BRANCH=unknown

# Labels for image metadata
LABEL org.opencontainers.image.source="https://github.com/jgfMechatronics/Agent-Home"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL git.branch="${GIT_BRANCH}"

# Don't buffer Python output (better for container logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching - this layer only rebuilds when deps change)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --locked --no-dev

# Copy application code
COPY . .

EXPOSE 8000

# Health check (using Python since slim image may not have curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server (uv run uses the .venv created by uv sync)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
