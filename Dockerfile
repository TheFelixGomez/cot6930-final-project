# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment (.venv)
# --no-dev ensures we don't install testing libraries (pytest, etc.) into our production image
RUN uv sync --frozen --no-cache --no-dev

# ==========================================
# Stage 2: Final Runtime (Tiny & Secure)
# ==========================================
FROM python:3.13-slim

# For hygiene, create a non-root user to run the app securely
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy the completely built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy only the necessary application code and resources, excluding tests and development files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser recommender/ ./recommender/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Put the virtual environment in the PATH so that there is no need for absolute paths
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]