# Multi-stage build for optimized image size
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=python3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (force use of system Python)
RUN uv sync --frozen --no-dev --python-preference only-system

# Copy application code
COPY app/ ./app/
COPY data/prompts/ ./data/prompts/

# Create data directories for persistence
RUN mkdir -p data/chroma_db data/reviews

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["uv", "run", "streamlit", "run", "app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
