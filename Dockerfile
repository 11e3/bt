# Multi-stage build for production-ready container
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r btuser && useradd -r -g btuser btuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Production stage
FROM base as production

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install in production mode
RUN pip install --no-deps -e .

# Change ownership and switch to non-root user
RUN chown -R btuser:btuser /app
USER btuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bt; print('BT Framework is healthy')" || exit 1

# Default command
CMD ["python", "-c", "import bt; print('BT Framework container started')"]
