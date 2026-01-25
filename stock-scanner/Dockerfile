# Build stage - compile dependencies
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Runtime stage - minimal image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY app/ /app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: run scheduler
CMD ["python", "-m", "stock_scanner.scheduler", "run"]
