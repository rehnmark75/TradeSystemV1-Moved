FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8095

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8095"]
