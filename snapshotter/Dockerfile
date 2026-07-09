FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY candle_snapshotter.py .

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/candle_snapshotter.py"]
