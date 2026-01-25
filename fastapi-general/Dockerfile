FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY worker/app /app/worker/app
COPY fastapi-general/app /app/app

ENV PYTHONPATH=/app/worker/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8008

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
