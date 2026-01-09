# Dockerfile (Streamlit)
FROM python:3.11-slim

WORKDIR /app
COPY .streamlit /app/.streamlit

# Install docker CLI for running backtests
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.4.1.tgz | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .

# Install streaming requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]