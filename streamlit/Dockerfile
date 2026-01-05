# Dockerfile (Streamlit)
FROM python:3.11-slim

WORKDIR /app
COPY .streamlit /app/.streamlit

# Install dependencies
COPY requirements.txt .

# Install streaming requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]