FROM python:3.11-slim

# Python runtime behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Non-root user
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# System deps for building some Python wheels (e.g., chroma deps)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps early to leverage Docker layer cache
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Ownership to non-root
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080
ENV PORT=8080

# Start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]