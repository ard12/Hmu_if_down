# Multi-stage edge deployment Dockerfile for Multi-Modal Fall Detection Hub
FROM python:3.11-slim

# Install OS runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy dependency specifications and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY hub/ /app/hub/
COPY config/ /app/config/
COPY src/ /app/src/
COPY main.py /app/

# Create persistent state directories
RUN mkdir -p /app/incidents /app/datasets

# CSI UDP: 5555 | mmWave UDP: 5556 | Web Telemetry HUD: 8000
EXPOSE 5555/udp
EXPOSE 5556/udp
EXPOSE 8000/tcp

# Healthcheck checking Web HUD responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default launch command runs dual-fusion hub with Web HUD enabled
CMD ["python", "-m", "hub.server", "--mode", "fusion", "--web", "--web-port", "8000"]
