FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy and install dependencies first (better caching)
COPY pyproject.toml ./
RUN uv pip install --system -e ".[docker]"

# Copy application code
COPY src/ ./src/
COPY saved/ ./saved/

# Create supervisor configuration directory
RUN mkdir -p /etc/supervisor/conf.d/

# Create supervisor configuration with your working ports
RUN echo '[supervisord]\n\
nodaemon=true\n\
\n\
[program:fastapi]\n\
command=uv run uvicorn src.app.service:app --host 0.0.0.0 --port 8080\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/var/log/fastapi.log\n\
stderr_logfile=/var/log/fastapi.log\n\
\n\
[program:streamlit]\n\
command=uv run streamlit run src/app/frontend.py --server.port=8503 --server.address=0.0.0.0\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/var/log/streamlit.log\n\
stderr_logfile=/var/log/streamlit.log\n\
' > /etc/supervisor/conf.d/supervisord.conf

# Add health check for FastAPI service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Expose both ports
EXPOSE 3000 8503

CMD ["supervisord"]
