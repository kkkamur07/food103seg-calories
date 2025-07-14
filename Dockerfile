FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (no DVC needed)
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml README.md uv.lock ./
RUN uv pip install --system -e ".[docker]"

# Copy application code
COPY src/ ./src/

# Create models directory (model will be copied by Cloud Build)
RUN mkdir -p saved/models/

# Copy model file (this will be provided by Cloud Build)
COPY saved/models/model.pth ./saved/models/

# Verify model was copied
RUN ls -la saved/models/ && echo "✅ Model file ready for inference"

# Create supervisor configuration
RUN mkdir -p /etc/supervisor/conf.d/
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
command=uv run streamlit run src/app/frontend.py --server.port=8501 --server.address=0.0.0.0\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/var/log/streamlit.log\n\
stderr_logfile=/var/log/streamlit.log\n\
' > /etc/supervisor/conf.d/supervisord.conf

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

EXPOSE 8080 8501
CMD ["supervisord"]
