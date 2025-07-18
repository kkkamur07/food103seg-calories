FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for dependency installation first
COPY pyproject.toml README.md uv.lock ./
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file (make sure this exists in your build context)
COPY saved/models/model.pth saved/models/model.pth

# Copy application code
COPY src/ ./src/
COPY static/favicon.ico ./static/favicon.ico
COPY src/app/fast_api.py .      

# Environment variables
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]

