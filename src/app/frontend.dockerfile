
FROM python:3.11-slim
EXPOSE 8505

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir /app
WORKDIR /app

COPY frontend_requirements.txt  /app/frontend_requirements.txt

RUN pip install --no-cache-dir -r /app/frontend_requirements.txt

COPY frontend.py /app/frontend.py



ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8505", "--server.address=0.0.0.0"]