FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

COPY . .

ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379/0

RUN mkdir -p /app/temp && chmod 777 /app/temp
RUN mkdir -p /app/analyzed_frames && chmod 777 /app/analyzed_frames

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]