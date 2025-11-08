# backend/Dockerfile
FROM python:3.10.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Will be overridden by docker-compose
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]