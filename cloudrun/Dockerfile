
# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System deps (rarely change)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Heavy deps (cache separately)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Python deps (cache on requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run sets PORT env var, defaulting to 8080)
EXPOSE 8080

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
