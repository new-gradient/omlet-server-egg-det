# Lightweight Python base
FROM python:3.11-slim

# Use non-root user for security
ENV USER=appuser UID=10001

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps (opencv/onnxruntime may need GLib/GL)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1-mesa-dri \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install python deps first (leverage docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create a non-root user and switch
RUN useradd -m -u ${UID} ${USER}
USER ${USER}

# Default environment
ENV ENABLE_CORS=true \
    DETECTION_THRESHOLD=0.3

# Expose Flask port
EXPOSE 5000

# Start with gunicorn
CMD ["gunicorn", "--workers", "4", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
