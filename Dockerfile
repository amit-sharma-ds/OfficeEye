# OfficeEye in a container (Render, or any Docker host).
# The cameras must be reachable from wherever this runs: on a cloud host
# that means the office cameras are NOT visible - see README, "Deploying".
FROM python:3.12-slim

# OpenCV needs libgl1/libglib even without a screen; tzdata for the clock.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*
# Events and reports use local time: the office's, not the server's UTC.
ENV TZ=Asia/Kolkata

WORKDIR /app
COPY requirements.txt .
# The CPU build of PyTorch: the default one pulls ~2 GB of GPU libraries.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app.py"]
