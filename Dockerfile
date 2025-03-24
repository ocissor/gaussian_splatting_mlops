FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install system dependencies required by Open3D
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

RUN ls -R /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

EXPOSE 8000
#CMD ["python", "src/components/train.py"]
#CMD ["dvc", "repro"]
