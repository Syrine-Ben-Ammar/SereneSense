# Installation Guide

Complete guide for installing SereneSense on various platforms.

## Table of Contents

- [System Requirements](#system-requirements)
- [Local Installation](#local-installation)
- [Docker Installation](#docker-installation)
- [Docker Compose Installation](#docker-compose-installation)
- [Jetson Orin Installation](#jetson-orin-installation)
- [Raspberry Pi Installation](#raspberry-pi-installation)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.10+
- **RAM**: 8GB minimum (16GB recommended for training)
- **Disk Space**: 10GB (for dependencies + data)
- **OS**: Linux, macOS, or Windows

### GPU Requirements (Optional)

- **CUDA**: 12.x
- **cuDNN**: 8.x
- **GPU Memory**: 4GB minimum (8GB recommended)
- **Supported GPUs**: NVIDIA RTX 3060+, A100, V100, etc.

### Edge Device Support

- **Jetson Orin Nano**: 8GB RAM, 64GB eMMC
- **Raspberry Pi 5**: 8GB RAM, 128GB microSD

## Local Installation

### Prerequisites

```bash
# Verify Python version
python3 --version  # Should be 3.10+

# Verify pip
pip --version

# Install system dependencies
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv libsndfile1 ffmpeg

# macOS
brew install python3 libsndfile ffmpeg

# Windows
# Download Python 3.10+ from python.org
# Download ffmpeg from https://ffmpeg.org/download.html
```

### Step 1: Create Virtual Environment

```bash
# Create venv
python3.10 -m venv venv

# Activate venv
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install SereneSense dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchaudio; print(f'TorchAudio version: {torchaudio.__version__}')"
python -c "from src.core.models.audioMAE.model import AudioMAE; print('✓ AudioMAE available')"

# Test CUDA (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then echo "✓ GPU support ready"; fi
```

### Step 4: Download Datasets (Optional)

```bash
# Download MAD dataset
python scripts/download_datasets.py --dataset mad --output data/raw

# Download other datasets
python scripts/download_datasets.py --dataset audioset --output data/raw
python scripts/download_datasets.py --dataset fsd50k --output data/raw
```

## Docker Installation

### Prerequisites

- Docker 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- (Optional) NVIDIA Docker ([Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

### Build Docker Image

```bash
# Build GPU image
docker build -t serenesense:latest .

# Build Jetson image
docker build -f Dockerfile.jetson -t serenesense:jetson .

# Build Raspberry Pi image
docker build -f Dockerfile.rpi -t serenesense:rpi .
```

### Run Container

```bash
# Interactive shell
docker run -it -v $(pwd)/data:/workspace/data serenesense:latest bash

# Run API server
docker run -d -p 8000:8000 -v $(pwd)/data:/workspace/data serenesense:latest

# With GPU support (requires nvidia-docker)
docker run -d --gpus all -p 8000:8000 serenesense:latest

# Check logs
docker logs <container_id>
```

## Docker Compose Installation

### Step 1: Create Environment File

```bash
# Copy and customize environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### Step 2: Start Services

```bash
# Development setup (with hot-reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production setup
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f
```

### Step 3: Access Services

```bash
# API endpoint
curl http://localhost:8000/health

# API documentation
# Open: http://localhost:8000/docs

# TensorBoard (monitoring)
# Open: http://localhost:6006

# MLflow (experiment tracking)
# Open: http://localhost:5000
```

### Step 4: Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Jetson Orin Installation

### Prerequisites

- Jetson Orin Nano with JetPack 5.x
- 64GB+ microSD card
- 5V/4A power supply

### Installation Steps

```bash
# 1. Flash JetPack 5.x to microSD
# Follow: https://developer.nvidia.com/embedded/jetpack

# 2. Boot Jetson and complete initial setup

# 3. Install Docker and nvidia-docker
sudo apt-get update
sudo apt-get install -y docker.io nvidia-docker2

# 4. Clone SereneSense repository
git clone https://github.com/serenesense/serenesense.git
cd serenesense

# 5. Build Jetson image
docker build -f Dockerfile.jetson -t serenesense:jetson .

# 6. Run container
docker run -d --gpus all -p 8000:8000 \
  -v $(pwd)/data:/workspace/data \
  serenesense:jetson

# 7. Test API
curl http://localhost:8000/health
```

### Performance Tuning

```bash
# Set maximum GPU clock
sudo jetson_clocks

# Monitor performance
jtop

# Check GPU memory
nvidia-smi

# Check power consumption
tegrastats
```

## Raspberry Pi Installation

### Prerequisites

- Raspberry Pi 5 with 8GB RAM
- 128GB microSD card with Raspberry Pi OS 64-bit
- 5V/5A USB-C power supply
- Optional: AI HAT+ for accelerated inference

### Installation Steps

```bash
# 1. Install Raspberry Pi OS 64-bit
# Follow: https://www.raspberrypi.com/software/

# 2. Update system
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3.10 python3-pip libsndfile1 ffmpeg

# 3. Install Docker
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 4. Clone SereneSense
git clone https://github.com/serenesense/serenesense.git
cd serenesense

# 5. Build RPi image
docker build -f Dockerfile.rpi -t serenesense:rpi .

# 6. Run container
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/workspace/data \
  --memory 2g \
  serenesense:rpi

# 7. Test API
curl http://localhost:8000/health
```

### Performance Optimization

```bash
# Disable unnecessary services
sudo systemctl disable bluetooth.service
sudo systemctl disable avahi-daemon.service

# Allocate swap for training
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Monitor performance
top
vcgencmd measure_temp  # Check temperature
```

## Troubleshooting

### Import Errors

```bash
# Error: "No module named 'torch'"
pip install -r requirements.txt

# Error: "CUDA out of memory"
# Reduce batch size in config:
# batch_size: 16  # Instead of 32
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify CUDA version
nvidia-smi

# Install correct CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

### Audio Processing

```bash
# Error: "librosa cannot open audio file"
# Install audio decoders:
sudo apt-get install -y libsndfile1 ffmpeg

# Test audio loading
python -c "import librosa; y, sr = librosa.load('test.wav'); print(f'Loaded: {len(y)} samples')"
```

### Docker Issues

```bash
# Error: "Docker daemon not running"
sudo systemctl start docker

# Error: "Permission denied while trying to connect to Docker daemon"
sudo usermod -aG docker $USER
# Then logout and login again

# View logs
docker logs <container_id>

# Clean up images
docker system prune
```

### Memory Issues

```bash
# Check available memory
free -h

# Reduce model size
# Use: model_name: 'audioMAE-small' instead of 'audioMAE-base'

# Enable swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### GPU Memory Issues

```bash
# Reduce batch size
batch_size: 8  # From 32

# Use mixed precision training
mixed_precision: true

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

## Verification Checklist

After installation, verify everything works:

- [ ] Python 3.10+ installed
- [ ] All dependencies installed (pip list)
- [ ] PyTorch version correct (python -c "import torch; print(torch.__version__)")
- [ ] CUDA available (if GPU) (python -c "import torch; print(torch.cuda.is_available())")
- [ ] AudioMAE model loads (python -c "from src.core.models.audioMAE.model import AudioMAE; print('OK')")
- [ ] Docker image builds (docker build -t test . && echo "OK")
- [ ] API server starts (docker-compose up -d && curl http://localhost:8000/health)

## Next Steps

1. **Download Data**: `python scripts/download_datasets.py --dataset mad`
2. **Prepare Data**: `python scripts/prepare_data.py --input data/raw --output data/processed`
3. **Run Training**: `python scripts/train_model.py --config configs/training/audioMAE.yaml`
4. **Start API**: `docker-compose up -d`
5. **Read Notebooks**: `jupyter notebook notebooks/`

For more details, see [QUICK_START_EXECUTION.md](QUICK_START_EXECUTION.md) or [PROJECT_COMPLETION_GUIDE.md](PROJECT_COMPLETION_GUIDE.md).
