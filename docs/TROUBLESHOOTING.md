# Troubleshooting Guide

Solutions to common issues when installing, using, and deploying SereneSense.

## Installation Issues

### Python Version Error

**Error**: `Error: Python 3.10+ is required`

**Solution**:
```bash
# Check Python version
python3 --version

# Install Python 3.10+
# Ubuntu/Debian
sudo apt-get install python3.10 python3.10-venv

# macOS
brew install python@3.10

# Set as default
alias python3=python3.10
```

### Pip Install Fails

**Error**: `pip: command not found` or `ModuleNotFoundError: pip`

**Solution**:
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Use pip3 explicitly
pip3 install -r requirements.txt

# Or use python -m pip
python -m pip install -r requirements.txt
```

### Virtual Environment Issues

**Error**: `ModuleNotFoundError` after creating venv

**Solution**:
```bash
# Ensure venv is activated
# Linux/macOS
source venv/bin/activate
echo $VIRTUAL_ENV  # Should show venv path

# Windows
venv\Scripts\activate
echo %VIRTUAL_ENV%

# Reinstall requirements in active venv
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## PyTorch Issues

### CUDA Not Detected

**Error**: `torch.cuda.is_available()` returns False

**Solution**:
```bash
# Check installed CUDA version
nvidia-smi

# Install PyTorch for correct CUDA version
# Go to https://pytorch.org/get-started/locally/
# Select your CUDA version and install command

# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory (CUDA)

**Error**: `CUDA out of memory. Tried to allocate X MB`

**Solution**:
1. **Reduce batch size**:
   ```yaml
   # configs/training/audioMAE.yaml
   training:
     batch_size: 8  # Reduce from 32
   ```

2. **Enable mixed precision**:
   ```yaml
   training:
     mixed_precision: true
   ```

3. **Clear CUDA cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Close other GPU applications**:
   ```bash
   # Check GPU usage
   nvidia-smi

   # Kill other processes
   fuser -k /dev/nvidia*
   ```

### Wrong CUDA/cuDNN Version

**Error**: `CUDA runtime error` or `cuDNN version mismatch`

**Solution**:
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
python -c "import torch.backends.cudnn as cudnn; print(cudnn.version())"

# Install matching versions
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Audio Processing Issues

### Audio File Format Error

**Error**: `Could not open audio file` or `format not supported`

**Solution**:
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS

# Convert to supported format (WAV, MP3, FLAC)
ffmpeg -i input.m4a -acodec pcm_s16le -ar 16000 output.wav

# Verify audio properties
ffprobe input.wav
```

### Resampling Error

**Error**: `resample requires >= 1 audio channel`

**Solution**:
```bash
# Ensure audio is mono
ffmpeg -i input.wav -ac 1 -ar 16000 output_mono.wav

# Verify in Python
import librosa
y, sr = librosa.load('input.wav', mono=True, sr=16000)
print(f"Shape: {y.shape}, Sample rate: {sr}")
```

### Silence Detection Issue

**Error**: Model fails on silent audio or very quiet signals

**Solution**:
```python
# Add noise floor
import numpy as np
audio = audio + np.random.randn(len(audio)) * 0.001  # Add small noise

# Or adjust silence threshold
import librosa
y_trimmed, _ = librosa.effects.trim(y, top_db=30)  # Increase top_db

# Ensure minimum amplitude
if np.max(np.abs(audio)) < 0.01:
    audio = audio * 10  # Amplify
```

## Model Issues

### Model Load Failed

**Error**: `FileNotFoundError: Model weights not found`

**Solution**:
```bash
# Download pre-trained models
python scripts/download_model.py --model audioMAE --output models/

# Or train your own
python scripts/train_model.py --config configs/training/audioMAE.yaml
```

### Inference Produces Same Output

**Error**: Model always predicts same class

**Solution**:
```python
# Ensure model is in eval mode
model.eval()

# Disable dropout/batch norm training behavior
with torch.no_grad():
    output = model(input_tensor)

# Check input normalization
audio = (audio - audio.mean()) / (audio.std() + 1e-8)
```

### Poor Prediction Accuracy

**Error**: Model accuracy <70% on validation set

**Solution**:
1. **Increase training time**:
   ```yaml
   training:
     epochs: 100  # Increase from 50
   ```

2. **Adjust learning rate**:
   ```yaml
   training:
     learning_rate: 0.0001  # Reduce from 0.001
   ```

3. **Add data augmentation**:
   ```yaml
   augmentation:
     enabled: true
     time_stretch: 0.1
     pitch_shift: 2
     gain: 0.05
   ```

4. **Check data quality**:
   ```bash
   # Verify dataset
   python scripts/validate_dataset.py --input data/processed/
   ```

## Docker Issues

### Docker Daemon Not Running

**Error**: `Cannot connect to Docker daemon`

**Solution**:
```bash
# Start Docker daemon
sudo systemctl start docker
sudo systemctl status docker

# Enable auto-start
sudo systemctl enable docker

# Check Docker is working
docker run hello-world
```

### Permission Denied

**Error**: `permission denied while trying to connect to Docker daemon`

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply changes (logout/login required)
newgrp docker

# Verify
docker ps
```

### Image Build Failed

**Error**: `Docker build failed` or `Error building image`

**Solution**:
```bash
# Check Docker disk space
docker system df

# Clean up unused images
docker system prune -a

# Build with verbose output
docker build -t serenesense:latest . --progress=plain

# Check Dockerfile syntax
docker build --dry-run -t serenesense:latest .
```

### Container Exit Immediately

**Error**: Container exits with error code

**Solution**:
```bash
# Check logs
docker logs <container_id>

# Run interactively for debugging
docker run -it serenesense:latest bash

# Check entrypoint
docker inspect serenesense:latest | grep -i entrypoint
```

## API Server Issues

### Port Already in Use

**Error**: `Address already in use` or `Bind error`

**Solution**:
```bash
# Check what's using the port
lsof -i :8000  # Unix/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Unix/Linux
taskkill /PID <PID> /F  # Windows

# Use different port
docker run -p 9000:8000 serenesense:latest
```

### API Endpoint Not Responding

**Error**: `Connection refused` or `timeout`

**Solution**:
```bash
# Test API connectivity
curl -v http://localhost:8000/health

# Check container logs
docker-compose logs serenesense-api

# Verify port mapping
docker-compose ps

# Test from inside container
docker-compose exec serenesense-api curl http://localhost:8000/health
```

### Slow Response Times

**Error**: API responses take >1 second

**Solution**:
1. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon
   ```

2. **Reduce input size**:
   ```python
   # Reduce audio length from 10s to 5s
   audio = audio[:80000]  # 5 seconds at 16kHz
   ```

3. **Enable quantized model**:
   ```yaml
   model:
     quantized: true
     precision: int8
   ```

4. **Check system resources**:
   ```bash
   top
   free -h
   ```

## Data Processing Issues

### Dataset Download Fails

**Error**: `Download failed` or `Connection timeout`

**Solution**:
```bash
# Resume download
python scripts/download_datasets.py --dataset mad --resume

# Use different mirror/source
python scripts/download_datasets.py --dataset mad --mirror backup

# Check network connectivity
ping google.com

# Increase timeout
python scripts/download_datasets.py --dataset mad --timeout 300
```

### Data Preparation Slow

**Error**: Preparation takes >30 minutes

**Solution**:
```bash
# Use multiple workers
python scripts/prepare_data.py --workers 8 --batch-size 64

# Skip augmentation
python scripts/prepare_data.py --no-augmentation

# Use SSD for storage
python scripts/prepare_data.py --output /mnt/ssd/data/

# Check disk speed
dd if=/dev/zero of=test.dat bs=1M count=1024 oflag=direct
```

### Out of Memory During Preparation

**Error**: `MemoryError` or `Killed`

**Solution**:
```bash
# Reduce batch size
python scripts/prepare_data.py --batch-size 16

# Process in chunks
python scripts/prepare_data.py --chunk-size 100  # Process 100 files at a time

# Clear temp files
rm -rf /tmp/*
```

## Performance Issues

### Training Too Slow

**Error**: Training takes >2 hours per epoch

**Solution**:
1. **Enable mixed precision**:
   ```yaml
   training:
     mixed_precision: true
   ```

2. **Increase batch size**:
   ```yaml
   training:
     batch_size: 64  # If GPU memory allows
   ```

3. **Use data caching**:
   ```bash
   # Pre-compute and cache
   python scripts/cache_dataset.py
   ```

4. **Profile training**:
   ```bash
   python -m cProfile -s cumtime scripts/train_model.py
   ```

### Inference Latency High

**Error**: Inference takes >100ms

**Solution**:
```bash
# Use optimized model
python scripts/optimize_for_edge.py --input models/audioMAE.pt --output models/audioMAE_opt.onnx

# Benchmark
python scripts/benchmark_performance.py --model models/audioMAE_opt.onnx
```

## Monitoring & Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Check System Resources

```bash
# CPU usage
top

# Memory usage
free -h

# Disk space
df -h

# GPU status
nvidia-smi
watch nvidia-smi  # Continuous monitoring
```

### Profile Code

```bash
# CPU profiling
python -m cProfile -s cumtime script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# GPU profiling
pip install nvidia-pytorch-nvml
nsys profile python script.py
```

## Getting Help

If your issue isn't listed above:

1. **Check logs**: `docker-compose logs`
2. **Search issues**: https://github.com/serenesense/serenesense/issues
3. **Read docs**: [INSTALLATION.md](INSTALLATION.md), [README.md](README.md)
4. **Open issue**: Provide logs, environment, and reproduction steps
5. **Ask community**: GitHub Discussions

## Reporting Issues

When reporting issues, include:

```markdown
- **Error message**: Full error traceback
- **Environment**: Python version, OS, hardware
- **Installation method**: pip, docker, source
- **Steps to reproduce**: Minimal reproducible example
- **Logs**: docker-compose logs or Python logs
```

For more help, see [CONTRIBUTING.md](CONTRIBUTING.md) - "Reporting Issues" section.
