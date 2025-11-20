# Raspberry Pi 5 Deployment Plan

**SereneSense Military Vehicle Sound Detection System**

---

## Executive Summary

This document outlines the deployment strategy for the AudioMAE model (82.15% validation accuracy) on Raspberry Pi 5 hardware for real-time military vehicle sound detection.

### Deployment Goals

| Goal | Target Metric | Priority |
|------|--------------|----------|
| **Accuracy** | ≥80% (maintain current 82.15%) | Critical |
| **Latency** | <500ms per inference | High |
| **Memory** | <2GB RAM usage | High |
| **Power** | <15W average consumption | Medium |
| **Throughput** | ≥2 FPS (real-time 10-second clips) | Medium |

### Current Status

- ✅ AudioMAE model trained (82.15% validation accuracy)
- ✅ Model checkpoint available: `outputs/checkpoint_audiomae_099.pth`
- ✅ Comprehensive documentation and analysis complete
- ⏳ Raspberry Pi 5 deployment: **NEXT PHASE**

---

## 1. Hardware Specifications

### Raspberry Pi 5

**Specifications:**
- **CPU**: Quad-core ARM Cortex-A76 @ 2.4GHz
- **GPU**: VideoCore VII (800MHz)
- **RAM**: 8GB LPDDR4X (recommended)
- **Storage**: 64GB+ microSD card (Class 10 or better)
- **Power**: 27W max (5V/5A USB-C)
- **Connectivity**: Gigabit Ethernet, Wi-Fi 6, Bluetooth 5.0

**Audio Input:**
- **USB Microphone**: High-quality USB audio interface
  - Recommended: Blue Yeti, Rode NT-USB, or similar
  - Sample rate: 16kHz minimum
  - Channels: Mono or stereo (will convert to mono)

**Cooling:**
- **Required**: Active cooling (fan)
- **Reason**: Continuous inference generates heat
- **Recommendation**: Official Raspberry Pi active cooler

---

## 2. Model Optimization Pipeline

### Phase 2.1: Export to ONNX

**Goal**: Convert PyTorch model to ONNX format for cross-platform compatibility.

**Steps:**
```python
import torch
import onnx
from core.models.audioMAE import AudioMAE, AudioMAEConfig

# Load trained model
config = AudioMAEConfig(
    num_classes=7,
    embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    patch_size=16,
    in_chans=1
)

model = AudioMAE(config)
checkpoint = torch.load('outputs/checkpoint_audiomae_099.pth')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 1, 128, 128)
torch.onnx.export(
    model,
    dummy_input,
    'outputs/audiomae_fp32.onnx',
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['spectrogram'],
    output_names=['logits'],
    dynamic_axes={
        'spectrogram': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)
```

**Deliverable**: `audiomae_fp32.onnx` (~424MB)

### Phase 2.2: Quantization (INT8)

**Goal**: Reduce model size and improve inference speed with minimal accuracy loss.

**Method**: Post-Training Quantization (PTQ)

**Steps:**
```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize model to INT8
model_fp32 = 'outputs/audiomae_fp32.onnx'
model_int8 = 'outputs/audiomae_int8.onnx'

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8,
    optimize_model=True,
    extra_options={
        'MatMulConstBOnly': True,
        'WeightSymmetric': True
    }
)
```

**Expected Results:**
- Model size: ~424MB → ~106MB (4× reduction)
- Inference speed: 2-4× faster
- Accuracy drop: <2% (80%+ target)

**Deliverable**: `audiomae_int8.onnx` (~106MB)

### Phase 2.3: Model Optimization

**Additional Optimizations:**

1. **Graph Optimization**:
   - Remove unnecessary operations
   - Fuse batch normalization into convolutions
   - Simplify reshape operations

2. **Operator Fusion**:
   - Combine consecutive operations
   - Reduce memory transfers

3. **Pruning** (Optional):
   - Remove low-magnitude weights
   - Target: 30-50% sparsity
   - Expected: 10-20% speedup with <1% accuracy loss

**Script**: `scripts/optimize_for_edge.py` (to be created)

---

## 3. Raspberry Pi 5 Setup

### Phase 3.1: Operating System Installation

**Recommended OS**: Raspberry Pi OS (64-bit) Lite
- **Version**: Latest stable release
- **Reason**: Lightweight, no desktop environment overhead

**Installation Steps:**
1. Download Raspberry Pi OS Lite (64-bit)
2. Flash to microSD card using Raspberry Pi Imager
3. Enable SSH in boot configuration
4. Boot Raspberry Pi and connect via SSH

### Phase 3.2: System Configuration

**Update System:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-pip \
    python3-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    python3-h5py \
    libportaudio2 \
    libasound2-dev \
    libsndfile1
```

**Configure Audio:**
```bash
# List audio devices
arecord -l

# Set default audio input (adjust card/device numbers)
sudo nano /etc/asound.conf

# Add:
pcm.!default {
    type hw
    card 1
    device 0
}
ctl.!default {
    type hw
    card 1
}
```

**Increase Swap (for compilation):**
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Phase 3.3: Python Environment Setup

**Install Python Dependencies:**
```bash
# Create virtual environment
python3 -m venv ~/serenesense-env
source ~/serenesense-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install numpy==1.24.3
pip install scipy
pip install librosa==0.10.1
pip install soundfile
pip install pyaudio

# Install ONNX Runtime (ARM64 optimized)
pip install onnxruntime==1.16.0

# Install additional utilities
pip install pyyaml
pip install tqdm
```

**Verify Installation:**
```bash
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
python3 -c "import librosa; print(librosa.__version__)"
python3 -c "import numpy; print(numpy.__version__)"
```

### Phase 3.4: Transfer Model and Code

**Copy Files to Raspberry Pi:**
```bash
# From development machine, transfer files
scp outputs/audiomae_int8.onnx pi@raspberrypi.local:~/serenesense/
scp -r src/core/data/preprocessing pi@raspberrypi.local:~/serenesense/
scp -r src/core/inference pi@raspberrypi.local:~/serenesense/
scp configs/models/audioMAE.yaml pi@raspberrypi.local:~/serenesense/
```

---

## 4. Real-Time Inference Pipeline

### Phase 4.1: Audio Capture

**Audio Capture Script**: `scripts/capture_audio.py`

```python
import pyaudio
import numpy as np
import wave
from pathlib import Path

class AudioCapture:
    def __init__(
        self,
        sample_rate=16000,
        duration=10.0,
        channels=1,
        chunk=1024
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16

        self.audio = pyaudio.PyAudio()

    def capture(self):
        """Capture audio from microphone."""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print("Recording...")
        frames = []

        for _ in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording complete.")

        stream.stop_stream()
        stream.close()

        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        return audio_data

    def save(self, audio_data, filepath):
        """Save audio to WAV file."""
        wf = wave.open(str(filepath), 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())
        wf.close()

    def __del__(self):
        self.audio.terminate()
```

### Phase 4.2: Feature Extraction

**Mel Spectrogram Generation** (same as training):

```python
import librosa
import numpy as np

class MelSpectrogramGenerator:
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        fmin=50,
        fmax=8000
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, audio):
        """Generate mel spectrogram from audio."""
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        return log_mel
```

### Phase 4.3: ONNX Inference

**Inference Engine**: `scripts/realtime_detector.py`

```python
import onnxruntime as ort
import numpy as np
import librosa
from scipy.ndimage import zoom

# MAD class names
MAD_CLASSES = {
    0: "Helicopter",
    1: "Fighter Aircraft",
    2: "Military Vehicle",
    3: "Truck",
    4: "Foot Movement",
    5: "Speech",
    6: "Background"
}

class RealtimeDetector:
    def __init__(self, model_path, target_size=(128, 128)):
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.target_size = target_size

        # Initialize spectrogram generator
        self.spec_generator = MelSpectrogramGenerator(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=128,
            fmin=50,
            fmax=8000
        )

    def preprocess(self, audio):
        """Convert audio to model input."""
        # Generate mel spectrogram
        mel_spec = self.spec_generator(audio)

        # Resize to target size
        if mel_spec.shape != self.target_size:
            zoom_factors = (
                self.target_size[0] / mel_spec.shape[0],
                self.target_size[1] / mel_spec.shape[1]
            )
            mel_spec = zoom(mel_spec, zoom_factors, order=1)

        # Add batch and channel dimensions: (1, 1, 128, 128)
        mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]

        return mel_spec.astype(np.float32)

    def predict(self, audio):
        """Perform inference on audio."""
        # Preprocess
        input_tensor = self.preprocess(audio)

        # Run inference
        outputs = self.session.run(None, {'spectrogram': input_tensor})
        logits = outputs[0]

        # Get prediction
        pred_class = np.argmax(logits, axis=1)[0]
        probabilities = self.softmax(logits[0])

        return {
            'class_id': int(pred_class),
            'class_name': MAD_CLASSES[pred_class],
            'confidence': float(probabilities[pred_class]),
            'probabilities': {
                MAD_CLASSES[i]: float(probabilities[i])
                for i in range(7)
            }
        }

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
```

### Phase 4.4: Main Detection Loop

**Main Script**: `scripts/run_detector.py`

```python
import time
from audio_capture import AudioCapture
from realtime_detector import RealtimeDetector

def main():
    print("=" * 80)
    print("SereneSense Real-Time Military Vehicle Detection")
    print("=" * 80)

    # Initialize components
    print("\nInitializing...")
    capture = AudioCapture(sample_rate=16000, duration=10.0)
    detector = RealtimeDetector(model_path='audiomae_int8.onnx')

    print("Ready! Press Ctrl+C to stop.\n")

    try:
        while True:
            # Capture audio
            print(f"[{time.strftime('%H:%M:%S')}] Capturing audio...")
            audio = capture.capture()

            # Detect
            print("Analyzing...")
            start_time = time.time()
            result = detector.predict(audio)
            inference_time = (time.time() - start_time) * 1000  # ms

            # Display result
            print("=" * 80)
            print(f"Detection: {result['class_name']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Inference Time: {inference_time:.1f}ms")
            print("\nAll Probabilities:")
            for class_name, prob in result['probabilities'].items():
                bar = '█' * int(prob * 50)
                print(f"  {class_name:20s}: {prob*100:5.1f}% {bar}")
            print("=" * 80)
            print()

            # Wait before next capture
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping detector...")
        print("Goodbye!")

if __name__ == "__main__":
    main()
```

---

## 5. Performance Benchmarking

### Phase 5.1: Metrics to Measure

**Inference Metrics:**
```python
import time
import psutil
import numpy as np

class PerformanceBenchmark:
    def __init__(self):
        self.latencies = []
        self.memory_usage = []
        self.cpu_usage = []

    def measure_inference(self, detector, audio):
        """Measure single inference performance."""
        # CPU usage before
        cpu_before = psutil.cpu_percent(interval=0.1)

        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Inference
        start_time = time.time()
        result = detector.predict(audio)
        latency = (time.time() - start_time) * 1000  # ms

        # CPU usage after
        cpu_after = psutil.cpu_percent(interval=0.1)
        cpu_avg = (cpu_before + cpu_after) / 2

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        # Record
        self.latencies.append(latency)
        self.memory_usage.append(mem_after)
        self.cpu_usage.append(cpu_avg)

        return {
            'latency_ms': latency,
            'memory_mb': mem_after,
            'cpu_percent': cpu_avg,
            'result': result
        }

    def get_statistics(self):
        """Get performance statistics."""
        return {
            'latency': {
                'mean': np.mean(self.latencies),
                'std': np.std(self.latencies),
                'min': np.min(self.latencies),
                'max': np.max(self.latencies),
                'p50': np.percentile(self.latencies, 50),
                'p95': np.percentile(self.latencies, 95),
                'p99': np.percentile(self.latencies, 99),
            },
            'memory': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
            },
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
            },
            'throughput_fps': 1000 / np.mean(self.latencies),
        }
```

### Phase 5.2: Benchmark Script

**Script**: `scripts/benchmark_model.py`

```bash
python scripts/benchmark_model.py \
    --model outputs/audiomae_int8.onnx \
    --iterations 100 \
    --output benchmarks/raspberry_pi5.json
```

**Expected Metrics:**

| Metric | FP32 Model | INT8 Model | Target |
|--------|-----------|-----------|---------|
| **Latency (mean)** | 800-1200ms | 200-400ms | <500ms |
| **Latency (p95)** | 1000-1500ms | 300-500ms | <600ms |
| **Memory** | 1.5-2.0GB | 800MB-1.2GB | <2GB |
| **CPU Usage** | 80-100% | 60-90% | <90% |
| **Throughput** | 0.8-1.2 FPS | 2-5 FPS | ≥2 FPS |
| **Accuracy** | 82.15% | ≥80% | ≥80% |

---

## 6. Testing & Validation

### Phase 6.1: Accuracy Validation

**Test on Validation Set:**
```bash
python scripts/evaluate_on_device.py \
    --model audiomae_int8.onnx \
    --data-dir /path/to/mad/validation \
    --output validation_results.json
```

**Verify:**
- INT8 accuracy ≥ 80% (target: maintain 82.15%)
- Per-class performance comparable to FP32
- No catastrophic failures on specific classes

### Phase 6.2: Real-World Testing

**Test Scenarios:**
1. **Single Vehicle Detection**: One vehicle type at a time
2. **Noisy Environment**: Background noise + vehicle sounds
3. **Multiple Vehicles**: Overlapping sounds
4. **Distance Variation**: Near and far vehicle sounds
5. **Weather Conditions**: Wind, rain, ambient noise

**Test Data Collection:**
```bash
# Record test samples
python scripts/record_test_samples.py \
    --duration 10 \
    --num-samples 50 \
    --output test_samples/
```

### Phase 6.3: Edge Case Analysis

**Failure Analysis:**
- Misclassifications: Which classes are confused?
- Low confidence predictions: When is model uncertain?
- Latency spikes: What causes slow inference?
- Resource issues: Memory leaks, CPU throttling?

---

## 7. Deployment Checklist

### Pre-Deployment

- [ ] Model exported to ONNX (FP32)
- [ ] Model quantized (INT8)
- [ ] Model validated on development machine
- [ ] Raspberry Pi 5 hardware acquired
- [ ] USB microphone tested and verified
- [ ] Cooling solution installed

### Setup

- [ ] Raspberry Pi OS installed (64-bit Lite)
- [ ] System updated and configured
- [ ] Python environment set up
- [ ] ONNX Runtime installed and verified
- [ ] Audio input configured and tested
- [ ] Model files transferred to device

### Testing

- [ ] Inference pipeline tested with sample data
- [ ] Real-time detection loop running
- [ ] Performance benchmarks collected
- [ ] Accuracy validation completed (≥80%)
- [ ] Edge cases identified and documented
- [ ] Latency target met (<500ms)
- [ ] Memory target met (<2GB)

### Production

- [ ] Continuous operation tested (24-hour run)
- [ ] Thermal performance verified (no throttling)
- [ ] Power consumption measured (<15W)
- [ ] Failure recovery mechanisms tested
- [ ] Logging and monitoring configured
- [ ] Documentation finalized

---

## 8. Monitoring & Maintenance

### Monitoring Dashboard

**Metrics to Track:**
- Inference latency (rolling average)
- Memory usage (current/peak)
- CPU temperature
- Detection counts per class
- Confidence distribution
- Error rate

**Implementation**: Simple web dashboard or log files

### Logging

**Log Format:**
```
[2025-11-20 14:32:15] DETECTION | Class: Helicopter | Confidence: 0.92 | Latency: 245ms | Memory: 1024MB
[2025-11-20 14:32:26] DETECTION | Class: Background | Confidence: 0.67 | Latency: 238ms | Memory: 1028MB
[2025-11-20 14:32:37] ERROR | Audio capture failed | Retrying...
```

### Maintenance Tasks

**Regular:**
- Check system logs for errors
- Monitor disk space (audio recordings)
- Verify model accuracy on new test data
- Update dependencies if needed

**As Needed:**
- Retrain model with new data
- Optimize model further if performance degrades
- Replace hardware components if failing

---

## 9. Troubleshooting

### Common Issues

**Issue 1: High Latency (>500ms)**
- **Cause**: Model too large, CPU throttling
- **Solution**: Use INT8 quantization, improve cooling, reduce batch size

**Issue 2: Out of Memory**
- **Cause**: Model + preprocessing exceeds RAM
- **Solution**: Use INT8 model, reduce spectrogram size, close other processes

**Issue 3: Audio Capture Errors**
- **Cause**: Incorrect audio device configuration
- **Solution**: Check `arecord -l`, verify ALSA configuration

**Issue 4: Low Accuracy**
- **Cause**: Quantization artifacts, incorrect preprocessing
- **Solution**: Verify INT8 accuracy on dev machine, check spectrogram generation

**Issue 5: CPU Throttling**
- **Cause**: Overheating
- **Solution**: Install active cooling, reduce inference frequency

---

## 10. Future Enhancements

### Short-Term

1. **Model Pruning**: Reduce model size by 30-50%
2. **Ensemble**: Combine CNN + AudioMAE for better accuracy
3. **Continuous Learning**: Update model with new field data
4. **Alert System**: Trigger alerts for specific vehicle detections

### Long-Term

1. **Hardware Acceleration**: Use Raspberry Pi AI HAT for faster inference
2. **Multi-Model**: Deploy multiple models for different scenarios
3. **Edge Training**: Fine-tune model on-device with new data
4. **Distributed Deployment**: Network of Raspberry Pi detectors

---

## 11. Expected Outcomes

### Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Accuracy** | ≥80% | ≥82% |
| **Latency** | <500ms | <300ms |
| **Memory** | <2GB | <1.5GB |
| **Power** | <15W | <12W |
| **Throughput** | ≥2 FPS | ≥3 FPS |
| **Uptime** | 99% (24 hours) | 99.9% (1 week) |

### Deliverables

1. **Optimized Model**: INT8 ONNX model (≈106MB)
2. **Deployment Scripts**: Complete inference pipeline
3. **Benchmarking Report**: Performance metrics and analysis
4. **Documentation**: Setup guide, troubleshooting, maintenance
5. **Demo Video**: Real-time detection on Raspberry Pi 5
6. **Test Results**: Accuracy validation and edge case analysis

---

## 12. Timeline Estimate

### Phase-by-Phase Breakdown

| Phase | Duration | Tasks |
|-------|----------|-------|
| **1. Model Optimization** | 1-2 days | ONNX export, INT8 quantization, validation |
| **2. Hardware Setup** | 1 day | OS installation, system configuration, environment setup |
| **3. Software Deployment** | 1-2 days | Transfer code, install dependencies, test pipeline |
| **4. Testing & Validation** | 2-3 days | Benchmark, accuracy tests, edge cases |
| **5. Documentation** | 1 day | Finalize guides, write report, create demo |
| **Total** | **6-9 days** | Complete deployment to production |

### Prerequisites

- Raspberry Pi 5 hardware available
- USB microphone available
- Access to development machine for model export
- Time for thorough testing (not rushed)

---

## 13. Success Criteria

**Minimum Viable Deployment:**
- ✅ Model running on Raspberry Pi 5
- ✅ Real-time audio capture working
- ✅ Inference latency <500ms
- ✅ Accuracy ≥80%
- ✅ Stable operation for 24 hours

**Production-Ready Deployment:**
- ✅ All minimum criteria met
- ✅ Comprehensive benchmarking completed
- ✅ Edge case testing performed
- ✅ Documentation finalized
- ✅ Demo video created
- ✅ Thesis chapter drafted

---

## 14. Conclusion

The AudioMAE model (82.15% validation accuracy) is **ready for deployment** to Raspberry Pi 5. The deployment plan provides:

1. **Clear optimization path**: PyTorch → ONNX → INT8 quantization
2. **Detailed setup guide**: OS installation to inference pipeline
3. **Performance targets**: Realistic metrics based on hardware capabilities
4. **Comprehensive testing**: Accuracy, latency, edge cases
5. **Production readiness**: Monitoring, maintenance, troubleshooting

**Next Step**: Begin Phase 1 (Model Optimization) after thesis documentation is complete.

---

**Document Version**: 1.0
**Date**: November 20, 2025
**Status**: READY FOR IMPLEMENTATION
**Contact**: sirine.ben.ammar32@gmail.com
