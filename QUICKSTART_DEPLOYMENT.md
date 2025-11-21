# Quick Start: Raspberry Pi 5 Deployment

**Get SereneSense running on Raspberry Pi 5 in 30 minutes!**

---

## Prerequisites

- ✅ Trained AudioMAE model (checkpoint available)
- ✅ Raspberry Pi 5 (4GB+ RAM, 8GB recommended)
- ✅ microSD card (32GB+, Class 10)
- ✅ USB microphone (16kHz capable)
- ✅ Power supply (5V/5A recommended)
- ✅ Active cooling (fan or heatsink)

---

## Part 1: Development PC (10 minutes)

### Step 1: Export and Quantize Model

```bash
cd SereneSense

# Export to ONNX (FP32)
python scripts/export_to_onnx.py
# Output: outputs/audiomae_fp32.onnx (424 MB)

# Quantize to INT8
python scripts/quantize_onnx.py
# Output: outputs/audiomae_int8.onnx (106 MB)

# Test deployment pipeline
python scripts/test_deployment.py
```

**Expected Time:** 5-10 minutes

### Step 2: Transfer Files to Raspberry Pi

```bash
# Create directory on RPi
ssh pi@<raspberry_pi_ip> "mkdir -p ~/serenity_deploy"

# Transfer all necessary files
scp outputs/audiomae_int8.onnx pi@<raspberry_pi_ip>:~/serenity_deploy/
scp scripts/rpi_*.py pi@<raspberry_pi_ip>:~/serenity_deploy/
scp scripts/rpi_requirements.txt pi@<raspberry_pi_ip>:~/serenity_deploy/
scp scripts/rpi_setup.sh pi@<raspberry_pi_ip>:~/serenity_deploy/
```

**Expected Time:** 2-3 minutes

---

## Part 2: Raspberry Pi Setup (15 minutes)

### Step 3: Run Automated Setup

```bash
# SSH into Raspberry Pi
ssh pi@<raspberry_pi_ip>

# Navigate to deployment directory
cd ~/serenity_deploy

# Make setup script executable
chmod +x rpi_setup.sh

# Run setup (installs all dependencies)
bash rpi_setup.sh
```

The setup script will:
1. ✓ Check system compatibility
2. ✓ Install system packages
3. ✓ Create virtual environment
4. ✓ Install Python dependencies
5. ✓ Verify installation

**Expected Time:** 15-20 minutes (mostly package installation)

### Step 4: Connect Microphone

1. Plug USB microphone into Raspberry Pi
2. Verify detection:
   ```bash
   lsusb  # Should show USB audio device
   arecord -l  # List recording devices
   ```

---

## Part 3: First Detection (5 minutes)

### Step 5: Run Real-Time Detection

```bash
cd ~/serenity_deploy
python3 rpi_deploy.py --mode realtime --verbose --max-detections 5
```

**Expected Output:**
```
==================================================================
Military Vehicle Sound Detector - Raspberry Pi 5
==================================================================

Initializing audio preprocessor...
✓ Detector initialized successfully!

Starting Continuous Detection
Press Ctrl+C to stop

[2025-11-21 14:30:00] Capturing audio...
Recording 10.0 seconds of audio...
[2025-11-21 14:30:10] Detection #1
  Predicted: Military Vehicle
  Confidence: 89.23%
  Inference time: 245.8 ms
  All probabilities:
    → Military Vehicle    : 89.23%
      Truck              : 6.12%
      Background         : 2.34%
      ...
----------------------------------------------------------------------
```

### Step 6: Test with Audio File (Optional)

```bash
# Test with your own audio file
python3 rpi_deploy.py --mode file --file test_audio.wav
```

---

## Common Commands

### Basic Detection
```bash
python3 rpi_deploy.py --mode realtime
```

### Detailed Output
```bash
python3 rpi_deploy.py --mode realtime --verbose
```

### Limited Detections
```bash
python3 rpi_deploy.py --mode realtime --max-detections 10
```

### High Confidence Only
```bash
python3 rpi_deploy.py --mode realtime --confidence 0.8
```

### File Testing
```bash
python3 rpi_deploy.py --mode file --file audio.wav
```

### Help
```bash
python3 rpi_deploy.py --help
```

---

## Expected Performance

| Metric | Expected Value |
|--------|----------------|
| **Accuracy** | 82%+ |
| **Inference Time** | 240-280 ms |
| **Total Latency** | 300-370 ms |
| **Memory Usage** | ~800 MB |
| **CPU Usage** | 40-60% |
| **Temperature** | 45-55°C (with cooling) |
| **Power Draw** | 8-12W |

---

## Troubleshooting

### Audio Not Capturing
```bash
# Check devices
arecord -l

# Test recording
arecord -d 5 test.wav
aplay test.wav
```

### PyAudio Installation Failed
```bash
sudo apt-get install python3-pyaudio
```

### ONNX Runtime Not Found
```bash
pip3 install --upgrade onnxruntime==1.16.0
```

### High Temperature / Throttling
```bash
# Check temperature
vcgencmd measure_temp

# Check throttling status
vcgencmd get_throttled
# 0x0 = no throttling (good)
```

---

## Complete Documentation

For comprehensive information, see:

- **[Full Deployment Guide](docs/RPi5_DEPLOYMENT_GUIDE.md)** - Complete step-by-step instructions
- **[Main README](README.md)** - Project overview and setup
- **[Training Results](docs/reports/FINAL_RESULTS.md)** - Model performance analysis
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

---

## What's Next?

1. **Test with Real Audio**
   - Record actual military vehicle sounds
   - Validate accuracy in field conditions

2. **Integrate Alerts**
   - Add LED indicators
   - Send network notifications
   - Log detections to database

3. **Optimize Performance**
   - Tune confidence thresholds
   - Adjust detection intervals
   - Enable continuous operation

4. **Deploy in Field**
   - Add weatherproof enclosure
   - Use battery power
   - Test in various conditions

---

## File Structure

```
~/serenity_deploy/
├── audiomae_int8.onnx           # Quantized model (106 MB)
├── rpi_preprocessing.py         # Preprocessing module
├── rpi_deploy.py                # Main deployment script
├── rpi_requirements.txt         # Python dependencies
└── rpi_setup.sh                 # Setup script
```

---

## Success Checklist

- [ ] Model exported to ONNX (FP32)
- [ ] Model quantized to INT8
- [ ] Files transferred to Raspberry Pi
- [ ] Setup script executed successfully
- [ ] USB microphone connected and detected
- [ ] First detection completed
- [ ] Performance metrics verified

---

**Congratulations!** 🎉

You now have a working military vehicle sound detection system on Raspberry Pi 5!

**Total Time:** ~30 minutes
**Accuracy:** 82%+
**Latency:** <500ms

For questions or issues, refer to the full documentation or create an issue on GitHub.

---

**Author:** SereneSense Team
**Version:** 1.0.0
**Date:** 2025-11-21
