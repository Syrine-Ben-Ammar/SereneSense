#!/bin/bash
# Raspberry Pi 5 Setup Script for SereneSense Deployment
# Author: SereneSense Team
# Date: 2025-11-21

set -e  # Exit on error

echo "=================================================================="
echo "SereneSense - Raspberry Pi 5 Setup"
echo "Military Vehicle Sound Detection System"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
echo -e "\n${YELLOW}[1/7] Checking system...${NC}"
if [ ! -f /proc/device-tree/model ]; then
    echo -e "${YELLOW}Warning: Not running on Raspberry Pi${NC}"
else
    MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $MODEL"

    if [[ ! $MODEL == *"Raspberry Pi 5"* ]]; then
        echo -e "${YELLOW}Warning: Optimized for Raspberry Pi 5, detected: $MODEL${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Update system
echo -e "\n${YELLOW}[2/7] Updating system packages...${NC}"
sudo apt-get update
echo -e "${GREEN}✓ System updated${NC}"

# Install system dependencies
echo -e "\n${YELLOW}[3/7] Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    portaudio19-dev \
    libsndfile1 \
    libatlas-base-dev \
    git \
    wget

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Create virtual environment (optional but recommended)
echo -e "\n${YELLOW}[4/7] Creating Python virtual environment...${NC}"
read -p "Create virtual environment? (recommended) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m venv ~/serenity_env
    source ~/serenity_env/bin/activate
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
    echo "To activate in future: source ~/serenity_env/bin/activate"
else
    echo "Skipping virtual environment"
fi

# Upgrade pip
echo -e "\n${YELLOW}[5/7] Upgrading pip...${NC}"
pip3 install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install Python dependencies
echo -e "\n${YELLOW}[6/7] Installing Python packages...${NC}"
echo "This may take 10-15 minutes on first installation..."

# Try installing from requirements file
if [ -f "rpi_requirements.txt" ]; then
    pip3 install -r rpi_requirements.txt
else
    # Fallback: install manually
    echo "Requirements file not found, installing packages manually..."
    pip3 install onnxruntime==1.16.0
    pip3 install numpy==1.24.3
    pip3 install librosa==0.10.1
    pip3 install soundfile==0.12.1
    pip3 install scipy==1.11.4

    # Try pyaudio (might fail on some systems)
    pip3 install pyaudio==0.2.14 || sudo apt-get install -y python3-pyaudio
fi

echo -e "${GREEN}✓ Python packages installed${NC}"

# Verify installation
echo -e "\n${YELLOW}[7/7] Verifying installation...${NC}"

python3 << EOF
import sys
print(f"Python version: {sys.version}")

try:
    import onnxruntime as ort
    print(f"✓ ONNX Runtime: {ort.__version__}")
except ImportError as e:
    print(f"✗ ONNX Runtime: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except ImportError as e:
    print(f"✗ Librosa: {e}")

try:
    import soundfile as sf
    print(f"✓ SoundFile: {sf.__version__}")
except ImportError as e:
    print(f"✗ SoundFile: {e}")

try:
    import pyaudio
    print(f"✓ PyAudio: installed")
except ImportError as e:
    print(f"✗ PyAudio: {e}")
EOF

echo -e "\n${GREEN}=================================================================="
echo "Setup Complete!"
echo "=================================================================="
echo -e "${NC}"

# Create deployment directory structure
echo -e "\n${YELLOW}Creating deployment directory...${NC}"
mkdir -p ~/serenity_deploy
cd ~/serenity_deploy

echo -e "\n${GREEN}Next Steps:${NC}"
echo "1. Transfer the following files to ~/serenity_deploy/:"
echo "   - audiomae_int8.onnx (quantized model)"
echo "   - rpi_preprocessing.py (preprocessing module)"
echo "   - rpi_deploy.py (deployment script)"
echo ""
echo "2. Example transfer command (from your PC):"
echo "   scp audiomae_int8.onnx rpi_preprocessing.py rpi_deploy.py pi@<raspberry_pi_ip>:~/serenity_deploy/"
echo ""
echo "3. Connect USB microphone to Raspberry Pi"
echo ""
echo "4. Test deployment:"
echo "   cd ~/serenity_deploy"
echo "   python3 rpi_deploy.py --mode realtime --verbose"
echo ""
echo "5. For file-based testing:"
echo "   python3 rpi_deploy.py --mode file --file test_audio.wav"
echo ""
echo "=================================================================="
echo "For help: python3 rpi_deploy.py --help"
echo "=================================================================="
