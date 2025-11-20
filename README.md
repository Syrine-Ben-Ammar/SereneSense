# 🎯 SereneSense: Military Vehicle Sound Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**SereneSense** is a research project for military vehicle sound detection using state-of-the-art transformer architectures. This thesis project implements and compares multiple deep learning approaches for audio classification on the MAD (Military Audio Detection) dataset.

## 📊 Project Status

### ✅ Phase 1: Model Development & Training (COMPLETE)
- **CNN Baseline**: 66.88% validation accuracy
- **CRNN Baseline**: 73.21% validation accuracy
- **AudioMAE (Best)**: **82.15% validation accuracy**
- **MAD Dataset**: Preprocessed 7,466 samples across 7 classes
- **Comprehensive Reports**: Training curves, metrics, and analysis

### 🔄 Phase 3: Deployment (NEXT STEP)
- Raspberry Pi 5 deployment
- Model optimization (ONNX, quantization)
- Real-time inference testing

## 🎯 Key Results

### Model Performance Comparison

| Model | Architecture | Validation Accuracy | Improvement | Parameters |
|-------|-------------|---------------------|-------------|------------|
| **Old Notebook CNN** | CNN + MFCC | ~66-68% | Baseline | 242K |
| **New CNN Baseline** | CNN + MFCC | **66.88%** | ✅ Reproduced | 242K |
| **New CRNN** | CNN + BiLSTM | **73.21%** | +6.3% | 1.5M |
| **AudioMAE** | ViT Transformer | **82.15%** | **+15.2%** | 111M |

### AudioMAE Performance Highlights
- **Validation Accuracy**: 82.15%
- **Training Accuracy**: 69.77%
- **Generalization Gap**: **+12.38%** (validation > training - excellent!)
- **Training Time**: 4 hours (100 epochs on GPU)
- **No Overfitting**: Validation loss (0.8693) < Training loss (0.9763)

### Generalization Analysis

| Model | Train Acc | Val Acc | Gap | Status |
|-------|-----------|---------|-----|--------|
| CNN | 68.45% | 66.88% | -1.57% | Slight overfit |
| CRNN | 74.89% | 73.21% | -1.68% | Slight overfit |
| **AudioMAE** | **69.77%** | **82.15%** | **+12.38%** | ✅ **Excellent** |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- Anaconda/Miniconda (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/Syrine-Ben-Ammar/SereneSense.git
cd SereneSense

# Create conda environment
conda create -n serenesense python=3.10
conda activate serenesense

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

```bash
# Prepare MAD dataset
python scripts/prepare_data.py --config configs/data/mad_dataset.yaml

# This will create:
# - data/processed/mad/train/ (5,464 samples)
# - data/processed/mad/validation/ (965 samples)
# - data/processed/mad/test/ (1,037 samples)
```

### Training

#### Train AudioMAE (Best Performance)
```bash
python scripts/train_model.py \
    --config configs/models/audioMAE.yaml \
    --data-dir data/raw/mad \
    --epochs 100 \
    --batch-size 16
```

#### Train Legacy Models (CNN/CRNN)
```bash
# Train CNN baseline
python scripts/train_legacy_model.py \
    --config configs/models/legacy_cnn_mfcc.yaml \
    --epochs 150

# Train CRNN baseline
python scripts/train_legacy_model.py \
    --config configs/models/legacy_crnn_mfcc.yaml \
    --epochs 100
```

### Evaluation

```bash
# Evaluate AudioMAE model
python scripts/evaluate_audiomae.py \
    --model outputs/checkpoint_audiomae_099.pth \
    --data-dir data/raw/mad

# Evaluate legacy models
python scripts/evaluate_legacy_model.py \
    --checkpoint outputs/phase1/cnn_baseline.pth \
    --config configs/models/legacy_cnn_mfcc.yaml
```

## 📊 Dataset: MAD (Military Audio Detection)

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 5,464 | 73.2% |
| Validation | 965 | 12.9% |
| Test | 1,037 | 13.9% |
| **Total** | **7,466** | **100%** |

### Military Vehicle Classes (7 Classes)

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Helicopter | Rotary-wing aircraft |
| 1 | Fighter Aircraft | Fixed-wing military jets |
| 2 | Military Vehicle | Armored vehicles, APCs |
| 3 | Truck | Military trucks |
| 4 | Foot Movement | Infantry, footsteps |
| 5 | Speech | Human speech |
| 6 | Background | Ambient noise |

## 🏗️ Architecture Overview

### AudioMAE (Audio Masked Autoencoder)

Our best-performing model uses a Vision Transformer (ViT) architecture adapted for audio:

**Key Components:**
- **Input**: Mel spectrograms (128×128) from 10-second audio clips
- **Encoder**: 12-layer ViT with 768-dim embeddings (12 attention heads)
- **Decoder**: 8-layer transformer with 512-dim embeddings (16 attention heads)
- **Patch Size**: 16×16
- **Parameters**: 111,089,927 (~424MB)

**Training Configuration:**
- **Optimizer**: AdamW (LR=1e-4, weight decay=0.05)
- **Scheduler**: Cosine annealing with warm restarts
- **Regularization**: Mixup (α=0.8), Label smoothing (0.1), Dropout (0.5)
- **Batch Size**: 16
- **Epochs**: 100

### CNN Baseline

**Architecture**: 3-layer CNN + MFCC features
- **Features**: 40 MFCC coefficients + Δ + ΔΔ
- **Layers**: 48 → 96 → 192 filters
- **Parameters**: ~242K
- **Accuracy**: 66.88%

### CRNN Baseline

**Architecture**: CNN + Bidirectional LSTM
- **Features**: 40 MFCC coefficients + Δ + ΔΔ
- **Temporal Modeling**: BiLSTM layers
- **Parameters**: ~1.5M
- **Accuracy**: 73.21%

## 🏗️ Project Structure

```
SereneSense/
├── src/core/                          # Core implementation
│   ├── models/
│   │   ├── audioMAE/                 # AudioMAE implementation (82.15%)
│   │   ├── legacy/                   # CNN/CRNN baselines
│   │   └── __init__.py
│   ├── data/
│   │   ├── loaders/                  # MAD dataset loader
│   │   ├── preprocessing/            # Spectrogram generation
│   │   └── augmentation/             # SpecAugment, mixup
│   ├── training/                      # Training pipeline
│   ├── inference/                     # Inference utilities
│   └── utils/                         # Logging, device management
│
├── scripts/                           # Training & evaluation scripts
│   ├── train_model.py                # AudioMAE training
│   ├── train_legacy_model.py         # CNN/CRNN training
│   ├── evaluate_audiomae.py          # AudioMAE evaluation
│   ├── evaluate_legacy_model.py      # Legacy evaluation
│   ├── prepare_data.py               # Data preprocessing
│   ├── prepare_mad_metadata.py       # MAD metadata generation
│   └── plot_training_history.py      # Visualization
│
├── configs/                           # Configuration files
│   ├── models/
│   │   ├── audioMAE.yaml             # AudioMAE config
│   │   ├── legacy_cnn_mfcc.yaml      # CNN config
│   │   └── legacy_crnn_mfcc.yaml     # CRNN config
│   └── data/
│       └── mad_dataset.yaml          # MAD dataset config
│
├── outputs/                           # Training outputs
│   ├── best_model_audiomae_000.pth   # Best AudioMAE model (424MB)
│   ├── checkpoint_audiomae_099.pth   # Final checkpoint
│   ├── phase1/                       # Legacy model checkpoints
│   ├── history/                      # Training history JSON
│   └── plots/                        # Training curves
│
├── docs/                              # Documentation
│   ├── reports/                       # Training reports
│   │   ├── FINAL_RESULTS.md          # Comprehensive results
│   │   ├── TRAINING_SUMMARY_REPORT.md
│   │   ├── COMPARISON_ANALYSIS.md    # Model comparisons
│   │   └── training_curves.png       # Visualizations
│   ├── LEGACY_MODELS.md              # CNN/CRNN documentation
│   └── DEPLOYMENT_PLAN.md            # Raspberry Pi deployment plan
│
├── OLD/                               # Reference notebooks
│   ├── train_mad_mfcc_gpu_v2.ipynb  # Original CNN notebook
│   └── train_mad_crnn_gpu.ipynb      # Original CRNN notebook
│
├── data/                              # Dataset storage
│   ├── raw/mad/                       # Raw MAD dataset
│   └── processed/mad/                 # Preprocessed HDF5 files
│
├── README.md                          # This file
├── RoadMap.txt                        # Project roadmap
└── requirements.txt                   # Dependencies
```

## 📈 Training Results

### AudioMAE Training Curves

All training visualizations available in `docs/reports/`:
- `training_curves.png` - Training/validation loss and accuracy (100 epochs)
- `performance_comparison.png` - Detailed performance analysis
- `training_metrics.csv` - Metrics table (Excel/LaTeX ready)

### Key Observations

1. **Excellent Generalization**: Validation accuracy (82.15%) exceeds training accuracy (69.77%) by 12.38%, indicating:
   - Strong regularization (Mixup, label smoothing)
   - No overfitting
   - Good model robustness

2. **Stable Training**: Loss curves show smooth convergence over 100 epochs

3. **Significant Improvement**: 15.2% accuracy gain over original CNN baseline

## 📚 Documentation

### Reports & Analysis
- **[FINAL_RESULTS.md](docs/reports/FINAL_RESULTS.md)**: Complete results analysis
- **[TRAINING_SUMMARY_REPORT.md](docs/reports/TRAINING_SUMMARY_REPORT.md)**: 10-section training breakdown
- **[COMPARISON_ANALYSIS.md](docs/reports/COMPARISON_ANALYSIS.md)**: Model comparison with old notebook
- **[LEGACY_MODELS.md](docs/LEGACY_MODELS.md)**: CNN/CRNN documentation

### Configuration Examples
- **[audioMAE.yaml](configs/models/audioMAE.yaml)**: AudioMAE training config
- **[mad_dataset.yaml](configs/data/mad_dataset.yaml)**: Dataset configuration

## 🚀 Next Steps: Raspberry Pi 5 Deployment

The next phase involves deploying the trained AudioMAE model to Raspberry Pi 5:

### Deployment Plan
1. **Model Optimization**
   - Export to ONNX format
   - Apply INT8 quantization
   - Optimize for edge inference

2. **Raspberry Pi 5 Setup**
   - Install PyTorch/ONNX Runtime
   - Configure audio input
   - Implement real-time detection

3. **Performance Benchmarking**
   - Measure inference latency
   - Test accuracy on device
   - Evaluate power consumption

See **[DEPLOYMENT_PLAN.md](docs/DEPLOYMENT_PLAN.md)** for detailed deployment steps.

## 🛠️ Development

### Running Tests
```bash
# Check model checkpoint
python scripts/inspect_checkpoint.py outputs/checkpoint_audiomae_099.pth

# Generate training report
python scripts/generate_training_report.py

# Plot training history
python scripts/plot_training_history.py --history outputs/history/audiomae.json
```

### Model Checkpoints

| Checkpoint | Epoch | Size | Val Acc | Location |
|------------|-------|------|---------|----------|
| Best Model | 0 | 424MB | ~82% | `outputs/best_model_audiomae_000.pth` |
| Final | 99 | 424MB | 82.15% | `outputs/checkpoint_audiomae_099.pth` |
| CNN Baseline | - | ~1MB | 66.88% | `outputs/phase1/cnn_baseline.pth` |
| CRNN Baseline | - | ~6MB | 73.21% | `outputs/phase1/crnn_baseline.pth` |

## 🎓 Academic Context

This project was developed as part of a thesis on military vehicle sound detection. Key contributions:

1. **Reproduction Study**: Successfully reproduced and validated old notebook CNN results (66-68% accuracy)

2. **Architecture Comparison**: Evaluated three approaches:
   - Traditional CNN + MFCC features
   - Enhanced CRNN with temporal modeling
   - Modern AudioMAE transformer architecture

3. **Generalization Analysis**: Demonstrated that AudioMAE achieves superior generalization (+12.38% gap) despite higher model complexity

4. **Practical Implementation**: Complete training pipeline with visualization and evaluation tools

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{benammar2025serenesense,
  title={SereneSense: Military Vehicle Sound Detection using Transformer Architectures},
  author={Ben Ammar, Syrine},
  year={2025},
  school={University},
  note={AudioMAE achieved 82.15\% validation accuracy on MAD dataset}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MAD Dataset**: Military Audio Dataset contributors
- **PyTorch**: For the deep learning framework
- **Meta AI**: For AudioMAE architecture (Huang et al., 2022)
- **Hugging Face**: For transformer implementations

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Email**: sirine.ben.ammar32@gmail.com

---

**Academic Research Project - 2025**

Built for military vehicle sound detection research using state-of-the-art transformer architectures.
