# SereneSense Project - Index & Navigation

Quick navigation guide for SereneSense documentation and code.

## 🎯 START HERE

Choose your path based on your needs:

| Goal | Action |
|------|--------|
| **New to SereneSense?** | Read [README.md](README.md) first (10 min) |
| **Want to install?** | Follow [docs/INSTALLATION.md](docs/INSTALLATION.md) (10 min) |
| **Working with legacy models?** | See [docs/LEGACY_MODELS.md](docs/LEGACY_MODELS.md) (15 min) |
| **Having issues?** | Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| **Contributing code?** | Read [CONTRIBUTING.md](CONTRIBUTING.md) (20 min) |

---

## 📚 Documentation Structure

### Root Level
```
README.md              → Project overview & features
CONTRIBUTING.md       → Code standards & contribution guidelines
INDEX.md              → This file - quick navigation
```

### docs/ Folder
```
INSTALLATION.md       → Setup instructions for all platforms
LEGACY_MODELS.md      → CNN/CRNN models documentation
TROUBLESHOOTING.md    → Common issues & solutions
```

---

## 📖 Documentation Guide

### For Everyone
| Document | Purpose | Time |
|----------|---------|------|
| [README.md](README.md) | Project overview, features, quick start | 10 min |

### For Users & Developers
| Document | Purpose | Time |
|----------|---------|------|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Setup guide for Docker, local, edge devices | 10 min |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | FAQ and common issues | 5 min |

### For Researchers & Comparison
| Document | Purpose | Time |
|----------|---------|------|
| [docs/LEGACY_MODELS.md](docs/LEGACY_MODELS.md) | CNN/CRNN models, training, comparison | 15 min |

### For Contributors
| Document | Purpose | Time |
|----------|---------|------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development guide, standards, workflow | 20 min |

---

## 📁 Project Structure

### Core Implementation
```
src/core/
├── models/                 # AI models (AudioMAE, AST, BEATs, legacy)
│   └── legacy/            # CNN & CRNN models (NEW)
├── data/                   # Data processing
│   ├── loaders/
│   ├── preprocessing/
│   │   └── legacy_mfcc.py  # MFCC extraction (NEW)
│   └── augmentation/
│       └── legacy_specaugment.py  # SpecAugment (NEW)
├── training/               # Training pipeline
├── inference/              # Inference engines
├── deployment/             # Edge & API deployment
├── evaluation/             # Benchmarking
└── utils/                  # Utilities
```

### Configuration
```
configs/
├── models/
│   ├── legacy_cnn_mfcc.yaml       # CNN config (NEW)
│   └── legacy_crnn_mfcc.yaml      # CRNN config (NEW)
├── data/
└── deployment/
```

### Scripts
```
scripts/
├── train_legacy_model.py          # Legacy model training (NEW)
├── compare_models.py              # Model comparison (NEW)
├── download_datasets.py
├── prepare_data.py
├── train_model.py
├── evaluate_model.py
├── optimize_for_edge.py
└── deploy_model.py
```

### Testing
```
tests/unit/
├── test_legacy_models.py          # Model tests (NEW - 24 tests)
├── test_legacy_features.py        # Feature tests (NEW - 20 tests)
└── test_legacy_augmentation.py    # Augmentation tests (NEW - 14 tests)
```

---

## 🎯 By Use Case

### I want to... → Do this

**Get started quickly**
1. Read [README.md](README.md)
2. Run: `docker-compose up -d` or follow [docs/INSTALLATION.md](docs/INSTALLATION.md)
3. Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) if needed

**Train a modern model**
1. Read [README.md](README.md) section "Getting Started"
2. Follow setup in [docs/INSTALLATION.md](docs/INSTALLATION.md)
3. Run: `python scripts/train_model.py --config configs/models/audioMAE.yaml`

**Train legacy models (CNN/CRNN)**
1. Read [docs/LEGACY_MODELS.md](docs/LEGACY_MODELS.md)
2. Run: `python scripts/train_legacy_model.py --model cnn`

**Compare models**
1. See [docs/LEGACY_MODELS.md](docs/LEGACY_MODELS.md) section "Comparison"
2. Run: `python scripts/compare_models.py --device cuda`

**Deploy on edge**
1. See [README.md](README.md) section "Deployment Options"
2. Follow [docs/INSTALLATION.md](docs/INSTALLATION.md) edge setup

**Contribute code**
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Follow standards and submission process

**Troubleshoot issues**
1. Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
2. Search for your error message

---

## 🔧 Quick Commands

### Setup & Installation
```bash
# Docker setup (recommended)
docker-compose up -d

# Local setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
python scripts/download_datasets.py --datasets mad
python scripts/prepare_data.py --config configs/data/mad_dataset.yaml
```

### Training
```bash
# Modern model (AudioMAE)
python scripts/train_model.py --config configs/models/audioMAE.yaml

# Legacy CNN
python scripts/train_legacy_model.py --model cnn --epochs 150

# Legacy CRNN
python scripts/train_legacy_model.py --model crnn --epochs 300
```

### Testing
```bash
# All tests
pytest tests/ -v

# Legacy model tests
pytest tests/unit/test_legacy_*.py -v

# With coverage
pytest tests/ --cov=core
```

### Comparison & Benchmarking
```bash
# Compare models
python scripts/compare_models.py --device cuda --output results.json
```

---

## 📊 Key Files Reference

### Documentation
| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Setup guide |
| [docs/LEGACY_MODELS.md](docs/LEGACY_MODELS.md) | Legacy models guide |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | FAQ & troubleshooting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide |

### Models
| Location | Description |
|----------|-------------|
| `src/core/models/audioMAE/` | AudioMAE implementation |
| `src/core/models/ast/` | AST implementation |
| `src/core/models/beats/` | BEATs implementation |
| `src/core/models/legacy/` | Legacy CNN/CRNN (NEW) |

### Configuration
| File | Description |
|------|-------------|
| `configs/models/audioMAE.yaml` | AudioMAE config |
| `configs/models/legacy_cnn_mfcc.yaml` | CNN config (NEW) |
| `configs/models/legacy_crnn_mfcc.yaml` | CRNN config (NEW) |

### Scripts
| File | Description |
|------|-------------|
| `scripts/train_model.py` | Train modern models |
| `scripts/train_legacy_model.py` | Train legacy models (NEW) |
| `scripts/compare_models.py` | Compare models (NEW) |
| `scripts/evaluate_model.py` | Evaluate models |
| `scripts/optimize_for_edge.py` | Optimize for edge |

---

## 🔗 Important Links

**Documentation**: [docs/](docs/) folder
**Source Code**: [src/core/](src/core/) folder
**Tests**: [tests/](tests/) folder
**Configuration**: [configs/](configs/) folder
**Scripts**: [scripts/](scripts/) folder

---

## ✅ What's Included

**Modern Models** (Production-ready)
- ✅ AudioMAE (91.07% accuracy on MAD)
- ✅ Audio Spectrogram Transformer (89.45% accuracy)
- ✅ BEATs (90.23% accuracy)

**Legacy Models** (For comparison & education)
- ✅ CNN MFCC (85% accuracy, 242K params)
- ✅ CRNN MFCC (87% accuracy, 1.5M params)

**Infrastructure**
- ✅ Training pipeline with MLOps
- ✅ Real-time inference (50+ FPS)
- ✅ Edge optimization (Jetson, RPi)
- ✅ FastAPI REST API
- ✅ Docker containerization

**Testing & Quality**
- ✅ 67,680 lines of tests
- ✅ 58 legacy model tests (NEW)
- ✅ >90% code coverage
- ✅ CI/CD workflows

---

## 📋 Document Updates

**Latest Changes:**
- Added legacy models integration (CNN, CRNN, MFCC, SpecAugment)
- Cleaned up documentation structure (removed 9 redundant files)
- Updated README.md with legacy models references
- Consolidated documentation from 12 files to 6 professional documents

**Current Version**: Production-ready v1.0.0

---

## 🚀 Next Steps

1. **Read** [README.md](README.md) for project overview
2. **Install** following [docs/INSTALLATION.md](docs/INSTALLATION.md)
3. **Choose** your model based on your needs
4. **Deploy** with confidence!

For questions, check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

**Last Updated**: November 2024
**Status**: Production Ready ✅
