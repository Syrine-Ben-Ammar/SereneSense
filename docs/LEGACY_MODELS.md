# Legacy Models: CNN and CRNN

This document covers the legacy CNN and CRNN models integrated for comparison and educational purposes.

## Overview

SereneSense includes legacy MFCC-based CNN and CRNN models alongside modern transformers. These models demonstrate the evolution from traditional deep learning (CNN/CRNN) to state-of-the-art transformer architectures.

| Model | Architecture | Parameters | Accuracy | Latency | Use Case |
|-------|--------------|-----------|----------|---------|----------|
| **CNN MFCC** | 3-layer CNN | 242K | ~85% | 20ms | Edge devices, baselines |
| **CRNN MFCC** | CNN + BiLSTM | 1.5M | ~87% | 120ms | Temporal analysis, research |
| **AudioMAE** | Vision Transformer | 300M+ | 91% | 8.2ms | Production (recommended) |

## Architecture Details

### CNN MFCC Model
- **Input**: MFCC + delta + delta-delta (40×92×3)
- **Architecture**: Conv(48) → Conv(96) → Conv(192) → GlobalAvgPool → Dense
- **Parameters**: ~242K
- **Strengths**: Lightweight, fast inference, suitable for edge
- **Limitations**: Lower accuracy, limited temporal context

### CRNN MFCC Model
- **Input**: MFCC + delta + delta-delta (40×124×3) for 4-second audio
- **Architecture**: Conv(48→96→192) + BiLSTM(128) + BiLSTM(64) → Pool → Dense
- **Parameters**: ~1.5M (6.3x larger than CNN)
- **Strengths**: Explicit temporal modeling, better accuracy
- **Limitations**: Slower inference, not suitable for real-time edge

## Training

### Dataset Preparation
```bash
# Download MAD dataset
python scripts/download_datasets.py --datasets mad

# Prepare data
python scripts/prepare_data.py --config configs/data/mad_dataset.yaml
```

### Training CNN
```bash
python scripts/train_legacy_model.py \
    --model cnn \
    --epochs 150 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --checkpoint models/cnn_best.pth
```

### Training CRNN
```bash
python scripts/train_legacy_model.py \
    --model crnn \
    --epochs 300 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --checkpoint models/crnn_best.pth
```

## Usage

### Import and Initialize
```python
from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelConfig
from core.models.legacy.legacy_config import LegacyModelType

# CNN
config = LegacyModelConfig(model_type=LegacyModelType.CNN)
model = CNNMFCCModel(config)

# CRNN
config = LegacyModelConfig(model_type=LegacyModelType.CRNN)
model = CRNNMFCCModel(config)
```

### Feature Extraction
```python
from core.data.preprocessing.legacy_mfcc import LegacyMFCCPreprocessor

# Prepare preprocessor
preprocessor = LegacyMFCCPreprocessor(
    sample_rate=16000,
    duration=3.0,  # 3s for CNN, 4s for CRNN
    n_mfcc=40,
    use_deltas=True,
    use_delta_deltas=True,
)

# Process audio file
features = preprocessor.process_file('audio.wav')
# Output shape: (40, n_frames, 3)
```

### Data Augmentation
```python
from core.data.augmentation.legacy_specaugment import LegacySpecAugmentTransform
import torch

# Create augmentation
augment = LegacySpecAugmentTransform(
    freq_mask_param=15,
    num_freq_masks=2,
    time_mask_param=10,
    num_time_masks=2,
    apply_prob=0.8,
    training=True,
)

# Apply to batch
batch = torch.randn(4, 3, 40, 92)
augmented = augment(batch)
```

### Inference
```python
import torch

# Prepare input
features = torch.randn(1, 3, 40, 92)

# Get predictions
with torch.no_grad():
    logits = model(features)
    probs = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1)

# Or use convenience methods
pred_class = model.predict(features)
probabilities = model.predict_proba(features)
```

## Configuration

### YAML Configuration
Configurations are available in `configs/models/`:
- `legacy_cnn_mfcc.yaml`: Complete CNN configuration
- `legacy_crnn_mfcc.yaml`: Complete CRNN configuration

### Python Configuration
```python
from core.models.legacy.legacy_config import (
    LegacyModelConfig,
    MFCCConfig,
    SpecAugmentConfig,
    CNNConfig,
    CRNNConfig,
)

config = LegacyModelConfig(
    model_type=LegacyModelType.CNN,
    mfcc=MFCCConfig(
        sample_rate=16000,
        n_mfcc=40,
        use_deltas=True,
        use_delta_deltas=True,
    ),
    spec_augment=SpecAugmentConfig(
        freq_mask_param=15,
        num_freq_masks=2,
    ),
    device='cuda',
)
```

## Comparison: Legacy vs Modern

### Accuracy Improvement
```
CNN MFCC:     85% ━━━━━━━
CRNN MFCC:    87% ━━━━━━━━
AudioMAE:     91% ━━━━━━━━━━  (+4-6% improvement)
```

### Feature Extraction
- **Legacy (MFCC)**: Handcrafted features designed for speech
  - 40 MFCC coefficients
  - Fixed representation regardless of task
  - Information loss during aggregation

- **Modern (Learned)**: Self-supervised pre-training on 2M+ samples
  - 128-bin mel-spectrograms with learned embeddings
  - Task-adaptive representations
  - Transfer learning from diverse audio

### Temporal Modeling
- **CNN**: Implicit via convolution (limited context)
- **CRNN**: Explicit via BiLSTM (full sequence context)
- **Transformer**: Attention-based (parallel, efficient)

## Testing

### Run All Legacy Model Tests
```bash
pytest tests/unit/test_legacy_*.py -v
```

### Specific Test Files
```bash
# Model tests (24 tests)
pytest tests/unit/test_legacy_models.py -v

# Feature extraction tests (20 tests)
pytest tests/unit/test_legacy_features.py -v

# Augmentation tests (14 tests)
pytest tests/unit/test_legacy_augmentation.py -v
```

## Performance Benchmarking

### Compare Models
```bash
python scripts/compare_models.py --device cuda --output results.json
```

This generates:
- Parameter counts
- Inference latency (CPU/GPU)
- Memory usage
- Throughput metrics
- Detailed comparison tables

## When to Use Legacy Models

### Use CNN When:
- ✅ Need <5ms latency
- ✅ Extreme resource constraints
- ✅ Educational purposes
- ✅ Baseline comparison

### Use CRNN When:
- ✅ Need better accuracy (~87%)
- ✅ Analyzing temporal patterns
- ✅ Research/experimentation
- ✅ Not constrained by latency

### Use Modern Transformers When:
- ✅ Production deployment (recommended)
- ✅ Best possible accuracy needed (91%+)
- ✅ Real-time requirements
- ✅ Edge deployment (with optimization)

## Key Differences from Notebooks

The integrated legacy models differ from original notebooks in:

1. **Modular Design**: Separated into reusable components
2. **Professional Code**: Type hints, docstrings, error handling
3. **Configuration**: YAML + Python dataclass config
4. **Testing**: Comprehensive unit test suite (58 tests)
5. **Integration**: Seamless with modern SereneSense components
6. **Documentation**: Professional guides and examples

## File Structure

```
src/core/models/legacy/
├── __init__.py                 # Module exports
├── base_legacy_model.py        # Abstract base class
├── legacy_config.py            # Configuration dataclasses
├── cnn_mfcc.py                 # CNN model
└── crnn_mfcc.py                # CRNN model

src/core/data/
├── preprocessing/legacy_mfcc.py
└── augmentation/legacy_specaugment.py

configs/models/
├── legacy_cnn_mfcc.yaml
└── legacy_crnn_mfcc.yaml

scripts/
├── train_legacy_model.py       # Training script
└── compare_models.py           # Comparison tool

tests/unit/
├── test_legacy_models.py       # Model tests (24)
├── test_legacy_features.py     # Feature tests (20)
└── test_legacy_augmentation.py # Augmentation tests (14)
```

## Troubleshooting

### Memory Issues During Training
```bash
# Reduce batch size
--batch-size 8

# Enable gradient checkpointing
# (Modify training script to enable)
```

### CUDA Out of Memory
```bash
# Use CPU training
--device cpu

# Or use mixed precision
# (Already enabled in training script)
```

### Feature Shape Mismatch
Ensure MFCC preprocessor duration matches model expectations:
- CNN: `duration=3.0` → output shape (40, 92, 3)
- CRNN: `duration=4.0` → output shape (40, 124, 3)

## References

- **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (2019)
- **MFCC**: Davis & Mermelstein, "Comparison of parametric representations for monosyllabic word recognition" (1980)
- **CRNN**: Choi et al., "Convolutional Recurrent Neural Networks for Music Classification" (2017)
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)

## See Also

- [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md): Detailed comparison between legacy and modern approaches
- [INSTALLATION.md](INSTALLATION.md): Setup instructions
- [README.md](../README.md): Main documentation
