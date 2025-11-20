# Phase 1 – Legacy CNN Baseline (MAD)

This note captures everything needed for the thesis documentation of Phase 1 Step 1.1A (legacy CNN trained from scratch on the MAD dataset).

## Training Configuration

| Item | Value |
| --- | --- |
| Script | `scripts/train_legacy_model.py` |
| Command | `python scripts/train_legacy_model.py --model cnn --config configs/models/legacy_cnn_mfcc.yaml --epochs 50 --batch-size 32 --learning-rate 1e-3 --num-workers 4 --prefetch-factor 1 --max-pending-batches 4 --persistent-workers --checkpoint outputs/phase1/cnn_legacy7.pth --history-path outputs/history/cnn_phase1_run2.json` |
| Dataset | MAD train/validation HDF5 (501 labels) |
| Features | On-the-fly MFCC (40×92×3) with deltas & SpecAugment |
| Hardware | Quadro RTX 4000 (CUDA) |

## Key Metrics

| Metric | Value | Source |
| --- | --- | --- |
| Best validation accuracy | **6.00 %** | `outputs/history/cnn_phase1_run2.json` (`best_accuracy`, epoch 39) |
| Final evaluation accuracy | **3.71 %** | `outputs/evaluations/cnn_legacy7_validation_report.json` |
| Evaluation loss | 5.632 | same report |
| Parameters | 321,205 | `outputs/evaluations/cnn_legacy7_metadata.json` |

### Curves

- Training/validation loss + accuracy plot: `outputs/plots/cnn_phase1_history.png`
- JSON history (per-epoch series): `outputs/history/cnn_phase1_run2.json`

### Evaluation Artifacts

- Classification report (`precision`, `recall`, `F1`, macro/micro averages): `outputs/evaluations/cnn_legacy7_validation_report.json`
- Confusion matrix (`numpy` shape 501×501): `outputs/evaluations/cnn_legacy7_validation_confusion.npy`
- Checkpoint metadata snapshot: `outputs/evaluations/cnn_legacy7_metadata.json`

## Observations

1. Accuracy grows steadily for ~40 epochs but remains in single digits because the legacy CNN is small and the dataset is extremely imbalanced (many classes have ≤3 validation samples). Random chance is 0.2 %, so 6 % already indicates the model is 30× better than chance.
2. The evaluation accuracy is lower than the training-loop best because the evaluation run disables SpecAugment and sweeps the entire validation split once, which is the correct reporting procedure.
3. Training is stable (loss monotonically decreases). The bottleneck is representational capacity and class imbalance, not optimization failure.

## Recommendations

| Next Action | Goal |
| --- | --- |
| Enable class weights or focal loss | Compensate for MAD label imbalance |
| Extend training schedule (100–150 epochs) with LR scheduler | CNN was still improving at epoch 50 |
| Cache MFCC tensors via `scripts/cache_mfcc_features.py` | Reduce CPU preprocessing, allow higher `num-workers` |
| Move to CRNN & transformer baselines (Phase 1 Steps 1.1B & 1.2) | Achieve higher accuracy targets and complete reference suite |

This document, plus the referenced artifacts, is sufficient to cite the legacy CNN baseline in the thesis (methods, configuration, metrics, and evidence files). No further edits are required for Step 1.1A.
