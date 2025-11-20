"""
Evaluate AudioMAE Model on MAD Dataset
Tests the trained model and shows accuracy, loss, and per-class performance.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.models.audioMAE import AudioMAE, AudioMAEConfig
from core.data.loaders.mad_loader import MADDataset
from core.data.preprocessing.spectrograms import MelSpectrogramGenerator

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

class SpectrogramResizeTransform:
    """Resize spectrograms to target size."""
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def __call__(self, spectrogram):
        if spectrogram.shape[-2:] != self.target_size:
            spectrogram = F.interpolate(
                spectrogram.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return spectrogram

def evaluate_model(model_path, data_dir, batch_size=16, device='cuda'):
    """
    Evaluate a trained AudioMAE model.

    Args:
        model_path: Path to model checkpoint
        data_dir: Path to MAD dataset
        batch_size: Batch size for evaluation
        device: Device to use (cuda/cpu)
    """
    print("="*80)
    print("AudioMAE Model Evaluation")
    print("="*80)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load checkpoint
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model config
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

    # Create model
    model = AudioMAE(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create spectrogram transform
    spec_transform = MelSpectrogramGenerator(
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        fmin=50,
        fmax=8000
    )
    resize_transform = SpectrogramResizeTransform(target_size=(128, 128))

    # Load validation dataset
    print(f"\nLoading validation dataset from: {data_dir}")
    val_dataset = MADDataset(
        data_dir=Path(data_dir),
        split='val'
    )

    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluation
    print(f"\nEvaluating...")
    all_preds = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Get audio and labels
            audio = batch['spectrograms'] if 'spectrograms' in batch else batch['audio']
            labels = batch['labels']

            # Generate spectrograms if needed
            if audio.dim() == 2 or (audio.dim() == 3 and audio.shape[1] != 1):
                # Raw audio - generate spectrograms
                specs = []
                for aud in audio:
                    if aud.dim() == 2:
                        aud = aud[0]  # Take first channel
                    spec = spec_transform(aud)
                    spec = resize_transform(spec)
                    specs.append(spec)
                audio = torch.stack(specs)

            # Ensure correct shape [B, 1, H, W]
            if audio.dim() == 3:
                audio = audio.unsqueeze(1)  # Add channel dimension

            audio = audio.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(audio)

            # Get predictions
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', outputs.get('cls_output')))
            else:
                logits = outputs

            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            all_losses.append(loss.item())

            # Get predictions
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    avg_loss = np.mean(all_losses)

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Loss: {avg_loss:.4f}")

    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for class_id, class_name in MAD_CLASSES.items():
        class_mask = all_labels == class_id
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            print(f"  {class_name:20s}: {class_acc:.2%} ({class_mask.sum():4d} samples)")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  (Rows: True Labels, Columns: Predictions)")
    print(f"  Classes: " + ", ".join([f"{i}:{MAD_CLASSES[i][:10]}" for i in range(7)]))

    confusion = np.zeros((7, 7), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion[true_label][pred_label] += 1

    print()
    for i in range(7):
        row_str = f"  {MAD_CLASSES[i][:15]:15s} | "
        row_str += " ".join([f"{confusion[i][j]:4d}" for j in range(7)])
        print(row_str)

    print(f"\n{'='*80}")

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'confusion_matrix': confusion
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AudioMAE model")
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/checkpoint_audiomae_019.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/mad',
        help='Path to MAD dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
