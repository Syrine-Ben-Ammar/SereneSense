"""
Quick Evaluation Using Training Data Pipeline
Uses the same data loading as training to ensure compatibility.
"""

import torch
from pathlib import Path
import sys
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.models.audioMAE import AudioMAE, AudioMAEConfig
from core.data.loaders.mad_loader import create_mad_dataloader

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

def evaluate():
    """Quick evaluation on validation set."""

    print("="*80)
    print("Quick AudioMAE Evaluation")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    checkpoint_path = "outputs/checkpoint_audiomae_099.pth"
    print(f"\nLoading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
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
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load validation data (using same pipeline as training)
    print("\nLoading validation data...")
    val_loader = create_mad_dataloader(
        data_dir=Path("data/raw/mad"),
        split='val',
        batch_size=32,
        num_workers=4
    )

    print(f"Validation batches: {len(val_loader)}")

    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # Per-class stats
    class_correct = {i: 0 for i in range(7)}
    class_total = {i: 0 for i in range(7)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get data
            spectrograms = batch['spectrograms'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            outputs = model(spectrograms)

            # Get predictions
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', outputs.get('cls_output')))
            else:
                logits = outputs

            preds = logits.argmax(dim=1)

            # Stats
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Per-class
            for i in range(7):
                mask = labels == i
                if mask.sum() > 0:
                    class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Progress: {batch_idx + 1}/{len(val_loader)} batches...")

    # Results
    accuracy = 100 * correct / total

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

    print(f"\nPer-Class Accuracy:")
    for i in range(7):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {MAD_CLASSES[i]:20s}: {class_acc:6.2f}% ({class_correct[i]:3d}/{class_total[i]:3d})")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  Rows=True, Cols=Predicted")

    confusion = np.zeros((7, 7), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    # Header
    print("      ", end="")
    for i in range(7):
        print(f"{i:5d}", end="")
    print()

    # Rows
    for i in range(7):
        print(f"  {i}: ", end="")
        for j in range(7):
            print(f"{confusion[i][j]:5d}", end="")
        print(f"  ({MAD_CLASSES[i]})")

    print("\n" + "="*80)

    return accuracy

if __name__ == "__main__":
    acc = evaluate()
    print(f"\nFinal Validation Accuracy: {acc:.2f}%\n")
