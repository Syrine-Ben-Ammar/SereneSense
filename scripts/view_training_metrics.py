"""
View Training Metrics from AudioMAE Checkpoints
Extracts and displays all training metrics, accuracy, and performance data.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def load_checkpoint_safe(checkpoint_path):
    """Load checkpoint with weights_only=False for older format."""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def display_checkpoint_metrics(checkpoint_path):
    """Display detailed metrics from a checkpoint."""
    checkpoint = load_checkpoint_safe(checkpoint_path)
    if checkpoint is None:
        return None

    print(f"\n{'='*80}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"{'='*80}")

    metrics_data = {}

    # Extract epoch
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        print(f"\nEpoch: {epoch + 1}")
        metrics_data['epoch'] = epoch + 1

    # Extract training metrics
    if 'train_metrics' in checkpoint:
        print(f"\n--- Training Metrics ---")
        train_metrics = checkpoint['train_metrics']

        if isinstance(train_metrics, dict):
            for key, value in train_metrics.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    latest = value[-1] if isinstance(value, list) else value
                    print(f"  {key}: {latest:.4f}")
                    metrics_data[f'train_{key}'] = latest
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                    metrics_data[f'train_{key}'] = value

    # Extract validation metrics
    if 'val_metrics' in checkpoint:
        print(f"\n--- Validation Metrics ---")
        val_metrics = checkpoint['val_metrics']

        if isinstance(val_metrics, dict):
            for key, value in val_metrics.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    latest = value[-1] if isinstance(value, list) else value
                    print(f"  {key}: {latest:.4f}")
                    metrics_data[f'val_{key}'] = latest
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                    metrics_data[f'val_{key}'] = value

    # Extract best metrics
    if 'best_metric' in checkpoint:
        print(f"\n--- Best Performance ---")
        print(f"  Best Metric Value: {checkpoint['best_metric']:.4f}")
        metrics_data['best_metric'] = checkpoint['best_metric']

    # Extract learning rate
    if 'optimizer' in checkpoint and 'param_groups' in checkpoint['optimizer']:
        lr = checkpoint['optimizer']['param_groups'][0]['lr']
        print(f"\n--- Optimizer ---")
        print(f"  Learning Rate: {lr:.2e}")
        metrics_data['learning_rate'] = lr

    # Extract training time
    if 'training_time' in checkpoint:
        print(f"\n--- Training Time ---")
        print(f"  Total Time: {checkpoint['training_time']:.2f}s ({checkpoint['training_time']/60:.1f} minutes)")
        metrics_data['training_time_seconds'] = checkpoint['training_time']

    # File info
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
    print(f"\n--- File Info ---")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return metrics_data

def compare_checkpoints(checkpoint_paths):
    """Compare metrics across multiple checkpoints."""
    print(f"\n{'='*80}")
    print("TRAINING PROGRESS COMPARISON")
    print(f"{'='*80}\n")

    all_metrics = []

    for cp_path in checkpoint_paths:
        metrics = display_checkpoint_metrics(cp_path)
        if metrics:
            all_metrics.append(metrics)

    if len(all_metrics) > 1:
        print(f"\n{'='*80}")
        print("IMPROVEMENT SUMMARY")
        print(f"{'='*80}\n")

        first = all_metrics[0]
        last = all_metrics[-1]

        # Compare key metrics
        for metric in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']:
            if metric in first and metric in last:
                change = last[metric] - first[metric]
                pct_change = (change / first[metric] * 100) if first[metric] != 0 else 0
                direction = "↑" if change > 0 else "↓"
                print(f"{metric}:")
                print(f"  Epoch {first['epoch']}: {first[metric]:.4f}")
                print(f"  Epoch {last['epoch']}: {last[metric]:.4f}")
                print(f"  Change: {direction} {abs(change):.4f} ({abs(pct_change):.1f}%)\n")

def main():
    """Main function to view all metrics."""
    outputs_dir = Path("outputs")

    print("\n" + "="*80)
    print("AudioMAE Training Metrics Viewer")
    print("="*80)

    # Find all AudioMAE checkpoints
    checkpoints = list(outputs_dir.glob("*audiomae*.pth"))
    checkpoints.extend(outputs_dir.glob("*audioMAE*.pth"))

    if not checkpoints:
        print("\n[X] No checkpoints found!")
        return

    # Sort by epoch number
    def get_epoch_num(path):
        try:
            # Extract number from checkpoint_audiomae_019.pth
            stem = path.stem
            if 'checkpoint' in stem:
                return int(stem.split('_')[-1])
            return -1
        except:
            return -1

    checkpoints.sort(key=get_epoch_num)

    print(f"\nFound {len(checkpoints)} checkpoint(s)\n")

    # Display all checkpoints
    compare_checkpoints(checkpoints)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Summary
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    if latest_checkpoint:
        epoch_num = get_epoch_num(latest_checkpoint)
        print(f"\nLatest Training Epoch: {epoch_num + 1}")
        print(f"Latest Checkpoint: {latest_checkpoint.name}")

        # Load latest to show final metrics
        latest = load_checkpoint_safe(latest_checkpoint)
        if latest:
            print(f"\nFinal Status:")
            if 'train_metrics' in latest and 'loss' in latest['train_metrics']:
                losses = latest['train_metrics']['loss']
                final_loss = losses[-1] if isinstance(losses, list) else losses
                print(f"  Final Training Loss: {final_loss:.4f}")

            if 'val_metrics' in latest and 'accuracy' in latest['val_metrics']:
                accs = latest['val_metrics']['accuracy']
                final_acc = accs[-1] if isinstance(accs, list) else accs
                print(f"  Final Validation Accuracy: {final_acc:.2%}")

    print()

if __name__ == "__main__":
    main()
