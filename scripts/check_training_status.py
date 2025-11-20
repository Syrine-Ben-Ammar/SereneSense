"""
Check AudioMAE Training Status
Quick script to verify if training completed successfully.
"""

import torch
from pathlib import Path
from datetime import datetime

def check_checkpoint(checkpoint_path):
    """Check a checkpoint file and display info."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(f"\n{'='*70}")
        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"{'='*70}")

        # Get basic info
        if isinstance(checkpoint, dict):
            print(f"\nCheckpoint Keys: {list(checkpoint.keys())}")

            # Epoch info
            if 'epoch' in checkpoint:
                print(f"\nEpoch: {checkpoint['epoch']}")

            # Model state
            if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                state_dict = checkpoint[state_key]
                num_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                print(f"Model Parameters: {num_params:,}")

            # Metrics
            if 'metrics' in checkpoint:
                print(f"\nMetrics:")
                for key, value in checkpoint['metrics'].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

            # Best metric
            if 'best_metric' in checkpoint:
                print(f"\nBest Metric: {checkpoint['best_metric']:.4f}")

            # Training time
            if 'training_time' in checkpoint:
                print(f"Training Time: {checkpoint['training_time']:.2f}s")

        else:
            print("Checkpoint is a model state dict (not a full checkpoint)")
            num_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
            print(f"Model Parameters: {num_params:,}")

        # File size and date
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        print(f"\nFile Size: {file_size_mb:.1f} MB")
        print(f"Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"Error loading {checkpoint_path.name}: {e}")
        return False

def main():
    """Main function to check all AudioMAE checkpoints."""
    outputs_dir = Path("outputs")

    print("\n" + "="*70)
    print("AudioMAE Training Status Check")
    print("="*70)

    # Find all AudioMAE checkpoints
    checkpoints = list(outputs_dir.glob("*audiomae*.pth"))
    checkpoints.extend(outputs_dir.glob("*audioMAE*.pth"))

    if not checkpoints:
        print("\n[X] No AudioMAE checkpoints found!")
        print("   Training may not have completed or saved any checkpoints.")
        return

    print(f"\n[OK] Found {len(checkpoints)} checkpoint(s)")

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Check each checkpoint
    for checkpoint_path in checkpoints:
        check_checkpoint(checkpoint_path)

    # Determine if training completed
    print("\n" + "="*70)
    print("TRAINING STATUS ANALYSIS")
    print("="*70)

    best_model = outputs_dir / "best_model_audiomae_000.pth"
    latest_checkpoint = checkpoints[0] if checkpoints else None

    if best_model.exists() and "checkpoint_audiomae_009" in str(latest_checkpoint):
        print("\n[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
        print("   - Best model saved")
        print("   - Reached epoch 10 (checkpoint_009)")
        print(f"   - Best model: {best_model}")
        print(f"   - Latest checkpoint: {latest_checkpoint}")
    elif latest_checkpoint:
        checkpoint_num = int(str(latest_checkpoint.stem).split('_')[-1])
        print(f"\n[WARNING] TRAINING PARTIALLY COMPLETED")
        print(f"   - Reached epoch {checkpoint_num + 1}")
        print(f"   - Latest checkpoint: {latest_checkpoint}")
        if checkpoint_num < 9:
            print(f"   - Missing {10 - checkpoint_num - 1} epoch(s) to complete 10 epochs")
    else:
        print("\n[ERROR] TRAINING STATUS UNKNOWN")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
