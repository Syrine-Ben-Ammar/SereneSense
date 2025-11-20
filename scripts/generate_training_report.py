"""
Generate Comprehensive Training Report for AudioMAE Model
Creates all metrics, graphs, and analysis for research report.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_checkpoint(checkpoint_path):
    """Load checkpoint safely."""
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)

def extract_training_history(checkpoints_dir):
    """Extract full training history from all checkpoints."""
    checkpoints_dir = Path(checkpoints_dir)

    # Find all checkpoints
    checkpoints = sorted(checkpoints_dir.glob("checkpoint_audiomae_*.pth"))

    if not checkpoints:
        print("No checkpoints found!")
        return None

    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
        'epoch_time': []
    }

    print(f"Found {len(checkpoints)} checkpoints. Extracting history...")

    for cp_path in checkpoints:
        try:
            checkpoint = load_checkpoint(cp_path)
            epoch = checkpoint.get('epoch', 0)

            # Extract training metrics
            train_metrics = checkpoint.get('train_metrics', {})
            val_metrics = checkpoint.get('val_metrics', {})

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_metrics.get('loss', 0))
            history['train_accuracy'].append(train_metrics.get('accuracy', 0) * 100)
            history['val_loss'].append(val_metrics.get('loss', 0))
            history['val_accuracy'].append(val_metrics.get('accuracy', 0) * 100)

            # Learning rate
            optimizer = checkpoint.get('optimizer', {})
            if 'param_groups' in optimizer:
                lr = optimizer['param_groups'][0]['lr']
                history['learning_rate'].append(lr)
            else:
                history['learning_rate'].append(0)

            # Epoch time
            history['epoch_time'].append(train_metrics.get('epoch_time', 0))

        except Exception as e:
            print(f"Error loading {cp_path.name}: {e}")

    return history

def plot_training_curves(history, output_dir):
    """Generate training/validation curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AudioMAE Training Progress - MAD Dataset', fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
    ax1.set_title('Loss Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Improvement', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    # 3. Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], linewidth=2, color='green', marker='o', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Generalization gap
    ax4 = axes[1, 1]
    gap = np.array(history['val_accuracy']) - np.array(history['train_accuracy'])
    ax4.plot(epochs, gap, linewidth=2, color='purple', marker='o', markersize=3)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='No Gap')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax4.set_title('Generalization Gap (Val Acc - Train Acc)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved training curves: {output_path}")
    plt.close()

def plot_performance_comparison(history, output_dir):
    """Create detailed performance comparison plots."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs = history['epoch']

    # 1. Loss comparison (last 20 epochs zoomed)
    ax1 = axes[0]
    start_idx = max(0, len(epochs) - 20)
    ax1.plot(epochs[start_idx:], history['train_loss'][start_idx:],
             label='Training Loss', linewidth=2.5, marker='o', markersize=5)
    ax1.plot(epochs[start_idx:], history['val_loss'][start_idx:],
             label='Validation Loss', linewidth=2.5, marker='s', markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence (Final 20 Epochs)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy comparison (last 20 epochs zoomed)
    ax2 = axes[1]
    ax2.plot(epochs[start_idx:], history['train_accuracy'][start_idx:],
             label='Training Accuracy', linewidth=2.5, marker='o', markersize=5)
    ax2.plot(epochs[start_idx:], history['val_accuracy'][start_idx:],
             label='Validation Accuracy', linewidth=2.5, marker='s', markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy (Final 20 Epochs)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved performance comparison: {output_path}")
    plt.close()

def generate_metrics_table(history, output_dir):
    """Generate comprehensive metrics table."""
    output_dir = Path(output_dir)

    # Calculate key metrics
    metrics = {
        'Metric': [],
        'Initial (Epoch 1)': [],
        'Final (Epoch 100)': [],
        'Best': [],
        'Improvement': []
    }

    # Training Loss
    metrics['Metric'].append('Training Loss')
    metrics['Initial (Epoch 1)'].append(f"{history['train_loss'][0]:.4f}")
    metrics['Final (Epoch 100)'].append(f"{history['train_loss'][-1]:.4f}")
    metrics['Best'].append(f"{min(history['train_loss']):.4f}")
    if history['train_loss'][0] != 0:
        improvement = ((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100)
        metrics['Improvement'].append(f"{improvement:.1f}%")
    else:
        metrics['Improvement'].append("N/A")

    # Validation Loss
    metrics['Metric'].append('Validation Loss')
    val_losses = [v for v in history['val_loss'] if v > 0]
    if val_losses:
        metrics['Initial (Epoch 1)'].append(f"{val_losses[0]:.4f}")
        metrics['Final (Epoch 100)'].append(f"{val_losses[-1]:.4f}")
        metrics['Best'].append(f"{min(val_losses):.4f}")
        if val_losses[0] != 0:
            improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0] * 100)
            metrics['Improvement'].append(f"{improvement:.1f}%")
        else:
            metrics['Improvement'].append("N/A")
    else:
        metrics['Initial (Epoch 1)'].append("N/A")
        metrics['Final (Epoch 100)'].append("N/A")
        metrics['Best'].append("N/A")
        metrics['Improvement'].append("N/A")

    # Training Accuracy
    metrics['Metric'].append('Training Accuracy (%)')
    metrics['Initial (Epoch 1)'].append(f"{history['train_accuracy'][0]:.2f}%")
    metrics['Final (Epoch 100)'].append(f"{history['train_accuracy'][-1]:.2f}%")
    metrics['Best'].append(f"{max(history['train_accuracy']):.2f}%")
    improvement = history['train_accuracy'][-1] - history['train_accuracy'][0]
    metrics['Improvement'].append(f"+{improvement:.2f}%")

    # Validation Accuracy
    metrics['Metric'].append('Validation Accuracy (%)')
    val_accs = [v for v in history['val_accuracy'] if v > 0]
    if val_accs:
        metrics['Initial (Epoch 1)'].append(f"{val_accs[0]:.2f}%")
        metrics['Final (Epoch 100)'].append(f"{val_accs[-1]:.2f}%")
        metrics['Best'].append(f"{max(val_accs):.2f}%")
        improvement = val_accs[-1] - val_accs[0]
        metrics['Improvement'].append(f"+{improvement:.2f}%")
    else:
        metrics['Initial (Epoch 1)'].append("N/A")
        metrics['Final (Epoch 100)'].append("N/A")
        metrics['Best'].append("N/A")
        metrics['Improvement'].append("N/A")

    # Learning Rate
    metrics['Metric'].append('Learning Rate')
    metrics['Initial (Epoch 1)'].append(f"{history['learning_rate'][0]:.2e}")
    metrics['Final (Epoch 100)'].append(f"{history['learning_rate'][-1]:.2e}")
    metrics['Best'].append('-')
    metrics['Improvement'].append('-')

    # Create DataFrame
    df = pd.DataFrame(metrics)

    # Save as CSV
    csv_path = output_dir / 'training_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved metrics table: {csv_path}")

    # Save as formatted text
    txt_path = output_dir / 'training_metrics.txt'
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AUDIOMAE TRAINING METRICS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
    print(f"[OK] Saved metrics text: {txt_path}")

    return df

def generate_statistics_report(history, output_dir):
    """Generate detailed statistics report."""
    output_dir = Path(output_dir)

    report = []
    report.append("="*80)
    report.append("AUDIOMAE TRAINING - STATISTICAL ANALYSIS")
    report.append("="*80)
    report.append("")

    # Overall statistics
    report.append("1. TRAINING OVERVIEW")
    report.append("-" * 40)
    report.append(f"   Total Epochs:           100")
    report.append(f"   Total Training Time:    {sum(history['epoch_time']):.1f}s ({sum(history['epoch_time'])/3600:.2f} hours)")
    report.append(f"   Average Epoch Time:     {np.mean(history['epoch_time']):.1f}s")
    report.append(f"   Dataset:                MAD (Military Audio Detection)")
    report.append(f"   Classes:                7")
    report.append("")

    # Loss statistics
    report.append("2. LOSS ANALYSIS")
    report.append("-" * 40)
    report.append(f"   Training Loss:")
    report.append(f"      Initial:   {history['train_loss'][0]:.4f}")
    report.append(f"      Final:     {history['train_loss'][-1]:.4f}")
    report.append(f"      Best:      {min(history['train_loss']):.4f}")
    report.append(f"      Mean:      {np.mean(history['train_loss']):.4f}")
    report.append(f"      Std Dev:   {np.std(history['train_loss']):.4f}")

    val_losses = [v for v in history['val_loss'] if v > 0]
    if val_losses:
        report.append(f"   Validation Loss:")
        report.append(f"      Initial:   {val_losses[0]:.4f}")
        report.append(f"      Final:     {val_losses[-1]:.4f}")
        report.append(f"      Best:      {min(val_losses):.4f}")
        report.append(f"      Mean:      {np.mean(val_losses):.4f}")
        report.append(f"      Std Dev:   {np.std(val_losses):.4f}")
    report.append("")

    # Accuracy statistics
    report.append("3. ACCURACY ANALYSIS")
    report.append("-" * 40)
    report.append(f"   Training Accuracy:")
    report.append(f"      Initial:   {history['train_accuracy'][0]:.2f}%")
    report.append(f"      Final:     {history['train_accuracy'][-1]:.2f}%")
    report.append(f"      Best:      {max(history['train_accuracy']):.2f}%")
    report.append(f"      Mean:      {np.mean(history['train_accuracy']):.2f}%")

    val_accs = [v for v in history['val_accuracy'] if v > 0]
    if val_accs:
        report.append(f"   Validation Accuracy:")
        report.append(f"      Initial:   {val_accs[0]:.2f}%")
        report.append(f"      Final:     {val_accs[-1]:.2f}%")
        report.append(f"      Best:      {max(val_accs):.2f}%")
        report.append(f"      Mean:      {np.mean(val_accs):.2f}%")
    report.append("")

    # Generalization analysis
    report.append("4. GENERALIZATION ANALYSIS")
    report.append("-" * 40)
    gap = np.array(history['val_accuracy']) - np.array(history['train_accuracy'])
    gap_filtered = gap[gap != -np.array(history['train_accuracy'])]  # Remove zeros
    if len(gap_filtered) > 0:
        report.append(f"   Generalization Gap (Val - Train):")
        report.append(f"      Final:     {gap[-1]:.2f}%")
        report.append(f"      Mean:      {np.mean(gap_filtered):.2f}%")
        report.append(f"      Best:      {max(gap_filtered):.2f}%")
        if gap[-1] > 0:
            report.append(f"   [OK] Model generalizes well (positive gap)")
        else:
            report.append(f"   [!] Model shows signs of overfitting (negative gap)")
    report.append("")

    # Learning rate analysis
    report.append("5. LEARNING RATE SCHEDULE")
    report.append("-" * 40)
    report.append(f"   Initial LR:    {history['learning_rate'][0]:.2e}")
    report.append(f"   Final LR:      {history['learning_rate'][-1]:.2e}")
    report.append(f"   Min LR:        {min(history['learning_rate']):.2e}")
    report.append(f"   Max LR:        {max(history['learning_rate']):.2e}")
    report.append("")

    # Performance improvement
    report.append("6. PERFORMANCE IMPROVEMENT")
    report.append("-" * 40)
    if history['train_loss'][0] != 0:
        train_loss_improvement = ((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100)
        report.append(f"   Training Loss:     {train_loss_improvement:.1f}% reduction")

    if val_losses and val_losses[0] != 0:
        val_loss_improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0] * 100)
        report.append(f"   Validation Loss:   {val_loss_improvement:.1f}% reduction")

    train_acc_improvement = history['train_accuracy'][-1] - history['train_accuracy'][0]
    report.append(f"   Training Acc:      +{train_acc_improvement:.2f}%")

    if val_accs:
        val_acc_improvement = val_accs[-1] - val_accs[0]
        report.append(f"   Validation Acc:    +{val_acc_improvement:.2f}%")
    report.append("")

    report.append("="*80)

    # Save report
    report_path = output_dir / 'statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"[OK] Saved statistics report: {report_path}")

    # Also print to console
    print("\n" + '\n'.join(report))

def main():
    """Generate comprehensive training report."""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE TRAINING REPORT")
    print("="*80 + "\n")

    # Directories
    checkpoints_dir = Path("outputs")
    reports_dir = Path("docs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Extract training history
    print("[*] Extracting training history from checkpoints...")
    history = extract_training_history(checkpoints_dir)

    if history is None or len(history['epoch']) == 0:
        print("[X] No training history found!")
        return

    print(f"[OK] Extracted {len(history['epoch'])} epochs of training data\n")

    # Generate all visualizations and reports
    print("[*] Generating training curves...")
    plot_training_curves(history, reports_dir)

    print("[*] Generating performance comparisons...")
    plot_performance_comparison(history, reports_dir)

    print("[*] Generating metrics table...")
    metrics_df = generate_metrics_table(history, reports_dir)

    print("[*] Generating statistics report...")
    generate_statistics_report(history, reports_dir)

    # Save full history as JSON
    history_path = reports_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history: {history_path}")

    print("\n" + "="*80)
    print("[SUCCESS] REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"\nAll reports saved to: {reports_dir.absolute()}")
    print("\nGenerated files:")
    print("  - training_curves.png            (Main training/validation curves)")
    print("  - performance_comparison.png     (Detailed final epoch analysis)")
    print("  - training_metrics.csv           (Metrics table for LaTeX/Excel)")
    print("  - training_metrics.txt           (Formatted metrics text)")
    print("  - statistics_report.txt          (Complete statistical analysis)")
    print("  - training_history.json          (Raw data for further analysis)")
    print("\n")

if __name__ == "__main__":
    main()
