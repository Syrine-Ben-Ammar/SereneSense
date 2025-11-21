"""
Export trained AudioMAE model to ONNX format for Raspberry Pi deployment.

This script:
1. Loads the trained AudioMAE checkpoint
2. Exports the model to ONNX format (FP32)
3. Validates the ONNX model output matches PyTorch
4. Saves the ONNX model for deployment

Author: SereneSense Team
Date: 2025-11-21
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from core.models.audioMAE import AudioMAE, AudioMAEConfig


def load_trained_model(checkpoint_path: str) -> AudioMAE:
    """
    Load the trained AudioMAE model from checkpoint.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file

    Returns:
        Loaded AudioMAE model in evaluation mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Create model configuration (same as training)
    config = AudioMAEConfig(
        num_classes=7,              # MAD dataset classes
        img_size=(128, 128),        # Spectrogram size
        patch_size=16,              # 16x16 patches
        embed_dim=768,              # Embedding dimension
        encoder_depth=12,           # 12 transformer layers
        encoder_num_heads=12,       # 12 attention heads
        decoder_embed_dim=512,      # Decoder embedding
        decoder_depth=8,            # 8 decoder layers
        decoder_num_heads=16,       # 16 decoder attention heads
        mlp_ratio=4.0,              # MLP hidden dimension ratio
        dropout=0.0,                # Dropout rate
        attention_dropout=0.0,      # Attention dropout
        drop_path=0.1               # Stochastic depth (drop_path_rate)
    )

    # Create model
    model = AudioMAE(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load state dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load weights (strict=False to ignore decoder weights if present)
    model.load_state_dict(state_dict, strict=False)

    # Set to evaluation mode
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


class ModelWrapper(torch.nn.Module):
    """Wrapper to make model return only logits for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # Return only logits
        if isinstance(output, dict):
            return output['logits']
        elif hasattr(output, 'logits'):
            return output.logits
        else:
            return output


def export_to_onnx(
    model: AudioMAE,
    output_path: str,
    opset_version: int = 14,
    validate: bool = True
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: Trained AudioMAE model
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (14 for broad compatibility)
        validate: Whether to validate the exported model
    """
    print(f"\nExporting model to ONNX format...")
    print(f"Output path: {output_path}")
    print(f"Opset version: {opset_version}")

    # Wrap model to return only logits
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    # Create dummy input (batch_size=1, channels=1, height=128, width=128)
    dummy_input = torch.randn(1, 1, 128, 128)

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,                  # Wrapped model
        dummy_input,                    # Dummy input
        output_path,                    # Output file path
        export_params=True,             # Store trained parameters
        opset_version=opset_version,    # ONNX version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['spectrogram'],    # Input tensor name
        output_names=['logits'],        # Output tensor name
        dynamic_axes={                  # Dynamic batch size
            'spectrogram': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"[OK] Model exported to ONNX successfully!")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] ONNX model size: {file_size_mb:.2f} MB")

    # Validate ONNX model
    if validate:
        print("\nValidating ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid!")

        # Print model info
        print(f"\nModel Information:")
        print(f"  Input: {onnx_model.graph.input[0].name}")
        print(f"  Output: {onnx_model.graph.output[0].name}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")


def validate_onnx_output(
    pytorch_model: AudioMAE,
    onnx_path: str,
    num_samples: int = 5,
    tolerance: float = 1e-5
) -> None:
    """
    Validate that ONNX model produces same output as PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_samples: Number of random samples to test
        tolerance: Maximum allowed difference
    """
    print(f"\nValidating ONNX model output against PyTorch...")
    print(f"Testing {num_samples} random samples with tolerance={tolerance}")

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Wrap model to get logits
    wrapped_model = ModelWrapper(pytorch_model)

    # Test with random inputs
    max_diff = 0.0
    for i in range(num_samples):
        # Generate random input
        test_input = torch.randn(1, 1, 128, 128)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = wrapped_model(test_input).numpy()

        # ONNX inference
        onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]

        # Calculate difference
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)

        print(f"  Sample {i+1}/{num_samples}: max_diff = {diff:.2e}")

    print(f"\nMaximum difference: {max_diff:.2e}")

    if max_diff < tolerance:
        print(f"[OK] ONNX model output matches PyTorch (within tolerance)!")
    else:
        print(f"[WARNING] Difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
        print(f"  This might be acceptable depending on use case.")


def main():
    """Main export pipeline."""

    print("=" * 70)
    print("AudioMAE ONNX Export Pipeline")
    print("=" * 70)

    # Configuration
    checkpoint_path = "outputs/checkpoint_audiomae_099.pth"  # Final checkpoint
    onnx_output_path = "outputs/audiomae_fp32.onnx"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the model has been trained and checkpoint exists.")
        return

    # Step 1: Load trained model
    print("\n[Step 1/3] Loading trained AudioMAE model...")
    model = load_trained_model(checkpoint_path)

    # Step 2: Export to ONNX
    print("\n[Step 2/3] Exporting to ONNX format...")
    export_to_onnx(
        model=model,
        output_path=onnx_output_path,
        opset_version=14,  # Good compatibility with ONNX Runtime
        validate=True
    )

    # Step 3: Validate output
    print("\n[Step 3/3] Validating ONNX output...")
    validate_onnx_output(
        pytorch_model=model,
        onnx_path=onnx_output_path,
        num_samples=10,
        tolerance=1e-5
    )

    print("\n" + "=" * 70)
    print("Export completed successfully!")
    print("=" * 70)
    print(f"\nONNX model saved to: {onnx_output_path}")
    print(f"Next step: Run quantization script to create INT8 model")
    print(f"  --> python scripts/quantize_onnx.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
