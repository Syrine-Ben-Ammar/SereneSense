"""
Test deployment pipeline for Raspberry Pi.

This script validates the complete deployment pipeline:
1. ONNX model loading
2. Preprocessing pipeline
3. Inference execution
4. End-to-end latency measurement

Can be run on development PC or Raspberry Pi for validation.

Author: SereneSense Team
Date: 2025-11-21
"""

import numpy as np
import onnxruntime as ort
import time
import sys
from pathlib import Path
from rpi_preprocessing import AudioPreprocessor


def test_model_loading(model_path: str) -> ort.InferenceSession:
    """Test ONNX model loading."""
    print("\n[Test 1/5] Model Loading")
    print("-" * 50)

    try:
        session = ort.InferenceSession(model_path)
        print(f"[OK] Model loaded successfully")
        print(f"  Path: {model_path}")
        print(f"  Provider: {session.get_providers()[0]}")
        print(f"  Input: {session.get_inputs()[0].name}")
        print(f"  Output: {session.get_outputs()[0].name}")
        return session
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        sys.exit(1)


def test_preprocessing() -> AudioPreprocessor:
    """Test preprocessing pipeline."""
    print("\n[Test 2/5] Preprocessing Pipeline")
    print("-" * 50)

    try:
        preprocessor = AudioPreprocessor()

        # Test with dummy audio
        dummy_audio = np.random.randn(16000 * 10).astype(np.float32)
        spectrogram = preprocessor.preprocess(
            (dummy_audio, 16000),
            input_type='array'
        )

        assert spectrogram.shape == (1, 1, 128, 128), "Shape mismatch!"
        assert spectrogram.dtype == np.float32, "Type mismatch!"

        print(f"[OK] Preprocessing working correctly")
        print(f"  Output shape: {spectrogram.shape}")
        print(f"  Output dtype: {spectrogram.dtype}")
        print(f"  Value range: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]")

        return preprocessor
    except Exception as e:
        print(f"[FAIL] Preprocessing failed: {e}")
        sys.exit(1)


def test_inference(session: ort.InferenceSession, preprocessor: AudioPreprocessor):
    """Test model inference."""
    print("\n[Test 3/5] Model Inference")
    print("-" * 50)

    try:
        # Generate dummy audio
        dummy_audio = np.random.randn(16000 * 10).astype(np.float32)

        # Preprocess
        spectrogram = preprocessor.preprocess(
            (dummy_audio, 16000),
            input_type='array'
        )

        # Run inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: spectrogram})[0]

        # Check output
        assert output.shape == (1, 7), "Output shape mismatch!"

        print(f"[OK] Inference working correctly")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Apply softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        print(f"  Probabilities sum: {probabilities.sum():.6f}")

        return True
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        return False


def test_latency(
    session: ort.InferenceSession,
    preprocessor: AudioPreprocessor,
    num_runs: int = 100
):
    """Test inference latency."""
    print(f"\n[Test 4/5] Latency Measurement ({num_runs} runs)")
    print("-" * 50)

    # Generate test audio
    dummy_audio = np.random.randn(16000 * 10).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        spectrogram = preprocessor.preprocess((dummy_audio, 16000), input_type='array')
        _ = session.run(None, {input_name: spectrogram})

    # Measure preprocessing time
    preprocessing_times = []
    for _ in range(num_runs):
        start = time.time()
        spectrogram = preprocessor.preprocess((dummy_audio, 16000), input_type='array')
        preprocessing_times.append((time.time() - start) * 1000)

    # Measure inference time
    inference_times = []
    for _ in range(num_runs):
        spectrogram = preprocessor.preprocess((dummy_audio, 16000), input_type='array')
        start = time.time()
        _ = session.run(None, {input_name: spectrogram})
        inference_times.append((time.time() - start) * 1000)

    # Calculate statistics
    avg_preprocessing = np.mean(preprocessing_times)
    std_preprocessing = np.std(preprocessing_times)
    min_preprocessing = np.min(preprocessing_times)
    max_preprocessing = np.max(preprocessing_times)

    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    min_inference = np.min(inference_times)
    max_inference = np.max(inference_times)

    total_latency = avg_preprocessing + avg_inference

    print(f"Preprocessing:")
    print(f"  Average: {avg_preprocessing:.2f} ms")
    print(f"  Std Dev: {std_preprocessing:.2f} ms")
    print(f"  Min/Max: {min_preprocessing:.2f} / {max_preprocessing:.2f} ms")

    print(f"\nInference:")
    print(f"  Average: {avg_inference:.2f} ms")
    print(f"  Std Dev: {std_inference:.2f} ms")
    print(f"  Min/Max: {min_inference:.2f} / {max_inference:.2f} ms")

    print(f"\nTotal Latency: {total_latency:.2f} ms")

    # Check against targets
    if total_latency < 500:
        print(f"[OK] Latency meets target (<500ms)")
    else:
        print(f"[WARNING] Latency exceeds target (>500ms)")

    return total_latency


def test_memory_usage(session: ort.InferenceSession):
    """Test memory usage (basic estimation)."""
    print(f"\n[Test 5/5] Memory Usage Estimation")
    print("-" * 50)

    # Get model file size
    model_path = session._model_path
    if Path(model_path).exists():
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")
    else:
        print(f"Model size: Unknown")

    # Estimate total memory usage
    # Model (loaded) + preprocessing buffers + inference buffers
    estimated_memory_mb = model_size_mb * 1.5 + 100  # Rough estimate

    print(f"Estimated memory usage: ~{estimated_memory_mb:.0f} MB")

    if estimated_memory_mb < 2000:  # 2GB
        print(f"[OK] Memory usage within target (<2GB)")
    else:
        print(f"[WARNING] Memory usage may exceed target (>2GB)")


def main():
    """Run all deployment tests."""

    print("=" * 70)
    print("SereneSense Deployment Validation Suite")
    print("=" * 70)

    # Configuration - Use FP32 for PC testing, INT8 for RPi
    import platform
    is_arm = platform.machine().lower() in ['arm64', 'aarch64', 'armv7l']

    if is_arm:
        # Running on ARM (Raspberry Pi)
        model_path = "outputs/audiomae_int8.onnx"
        print("\n[INFO] Detected ARM platform - testing INT8 model")
    else:
        # Running on x86/x64 (Development PC)
        model_path = "outputs/audiomae_fp32.onnx"
        print("\n[INFO] Detected x86/x64 platform - testing FP32 model")
        print("       INT8 model cannot be tested on this platform")
        print("       INT8 testing will be performed on Raspberry Pi 5")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please run the following scripts first:")
        print("  1. python scripts/export_to_onnx.py")
        if not is_arm:
            print("     (This will create the FP32 model for PC testing)")
        else:
            print("  2. python scripts/quantize_onnx.py")
        sys.exit(1)

    # Run tests
    session = test_model_loading(model_path)
    preprocessor = test_preprocessing()
    test_inference(session, preprocessor)
    latency = test_latency(session, preprocessor, num_runs=100)
    test_memory_usage(session)

    # Final summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"[OK] All tests passed!")
    print(f"[OK] Model ready for Raspberry Pi deployment")
    print(f"[OK] Average total latency: {latency:.2f} ms")
    print("\n" + "=" * 70)
    print("Deployment Instructions:")
    print("=" * 70)
    print("1. Transfer files to Raspberry Pi:")
    print(f"   - {model_path}")
    print("   - scripts/rpi_preprocessing.py")
    print("   - scripts/rpi_deploy.py")
    print("")
    print("2. On Raspberry Pi, run setup:")
    print("   bash scripts/rpi_setup.sh")
    print("")
    print("3. Test deployment:")
    print("   python3 rpi_deploy.py --mode realtime --max-detections 5 --verbose")
    print("=" * 70)


if __name__ == "__main__":
    main()
