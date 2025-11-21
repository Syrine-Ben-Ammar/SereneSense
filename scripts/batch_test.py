"""
Batch testing script for validating model deployment.

This script:
1. Tests the model on multiple audio files
2. Generates performance reports
3. Validates accuracy against ground truth labels
4. Provides detailed metrics

Author: SereneSense Team
Date: 2025-11-21
"""

import numpy as np
import onnxruntime as ort
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from rpi_preprocessing import AudioPreprocessor


class BatchTester:
    """Batch testing for deployed models."""

    CLASS_LABELS = [
        "Helicopter",
        "Fighter Aircraft",
        "Military Vehicle",
        "Truck",
        "Footsteps",
        "Speech",
        "Background"
    ]

    def __init__(self, model_path: str):
        """
        Initialize batch tester.

        Args:
            model_path: Path to ONNX model
        """
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.preprocessor = AudioPreprocessor()

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def predict_file(self, audio_path: str) -> Tuple[int, float, np.ndarray, float]:
        """
        Run prediction on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (predicted_class, confidence, probabilities, inference_time)
        """
        # Preprocess
        start_time = time.time()
        spectrogram = self.preprocessor.preprocess(audio_path, input_type='file')
        preprocess_time = time.time() - start_time

        # Inference
        start_time = time.time()
        logits = self.session.run([self.session.get_outputs()[0].name], {self.input_name: spectrogram})[0]
        inference_time = time.time() - start_time

        # Post-process
        probabilities = self.softmax(logits[0])
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return predicted_class, confidence, probabilities, inference_time

    def test_files(
        self,
        audio_files: List[str],
        labels: List[int] = None
    ) -> Dict:
        """
        Test on multiple audio files.

        Args:
            audio_files: List of audio file paths
            labels: Optional ground truth labels

        Returns:
            Dictionary with results and metrics
        """
        results = {
            'files': [],
            'predictions': [],
            'confidences': [],
            'inference_times': [],
            'correct': 0,
            'total': 0
        }

        print(f"\nTesting {len(audio_files)} files...")
        print("-" * 70)

        for i, audio_path in enumerate(audio_files):
            try:
                # Predict
                pred_class, confidence, probabilities, inf_time = self.predict_file(audio_path)

                # Store results
                results['files'].append(str(audio_path))
                results['predictions'].append(pred_class)
                results['confidences'].append(confidence)
                results['inference_times'].append(inf_time)

                # Check accuracy if labels provided
                if labels is not None:
                    true_label = labels[i]
                    is_correct = (pred_class == true_label)
                    results['correct'] += int(is_correct)
                    marker = "✓" if is_correct else "✗"
                else:
                    true_label = None
                    marker = " "

                results['total'] += 1

                # Print result
                pred_name = self.CLASS_LABELS[pred_class]
                true_name = self.CLASS_LABELS[true_label] if true_label is not None else "Unknown"

                print(f"{marker} File {i+1}/{len(audio_files)}: {Path(audio_path).name}")
                print(f"    Predicted: {pred_name} ({confidence:.2%})")
                if labels is not None:
                    print(f"    True Label: {true_name}")
                print(f"    Inference: {inf_time*1000:.1f} ms")

            except Exception as e:
                print(f"✗ Error processing {audio_path}: {e}")

        print("-" * 70)

        # Calculate metrics
        if labels is not None:
            accuracy = results['correct'] / results['total'] * 100 if results['total'] > 0 else 0
            results['accuracy'] = accuracy
            print(f"\nAccuracy: {results['correct']}/{results['total']} ({accuracy:.2f}%)")

        avg_inference = np.mean(results['inference_times']) * 1000
        std_inference = np.std(results['inference_times']) * 1000
        print(f"Average Inference: {avg_inference:.2f} ± {std_inference:.2f} ms")

        avg_confidence = np.mean(results['confidences']) * 100
        print(f"Average Confidence: {avg_confidence:.2f}%")

        return results

    def generate_confusion_matrix(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            predictions: Predicted classes
            labels: True labels

        Returns:
            Confusion matrix (num_classes x num_classes)
        """
        num_classes = len(self.CLASS_LABELS)
        confusion = np.zeros((num_classes, num_classes), dtype=int)

        for pred, true in zip(predictions, labels):
            confusion[true, pred] += 1

        return confusion

    def print_confusion_matrix(self, confusion: np.ndarray):
        """Print confusion matrix in readable format."""
        print("\nConfusion Matrix:")
        print("=" * 70)

        # Header
        print("True \\ Pred", end="")
        for i in range(len(self.CLASS_LABELS)):
            print(f"\t{i}", end="")
        print()

        # Rows
        for i in range(len(self.CLASS_LABELS)):
            print(f"{i}: {self.CLASS_LABELS[i][:12]:<12}", end="")
            for j in range(len(self.CLASS_LABELS)):
                print(f"\t{confusion[i, j]}", end="")
            print()

        print("=" * 70)

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def load_file_list(file_list_path: str) -> Tuple[List[str], List[int]]:
    """
    Load file list with labels from text file.

    Format: file_path,label
    Example: audio/helicopter_001.wav,0

    Args:
        file_list_path: Path to file list

    Returns:
        Tuple of (audio_files, labels)
    """
    audio_files = []
    labels = []

    with open(file_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) == 2:
                audio_path, label = parts
                audio_files.append(audio_path.strip())
                labels.append(int(label.strip()))
            else:
                audio_files.append(parts[0].strip())

    return audio_files, labels if labels else None


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Batch testing for deployed AudioMAE model"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/audiomae_int8.onnx',
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--file-list',
        type=str,
        help='Path to file list (format: file_path,label per line)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Audio files to test'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.wav',
        help='File pattern for directory mode (default: *.wav)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='batch_results.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return

    # Get file list
    audio_files = []
    labels = None

    if args.file_list:
        audio_files, labels = load_file_list(args.file_list)
    elif args.files:
        audio_files = args.files
    elif args.directory:
        directory = Path(args.directory)
        audio_files = list(directory.glob(args.pattern))
        audio_files = [str(f) for f in audio_files]
    else:
        print("Error: Must specify --file-list, --files, or --directory")
        return

    if not audio_files:
        print("Error: No audio files found")
        return

    # Create tester
    tester = BatchTester(args.model)

    # Run tests
    print("=" * 70)
    print("Batch Testing")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Files: {len(audio_files)}")
    print(f"Labels: {'Yes' if labels else 'No'}")

    results = tester.test_files(audio_files, labels)

    # Generate confusion matrix if labels available
    if labels is not None:
        confusion = tester.generate_confusion_matrix(results['predictions'], labels)
        tester.print_confusion_matrix(confusion)

    # Save results
    tester.save_results(results, args.output)

    print("\n" + "=" * 70)
    print("Batch Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
