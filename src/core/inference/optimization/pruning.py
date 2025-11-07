#
# Plan:
# 1. Create comprehensive pruning system for model compression
# 2. Support structured and unstructured pruning methods
# 3. Magnitude-based, gradient-based, and importance-based pruning
# 4. Progressive pruning with fine-tuning
# 5. Accuracy-aware pruning with threshold monitoring
# 6. Channel pruning and filter pruning for structured compression
# 7. Performance benchmarking and validation
#

"""
Model Pruning for Military Vehicle Detection
Achieves significant model compression while preserving accuracy.

Features:
- Structured and unstructured pruning
- Magnitude, gradient, and importance-based methods
- Progressive pruning with fine-tuning
- Channel and filter pruning
- Accuracy-aware pruning
- Edge device optimization
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import time
import json
from collections import defaultdict, OrderedDict
import copy

from core.utils.config_parser import ConfigParser
from core.data.loaders.mad_loader import MADDataset
from core.core.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Pruning configuration"""

    # Pruning method
    method: str = "magnitude"  # magnitude, gradient, random, fisher
    structure: str = "unstructured"  # unstructured, structured, channel

    # Pruning ratios
    target_sparsity: float = 0.5  # Target sparsity (50% of weights pruned)
    layer_wise_ratios: Dict[str, float] = None  # Per-layer pruning ratios

    # Progressive pruning
    progressive: bool = True
    num_pruning_steps: int = 10
    pruning_schedule: str = "polynomial"  # linear, polynomial, exponential

    # Fine-tuning settings
    finetune_epochs: int = 5
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 32

    # Accuracy constraints
    accuracy_threshold: float = 0.95  # Minimum accuracy to maintain
    early_stopping: bool = True
    patience: int = 3

    # Structured pruning settings
    channel_pruning_ratio: float = 0.3
    filter_importance_metric: str = "l1_norm"  # l1_norm, l2_norm, geometric_median

    # Dataset settings
    validation_dataset_path: Optional[str] = None
    num_validation_samples: int = 1000

    # Output settings
    output_path: str = "model_pruned.pth"
    save_masks: bool = True

    # Performance settings
    benchmark_enabled: bool = True


class PruningOptimizer:
    """
    Model pruning optimizer for military vehicle detection.
    Supports various pruning strategies for edge deployment optimization.
    """

    def __init__(self, config: PruningConfig):
        """
        Initialize pruning optimizer.

        Args:
            config: Pruning configuration
        """
        self.config = config

        # Initialize audio processor for validation
        self.audio_processor = AudioProcessor(
            {
                "sample_rate": 16000,
                "n_mels": 128,
                "n_fft": 1024,
                "hop_length": 512,
                "win_length": 1024,
                "normalize": True,
            }
        )

        # Validation dataset
        self.validation_loader = None
        if config.validation_dataset_path:
            self._setup_validation_dataset()

        logger.info(f"Pruning optimizer initialized:")
        logger.info(f"  Method: {config.method}")
        logger.info(f"  Structure: {config.structure}")
        logger.info(f"  Target sparsity: {config.target_sparsity}")
        logger.info(f"  Progressive: {config.progressive}")

    def _setup_validation_dataset(self):
        """Setup validation dataset for accuracy monitoring"""
        try:
            dataset = MADDataset(
                data_dir=self.config.validation_dataset_path, split="val", transform=None
            )

            # Subsample for faster validation
            indices = np.random.choice(
                len(dataset), min(self.config.num_validation_samples, len(dataset)), replace=False
            )

            subset = torch.utils.data.Subset(dataset, indices)
            self.validation_loader = torch.utils.data.DataLoader(
                subset, batch_size=self.config.finetune_batch_size, shuffle=False, num_workers=0
            )

            logger.info(f"Validation dataset loaded: {len(subset)} samples")

        except Exception as e:
            logger.warning(f"Failed to load validation dataset: {e}")
            self.validation_loader = None

    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Prune model using specified method and configuration.

        Args:
            model: PyTorch model to prune

        Returns:
            Pruned model
        """
        logger.info("Starting model pruning...")

        # Create a copy of the model for pruning
        pruned_model = copy.deepcopy(model)

        if self.config.progressive:
            pruned_model = self._progressive_pruning(pruned_model)
        else:
            pruned_model = self._one_shot_pruning(pruned_model)

        # Final validation
        if self.validation_loader:
            final_accuracy = self._evaluate_model(pruned_model)
            logger.info(f"Final pruned model accuracy: {final_accuracy:.3f}")

        # Benchmark performance
        if self.config.benchmark_enabled:
            self._benchmark_performance(model, pruned_model)

        # Save pruned model
        self._save_pruned_model(pruned_model)

        # Print pruning statistics
        self._print_pruning_stats(pruned_model)

        logger.info("Model pruning completed successfully")
        return pruned_model

    def _progressive_pruning(self, model: nn.Module) -> nn.Module:
        """
        Apply progressive pruning with fine-tuning.

        Args:
            model: Model to prune

        Returns:
            Progressively pruned model
        """
        logger.info("Applying progressive pruning...")

        # Calculate pruning schedule
        sparsity_schedule = self._get_pruning_schedule()

        best_accuracy = 0.0
        patience_counter = 0

        for step, target_sparsity in enumerate(sparsity_schedule):
            logger.info(
                f"Pruning step {step + 1}/{len(sparsity_schedule)}, "
                f"target sparsity: {target_sparsity:.3f}"
            )

            # Apply pruning for this step
            if self.config.structure == "unstructured":
                self._apply_unstructured_pruning(model, target_sparsity)
            elif self.config.structure == "structured":
                self._apply_structured_pruning(model, target_sparsity)
            elif self.config.structure == "channel":
                self._apply_channel_pruning(model, target_sparsity)

            # Fine-tune after pruning
            if self.validation_loader:
                model = self._fine_tune_model(model)

                # Check accuracy
                accuracy = self._evaluate_model(model)
                logger.info(f"  Accuracy after fine-tuning: {accuracy:.3f}")

                # Early stopping based on accuracy
                if accuracy < self.config.accuracy_threshold:
                    if self.config.early_stopping:
                        logger.warning(f"Accuracy below threshold, stopping at step {step + 1}")
                        break

                # Track best accuracy for patience
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                    if patience_counter >= self.config.patience:
                        logger.info(
                            f"Early stopping due to no improvement for {self.config.patience} steps"
                        )
                        break

        return model

    def _one_shot_pruning(self, model: nn.Module) -> nn.Module:
        """
        Apply one-shot pruning to target sparsity.

        Args:
            model: Model to prune

        Returns:
            Pruned model
        """
        logger.info("Applying one-shot pruning...")

        if self.config.structure == "unstructured":
            self._apply_unstructured_pruning(model, self.config.target_sparsity)
        elif self.config.structure == "structured":
            self._apply_structured_pruning(model, self.config.target_sparsity)
        elif self.config.structure == "channel":
            self._apply_channel_pruning(model, self.config.target_sparsity)

        # Fine-tune after pruning
        if self.validation_loader:
            model = self._fine_tune_model(model)

        return model

    def _get_pruning_schedule(self) -> List[float]:
        """
        Generate pruning schedule for progressive pruning.

        Returns:
            List of sparsity values for each step
        """
        if self.config.pruning_schedule == "linear":
            schedule = np.linspace(
                0, self.config.target_sparsity, self.config.num_pruning_steps + 1
            )[1:]

        elif self.config.pruning_schedule == "polynomial":
            # Polynomial schedule (cubic by default)
            steps = np.arange(1, self.config.num_pruning_steps + 1)
            schedule = self.config.target_sparsity * (steps / self.config.num_pruning_steps) ** 3

        elif self.config.pruning_schedule == "exponential":
            # Exponential schedule
            steps = np.arange(1, self.config.num_pruning_steps + 1)
            schedule = self.config.target_sparsity * (
                1 - np.exp(-3 * steps / self.config.num_pruning_steps)
            )

        else:
            # Default to linear
            schedule = np.linspace(
                0, self.config.target_sparsity, self.config.num_pruning_steps + 1
            )[1:]

        return schedule.tolist()

    def _apply_unstructured_pruning(self, model: nn.Module, sparsity: float):
        """
        Apply unstructured pruning to model.

        Args:
            model: Model to prune
            sparsity: Target sparsity level
        """
        # Get modules to prune (typically Conv2d and Linear layers)
        modules_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                modules_to_prune.append((module, "weight"))

        if not modules_to_prune:
            logger.warning("No modules found for pruning")
            return

        # Apply pruning based on method
        if self.config.method == "magnitude":
            prune.global_unstructured(
                modules_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity
            )

        elif self.config.method == "random":
            prune.global_unstructured(
                modules_to_prune, pruning_method=prune.RandomUnstructured, amount=sparsity
            )

        elif self.config.method == "gradient":
            # Gradient-based pruning (requires gradients)
            # This is a simplified implementation
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(module, param_name, amount=sparsity)

        logger.info(f"Applied unstructured {self.config.method} pruning: {sparsity:.3f} sparsity")

    def _apply_structured_pruning(self, model: nn.Module, sparsity: float):
        """
        Apply structured pruning to model.

        Args:
            model: Model to prune
            sparsity: Target sparsity level
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune filters based on importance
                num_filters = module.out_channels
                num_to_prune = int(num_filters * sparsity)

                if num_to_prune > 0:
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=num_to_prune,
                        n=1,  # L1 norm
                        dim=0,  # Prune along output channel dimension
                    )

        logger.info(f"Applied structured pruning: {sparsity:.3f} filter sparsity")

    def _apply_channel_pruning(self, model: nn.Module, sparsity: float):
        """
        Apply channel pruning to model.

        Args:
            model: Model to prune
            sparsity: Target sparsity level
        """
        # This is a simplified channel pruning implementation
        # In practice, channel pruning requires careful handling of dependencies

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                # Calculate channel importance
                importance_scores = self._calculate_channel_importance(module)

                # Determine channels to prune
                num_channels = module.out_channels
                num_to_prune = int(num_channels * sparsity)

                if num_to_prune > 0:
                    # Get indices of least important channels
                    _, indices_to_prune = torch.topk(importance_scores, num_to_prune, largest=False)

                    # Create mask for channel pruning
                    mask = torch.ones(num_channels, dtype=torch.bool)
                    mask[indices_to_prune] = False

                    # Apply custom channel pruning
                    self._prune_channels(module, mask)

        logger.info(f"Applied channel pruning: {sparsity:.3f} channel sparsity")

    def _calculate_channel_importance(self, module: nn.Conv2d) -> torch.Tensor:
        """
        Calculate importance scores for channels.

        Args:
            module: Convolutional module

        Returns:
            Importance scores for each output channel
        """
        weight = module.weight.data

        if self.config.filter_importance_metric == "l1_norm":
            # L1 norm of filters
            importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)

        elif self.config.filter_importance_metric == "l2_norm":
            # L2 norm of filters
            importance = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)

        elif self.config.filter_importance_metric == "geometric_median":
            # Geometric median (simplified as mean for efficiency)
            importance = torch.mean(torch.abs(weight.view(weight.size(0), -1)), dim=1)

        else:
            # Default to L1 norm
            importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)

        return importance

    def _prune_channels(self, module: nn.Conv2d, channel_mask: torch.Tensor):
        """
        Prune channels from convolutional module.

        Args:
            module: Convolutional module
            channel_mask: Boolean mask indicating which channels to keep
        """
        # This is a simplified implementation
        # In practice, channel pruning requires modifying the model architecture

        # Zero out pruned channels
        with torch.no_grad():
            module.weight.data[~channel_mask] = 0
            if module.bias is not None:
                module.bias.data[~channel_mask] = 0

    def _fine_tune_model(self, model: nn.Module) -> nn.Module:
        """
        Fine-tune pruned model to recover accuracy.

        Args:
            model: Pruned model

        Returns:
            Fine-tuned model
        """
        if not self.validation_loader:
            logger.warning("No validation dataset available for fine-tuning")
            return model

        logger.info("Fine-tuning pruned model...")

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.finetune_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.finetune_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, labels) in enumerate(self.validation_loader):
                # Process audio to spectrograms
                spectrograms = []
                for audio in data:
                    spec = self.audio_processor.to_spectrogram(audio)
                    spectrograms.append(spec)

                batch_spectrograms = torch.stack(spectrograms)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_spectrograms)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    logger.info(f"  Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / num_batches
            logger.info(f"  Fine-tuning epoch {epoch + 1} completed, Avg Loss: {avg_loss:.4f}")

        model.eval()
        return model

    def _evaluate_model(self, model: nn.Module) -> float:
        """
        Evaluate model accuracy on validation dataset.

        Args:
            model: Model to evaluate

        Returns:
            Accuracy score
        """
        if not self.validation_loader:
            return 0.0

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in self.validation_loader:
                # Process audio to spectrograms
                spectrograms = []
                for audio in data:
                    spec = self.audio_processor.to_spectrogram(audio)
                    spectrograms.append(spec)

                batch_spectrograms = torch.stack(spectrograms)

                # Forward pass
                outputs = model(batch_spectrograms)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def _benchmark_performance(self, original_model: nn.Module, pruned_model: nn.Module):
        """
        Benchmark performance comparison between original and pruned models.

        Args:
            original_model: Original model
            pruned_model: Pruned model
        """
        logger.info("Benchmarking pruned model performance...")

        # Test input
        test_input = torch.randn(1, 1, 128, 128)

        # Benchmark original model
        original_times = []
        original_model.eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = original_model(test_input)

        # Benchmark
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = original_model(test_input)
            end_time = time.time()
            original_times.append(end_time - start_time)

        # Benchmark pruned model
        pruned_times = []
        pruned_model.eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = pruned_model(test_input)

        # Benchmark
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = pruned_model(test_input)
            end_time = time.time()
            pruned_times.append(end_time - start_time)

        # Calculate statistics
        original_avg = np.mean(original_times) * 1000  # ms
        pruned_avg = np.mean(pruned_times) * 1000  # ms
        speedup = original_avg / pruned_avg

        # Model size comparison
        original_size = self._get_model_size(original_model)
        pruned_size = self._get_model_size(pruned_model)
        compression_ratio = original_size / pruned_size

        # Sparsity statistics
        sparsity = self._calculate_sparsity(pruned_model)

        # Log results
        logger.info(f"Pruning Performance Results:")
        logger.info(f"  Original model: {original_avg:.2f}ms")
        logger.info(f"  Pruned model: {pruned_avg:.2f}ms")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Original size: {original_size / 1024 / 1024:.2f}MB")
        logger.info(f"  Pruned size: {pruned_size / 1024 / 1024:.2f}MB")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"  Model sparsity: {sparsity:.3f}")

        # Save benchmark results
        benchmark_data = {
            "original_latency_ms": original_avg,
            "pruned_latency_ms": pruned_avg,
            "speedup": speedup,
            "original_size_mb": original_size / 1024 / 1024,
            "pruned_size_mb": pruned_size / 1024 / 1024,
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
            "method": self.config.method,
            "structure": self.config.structure,
        }

        benchmark_file = self.config.output_path.replace(".pth", "_benchmark.json")
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f, indent=2)

        logger.info(f"Benchmark results saved: {benchmark_file}")

    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """
        Calculate overall sparsity of the model.

        Args:
            model: Model to analyze

        Returns:
            Sparsity ratio (0 to 1)
        """
        total_params = 0
        zero_params = 0

        for name, param in model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0.0
        return sparsity

    def _print_pruning_stats(self, model: nn.Module):
        """Print detailed pruning statistics"""
        logger.info("Pruning Statistics:")

        total_params = 0
        pruned_params = 0

        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module, (nn.Conv2d, nn.Linear)):
                module_params = module.weight.numel()
                module_pruned = (module.weight == 0).sum().item()

                module_sparsity = module_pruned / module_params if module_params > 0 else 0.0

                logger.info(
                    f"  {name}: {module_sparsity:.3f} sparsity "
                    f"({module_pruned}/{module_params} pruned)"
                )

                total_params += module_params
                pruned_params += module_pruned

        overall_sparsity = pruned_params / total_params if total_params > 0 else 0.0
        logger.info(
            f"  Overall: {overall_sparsity:.3f} sparsity "
            f"({pruned_params}/{total_params} pruned)"
        )

    def _save_pruned_model(self, model: nn.Module):
        """Save pruned model to disk"""
        # Save model state
        torch.save(model.state_dict(), self.config.output_path)
        logger.info(f"Pruned model saved: {self.config.output_path}")

        # Save pruning masks if requested
        if self.config.save_masks:
            masks = {}

            for name, module in model.named_modules():
                if hasattr(module, "weight_mask"):
                    masks[f"{name}.weight_mask"] = module.weight_mask
                if hasattr(module, "bias_mask"):
                    masks[f"{name}.bias_mask"] = module.bias_mask

            if masks:
                mask_path = self.config.output_path.replace(".pth", "_masks.pth")
                torch.save(masks, mask_path)
                logger.info(f"Pruning masks saved: {mask_path}")


def create_pruning_optimizer(config_path: str = None) -> PruningOptimizer:
    """
    Create pruning optimizer from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured pruning optimizer
    """
    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        config = PruningConfig(**config_dict.get("pruning", {}))
    else:
        config = PruningConfig()

    return PruningOptimizer(config)


if __name__ == "__main__":
    # Demo: Model pruning
    import argparse

    parser = argparse.ArgumentParser(description="Model Pruning")
    parser.add_argument("--model-path", required=True, help="PyTorch model path")
    parser.add_argument(
        "--method",
        default="magnitude",
        choices=["magnitude", "gradient", "random"],
        help="Pruning method",
    )
    parser.add_argument(
        "--structure",
        default="unstructured",
        choices=["unstructured", "structured", "channel"],
        help="Pruning structure",
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity")
    parser.add_argument("--output", default="model_pruned.pth", help="Output model path")
    parser.add_argument("--validation-data", help="Path to validation dataset")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Load model
        model = torch.load(args.model_path, map_location="cpu")
        model.eval()

        # Create pruning config
        config = PruningConfig(
            method=args.method,
            structure=args.structure,
            target_sparsity=args.sparsity,
            validation_dataset_path=args.validation_data,
            output_path=args.output,
        )

        # Create optimizer and prune
        optimizer = PruningOptimizer(config)
        pruned_model = optimizer.prune_model(model)

        print(f"✅ Model pruning completed: {args.output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Model pruning failed: {e}")
