#
# Plan:
# 1. Create comprehensive metrics calculation system for audio classification
# 2. Implement TrainingMetrics and ValidationMetrics classes
# 3. Add AudioClassificationMetrics for military vehicle detection
# 4. Create specific metric calculators (Accuracy, F1, AUC, TopK)
# 5. Implement audio-specific metrics (Spectral, Perceptual, Diversity)
# 6. Add confusion matrix and classification report generation
# 7. Create metric aggregation and statistical analysis tools
# 8. Support for multi-class and multi-label classification metrics
#

"""
Training Metrics for SereneSense

This module implements comprehensive metrics calculation and analysis
for training and evaluating SereneSense audio classification models.

Key Features:
- Standard classification metrics (accuracy, precision, recall, F1)
- Audio-specific metrics (spectral quality, perceptual measures)
- Confusion matrix and classification reports
- Top-K accuracy and ranking metrics
- Statistical significance testing
- Metric aggregation and visualization
- Class-wise and overall performance analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
import warnings

logger = logging.getLogger(__name__)


class MetricCalculator:
    """
    Base metric calculator for audio classification tasks.

    Provides comprehensive metrics calculation including standard
    classification metrics and audio-specific measures.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        average: str = "weighted",
        task_type: str = "multiclass",
    ):
        """
        Initialize metric calculator.

        Args:
            num_classes: Number of classes
            class_names: Names of classes for reporting
            average: Averaging strategy for multi-class metrics
            task_type: Type of task ('multiclass', 'multilabel', 'binary')
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.average = average
        self.task_type = task_type

        # Metric history
        self.reset_metrics()

    def reset_metrics(self):
        """Reset accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
    ):
        """
        Update metrics with new batch.

        Args:
            predictions: Model predictions [B] or [B, C]
            targets: Ground truth targets [B]
            probabilities: Class probabilities [B, C] (optional)
            loss: Batch loss (optional)
        """
        # Convert to numpy for sklearn compatibility
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()

        # Store for later computation
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

        if probabilities is not None:
            self.probabilities.extend(probabilities)

        if loss is not None:
            self.losses.append(loss)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.

        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions or not self.targets:
            return {}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {}

        # Basic classification metrics
        try:
            metrics["accuracy"] = accuracy_score(targets, predictions)
            metrics["precision"] = precision_score(
                targets, predictions, average=self.average, zero_division=0
            )
            metrics["recall"] = recall_score(
                targets, predictions, average=self.average, zero_division=0
            )
            metrics["f1_score"] = f1_score(
                targets, predictions, average=self.average, zero_division=0
            )
            metrics["matthews_corrcoef"] = matthews_corrcoef(targets, predictions)
        except Exception as e:
            logger.warning(f"Error computing basic metrics: {e}")

        # Per-class metrics
        try:
            precision_per_class = precision_score(
                targets, predictions, average=None, zero_division=0
            )
            recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
            f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)

            for i, class_name in enumerate(self.class_names):
                if i < len(precision_per_class):
                    metrics[f"precision_{class_name}"] = precision_per_class[i]
                    metrics[f"recall_{class_name}"] = recall_per_class[i]
                    metrics[f"f1_{class_name}"] = f1_per_class[i]
        except Exception as e:
            logger.warning(f"Error computing per-class metrics: {e}")

        # Probability-based metrics
        if self.probabilities:
            try:
                probabilities = np.array(self.probabilities)

                if self.task_type == "multiclass" and self.num_classes > 2:
                    # Multi-class AUC (one-vs-rest)
                    targets_binarized = label_binarize(targets, classes=range(self.num_classes))
                    if targets_binarized.shape[1] > 1:  # Ensure we have multiple classes
                        metrics["auc_ovr"] = roc_auc_score(
                            targets_binarized,
                            probabilities,
                            average=self.average,
                            multi_class="ovr",
                        )
                        metrics["auc_ovo"] = roc_auc_score(
                            targets_binarized,
                            probabilities,
                            average=self.average,
                            multi_class="ovo",
                        )
                elif self.task_type == "binary" or self.num_classes == 2:
                    # Binary AUC
                    if probabilities.shape[1] == 2:
                        metrics["auc"] = roc_auc_score(targets, probabilities[:, 1])
                    else:
                        metrics["auc"] = roc_auc_score(targets, probabilities)

                # Average precision
                if self.task_type == "multiclass":
                    targets_binarized = label_binarize(targets, classes=range(self.num_classes))
                    if targets_binarized.shape[1] > 1:
                        metrics["avg_precision"] = average_precision_score(
                            targets_binarized, probabilities, average=self.average
                        )

            except Exception as e:
                logger.warning(f"Error computing probability-based metrics: {e}")

        # Top-K accuracy
        if self.probabilities:
            try:
                probabilities = np.array(self.probabilities)
                for k in [3, 5]:
                    if k <= self.num_classes:
                        top_k_acc = self._compute_top_k_accuracy(targets, probabilities, k)
                        metrics[f"top_{k}_accuracy"] = top_k_acc
            except Exception as e:
                logger.warning(f"Error computing top-k accuracy: {e}")

        # Loss metrics
        if self.losses:
            metrics["avg_loss"] = np.mean(self.losses)
            metrics["loss_std"] = np.std(self.losses)

        return metrics

    def _compute_top_k_accuracy(
        self, targets: np.ndarray, probabilities: np.ndarray, k: int
    ) -> float:
        """Compute top-K accuracy"""
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
        return correct / len(targets)

    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix"""
        if not self.predictions or not self.targets:
            return None

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        return confusion_matrix(targets, predictions, labels=range(self.num_classes))

    def get_classification_report(self) -> Optional[str]:
        """Get detailed classification report"""
        if not self.predictions or not self.targets:
            return None

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        return classification_report(
            targets,
            predictions,
            target_names=self.class_names,
            labels=range(self.num_classes),
            zero_division=0,
        )


class TrainingMetrics:
    """
    Specialized metrics calculator for training phase.

    Tracks training-specific metrics including loss curves,
    learning rates, and optimization statistics.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize training metrics.

        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.base_calculator = MetricCalculator(num_classes, class_names)
        self.batch_times = []
        self.learning_rates = []
        self.gradient_norms = []

    def update_batch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        batch_time: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
    ):
        """Update training metrics for a batch"""
        self.base_calculator.update(predictions, targets, loss=loss)
        self.batch_times.append(batch_time)
        self.learning_rates.append(learning_rate)

        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute training metrics"""
        metrics = self.base_calculator.compute_metrics()

        # Add training-specific metrics
        if self.batch_times:
            metrics["avg_batch_time"] = np.mean(self.batch_times)
            metrics["total_time"] = np.sum(self.batch_times)

        if self.learning_rates:
            metrics["learning_rate"] = self.learning_rates[-1]  # Latest LR
            metrics["avg_learning_rate"] = np.mean(self.learning_rates)

        if self.gradient_norms:
            metrics["avg_gradient_norm"] = np.mean(self.gradient_norms)
            metrics["max_gradient_norm"] = np.max(self.gradient_norms)

        return metrics

    def reset(self):
        """Reset all metrics"""
        self.base_calculator.reset_metrics()
        self.batch_times.clear()
        self.learning_rates.clear()
        self.gradient_norms.clear()


class ValidationMetrics:
    """
    Specialized metrics calculator for validation phase.

    Focuses on generalization metrics and model performance
    on unseen data.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize validation metrics.

        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.base_calculator = MetricCalculator(num_classes, class_names)
        self.confidence_scores = []

    def update_batch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
        loss: float,
    ):
        """Update validation metrics for a batch"""
        self.base_calculator.update(predictions, targets, probabilities, loss)

        # Compute confidence scores
        max_probs = torch.max(probabilities, dim=1)[0]
        self.confidence_scores.extend(max_probs.detach().cpu().numpy())

    def compute_metrics(self) -> Dict[str, float]:
        """Compute validation metrics"""
        metrics = self.base_calculator.compute_metrics()

        # Add validation-specific metrics
        if self.confidence_scores:
            metrics["avg_confidence"] = np.mean(self.confidence_scores)
            metrics["confidence_std"] = np.std(self.confidence_scores)

            # Calibration metrics
            predictions = np.array(self.base_calculator.predictions)
            targets = np.array(self.base_calculator.targets)

            # Expected Calibration Error (ECE)
            ece = self._compute_expected_calibration_error(
                predictions, targets, self.confidence_scores
            )
            metrics["expected_calibration_error"] = ece

        return metrics

    def _compute_expected_calibration_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def reset(self):
        """Reset all metrics"""
        self.base_calculator.reset_metrics()
        self.confidence_scores.clear()


class AudioClassificationMetrics:
    """
    Specialized metrics for audio classification tasks.

    Includes audio-specific metrics and analysis tailored
    for military vehicle sound detection.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize audio classification metrics.

        Args:
            num_classes: Number of classes
            class_names: Names of classes
            class_weights: Weights for class imbalance handling
        """
        self.base_calculator = MetricCalculator(num_classes, class_names)
        self.class_weights = class_weights or {}

        # Audio-specific metrics
        self.prediction_entropy = []
        self.class_distribution = defaultdict(int)

    def update_batch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
        audio_features: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Update audio classification metrics"""
        self.base_calculator.update(predictions, targets, probabilities)

        # Compute prediction entropy
        entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        self.prediction_entropy.extend(entropies.detach().cpu().numpy())

        # Update class distribution
        for pred in predictions.detach().cpu().numpy():
            self.class_distribution[int(pred)] += 1

    def compute_metrics(self) -> Dict[str, float]:
        """Compute audio classification metrics"""
        metrics = self.base_calculator.compute_metrics()

        # Add audio-specific metrics
        if self.prediction_entropy:
            metrics["avg_prediction_entropy"] = np.mean(self.prediction_entropy)
            metrics["prediction_entropy_std"] = np.std(self.prediction_entropy)

        # Class distribution analysis
        total_predictions = sum(self.class_distribution.values())
        if total_predictions > 0:
            for class_idx, count in self.class_distribution.items():
                class_name = self.base_calculator.class_names[class_idx]
                metrics[f"pred_ratio_{class_name}"] = count / total_predictions

        # Weighted metrics for class imbalance
        if self.class_weights:
            weighted_acc = self._compute_weighted_accuracy()
            if weighted_acc is not None:
                metrics["weighted_accuracy"] = weighted_acc

        return metrics

    def _compute_weighted_accuracy(self) -> Optional[float]:
        """Compute weighted accuracy for class imbalance"""
        if not self.base_calculator.predictions or not self.base_calculator.targets:
            return None

        predictions = np.array(self.base_calculator.predictions)
        targets = np.array(self.base_calculator.targets)

        weighted_correct = 0
        total_weight = 0

        for target, pred in zip(targets, predictions):
            weight = self.class_weights.get(target, 1.0)
            if target == pred:
                weighted_correct += weight
            total_weight += weight

        return weighted_correct / total_weight if total_weight > 0 else 0.0

    def get_class_performance_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get detailed per-class performance analysis"""
        cm = self.base_calculator.get_confusion_matrix()
        if cm is None:
            return {}

        analysis = {}

        for i, class_name in enumerate(self.base_calculator.class_names):
            if i < cm.shape[0]:
                # True positives, false positives, false negatives
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn

                # Compute metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                analysis[class_name] = {
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "f1_score": f1,
                    "support": tp + fn,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn,
                }

        return analysis

    def reset(self):
        """Reset all metrics"""
        self.base_calculator.reset_metrics()
        self.prediction_entropy.clear()
        self.class_distribution.clear()


class AccuracyCalculator:
    """Specialized accuracy calculator with various accuracy types"""

    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute standard accuracy"""
        correct = (predictions == targets).float()
        return correct.mean().item()

    @staticmethod
    def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Compute top-K accuracy"""
        _, top_k_preds = torch.topk(logits, k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1).float()
        return correct.mean().item()

    @staticmethod
    def compute_balanced_accuracy(
        predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> float:
        """Compute balanced accuracy (average of per-class accuracies)"""
        class_accuracies = []

        for class_idx in range(num_classes):
            class_mask = targets == class_idx
            if class_mask.any():
                class_predictions = predictions[class_mask]
                class_targets = targets[class_mask]
                class_acc = (class_predictions == class_targets).float().mean().item()
                class_accuracies.append(class_acc)

        return np.mean(class_accuracies) if class_accuracies else 0.0


class F1ScoreCalculator:
    """Specialized F1 score calculator with macro/micro/weighted variants"""

    @staticmethod
    def compute_f1_score(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        average: str = "weighted",
    ) -> float:
        """Compute F1 score with specified averaging"""
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        return f1_score(targets_np, predictions_np, average=average, zero_division=0)

    @staticmethod
    def compute_per_class_f1(
        predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> Dict[int, float]:
        """Compute per-class F1 scores"""
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        f1_scores = f1_score(targets_np, predictions_np, average=None, zero_division=0)

        return {i: f1_scores[i] for i in range(min(len(f1_scores), num_classes))}


class AUCCalculator:
    """Specialized AUC calculator for multi-class problems"""

    @staticmethod
    def compute_multiclass_auc(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        average: str = "weighted",
    ) -> float:
        """Compute multi-class AUC using one-vs-rest approach"""
        probabilities_np = probabilities.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Binarize targets
        targets_binarized = label_binarize(targets_np, classes=range(num_classes))

        if targets_binarized.shape[1] == 1:
            # Binary case
            return roc_auc_score(targets_binarized.ravel(), probabilities_np[:, 1])
        else:
            # Multi-class case
            return roc_auc_score(
                targets_binarized, probabilities_np, average=average, multi_class="ovr"
            )

    @staticmethod
    def compute_per_class_auc(
        probabilities: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> Dict[int, float]:
        """Compute per-class AUC scores"""
        probabilities_np = probabilities.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Binarize targets
        targets_binarized = label_binarize(targets_np, classes=range(num_classes))

        per_class_auc = {}
        for i in range(num_classes):
            if targets_binarized.shape[1] > 1:
                try:
                    auc = roc_auc_score(targets_binarized[:, i], probabilities_np[:, i])
                    per_class_auc[i] = auc
                except ValueError:
                    # Handle case where class is not present in targets
                    per_class_auc[i] = 0.0

        return per_class_auc


class TopKAccuracyCalculator:
    """Specialized top-K accuracy calculator"""

    @staticmethod
    def compute_top_k_accuracies(
        logits: torch.Tensor, targets: torch.Tensor, k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """Compute top-K accuracies for multiple K values"""
        results = {}

        for k in k_values:
            if k <= logits.size(1):  # Ensure K doesn't exceed number of classes
                acc = AccuracyCalculator.compute_top_k_accuracy(logits, targets, k)
                results[f"top_{k}_accuracy"] = acc

        return results


class SpectralMetrics:
    """Metrics for evaluating spectral quality of audio predictions"""

    @staticmethod
    def compute_spectral_convergence(
        pred_spectrogram: torch.Tensor, target_spectrogram: torch.Tensor
    ) -> float:
        """Compute spectral convergence between predicted and target spectrograms"""
        diff = pred_spectrogram - target_spectrogram
        convergence = torch.norm(diff) / torch.norm(target_spectrogram)
        return convergence.item()

    @staticmethod
    def compute_spectral_distortion(
        pred_spectrogram: torch.Tensor, target_spectrogram: torch.Tensor
    ) -> float:
        """Compute spectral distortion metric"""
        # Log magnitude spectrograms
        pred_log = torch.log(pred_spectrogram + 1e-8)
        target_log = torch.log(target_spectrogram + 1e-8)

        # Mean squared error in log domain
        mse = F.mse_loss(pred_log, target_log)
        return mse.item()


class PerceptualMetrics:
    """Perceptual quality metrics for audio evaluation"""

    @staticmethod
    def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
        """Compute Signal-to-Noise Ratio"""
        signal_power = torch.mean(signal**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr.item()

    @staticmethod
    def compute_pesq_like_metric(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> float:
        """Compute PESQ-like perceptual metric (simplified version)"""
        # This is a simplified version - real PESQ requires specialized libraries
        mse = F.mse_loss(pred_audio, target_audio)
        pesq_like = -10 * torch.log10(mse + 1e-8)
        return pesq_like.item()


class DiversityMetrics:
    """Metrics for evaluating prediction diversity and model confidence"""

    @staticmethod
    def compute_prediction_entropy(probabilities: torch.Tensor) -> float:
        """Compute average prediction entropy"""
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        return entropy.mean().item()

    @staticmethod
    def compute_confidence_histogram(
        probabilities: torch.Tensor, num_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Compute confidence histogram for calibration analysis"""
        max_probs = torch.max(probabilities, dim=1)[0].detach().cpu().numpy()

        hist, bin_edges = np.histogram(max_probs, bins=num_bins, range=(0, 1))

        return {
            "histogram": hist,
            "bin_edges": bin_edges,
            "mean_confidence": np.mean(max_probs),
            "std_confidence": np.std(max_probs),
        }

    @staticmethod
    def compute_class_distribution_divergence(
        pred_distribution: torch.Tensor, target_distribution: torch.Tensor
    ) -> float:
        """Compute KL divergence between predicted and target class distributions"""
        # Add small epsilon for numerical stability
        pred_dist = pred_distribution + 1e-8
        target_dist = target_distribution + 1e-8

        # Normalize
        pred_dist = pred_dist / pred_dist.sum()
        target_dist = target_dist / target_dist.sum()

        # Compute KL divergence
        kl_div = torch.sum(target_dist * torch.log(target_dist / pred_dist))
        return kl_div.item()


class MetricAggregator:
    """
    Utility class for aggregating metrics across multiple batches or epochs.
    """

    def __init__(self):
        self.metric_history = defaultdict(list)
        self.current_metrics = {}

    def update(self, metrics: Dict[str, float]):
        """Update with new metrics"""
        self.current_metrics = metrics.copy()
        for key, value in metrics.items():
            self.metric_history[key].append(value)

    def get_running_average(self, window_size: int = 10) -> Dict[str, float]:
        """Get running average of metrics over specified window"""
        running_avg = {}
        for key, values in self.metric_history.items():
            if values:
                window_values = values[-window_size:] if len(values) >= window_size else values
                running_avg[key] = np.mean(window_values)
        return running_avg

    def get_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics for all metrics"""
        stats = {}
        for key, values in self.metric_history.items():
            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                    "trend": (
                        np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0
                    ),
                }
        return stats

    def get_best_metrics(self, higher_is_better: List[str] = None) -> Dict[str, Tuple[float, int]]:
        """Get best values and epochs for each metric"""
        higher_is_better = higher_is_better or [
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "auc",
        ]

        best_metrics = {}
        for key, values in self.metric_history.items():
            if values:
                if any(metric in key.lower() for metric in higher_is_better):
                    best_value = np.max(values)
                    best_epoch = np.argmax(values)
                else:
                    best_value = np.min(values)
                    best_epoch = np.argmin(values)

                best_metrics[key] = (best_value, best_epoch)

        return best_metrics

    def reset(self):
        """Reset all accumulated metrics"""
        self.metric_history.clear()
        self.current_metrics.clear()
