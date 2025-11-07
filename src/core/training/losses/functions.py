#
# Plan:
# 1. Implement advanced loss functions for audio classification
# 2. Create LabelSmoothingCrossEntropy for regularization
# 3. Add FocalLoss for handling class imbalance
# 4. Implement ArcFaceLoss for better feature learning
# 5. Add audio-specific losses (spectral, perceptual)
# 6. Create MixupLoss and CutMixLoss for augmentation
# 7. Implement KnowledgeDistillationLoss for model compression
# 8. Add CombinedLoss and AdaptiveLossWeighting for multi-objective training
#

"""
Loss Functions for SereneSense Audio Classification

This module implements a comprehensive collection of loss functions
optimized for audio classification tasks, particularly for military
vehicle sound detection.

Key Features:
- Advanced classification losses (Label Smoothing, Focal, ArcFace)
- Audio-specific losses (Spectral Convergence, Perceptual)
- Data augmentation losses (Mixup, CutMix)
- Knowledge distillation support
- Adaptive loss weighting for multi-objective training
- Class imbalance handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss for regularization.

    Prevents overconfident predictions and improves generalization
    by distributing some probability mass to incorrect classes.
    """

    def __init__(self, smoothing: float = 0.1, temperature: float = 1.0):
        """
        Initialize label smoothing cross entropy loss.

        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)
            temperature: Temperature scaling for logits
        """
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.

        Args:
            logits: Model predictions [B, C]
            targets: Ground truth labels [B]

        Returns:
            Label smoothing cross entropy loss
        """
        # Apply temperature scaling
        logits = logits / self.temperature

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        num_classes = logits.size(-1)
        batch_size = targets.size(0)

        # One-hot encoding
        targets_one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)

        # Apply label smoothing
        smooth_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        # Compute loss
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Down-weights easy examples and focuses learning on hard examples,
    particularly useful for military vehicle detection with imbalanced classes.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for rare class (typically inverse of class frequency)
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [B, C]
            targets: Ground truth labels [B]

        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Apply focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_term = self.alpha

        # Compute focal loss
        focal_loss = alpha_term * focal_term * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss for improved feature learning.

    Adds angular margin penalty to improve intra-class compactness
    and inter-class separability for military vehicle classification.
    """

    def __init__(
        self, in_features: int, out_features: int, margin: float = 0.5, scale: float = 64.0
    ):
        """
        Initialize ArcFace loss.

        Args:
            in_features: Input feature dimension
            out_features: Number of classes
            margin: Angular margin penalty
            scale: Feature scale
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            features: Input features [B, D]
            targets: Ground truth labels [B]

        Returns:
            ArcFace loss
        """
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Compute phi (cosine with margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        # Apply margin penalty
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return F.cross_entropy(output, targets)


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss for audio quality assessment.

    Measures the difference between spectrograms in the frequency domain,
    useful for ensuring realistic audio generation and reconstruction.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        """
        Initialize spectral convergence loss.

        Args:
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral convergence loss.

        Args:
            pred_audio: Predicted audio [B, T]
            target_audio: Target audio [B, T]

        Returns:
            Spectral convergence loss
        """
        # Compute spectrograms
        pred_spec = torch.stft(
            pred_audio, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )
        target_spec = torch.stft(
            target_audio, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )

        # Compute magnitudes
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)

        # Compute spectral convergence
        convergence = torch.norm(pred_mag - target_mag, p="fro") / torch.norm(target_mag, p="fro")

        return convergence


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss for comprehensive audio quality assessment.

    Computes spectral losses at multiple time-frequency resolutions
    to capture both fine-grained and coarse-grained audio features.
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
    ):
        """
        Initialize multi-resolution STFT loss.

        Args:
            fft_sizes: List of FFT sizes for different resolutions
            hop_sizes: List of hop sizes for different resolutions
            win_lengths: List of window lengths for different resolutions
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

    def stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        """Compute STFT"""
        return torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),
            return_complex=True,
        )

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred_audio: Predicted audio [B, T]
            target_audio: Target audio [B, T]

        Returns:
            Multi-resolution STFT loss
        """
        total_loss = 0.0

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute spectrograms
            pred_spec = self.stft(pred_audio, fft_size, hop_size, win_length)
            target_spec = self.stft(target_audio, fft_size, hop_size, win_length)

            # Magnitude loss
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            mag_loss = F.l1_loss(pred_mag, target_mag)

            # Spectral convergence loss
            sc_loss = torch.norm(pred_mag - target_mag, p="fro") / torch.norm(target_mag, p="fro")

            total_loss += mag_loss + sc_loss

        return total_loss / len(self.fft_sizes)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss for audio using pre-trained networks.

    Uses features from pre-trained audio networks to compute
    perceptually meaningful losses for audio generation tasks.
    """

    def __init__(self, feature_network: nn.Module, layer_weights: List[float] = None):
        """
        Initialize perceptual loss.

        Args:
            feature_network: Pre-trained network for feature extraction
            layer_weights: Weights for different network layers
        """
        super().__init__()
        self.feature_network = feature_network
        self.feature_network.eval()

        # Freeze feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.layer_weights = layer_weights or [1.0]

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred_audio: Predicted audio
            target_audio: Target audio

        Returns:
            Perceptual loss
        """
        # Extract features
        pred_features = self.extract_features(pred_audio)
        target_features = self.extract_features(target_audio)

        # Compute perceptual loss
        total_loss = 0.0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            weight = self.layer_weights[i] if i < len(self.layer_weights) else 1.0
            loss = F.mse_loss(pred_feat, target_feat)
            total_loss += weight * loss

        return total_loss

    def extract_features(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from audio using pre-trained network"""
        # This would be implemented based on the specific feature network
        # For now, return dummy features
        return [audio]


class MixupLoss(nn.Module):
    """
    Mixup Loss for training with mixed samples.

    Handles loss computation for linearly interpolated samples
    and their corresponding mixed labels.
    """

    def __init__(self, criterion: nn.Module):
        """
        Initialize mixup loss.

        Args:
            criterion: Base loss function to apply to mixed samples
        """
        super().__init__()
        self.criterion = criterion

    def forward(
        self, logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """
        Compute mixup loss.

        Args:
            logits: Model predictions for mixed samples
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixing coefficient

        Returns:
            Mixup loss
        """
        loss_a = self.criterion(logits, targets_a)
        loss_b = self.criterion(logits, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


class CutMixLoss(nn.Module):
    """
    CutMix Loss for training with cut-and-paste augmentation.

    Similar to Mixup but for spatially mixed samples where
    rectangular regions are cut and pasted between samples.
    """

    def __init__(self, criterion: nn.Module):
        """
        Initialize CutMix loss.

        Args:
            criterion: Base loss function to apply to mixed samples
        """
        super().__init__()
        self.criterion = criterion

    def forward(
        self, logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """
        Compute CutMix loss.

        Args:
            logits: Model predictions for mixed samples
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixing coefficient (based on cut area)

        Returns:
            CutMix loss
        """
        loss_a = self.criterion(logits, targets_a)
        loss_b = self.criterion(logits, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for model compression.

    Enables training of smaller student models using knowledge
    from larger teacher models, useful for edge deployment.
    """

    def __init__(
        self, temperature: float = 4.0, alpha: float = 0.7, student_criterion: nn.Module = None
    ):
        """
        Initialize knowledge distillation loss.

        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss vs student loss
            student_criterion: Loss function for student predictions
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.student_criterion = student_criterion or nn.CrossEntropyLoss()

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            targets: Ground truth labels

        Returns:
            Knowledge distillation loss
        """
        # Distillation loss (KL divergence between teacher and student)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            self.temperature**2
        )

        # Student loss (cross entropy with ground truth)
        student_loss = self.student_criterion(student_logits, targets)

        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss for multi-objective training.

    Allows combining multiple loss functions with configurable weights
    for comprehensive training objectives.
    """

    def __init__(self, loss_configs: Dict[str, Dict[str, Any]] = None):
        """
        Initialize combined loss.

        Args:
            loss_configs: Dictionary of loss configurations
                Format: {loss_name: {'type': 'loss_type', 'weight': weight, 'params': {...}}}
        """
        super().__init__()

        # Default configuration for audio classification
        if loss_configs is None:
            loss_configs = {
                "classification": {
                    "type": "label_smoothing_ce",
                    "weight": 1.0,
                    "params": {"smoothing": 0.1},
                },
                "focal": {"type": "focal", "weight": 0.5, "params": {"alpha": 1.0, "gamma": 2.0}},
            }

        self.loss_functions = nn.ModuleDict()
        self.loss_weights = {}

        # Initialize loss functions
        for loss_name, config in loss_configs.items():
            loss_type = config["type"]
            weight = config["weight"]
            params = config.get("params", {})

            if loss_type == "label_smoothing_ce":
                loss_fn = LabelSmoothingCrossEntropy(**params)
            elif loss_type == "focal":
                loss_fn = FocalLoss(**params)
            elif loss_type == "ce":
                loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.loss_functions[loss_name] = loss_fn
            self.loss_weights[loss_name] = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}
        total_loss = 0.0

        for loss_name, loss_fn in self.loss_functions.items():
            loss_value = loss_fn(logits, targets)
            weighted_loss = self.loss_weights[loss_name] * loss_value

            losses[f"{loss_name}_loss"] = loss_value
            total_loss += weighted_loss

        losses["total_loss"] = total_loss
        return losses


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive Loss Weighting for balancing multiple objectives.

    Automatically adjusts loss weights during training based on
    the relative magnitudes and gradients of different loss components.
    """

    def __init__(self, loss_functions: Dict[str, nn.Module], adaptation_rate: float = 0.1):
        """
        Initialize adaptive loss weighting.

        Args:
            loss_functions: Dictionary of loss functions
            adaptation_rate: Rate at which to adapt weights
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.adaptation_rate = adaptation_rate

        # Initialize weights
        self.loss_weights = nn.Parameter(torch.ones(len(loss_functions)) / len(loss_functions))

        # Track loss history for adaptation
        self.loss_history = {name: [] for name in loss_functions.keys()}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute adaptively weighted loss.

        Args:
            logits: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}
        individual_losses = []

        # Compute individual losses
        for i, (loss_name, loss_fn) in enumerate(self.loss_functions.items()):
            loss_value = loss_fn(logits, targets)
            losses[f"{loss_name}_loss"] = loss_value
            individual_losses.append(loss_value)

            # Update loss history
            self.loss_history[loss_name].append(loss_value.item())
            if len(self.loss_history[loss_name]) > 100:  # Keep only recent history
                self.loss_history[loss_name].pop(0)

        # Compute weighted sum
        individual_losses = torch.stack(individual_losses)
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = torch.sum(weights * individual_losses)

        losses["total_loss"] = total_loss
        losses["loss_weights"] = weights

        return losses

    def update_weights(self):
        """Update loss weights based on recent loss history"""
        if not self.training:
            return

        # Compute relative loss magnitudes
        avg_losses = []
        for loss_name in self.loss_functions.keys():
            if self.loss_history[loss_name]:
                avg_loss = np.mean(self.loss_history[loss_name][-10:])  # Recent average
                avg_losses.append(avg_loss)
            else:
                avg_losses.append(1.0)

        # Adapt weights (higher weight for smaller losses to balance training)
        avg_losses = torch.tensor(avg_losses, device=self.loss_weights.device)
        inverse_losses = 1.0 / (avg_losses + 1e-8)
        target_weights = inverse_losses / inverse_losses.sum()

        # Exponential moving average update
        with torch.no_grad():
            self.loss_weights.data = (
                1 - self.adaptation_rate
            ) * self.loss_weights.data + self.adaptation_rate * target_weights
