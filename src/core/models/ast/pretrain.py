#
# Plan:
# 1. Implement AST pre-training module with self-supervised learning
# 2. Create PreTrainingConfig for pre-training specific settings
# 3. Implement MaskedSpectrogramModeling for self-supervised learning
# 4. Add ContrastiveLearning for improved representations
# 5. Create ASTPreTrainer class for managing the pre-training process
# 6. Support for different pre-training strategies (MSM, contrastive, hybrid)
# 7. Integration with AudioSet and other large-scale datasets
# 8. Progressive training and curriculum learning capabilities
#

"""
AST Pre-training Module for Self-Supervised Learning.

This module implements various pre-training strategies for Audio Spectrogram
Transformer (AST) models, including masked spectrogram modeling and 
contrastive learning approaches optimized for audio representations.

Key Features:
- Masked Spectrogram Modeling (MSM)
- Contrastive learning frameworks
- Multi-scale temporal modeling
- Progressive training strategies
- Integration with large-scale audio datasets

Reference:
AST: Audio Spectrogram Transformer
https://arxiv.org/abs/2104.01778
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import random
from pathlib import Path

from .model import AudioSpectrogramTransformer, ASTConfig
from ..base_model import BaseModel, ModelOutput
from ...utils.metrics import MetricCalculator
from ...utils.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class PreTrainingConfig:
    """Configuration for AST pre-training strategies"""
    
    # Pre-training strategy
    strategy: str = "masked_spectrogram_modeling"  # "msm", "contrastive", "hybrid"
    
    # Masked Spectrogram Modeling settings
    mask_ratio: float = 0.75                # Ratio of patches to mask
    mask_strategy: str = "random"           # "random", "block", "frequency", "time"
    min_mask_patches: int = 16              # Minimum number of patches to mask
    max_mask_patches: Optional[int] = None  # Maximum number of patches to mask
    
    # Block masking parameters
    block_mask_ratio: float = 0.3           # Ratio for block masking
    min_block_size: int = 4                 # Minimum block size
    max_block_size: int = 16                # Maximum block size
    
    # Frequency/Time masking
    freq_mask_ratio: float = 0.15           # Frequency masking ratio
    time_mask_ratio: float = 0.15           # Time masking ratio
    
    # Contrastive learning settings
    temperature: float = 0.07               # Temperature for contrastive loss
    negative_samples: int = 4096            # Number of negative samples
    momentum: float = 0.999                 # Momentum for moving averages
    
    # Progressive training
    progressive_training: bool = False      # Enable progressive training
    curriculum_schedule: List[float] = None # Curriculum learning schedule
    
    # Data augmentation
    spec_augment: bool = True               # Enable SpecAugment during pre-training
    mixup_alpha: float = 0.2                # Mixup parameter
    cutmix_alpha: float = 1.0               # CutMix parameter
    
    # Training hyperparameters
    learning_rate: float = 1e-4             # Base learning rate
    warmup_epochs: int = 10                 # Warmup epochs
    weight_decay: float = 1e-4              # Weight decay
    batch_size: int = 32                    # Batch size
    
    # Model specific
    reconstruction_loss_weight: float = 1.0  # Weight for reconstruction loss
    contrastive_loss_weight: float = 0.1     # Weight for contrastive loss


class MaskGenerator:
    """
    Generate various types of masks for masked spectrogram modeling.
    """
    
    def __init__(self, config: PreTrainingConfig):
        self.config = config
    
    def generate_random_mask(self, num_patches: int, mask_ratio: float) -> torch.Tensor:
        """Generate random mask"""
        num_masked = int(num_patches * mask_ratio)
        num_masked = max(self.config.min_mask_patches, num_masked)
        if self.config.max_mask_patches:
            num_masked = min(self.config.max_mask_patches, num_masked)
        
        # Random shuffle
        shuffle_indices = torch.randperm(num_patches)
        mask = torch.zeros(num_patches, dtype=torch.bool)
        mask[shuffle_indices[:num_masked]] = True
        
        return mask
    
    def generate_block_mask(self, grid_size: Tuple[int, int], mask_ratio: float) -> torch.Tensor:
        """Generate block-based mask"""
        h, w = grid_size
        mask = torch.zeros(h, w, dtype=torch.bool)
        
        target_masked = int(h * w * mask_ratio)
        masked_count = 0
        
        while masked_count < target_masked:
            # Random block position and size
            block_h = random.randint(self.config.min_block_size, 
                                   min(self.config.max_block_size, h))
            block_w = random.randint(self.config.min_block_size, 
                                   min(self.config.max_block_size, w))
            
            start_h = random.randint(0, h - block_h)
            start_w = random.randint(0, w - block_w)
            
            # Apply mask
            mask[start_h:start_h + block_h, start_w:start_w + block_w] = True
            masked_count = mask.sum().item()
        
        return mask.flatten()
    
    def generate_frequency_mask(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Generate frequency-based mask"""
        h, w = grid_size
        mask = torch.zeros(h, w, dtype=torch.bool)
        
        num_freq_bands = int(h * self.config.freq_mask_ratio)
        freq_indices = torch.randperm(h)[:num_freq_bands]
        
        mask[freq_indices, :] = True
        return mask.flatten()
    
    def generate_time_mask(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Generate time-based mask"""
        h, w = grid_size
        mask = torch.zeros(h, w, dtype=torch.bool)
        
        num_time_steps = int(w * self.config.time_mask_ratio)
        time_indices = torch.randperm(w)[:num_time_steps]
        
        mask[:, time_indices] = True
        return mask.flatten()
    
    def generate_mask(self, num_patches: int, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Generate mask based on configured strategy"""
        if self.config.mask_strategy == "random":
            return self.generate_random_mask(num_patches, self.config.mask_ratio)
        elif self.config.mask_strategy == "block":
            return self.generate_block_mask(grid_size, self.config.mask_ratio)
        elif self.config.mask_strategy == "frequency":
            return self.generate_frequency_mask(grid_size)
        elif self.config.mask_strategy == "time":
            return self.generate_time_mask(grid_size)
        else:
            raise ValueError(f"Unknown mask strategy: {self.config.mask_strategy}")


class MaskedSpectrogramModeling(nn.Module):
    """
    Masked Spectrogram Modeling for self-supervised AST pre-training.
    """
    
    def __init__(self, config: ASTConfig, pretrain_config: PreTrainingConfig):
        super().__init__()
        self.config = config
        self.pretrain_config = pretrain_config
        
        # Mask generator
        self.mask_generator = MaskGenerator(pretrain_config)
        
        # Reconstruction head
        patch_dim = config.patch_size[0] * config.patch_size[1] * config.in_channels
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, patch_dim)
        )
        
        # Mask token for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        torch.nn.init.trunc_normal_(self.mask_token, std=config.init_std)
    
    def patchify(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Convert spectrograms to patches"""
        B, C, H, W = spectrograms.shape
        ph, pw = self.config.patch_size
        
        # Reshape to patches
        x = spectrograms.reshape(B, C, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // ph) * (W // pw), C * ph * pw)
        
        return x
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert patches back to spectrograms"""
        B, N, patch_dim = patches.shape
        C = self.config.in_channels
        ph, pw = self.config.patch_size
        H, W = self.config.input_size
        
        h, w = H // ph, W // pw
        assert N == h * w
        
        x = patches.reshape(B, h, w, C, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, H, W)
        
        return x
    
    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mask to input tokens and return visible and masked tokens"""
        B, N, D = x.shape
        
        # Separate visible and masked tokens
        visible_tokens = x[~mask].reshape(B, -1, D)
        masked_indices = mask.nonzero(as_tuple=False)[:, 1]  # Get patch indices
        
        return visible_tokens, masked_indices
    
    def forward(
        self, 
        model: AudioSpectrogramTransformer, 
        spectrograms: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for masked spectrogram modeling.
        
        Args:
            model: AST model
            spectrograms: Input spectrograms [B, C, H, W]
            
        Returns:
            Dictionary with loss and reconstruction metrics
        """
        B = spectrograms.shape[0]
        
        # Get patch embeddings
        patch_embeddings = model.patch_embed(spectrograms)  # [B, N, D]
        N = patch_embeddings.shape[1]
        
        # Generate mask
        grid_size = model.patch_embed.grid_size
        mask = self.mask_generator.generate_mask(N, grid_size)
        mask = mask.unsqueeze(0).expand(B, -1)  # [B, N]
        
        # Add class token and positional embeddings to visible tokens only
        visible_mask = ~mask
        cls_tokens = model.cls_token.expand(B, -1, -1)
        
        # Get visible tokens
        visible_tokens = patch_embeddings[visible_mask].reshape(B, -1, patch_embeddings.shape[-1])
        
        # Add class token to visible tokens
        visible_tokens_with_cls = torch.cat([cls_tokens, visible_tokens], dim=1)
        
        # Add positional embeddings (only for visible patches + cls)
        visible_pos_indices = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=spectrograms.device),  # CLS token
            torch.where(visible_mask)[1].reshape(B, -1) + 1  # +1 for CLS offset
        ], dim=1)
        
        visible_pos_embed = model.pos_embed[:, visible_pos_indices.flatten()].reshape(
            B, visible_pos_indices.shape[1], -1
        )
        visible_tokens_with_cls = visible_tokens_with_cls + visible_pos_embed
        
        # Forward through transformer blocks
        x = model.pos_dropout(visible_tokens_with_cls)
        for block in model.blocks:
            x = block(x)
        x = model.norm(x)
        
        # Remove class token for reconstruction
        encoder_output = x[:, 1:]  # Remove CLS token
        
        # Reconstruct masked patches
        masked_tokens = self.mask_token.expand(B, mask.sum(dim=1).max(), -1)
        
        # Combine visible and masked tokens for reconstruction
        full_tokens = torch.zeros(B, N, encoder_output.shape[-1], device=spectrograms.device)
        full_tokens[visible_mask] = encoder_output.flatten(0, 1)
        full_tokens[mask] = masked_tokens.flatten(0, 1)[:mask.sum()]
        
        # Reconstruction head
        reconstructed_patches = self.reconstruction_head(full_tokens)  # [B, N, patch_dim]
        
        # Compute reconstruction loss (only on masked patches)
        target_patches = self.patchify(spectrograms)  # [B, N, patch_dim]
        
        # MSE loss on masked patches only
        masked_target = target_patches[mask]
        masked_pred = reconstructed_patches[mask]
        
        reconstruction_loss = F.mse_loss(masked_pred, masked_target)
        
        # Additional metrics
        with torch.no_grad():
            # Compute PSNR for masked patches
            mse = F.mse_loss(masked_pred, masked_target, reduction='mean')
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(
                masked_pred.flatten(1), 
                masked_target.flatten(1), 
                dim=1
            ).mean()
        
        return {
            'loss': reconstruction_loss,
            'reconstruction_loss': reconstruction_loss,
            'psnr': psnr,
            'cosine_similarity': cos_sim,
            'mask_ratio': mask.float().mean(),
            'reconstructed_patches': reconstructed_patches,
            'target_patches': target_patches,
            'mask': mask
        }


class ContrastiveLearning(nn.Module):
    """
    Contrastive learning framework for AST pre-training.
    """
    
    def __init__(self, config: ASTConfig, pretrain_config: PreTrainingConfig):
        super().__init__()
        self.config = config
        self.pretrain_config = pretrain_config
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 256)  # Projection dimension
        )
        
        # Momentum encoder (optional)
        if pretrain_config.momentum > 0:
            self.momentum_encoder = None  # Will be set externally
            self.momentum = pretrain_config.momentum
    
    def forward(
        self, 
        model: AudioSpectrogramTransformer,
        spectrograms1: torch.Tensor,
        spectrograms2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            model: AST model
            spectrograms1: First augmented view [B, C, H, W]
            spectrograms2: Second augmented view [B, C, H, W]
            
        Returns:
            Dictionary with contrastive loss and metrics
        """
        # Extract features from both views
        features1 = model.forward_features(spectrograms1)  # [B, N+1, D]
        features2 = model.forward_features(spectrograms2)  # [B, N+1, D]
        
        # Use CLS token as global representation
        global_features1 = features1[:, 0]  # [B, D]
        global_features2 = features2[:, 0]  # [B, D]
        
        # Project to contrastive space
        z1 = self.projection_head(global_features1)  # [B, projection_dim]
        z2 = self.projection_head(global_features2)  # [B, projection_dim]
        
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute contrastive loss (SimCLR style)
        batch_size = z1.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)  # [2B, projection_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(representations, representations.t()) / self.pretrain_config.temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(representations.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=representations.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Additional metrics
        with torch.no_grad():
            # Positive pair similarity
            pos_sim = F.cosine_similarity(z1, z2, dim=1).mean()
            
            # Top-1 accuracy (how often positive pairs are most similar)
            _, top_indices = similarity_matrix.topk(1, dim=1)
            accuracy = (top_indices.squeeze() == labels).float().mean()
        
        return {
            'loss': contrastive_loss,
            'contrastive_loss': contrastive_loss,
            'positive_similarity': pos_sim,
            'contrastive_accuracy': accuracy
        }


class ASTPreTrainer(BaseModel):
    """
    Pre-trainer for Audio Spectrogram Transformer models.
    
    Supports multiple pre-training strategies including masked spectrogram
    modeling and contrastive learning.
    """
    
    def __init__(
        self, 
        config: Union[Dict[str, Any], ASTConfig],
        pretrain_config: Optional[PreTrainingConfig] = None
    ):
        super().__init__()
        
        # Convert config if needed
        if isinstance(config, dict):
            config = ASTConfig(**config)
        
        self.config = config
        self.pretrain_config = pretrain_config or PreTrainingConfig()
        
        # Initialize AST model
        self.model = AudioSpectrogramTransformer(config)
        
        # Initialize pre-training modules
        if self.pretrain_config.strategy in ["masked_spectrogram_modeling", "msm", "hybrid"]:
            self.msm_module = MaskedSpectrogramModeling(config, self.pretrain_config)
        
        if self.pretrain_config.strategy in ["contrastive", "hybrid"]:
            self.contrastive_module = ContrastiveLearning(config, self.pretrain_config)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricCalculator(
            task_type='pretraining',
            metric_names=['loss', 'psnr', 'contrastive_accuracy']
        )
        
        logger.info(f"Initialized AST pre-trainer with strategy: {self.pretrain_config.strategy}")
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pre-training.
        
        Args:
            batch: Input batch containing spectrograms and optional augmented views
            
        Returns:
            Dictionary with losses and metrics
        """
        spectrograms = batch['spectrograms']  # [B, C, H, W]
        
        total_loss = 0.0
        output = {}
        
        # Masked Spectrogram Modeling
        if hasattr(self, 'msm_module'):
            msm_output = self.msm_module(self.model, spectrograms)
            msm_loss = msm_output['loss'] * self.pretrain_config.reconstruction_loss_weight
            total_loss += msm_loss
            
            output.update({
                'msm_loss': msm_loss,
                'psnr': msm_output['psnr'],
                'cosine_similarity': msm_output['cosine_similarity'],
                'mask_ratio': msm_output['mask_ratio']
            })
        
        # Contrastive Learning
        if hasattr(self, 'contrastive_module') and 'spectrograms_aug' in batch:
            contrastive_output = self.contrastive_module(
                self.model, spectrograms, batch['spectrograms_aug']
            )
            contrastive_loss = contrastive_output['loss'] * self.pretrain_config.contrastive_loss_weight
            total_loss += contrastive_loss
            
            output.update({
                'contrastive_loss': contrastive_loss,
                'positive_similarity': contrastive_output['positive_similarity'],
                'contrastive_accuracy': contrastive_output['contrastive_accuracy']
            })
        
        output['loss'] = total_loss
        return output
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step for pre-training"""
        output = self.forward(batch)
        
        # Calculate additional metrics
        metrics = self.metrics_calculator.calculate_batch_metrics(
            predictions=None,  # No predictions in pre-training
            targets=None,      # No targets in pre-training
            losses=output,
            prefix='train'
        )
        
        return {**output, **metrics}
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step for pre-training"""
        output = self.forward(batch)
        
        # Calculate additional metrics
        metrics = self.metrics_calculator.calculate_batch_metrics(
            predictions=None,
            targets=None,
            losses=output,
            prefix='val'
        )
        
        return {**output, **metrics}
    
    def save_pretrained_model(self, save_path: str):
        """Save pre-trained model weights"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'pretrain_config': self.pretrain_config.__dict__
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved pre-trained model to {save_path}")
    
    def load_pretrained_model(self, load_path: str):
        """Load pre-trained model weights"""
        checkpoint = torch.load(load_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pre-trained model from {load_path}")
    
    def get_encoder(self) -> AudioSpectrogramTransformer:
        """Get the encoder model for fine-tuning"""
        return self.model