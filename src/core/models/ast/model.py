#
# Plan:
# 1. Implement Audio Spectrogram Transformer (AST) with Vision Transformer backbone
# 2. Create ASTConfig dataclass for model configuration
# 3. Implement patch embedding specifically for audio spectrograms
# 4. Create AST transformer blocks with audio-optimized attention
# 5. Add positional embedding for 2D audio patches
# 6. Implement classification head for military vehicle detection
# 7. Support for different input sizes and patch configurations
# 8. Add model variants (base, small, tiny) for different deployment scenarios
#

"""
Audio Spectrogram Transformer (AST) for Military Vehicle Sound Detection.

This module implements the Audio Spectrogram Transformer architecture, which adapts
the Vision Transformer (ViT) backbone for audio classification tasks. AST treats
audio spectrograms as images and applies patch-based attention mechanisms.

Key Features:
- Vision Transformer backbone adapted for audio
- Patch-based spectrogram processing
- State-of-the-art performance on audio classification
- Optimized for military vehicle sound detection

Reference:
AST: Audio Spectrogram Transformer
https://arxiv.org/abs/2104.01778
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

from ..base_model import BaseModel, ModelOutput
from ...utils.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class ASTConfig:
    """Audio Spectrogram Transformer configuration"""
    
    # Input dimensions
    input_size: Tuple[int, int] = (128, 128)  # (freq_bins, time_frames)
    patch_size: Tuple[int, int] = (16, 16)    # Patch size for tokenization
    in_channels: int = 1                       # Input channels (mono audio)
    
    # Model architecture
    embed_dim: int = 768                       # Embedding dimension
    depth: int = 12                            # Number of transformer layers
    num_heads: int = 12                        # Number of attention heads
    mlp_ratio: float = 4.0                     # MLP expansion ratio
    
    # Classification
    num_classes: int = 7                       # Number of military vehicle classes
    global_pool: str = "token"                 # "token", "avg", "max"
    
    # Regularization
    dropout: float = 0.1                       # General dropout rate
    attention_dropout: float = 0.1             # Attention dropout rate
    path_dropout: float = 0.1                  # Stochastic depth dropout
    
    # Initialization
    init_std: float = 0.02                     # Weight initialization std
    
    # Audio-specific settings
    frequency_masking: bool = True             # Enable frequency masking during training
    time_masking: bool = True                  # Enable time masking during training
    mixup_alpha: float = 0.2                   # Mixup augmentation parameter
    
    # Pre-training settings
    pretrained: bool = False                   # Whether to use pre-trained weights
    pretrained_path: Optional[str] = None      # Path to pre-trained model


class PatchEmbedding(nn.Module):
    """
    Audio patch embedding for AST.
    Converts 2D audio spectrograms into patch embeddings.
    """
    
    def __init__(self, config: ASTConfig):
        super().__init__()
        self.config = config
        
        # Calculate number of patches
        self.grid_size = (
            config.input_size[0] // config.patch_size[0],
            config.input_size[1] // config.patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Patch embedding layer
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        logger.debug(f"Initialized patch embedding: {self.num_patches} patches of size {config.patch_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input spectrogram [B, C, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        
        # Validate input dimensions
        assert H == self.config.input_size[0] and W == self.config.input_size[1], \
            f"Input size mismatch: expected {self.config.input_size}, got {(H, W)}"
        
        # Extract patches and project to embedding space
        x = self.projection(x)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention optimized for audio spectrograms.
    """
    
    def __init__(self, config: ASTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.projection = nn.Linear(config.embed_dim, config.embed_dim)
        self.projection_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor [B, N, embed_dim]
            mask: Optional attention mask [B, N, N]
            
        Returns:
            Output tensor [B, N, embed_dim]
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        x = (attention_probs @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.projection(x)
        x = self.projection_dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron block for transformer.
    """
    
    def __init__(self, config: ASTConfig):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout2 = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for regularization.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    """
    
    def __init__(self, config: ASTConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = MultiHeadAttention(config)
        
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.drop_path(self.attention(self.norm1(x), mask))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class AudioSpectrogramTransformer(BaseModel):
    """
    Audio Spectrogram Transformer for military vehicle sound detection.
    
    This model adapts the Vision Transformer architecture for audio classification
    by treating spectrograms as images and applying patch-based attention.
    """
    
    def __init__(self, config: Union[Dict[str, Any], ASTConfig]):
        super().__init__()
        
        # Convert config if needed
        if isinstance(config, dict):
            config = ASTConfig(**config)
        
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim)
        )
        self.pos_dropout = nn.Dropout(config.dropout)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.path_dropout, config.depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, drop_path_rate=dpr[i])
            for i in range(config.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Classification head
        if config.global_pool == "token":
            self.head = nn.Linear(config.embed_dim, config.num_classes)
        elif config.global_pool in ["avg", "max"]:
            self.head = nn.Linear(config.embed_dim, config.num_classes)
        else:
            raise ValueError(f"Unknown global_pool: {config.global_pool}")
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized AST model with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=self.config.init_std)
        torch.nn.init.trunc_normal_(self.cls_token, std=self.config.init_std)
        
        # Initialize other layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.trunc_normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extraction layers.
        
        Args:
            x: Input spectrogram [B, C, H, W]
            
        Returns:
            Feature tokens [B, num_patches + 1, embed_dim]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        return x
    
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Feature tokens [B, num_patches + 1, embed_dim]
            
        Returns:
            Class logits [B, num_classes]
        """
        if self.config.global_pool == "token":
            # Use class token
            x = x[:, 0]  # [B, embed_dim]
        elif self.config.global_pool == "avg":
            # Global average pooling (excluding class token)
            x = x[:, 1:].mean(dim=1)  # [B, embed_dim]
        elif self.config.global_pool == "max":
            # Global max pooling (excluding class token)
            x = x[:, 1:].max(dim=1)[0]  # [B, embed_dim]
        
        # Classification layer
        x = self.head(x)  # [B, num_classes]
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> ModelOutput:
        """
        Forward pass of AST model.
        
        Args:
            x: Input spectrogram [B, C, H, W]
            labels: Ground truth labels [B] (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            ModelOutput containing logits, loss, and optional features
        """
        # Feature extraction
        features = self.forward_features(x)  # [B, num_patches + 1, embed_dim]
        
        # Classification
        logits = self.forward_head(features)  # [B, num_classes]
        
        # Prepare output
        output = ModelOutput(logits=logits)
        
        if return_features:
            output.features = features
            output.cls_token = features[:, 0]  # Class token features
            output.patch_features = features[:, 1:]  # Patch features
        
        # Compute loss if labels provided
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            output.loss = criterion(logits, labels)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> List[torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input spectrogram [B, C, H, W]
            layer_idx: Specific layer index to extract (None for all layers)
            
        Returns:
            List of attention maps [B, num_heads, num_patches + 1, num_patches + 1]
        """
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Extract attention weights from the attention module
            if hasattr(module, 'attention_probs'):
                attention_maps.append(module.attention_probs.detach())
        
        # Register hooks
        hooks = []
        for i, block in enumerate(self.blocks):
            if layer_idx is None or i == layer_idx:
                hooks.append(block.attention.register_forward_hook(hook_fn))
        
        try:
            # Forward pass
            with torch.no_grad():
                self.forward_features(x)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return attention_maps
    
    def interpolate_pos_embed(self, new_size: Tuple[int, int]):
        """
        Interpolate positional embeddings for different input sizes.
        
        Args:
            new_size: New input size (H, W)
        """
        old_size = self.config.input_size
        if new_size == old_size:
            return
        
        logger.info(f"Interpolating positional embeddings from {old_size} to {new_size}")
        
        # Calculate new number of patches
        new_grid_size = (
            new_size[0] // self.config.patch_size[0],
            new_size[1] // self.config.patch_size[1]
        )
        new_num_patches = new_grid_size[0] * new_grid_size[1]
        
        # Extract class token and patch embeddings
        pos_embed = self.pos_embed.data
        cls_pos_embed = pos_embed[:, 0:1, :]  # Class token
        patch_pos_embed = pos_embed[:, 1:, :]  # Patch embeddings
        
        # Reshape and interpolate patch embeddings
        old_grid_size = self.patch_embed.grid_size
        patch_pos_embed = patch_pos_embed.reshape(
            1, old_grid_size[0], old_grid_size[1], -1
        ).permute(0, 3, 1, 2)  # [1, embed_dim, old_h, old_w]
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=new_grid_size,
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)  # [1, new_num_patches, embed_dim]
        
        # Concatenate class token and interpolated patch embeddings
        new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
        
        # Update positional embeddings
        self.pos_embed = nn.Parameter(new_pos_embed)
        
        # Update patch embedding configuration
        self.patch_embed.grid_size = new_grid_size
        self.patch_embed.num_patches = new_num_patches
        self.config.input_size = new_size
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained weights from ImageNet or audio pre-training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce parameter matching
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove head weights if num_classes doesn't match
            if 'head.weight' in state_dict:
                if state_dict['head.weight'].shape[0] != self.config.num_classes:
                    logger.warning("Removing pre-trained head weights due to class mismatch")
                    del state_dict['head.weight']
                    del state_dict['head.bias']
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.info(f"Loaded pre-trained weights from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained weights: {e}")
            raise
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name: str, 
        num_classes: int = 7,
        **kwargs
    ) -> 'AudioSpectrogramTransformer':
        """
        Create AST model with pre-trained weights.
        
        Args:
            model_name: Name of pre-trained model
            num_classes: Number of output classes
            **kwargs: Additional configuration parameters
            
        Returns:
            AST model with pre-trained weights
        """
        # Define pre-trained model configurations
        pretrained_configs = {
            'ast_base_patch16_384': {
                'input_size': (128, 128),
                'patch_size': (16, 16),
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'num_classes': num_classes
            },
            'ast_small_patch16_224': {
                'input_size': (128, 128),
                'patch_size': (16, 16),
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'num_classes': num_classes
            }
        }
        
        if model_name not in pretrained_configs:
            raise ValueError(f"Unknown pre-trained model: {model_name}")
        
        # Create configuration
        config_dict = pretrained_configs[model_name]
        config_dict.update(kwargs)
        config = ASTConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load pre-trained weights if available
        # Note: This would typically download from a model hub
        logger.info(f"Created AST model '{model_name}' (pre-trained weights not loaded)")
        
        return model


def create_ast_variants():
    """Create different AST model variants for various deployment scenarios"""
    variants = {
        'ast_tiny': ASTConfig(
            input_size=(128, 128),
            patch_size=(16, 16),
            embed_dim=192,
            depth=12,
            num_heads=3,
            num_classes=7
        ),
        'ast_small': ASTConfig(
            input_size=(128, 128),
            patch_size=(16, 16),
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=7
        ),
        'ast_base': ASTConfig(
            input_size=(128, 128),
            patch_size=(16, 16),
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=7
        )
    }
    
    return {name: AudioSpectrogramTransformer(config) for name, config in variants.items()}