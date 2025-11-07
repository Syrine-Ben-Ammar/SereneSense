#
# Plan:
# 1. Implement AST fine-tuning module for military vehicle classification
# 2. Create FineTuningConfig for AST-specific fine-tuning settings
# 3. Implement LayerWiseOptimizer for layer-wise learning rate scheduling
# 4. Create ASTFineTuner class with various fine-tuning strategies
# 5. Support for progressive unfreezing and layer-wise training
# 6. Add domain adaptation techniques for military audio
# 7. Implement advanced augmentation during fine-tuning
# 8. Include attention visualization and interpretability tools
#

"""
AST Fine-tuning Module for Military Vehicle Sound Detection.

This module implements advanced fine-tuning strategies for Audio Spectrogram
Transformer (AST) models, optimized for military vehicle classification tasks.

Key Features:
- Layer-wise learning rate scheduling
- Progressive unfreezing strategies
- Domain adaptation techniques
- Advanced data augmentation
- Attention visualization tools

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
from pathlib import Path
import warnings

from .model import AudioSpectrogramTransformer, ASTConfig
from ..base_model import BaseModel, ModelOutput
from ...utils.metrics import MetricCalculator
from ...utils.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for AST fine-tuning strategies"""
    
    # Fine-tuning strategy
    strategy: str = "end_to_end"            # "linear_probe", "end_to_end", "progressive"
    
    # Layer freezing
    freeze_patch_embedding: bool = False    # Freeze patch embedding layer
    freeze_positional_embedding: bool = False  # Freeze positional embeddings
    num_frozen_layers: int = 0              # Number of initial layers to freeze
    
    # Progressive unfreezing
    progressive_unfreezing: bool = False    # Enable progressive unfreezing
    unfreeze_schedule: List[int] = None     # Epochs at which to unfreeze layers
    unfreeze_layers_per_step: int = 2       # Layers to unfreeze per step
    
    # Learning rates
    base_learning_rate: float = 1e-4        # Base learning rate
    head_learning_rate: float = 1e-3        # Classification head learning rate
    layer_decay: float = 0.75               # Layer-wise learning rate decay
    warmup_epochs: int = 5                  # Learning rate warmup epochs
    
    # Regularization
    head_dropout: float = 0.5               # Classification head dropout
    attention_dropout: float = 0.1          # Attention dropout during fine-tuning
    path_dropout: float = 0.1               # Path dropout for regularization
    
    # Data augmentation
    spec_augment: bool = True               # Enable SpecAugment
    mixup_alpha: float = 0.2                # Mixup parameter
    cutmix_alpha: float = 1.0               # CutMix parameter
    label_smoothing: float = 0.1            # Label smoothing parameter
    
    # Domain adaptation
    domain_adaptation: bool = False         # Enable domain adaptation
    adaptation_lambda: float = 0.1          # Domain adaptation weight
    
    # Advanced techniques
    knowledge_distillation: bool = False    # Enable knowledge distillation
    teacher_model_path: Optional[str] = None # Path to teacher model
    distillation_alpha: float = 0.7         # Distillation loss weight
    distillation_temperature: float = 4.0   # Distillation temperature
    
    # Few-shot learning
    few_shot_mode: bool = False             # Enable few-shot learning
    support_shots: int = 5                  # Shots per class in support set
    query_shots: int = 15                   # Shots per class in query set
    
    # Attention analysis
    save_attention_maps: bool = False       # Save attention maps during training


class LayerWiseOptimizer:
    """
    Layer-wise learning rate scheduling for AST fine-tuning.
    
    Implements layer-wise learning rate decay where deeper layers
    (closer to output) get higher learning rates.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        base_lr: float, 
        layer_decay: float = 0.75,
        head_lr_mult: float = 10.0
    ):
        """
        Initialize layer-wise optimizer.
        
        Args:
            model: AST model
            base_lr: Base learning rate
            layer_decay: Decay factor for earlier layers
            head_lr_mult: Multiplier for classification head
        """
        self.model = model
        self.base_lr = base_lr
        self.layer_decay = layer_decay
        self.head_lr_mult = head_lr_mult
        
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with layer-wise learning rates"""
        parameter_groups = []
        
        # Classification head - highest learning rate
        if hasattr(self.model, 'head'):
            head_params = list(self.model.head.parameters())
            if head_params:
                parameter_groups.append({
                    'params': head_params,
                    'lr': self.base_lr * self.head_lr_mult,
                    'name': 'classification_head'
                })
        
        # Layer norm and head components
        norm_params = []
        if hasattr(self.model, 'norm'):
            norm_params.extend(list(self.model.norm.parameters()))
        
        if norm_params:
            parameter_groups.append({
                'params': norm_params,
                'lr': self.base_lr,
                'name': 'layer_norm'
            })
        
        # Transformer blocks (reverse order - later layers get higher LR)
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(reversed(self.model.blocks)):
                layer_lr = self.base_lr * (self.layer_decay ** i)
                parameter_groups.append({
                    'params': list(block.parameters()),
                    'lr': layer_lr,
                    'name': f'transformer_layer_{len(self.model.blocks)-1-i}'
                })
        
        # Patch embedding - lowest learning rate
        if hasattr(self.model, 'patch_embed'):
            patch_embed_params = list(self.model.patch_embed.parameters())
            if patch_embed_params:
                parameter_groups.append({
                    'params': patch_embed_params,
                    'lr': self.base_lr * (self.layer_decay ** len(self.model.blocks)),
                    'name': 'patch_embedding'
                })
        
        # Positional embeddings and other parameters
        other_params = []
        handled_params = set()
        
        # Collect already handled parameters
        for group in parameter_groups:
            for param in group['params']:
                handled_params.add(id(param))
        
        # Find remaining parameters
        for name, param in self.model.named_parameters():
            if id(param) not in handled_params:
                other_params.append(param)
        
        if other_params:
            parameter_groups.append({
                'params': other_params,
                'lr': self.base_lr,
                'name': 'other_parameters'
            })
        
        return parameter_groups


class AttentionAnalyzer:
    """
    Analyzer for AST attention maps and interpretability.
    """
    
    def __init__(self, model: AudioSpectrogramTransformer):
        self.model = model
        self.attention_maps = []
        
    def extract_attention_maps(self, x: torch.Tensor, layer_indices: Optional[List[int]] = None) -> List[torch.Tensor]:
        """
        Extract attention maps from specified layers.
        
        Args:
            x: Input spectrogram [B, C, H, W]
            layer_indices: Specific layers to extract (None for all)
            
        Returns:
            List of attention maps
        """
        attention_maps = []
        
        def attention_hook(module, input, output):
            # The attention module should store attention weights
            if hasattr(module, 'attention_weights'):
                attention_maps.append(module.attention_weights.detach())
        
        # Register hooks
        hooks = []
        for i, block in enumerate(self.model.blocks):
            if layer_indices is None or i in layer_indices:
                hooks.append(block.attention.register_forward_hook(attention_hook))
        
        try:
            with torch.no_grad():
                self.model.forward_features(x)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return attention_maps
    
    def analyze_attention_patterns(self, attention_maps: List[torch.Tensor]) -> Dict[str, float]:
        """
        Analyze attention patterns for interpretability.
        
        Args:
            attention_maps: List of attention maps from different layers
            
        Returns:
            Dictionary with attention statistics
        """
        stats = {}
        
        for i, attn_map in enumerate(attention_maps):
            # attn_map shape: [B, num_heads, N+1, N+1]
            
            # CLS token attention (how much CLS attends to patches)
            cls_attention = attn_map[:, :, 0, 1:]  # [B, num_heads, N]
            
            # Average attention entropy (diversity of attention)
            entropy = -torch.sum(cls_attention * torch.log(cls_attention + 1e-8), dim=-1)
            
            stats[f'layer_{i}_attention_entropy'] = entropy.mean().item()
            stats[f'layer_{i}_max_attention'] = cls_attention.max().item()
            stats[f'layer_{i}_min_attention'] = cls_attention.min().item()
            stats[f'layer_{i}_attention_std'] = cls_attention.std().item()
        
        return stats


class ASTFineTuner(BaseModel):
    """
    Fine-tuning wrapper for Audio Spectrogram Transformer models.
    
    Supports multiple fine-tuning strategies optimized for military
    vehicle sound detection tasks.
    """
    
    def __init__(
        self, 
        config: Union[Dict[str, Any], ASTConfig],
        finetune_config: Optional[FineTuningConfig] = None,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize AST fine-tuner.
        
        Args:
            config: Model configuration
            finetune_config: Fine-tuning specific configuration
            pretrained_path: Path to pre-trained model weights
        """
        super().__init__()
        
        # Convert config if needed
        if isinstance(config, dict):
            config = ASTConfig(**config)
        
        self.config = config
        self.finetune_config = finetune_config or FineTuningConfig()
        
        # Initialize model
        self.model = AudioSpectrogramTransformer(config)
        
        # Load pre-trained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
        
        # Setup fine-tuning strategy
        self._setup_fine_tuning()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricCalculator(
            num_classes=config.num_classes,
            class_names=self._get_class_names()
        )
        
        # Initialize attention analyzer
        self.attention_analyzer = AttentionAnalyzer(self.model)
        
        # Track training state
        self.current_epoch = 0
        self.frozen_layers = set()
        
        logger.info(f"Initialized AST fine-tuner with strategy: {self.finetune_config.strategy}")
    
    def _get_class_names(self) -> List[str]:
        """Get military vehicle class names"""
        return [
            "helicopter", "fighter_aircraft", "military_vehicle",
            "truck", "footsteps", "speech", "background"
        ]
    
    def load_pretrained_weights(self, pretrained_path: str, strict: bool = False):
        """
        Load pre-trained AST weights.
        
        Args:
            pretrained_path: Path to pre-trained model checkpoint
            strict: Whether to strictly enforce parameter matching
        """
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle head weight mismatch
            if 'head.weight' in state_dict:
                if state_dict['head.weight'].shape[0] != self.config.num_classes:
                    logger.warning("Removing pre-trained head weights due to class number mismatch")
                    del state_dict['head.weight']
                    if 'head.bias' in state_dict:
                        del state_dict['head.bias']
            
            # Load weights
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=strict
            )
            
            if missing_keys:
                logger.warning(f"Missing keys in pre-trained weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in pre-trained weights: {unexpected_keys}")
            
            logger.info(f"Successfully loaded pre-trained weights from {pretrained_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained weights: {e}")
            raise
    
    def _setup_fine_tuning(self):
        """Setup fine-tuning strategy based on configuration"""
        strategy = self.finetune_config.strategy
        
        if strategy == "linear_probe":
            self._setup_linear_probe()
        elif strategy == "end_to_end":
            self._setup_end_to_end()
        elif strategy == "progressive":
            self._setup_progressive_unfreezing()
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
    
    def _setup_linear_probe(self):
        """Setup linear probing: freeze encoder, train only classification head"""
        # Freeze all encoder parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classification head
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        logger.info("Setup linear probing: encoder frozen, head trainable")
    
    def _setup_end_to_end(self):
        """Setup end-to-end fine-tuning: train entire model"""
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Optionally freeze some components
        if self.finetune_config.freeze_patch_embedding:
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False
        
        if self.finetune_config.freeze_positional_embedding:
            self.model.pos_embed.requires_grad = False
            self.model.cls_token.requires_grad = False
        
        # Freeze initial layers if specified
        if self.finetune_config.num_frozen_layers > 0:
            for i in range(self.finetune_config.num_frozen_layers):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = False
                self.frozen_layers.add(i)
        
        logger.info("Setup end-to-end fine-tuning")
    
    def _setup_progressive_unfreezing(self):
        """Setup progressive unfreezing: start frozen, gradually unfreeze"""
        # Start with everything frozen except classification head
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        # Mark all layers as frozen initially
        for i in range(len(self.model.blocks)):
            self.frozen_layers.add(i)
        
        logger.info("Setup progressive unfreezing: starting with frozen encoder")
    
    def update_unfreezing_schedule(self, epoch: int):
        """Update layer freezing based on progressive unfreezing schedule"""
        if not self.finetune_config.progressive_unfreezing:
            return
        
        schedule = self.finetune_config.unfreeze_schedule
        if schedule is None:
            # Default schedule: unfreeze every 10 epochs
            schedule = list(range(10, 100, 10))
        
        if epoch in schedule:
            layers_to_unfreeze = self.finetune_config.unfreeze_layers_per_step
            
            # Unfreeze from the last layers (closest to output) first
            frozen_layers_list = sorted(list(self.frozen_layers), reverse=True)
            
            for i in range(min(layers_to_unfreeze, len(frozen_layers_list))):
                layer_idx = frozen_layers_list[i]
                
                # Unfreeze the layer
                for param in self.model.blocks[layer_idx].parameters():
                    param.requires_grad = True
                
                self.frozen_layers.remove(layer_idx)
                logger.info(f"Unfroze transformer layer {layer_idx} at epoch {epoch}")
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer with layer-wise learning rates"""
        if self.finetune_config.layer_decay != 1.0:
            layer_optimizer = LayerWiseOptimizer(
                self.model,
                self.finetune_config.base_learning_rate,
                self.finetune_config.layer_decay,
                self.finetune_config.head_learning_rate / self.finetune_config.base_learning_rate
            )
            return layer_optimizer.get_parameter_groups()
        else:
            # Simple parameter grouping
            return [
                {
                    'params': [p for p in self.model.parameters() if p.requires_grad],
                    'lr': self.finetune_config.base_learning_rate
                }
            ]
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for fine-tuning.
        
        Args:
            x: Input audio spectrograms [B, C, H, W]
            labels: Ground truth labels [B]
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing logits, loss, and optionally features/attention
        """
        # Update unfreezing schedule
        self.update_unfreezing_schedule(self.current_epoch)
        
        # Forward pass
        model_output = self.model(x, labels, return_features)
        output = {
            'logits': model_output.logits,
            'loss': model_output.loss
        }
        
        if return_features and hasattr(model_output, 'features'):
            output['features'] = model_output.features
            output['cls_token'] = model_output.cls_token
            output['patch_features'] = model_output.patch_features
        
        # Extract attention maps if requested
        if return_attention:
            attention_maps = self.attention_analyzer.extract_attention_maps(x)
            output['attention_maps'] = attention_maps
            output['attention_stats'] = self.attention_analyzer.analyze_attention_patterns(attention_maps)
        
        return output
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step with data augmentation.
        
        Args:
            batch: Input batch containing 'spectrograms' and 'labels'
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        spectrograms = batch['spectrograms']  # [B, C, H, W]
        labels = batch['labels']  # [B]
        
        # Apply augmentation if configured
        if self.training and (self.finetune_config.mixup_alpha > 0 or self.finetune_config.cutmix_alpha > 0):
            spectrograms, labels = self._apply_augmentation(spectrograms, labels)
        
        # Forward pass
        output = self.forward(spectrograms, labels)
        
        loss = output['loss']
        logits = output['logits']
        
        # Calculate metrics
        with torch.no_grad():
            if isinstance(labels, tuple):  # Mixed labels from augmentation
                labels = labels[0] if isinstance(labels[0], torch.Tensor) else batch['labels']
            
            metrics = self.metrics_calculator.calculate_batch_metrics(
                logits, labels, prefix='train'
            )
        
        return {
            'loss': loss,
            'logits': logits,
            **metrics
        }
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step"""
        spectrograms = batch['spectrograms']
        labels = batch['labels']
        
        # Forward pass
        output = self.forward(spectrograms, labels, return_attention=self.finetune_config.save_attention_maps)
        
        loss = output['loss']
        logits = output['logits']
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_batch_metrics(
            logits, labels, prefix='val'
        )
        
        # Add attention statistics if available
        if 'attention_stats' in output:
            metrics.update(output['attention_stats'])
        
        return {
            'loss': loss,
            'logits': logits,
            **metrics
        }
    
    def _apply_augmentation(
        self, 
        spectrograms: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Apply mixup or cutmix augmentation"""
        if np.random.random() < 0.5 and self.finetune_config.mixup_alpha > 0:
            return self._mixup(spectrograms, labels)
        elif self.finetune_config.cutmix_alpha > 0:
            return self._cutmix(spectrograms, labels)
        else:
            return spectrograms, labels
    
    def _mixup(
        self, 
        spectrograms: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
        """Apply mixup augmentation"""
        alpha = self.finetune_config.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        
        batch_size = spectrograms.size(0)
        index = torch.randperm(batch_size, device=spectrograms.device)
        
        mixed_spectrograms = lam * spectrograms + (1 - lam) * spectrograms[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_spectrograms, (labels_a, labels_b, lam)
    
    def _cutmix(
        self, 
        spectrograms: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
        """Apply cutmix augmentation"""
        alpha = self.finetune_config.cutmix_alpha
        lam = np.random.beta(alpha, alpha)
        
        batch_size = spectrograms.size(0)
        index = torch.randperm(batch_size, device=spectrograms.device)
        
        _, _, H, W = spectrograms.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_spectrograms = spectrograms.clone()
        mixed_spectrograms[:, :, bby1:bby2, bbx1:bbx2] = spectrograms[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        labels_a, labels_b = labels, labels[index]
        
        return mixed_spectrograms, (labels_a, labels_b, lam)
    
    def predict(self, spectrograms: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions on spectrogram input.
        
        Args:
            spectrograms: Input spectrograms [B, C, H, W]
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(spectrograms, return_features=True)
            
            logits = output['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'logits': logits,
                'features': output.get('features')
            }
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'finetune_config': self.finetune_config.__dict__,
            'frozen_layers': list(self.frozen_layers)
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = False) -> Optional[Dict]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.frozen_layers = set(checkpoint.get('frozen_layers', []))
        
        logger.info(f"Loaded checkpoint from {filepath}")
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return checkpoint['optimizer_state_dict']
        
        return None