#
# Plan:
# 1. Implement BEATs fine-tuning module for military vehicle classification
# 2. Create FineTuningConfig for BEATs-specific fine-tuning settings
# 3. Implement LayerWiseOptimizer for layer-wise learning rate scheduling
# 4. Create SemanticAdapter for domain adaptation capabilities
# 5. Add BEATsFineTuner class with various fine-tuning strategies
# 6. Support for tokenizer adaptation and semantic alignment
# 7. Implement advanced regularization and augmentation
# 8. Include attention analysis and interpretability tools
#

"""
BEATs Fine-tuning Module for Military Vehicle Sound Detection.

This module implements advanced fine-tuning strategies for BEATs models,
optimized for military vehicle classification with semantic adaptation
and domain transfer capabilities.

Key Features:
- Semantic adaptation for domain transfer
- Tokenizer fine-tuning and alignment
- Layer-wise learning rate optimization
- Advanced regularization techniques
- Attention visualization and analysis

Reference:
BEATs: Bidirectional Encoder representation from Audio Transformers
https://arxiv.org/abs/2212.09058
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

from .model import BEATsModel, BEATsConfig, AcousticTokenizer
from ..base_model import BaseModel, ModelOutput
from ...utils.metrics import MetricCalculator
from ...utils.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for BEATs fine-tuning strategies"""
    
    # Fine-tuning strategy
    strategy: str = "semantic_adaptation"       # "linear_probe", "end_to_end", "semantic_adaptation"
    
    # Layer freezing
    freeze_encoder: bool = False                # Freeze entire encoder
    freeze_tokenizer: bool = False              # Freeze acoustic tokenizer
    freeze_embeddings: bool = False             # Freeze patch and positional embeddings
    num_frozen_layers: int = 0                  # Number of initial layers to freeze
    
    # Progressive unfreezing
    progressive_unfreezing: bool = False        # Enable progressive unfreezing
    unfreeze_schedule: List[int] = None         # Epochs at which to unfreeze layers
    unfreeze_layers_per_step: int = 2           # Layers to unfreeze per step
    
    # Learning rates
    base_learning_rate: float = 1e-4            # Base learning rate
    head_learning_rate: float = 1e-3            # Classification head learning rate
    tokenizer_learning_rate: float = 5e-5       # Tokenizer learning rate
    layer_decay: float = 0.75                   # Layer-wise learning rate decay
    warmup_epochs: int = 5                      # Learning rate warmup epochs
    
    # Semantic adaptation
    semantic_adaptation: bool = True            # Enable semantic adaptation
    adaptation_layers: List[int] = None         # Layers to apply adaptation
    adaptation_strength: float = 0.1            # Strength of adaptation regularization
    
    # Tokenizer adaptation
    adapt_tokenizer: bool = True                # Adapt tokenizer for target domain
    tokenizer_adaptation_weight: float = 0.05   # Weight for tokenizer adaptation loss
    token_alignment_weight: float = 0.02        # Weight for token alignment loss
    
    # Regularization
    head_dropout: float = 0.5                   # Classification head dropout
    token_dropout: float = 0.1                  # Token dropout probability
    feature_dropout: float = 0.1                # Feature dropout probability
    
    # Data augmentation
    spec_augment: bool = True                   # Enable SpecAugment
    mixup_alpha: float = 0.2                    # Mixup parameter
    cutmix_alpha: float = 1.0                   # CutMix parameter
    label_smoothing: float = 0.1                # Label smoothing parameter
    
    # Advanced techniques
    knowledge_distillation: bool = False        # Enable knowledge distillation
    teacher_model_path: Optional[str] = None    # Path to teacher model
    distillation_alpha: float = 0.7             # Distillation loss weight
    distillation_temperature: float = 4.0       # Distillation temperature
    
    # Few-shot learning
    few_shot_mode: bool = False                 # Enable few-shot learning
    support_shots: int = 5                      # Shots per class in support set
    query_shots: int = 15                       # Shots per class in query set
    prototype_learning: bool = False            # Use prototype-based learning
    
    # Analysis and visualization
    save_attention_maps: bool = False           # Save attention maps during training
    analyze_tokens: bool = False                # Analyze token usage patterns
    semantic_probing: bool = False              # Perform semantic probing


class LayerWiseOptimizer:
    """
    Layer-wise learning rate scheduling for BEATs fine-tuning.
    
    Implements sophisticated learning rate scheduling that considers
    the hierarchical nature of transformer representations.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        base_lr: float, 
        layer_decay: float = 0.75,
        head_lr_mult: float = 10.0,
        tokenizer_lr_mult: float = 0.5
    ):
        """
        Initialize layer-wise optimizer.
        
        Args:
            model: BEATs model
            base_lr: Base learning rate
            layer_decay: Decay factor for earlier layers
            head_lr_mult: Multiplier for classification head
            tokenizer_lr_mult: Multiplier for tokenizer
        """
        self.model = model
        self.base_lr = base_lr
        self.layer_decay = layer_decay
        self.head_lr_mult = head_lr_mult
        self.tokenizer_lr_mult = tokenizer_lr_mult
        
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
        
        # Acoustic tokenizer - specialized learning rate
        if hasattr(self.model, 'tokenizer'):
            tokenizer_params = list(self.model.tokenizer.parameters())
            if tokenizer_params:
                parameter_groups.append({
                    'params': tokenizer_params,
                    'lr': self.base_lr * self.tokenizer_lr_mult,
                    'name': 'acoustic_tokenizer'
                })
        
        # Encoder layers (reverse order - later layers get higher LR)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'blocks'):
            for i, block in enumerate(reversed(self.model.encoder.blocks)):
                layer_lr = self.base_lr * (self.layer_decay ** i)
                parameter_groups.append({
                    'params': list(block.parameters()),
                    'lr': layer_lr,
                    'name': f'encoder_layer_{len(self.model.encoder.blocks)-1-i}'
                })
        
        # Patch embedding - lower learning rate
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'patch_embed'):
            patch_embed_params = list(self.model.encoder.patch_embed.parameters())
            if patch_embed_params:
                parameter_groups.append({
                    'params': patch_embed_params,
                    'lr': self.base_lr * (self.layer_decay ** len(self.model.encoder.blocks)),
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


class SemanticAdapter(nn.Module):
    """
    Semantic adapter for domain adaptation in BEATs fine-tuning.
    
    This module helps adapt pre-trained semantic representations
    to the target domain (military vehicle sounds).
    """
    
    def __init__(self, config: BEATsConfig, finetune_config: FineTuningConfig):
        super().__init__()
        self.config = config
        self.finetune_config = finetune_config
        
        # Adaptation layers
        adaptation_layers = finetune_config.adaptation_layers or [6, 9, 11]  # Middle and late layers
        self.adaptation_layers = adaptation_layers
        
        # Domain adaptation modules
        self.domain_adapters = nn.ModuleDict({
            str(layer_idx): nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.embed_dim, config.embed_dim)
            ) for layer_idx in adaptation_layers
        })
        
        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, 2)  # Source vs Target domain
        )
        
        # Semantic alignment module
        self.semantic_aligner = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads // 2,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(
        self, 
        features: torch.Tensor, 
        layer_idx: int,
        domain_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply semantic adaptation to features.
        
        Args:
            features: Input features [B, N, D]
            layer_idx: Current layer index
            domain_labels: Domain labels for adversarial training
            
        Returns:
            Dictionary with adapted features and losses
        """
        output = {'features': features}
        
        # Apply domain adaptation if this layer is configured
        if str(layer_idx) in self.domain_adapters:
            # Domain adaptation
            adapted_features = self.domain_adapters[str(layer_idx)](features)
            residual_features = features + self.finetune_config.adaptation_strength * adapted_features
            output['features'] = residual_features
            
            # Domain adversarial loss
            if domain_labels is not None and self.training:
                # Global average pooling for domain classification
                pooled_features = features.mean(dim=1)  # [B, D]
                
                # Gradient reversal (implemented as negative scaling)
                reversed_features = -pooled_features + pooled_features.detach()
                domain_logits = self.domain_classifier(reversed_features)
                
                domain_loss = F.cross_entropy(domain_logits, domain_labels)
                output['domain_loss'] = domain_loss
            
            # Semantic alignment loss
            if hasattr(self, '_source_features'):
                aligned_features, attention_weights = self.semantic_aligner(
                    residual_features, self._source_features, self._source_features
                )
                alignment_loss = F.mse_loss(aligned_features, residual_features)
                output['alignment_loss'] = alignment_loss
        
        return output
    
    def set_source_features(self, source_features: torch.Tensor):
        """Set source domain features for alignment"""
        self._source_features = source_features.detach()


class TokenAnalyzer:
    """
    Analyzer for acoustic token usage patterns and semantic analysis.
    """
    
    def __init__(self, tokenizer: AcousticTokenizer):
        self.tokenizer = tokenizer
        self.token_usage_history = {}
        
    def analyze_token_usage(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze token usage patterns across different classes.
        
        Args:
            features: Input features [B, N, D]
            labels: Class labels [B]
            
        Returns:
            Dictionary with token analysis results
        """
        with torch.no_grad():
            # Get token predictions
            _, token_ids, _ = self.tokenizer(features, training=False)
            
            # Analyze token distribution per class
            token_stats = {}
            for class_idx in torch.unique(labels):
                class_mask = labels == class_idx
                class_tokens = token_ids[class_mask]  # [N_class, N]
                
                # Token frequency distribution
                token_freq = torch.bincount(class_tokens.flatten(), 
                                          minlength=self.tokenizer.config.tokenizer_vocab_size)
                token_freq = token_freq.float() / token_freq.sum()
                
                # Token entropy (diversity)
                token_entropy = -torch.sum(token_freq * torch.log(token_freq + 1e-8))
                
                # Most frequent tokens
                top_tokens = torch.topk(token_freq, k=10)[1]
                
                token_stats[f'class_{class_idx.item()}'] = {
                    'token_frequency': token_freq,
                    'token_entropy': token_entropy.item(),
                    'top_tokens': top_tokens.tolist(),
                    'num_unique_tokens': (token_freq > 0).sum().item()
                }
        
        return token_stats
    
    def visualize_token_attention(
        self, 
        features: torch.Tensor, 
        token_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Visualize attention patterns for different tokens.
        
        Args:
            features: Input features [B, N, D]
            token_ids: Token IDs [B, N]
            
        Returns:
            Dictionary with attention visualization data
        """
        # This would implement token-based attention analysis
        # For brevity, returning placeholder
        return {
            'token_attention_maps': torch.zeros(features.shape[:2]),
            'token_similarity_matrix': torch.zeros(token_ids.shape[1], token_ids.shape[1])
        }


class BEATsFineTuner(BaseModel):
    """
    Fine-tuning wrapper for BEATs models with semantic adaptation.
    
    Supports multiple fine-tuning strategies optimized for military
    vehicle sound detection with domain adaptation capabilities.
    """
    
    def __init__(
        self, 
        config: Union[Dict[str, Any], BEATsConfig],
        finetune_config: Optional[FineTuningConfig] = None,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize BEATs fine-tuner.
        
        Args:
            config: Model configuration
            finetune_config: Fine-tuning specific configuration
            pretrained_path: Path to pre-trained model weights
        """
        super().__init__()
        
        # Convert config if needed
        if isinstance(config, dict):
            config = BEATsConfig(**config)
        
        self.config = config
        self.finetune_config = finetune_config or FineTuningConfig()
        
        # Initialize model
        self.model = BEATsModel(config)
        
        # Load pre-trained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
        
        # Setup fine-tuning strategy
        self._setup_fine_tuning()
        
        # Initialize semantic adapter
        if self.finetune_config.semantic_adaptation:
            self.semantic_adapter = SemanticAdapter(config, self.finetune_config)
        else:
            self.semantic_adapter = None
        
        # Initialize token analyzer
        if self.finetune_config.analyze_tokens:
            self.token_analyzer = TokenAnalyzer(self.model.tokenizer)
        else:
            self.token_analyzer = None
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricCalculator(
            num_classes=config.num_classes,
            class_names=self._get_class_names()
        )
        
        # Track training state
        self.current_epoch = 0
        self.frozen_layers = set()
        
        logger.info(f"Initialized BEATs fine-tuner with strategy: {self.finetune_config.strategy}")
    
    def _get_class_names(self) -> List[str]:
        """Get military vehicle class names"""
        return [
            "helicopter", "fighter_aircraft", "military_vehicle",
            "truck", "footsteps", "speech", "background"
        ]
    
    def load_pretrained_weights(self, pretrained_path: str, strict: bool = False):
        """
        Load pre-trained BEATs weights.
        
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
                    keys_to_remove = [k for k in state_dict.keys() if k.startswith('head.')]
                    for key in keys_to_remove:
                        del state_dict[key]
            
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
        elif strategy == "semantic_adaptation":
            self._setup_semantic_adaptation()
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
    
    def _setup_linear_probe(self):
        """Setup linear probing: freeze encoder, train only classification head"""
        # Freeze encoder and tokenizer
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
        if self.finetune_config.freeze_tokenizer:
            for param in self.model.tokenizer.parameters():
                param.requires_grad = False
        
        # Ensure classification head is trainable
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        logger.info("Setup linear probing: encoder frozen, head trainable")
    
    def _setup_end_to_end(self):
        """Setup end-to-end fine-tuning: train entire model"""
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Apply specific freezing based on configuration
        if self.finetune_config.freeze_tokenizer:
            for param in self.model.tokenizer.parameters():
                param.requires_grad = False
        
        if self.finetune_config.freeze_embeddings:
            self.model.encoder.pos_embed.requires_grad = False
            for param in self.model.encoder.patch_embed.parameters():
                param.requires_grad = False
        
        # Freeze initial layers if specified
        if self.finetune_config.num_frozen_layers > 0:
            for i in range(self.finetune_config.num_frozen_layers):
                for param in self.model.encoder.blocks[i].parameters():
                    param.requires_grad = False
                self.frozen_layers.add(i)
        
        logger.info("Setup end-to-end fine-tuning")
    
    def _setup_semantic_adaptation(self):
        """Setup semantic adaptation fine-tuning"""
        # Start similar to end-to-end
        self._setup_end_to_end()
        
        # Additional setup for semantic adaptation
        if self.semantic_adapter:
            for param in self.semantic_adapter.parameters():
                param.requires_grad = True
        
        logger.info("Setup semantic adaptation fine-tuning")
    
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
                for param in self.model.encoder.blocks[layer_idx].parameters():
                    param.requires_grad = True
                
                self.frozen_layers.remove(layer_idx)
                logger.info(f"Unfroze encoder layer {layer_idx} at epoch {epoch}")
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer with layer-wise learning rates"""
        if self.finetune_config.layer_decay != 1.0:
            layer_optimizer = LayerWiseOptimizer(
                self.model,
                self.finetune_config.base_learning_rate,
                self.finetune_config.layer_decay,
                self.finetune_config.head_learning_rate / self.finetune_config.base_learning_rate,
                self.finetune_config.tokenizer_learning_rate / self.finetune_config.base_learning_rate
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
        domain_labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_tokens: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for fine-tuning.
        
        Args:
            x: Input audio spectrograms [B, C, H, W]
            labels: Ground truth labels [B]
            domain_labels: Domain labels for adaptation [B]
            return_features: Whether to return intermediate features
            return_tokens: Whether to return token information
            
        Returns:
            Dictionary containing logits, loss, and optional features/tokens
        """
        # Update unfreezing schedule
        self.update_unfreezing_schedule(self.current_epoch)
        
        # Forward pass through model
        model_output = self.model(
            x, 
            labels=labels, 
            use_tokenizer=self.finetune_config.adapt_tokenizer,
            return_features=return_features or return_tokens
        )
        
        output = {
            'logits': model_output.logits,
            'loss': model_output.loss if hasattr(model_output, 'loss') else None
        }
        
        # Apply semantic adaptation if enabled
        if self.semantic_adapter and hasattr(model_output, 'features'):
            # Apply adaptation to features from middle layers
            adaptation_losses = []
            for layer_idx in self.finetune_config.adaptation_layers or []:
                if layer_idx < len(self.model.encoder.blocks):
                    # Extract features from specific layer
                    layer_features = self.model.extract_features(x, layer_idx)
                    
                    # Apply semantic adaptation
                    adapted_output = self.semantic_adapter(
                        layer_features, layer_idx, domain_labels
                    )
                    
                    # Collect adaptation losses
                    if 'domain_loss' in adapted_output:
                        adaptation_losses.append(adapted_output['domain_loss'])
                    if 'alignment_loss' in adapted_output:
                        adaptation_losses.append(adapted_output['alignment_loss'])
            
            # Add adaptation losses to total loss
            if adaptation_losses and output['loss'] is not None:
                adaptation_loss = torch.stack(adaptation_losses).mean()
                output['loss'] += self.finetune_config.adaptation_strength * adaptation_loss
                output['adaptation_loss'] = adaptation_loss
        
        # Add token analysis if enabled
        if self.token_analyzer and return_tokens and labels is not None:
            token_stats = self.token_analyzer.analyze_token_usage(
                model_output.features, labels
            )
            output['token_stats'] = token_stats
        
        # Add tokenizer adaptation loss if enabled
        if (self.finetune_config.adapt_tokenizer and 
            hasattr(model_output, 'tokenizer_loss') and
            output['loss'] is not None):
            tokenizer_loss = model_output.tokenizer_loss
            output['loss'] += self.finetune_config.tokenizer_adaptation_weight * tokenizer_loss
            output['tokenizer_loss'] = tokenizer_loss
        
        # Return additional outputs
        if return_features and hasattr(model_output, 'features'):
            output['features'] = model_output.features
            output['pooled_features'] = model_output.pooled_features
        
        if return_tokens and hasattr(model_output, 'token_ids'):
            output['token_ids'] = model_output.token_ids
        
        return output
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step with data augmentation and semantic adaptation.
        
        Args:
            batch: Input batch containing 'spectrograms', 'labels', and optionally 'domain_labels'
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        spectrograms = batch['spectrograms']  # [B, C, H, W]
        labels = batch['labels']  # [B]
        domain_labels = batch.get('domain_labels')  # [B] (optional)
        
        # Apply augmentation if configured
        if self.training and (self.finetune_config.mixup_alpha > 0 or self.finetune_config.cutmix_alpha > 0):
            spectrograms, labels = self._apply_augmentation(spectrograms, labels)
        
        # Forward pass
        output = self.forward(
            spectrograms, 
            labels, 
            domain_labels=domain_labels,
            return_tokens=self.finetune_config.analyze_tokens
        )
        
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
            **metrics,
            **{k: v for k, v in output.items() if k.endswith('_loss')}
        }
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step"""
        spectrograms = batch['spectrograms']
        labels = batch['labels']
        domain_labels = batch.get('domain_labels')
        
        # Forward pass
        output = self.forward(
            spectrograms, 
            labels, 
            domain_labels=domain_labels,
            return_tokens=self.finetune_config.analyze_tokens
        )
        
        loss = output['loss']
        logits = output['logits']
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_batch_metrics(
            logits, labels, prefix='val'
        )
        
        return {
            'loss': loss,
            'logits': logits,
            **metrics,
            **{k: v for k, v in output.items() if k.endswith('_loss')}
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
            output = self.forward(spectrograms, return_features=True, return_tokens=True)
            
            logits = output['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            result = {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'logits': logits
            }
            
            if 'features' in output:
                result['features'] = output['features']
            if 'token_ids' in output:
                result['token_ids'] = output['token_ids']
            
            return result
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'finetune_config': self.finetune_config.__dict__,
            'frozen_layers': list(self.frozen_layers)
        }
        
        if self.semantic_adapter:
            checkpoint['semantic_adapter_state_dict'] = self.semantic_adapter.state_dict()
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = False) -> Optional[Dict]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.semantic_adapter and 'semantic_adapter_state_dict' in checkpoint:
            self.semantic_adapter.load_state_dict(checkpoint['semantic_adapter_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.frozen_layers = set(checkpoint.get('frozen_layers', []))
        
        logger.info(f"Loaded checkpoint from {filepath}")
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return checkpoint['optimizer_state_dict']
        
        return None