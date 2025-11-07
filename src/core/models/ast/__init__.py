"""
Audio Spectrogram Transformer (AST) Module for Military Vehicle Detection.

This module implements the Audio Spectrogram Transformer architecture
optimized for military vehicle sound classification. AST uses Vision
Transformer backbone adapted for audio spectrograms with patch-based
attention mechanisms.

Key Features:
- Vision Transformer backbone for audio classification
- Patch-based spectrogram processing
- Transfer learning from ImageNet pre-trained models
- Optimized for military vehicle sound detection
- Support for various input spectrogram sizes

Performance Benchmarks:
- AudioSet: 0.459 mAP (state-of-the-art)
- ESC-50: 95.6% accuracy
- Speech Commands V2: 98.11% accuracy
- MAD Dataset: 91.07% accuracy (military vehicles)

Reference:
AST: Audio Spectrogram Transformer
https://arxiv.org/abs/2104.01778
"""

from .model import (
    AudioSpectrogramTransformer,
    ASTConfig,
    PatchEmbedding as ASTPatchEmbedding,
    TransformerBlock as ASTTransformerBlock
)

from .pretrain import (
    ASTPreTrainer,
    PreTrainingConfig as ASTPreTrainingConfig,
    MaskedSpectrogramModeling,
    ContrastiveLearning
)

from .finetune import (
    ASTFineTuner,
    FineTuningConfig as ASTFineTuningConfig,
    LayerWiseOptimizer as ASTLayerWiseOptimizer
)

__all__ = [
    # Core model components
    'AudioSpectrogramTransformer',
    'ASTConfig',
    'ASTPatchEmbedding',
    'ASTTransformerBlock',
    
    # Pre-training components
    'ASTPreTrainer',
    'ASTPreTrainingConfig',
    'MaskedSpectrogramModeling',
    'ContrastiveLearning',
    
    # Fine-tuning components
    'ASTFineTuner',
    'ASTFineTuningConfig',
    'ASTLayerWiseOptimizer'
]

# Model variants and configurations
AST_MODEL_CONFIGS = {
    'ast_base_patch16_384': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'description': 'Base AST model for military vehicle detection'
    },
    
    'ast_small_patch16_224': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'description': 'Small AST model for edge deployment'
    },
    
    'ast_tiny_patch16_224': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'description': 'Tiny AST model for mobile deployment'
    }
}

def create_ast_model(variant: str = 'ast_base_patch16_384', **kwargs):
    """
    Create an AST model with predefined configuration.
    
    Args:
        variant: Model variant name
        **kwargs: Additional configuration parameters
        
    Returns:
        AudioSpectrogramTransformer model instance
    """
    if variant not in AST_MODEL_CONFIGS:
        raise ValueError(f"Unknown AST variant: {variant}. Available variants: {list(AST_MODEL_CONFIGS.keys())}")
    
    # Get base configuration
    config_dict = AST_MODEL_CONFIGS[variant].copy()
    config_dict.update(kwargs)
    
    # Create configuration object
    config = ASTConfig(**config_dict)
    
    # Create and return model
    return AudioSpectrogramTransformer(config)