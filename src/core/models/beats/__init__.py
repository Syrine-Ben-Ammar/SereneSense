"""
BEATs (Bidirectional Encoder representation from Audio Transformers) Module.

This module implements the BEATs architecture for military vehicle sound detection.
BEATs represents Microsoft's breakthrough approach to audio understanding using
bidirectional encoder representations with acoustic tokenization.

Key Features:
- Bidirectional encoder representations
- Acoustic tokenization for semantic learning
- Iterative audio pre-training
- State-of-the-art audio understanding
- Optimized for military vehicle classification

Performance Highlights:
- Outperforms AudioMAE and AST on multiple benchmarks
- Achieves superior performance on audio classification tasks
- Efficient semantic representation learning
- Robust to noise and domain variations

Reference:
BEATs: Bidirectional Encoder representation from Audio Transformers
https://arxiv.org/abs/2212.09058
"""

from .model import (
    BEATsModel,
    BEATsConfig,
    AcousticTokenizer,
    BidirectionalEncoder,
    BEATsTransformerBlock
)

from .pretrain import (
    BEATsPreTrainer,
    PreTrainingConfig as BEATsPreTrainingConfig,
    AcousticTokenizerTrainer,
    IterativePreTraining
)

from .finetune import (
    BEATsFineTuner,
    FineTuningConfig as BEATsFineTuningConfig,
    LayerWiseOptimizer as BEATsLayerWiseOptimizer,
    SemanticAdapter
)

__all__ = [
    # Core model components
    'BEATsModel',
    'BEATsConfig',
    'AcousticTokenizer',
    'BidirectionalEncoder', 
    'BEATsTransformerBlock',
    
    # Pre-training components
    'BEATsPreTrainer',
    'BEATsPreTrainingConfig',
    'AcousticTokenizerTrainer',
    'IterativePreTraining',
    
    # Fine-tuning components
    'BEATsFineTuner',
    'BEATsFineTuningConfig',
    'BEATsLayerWiseOptimizer',
    'SemanticAdapter'
]

# Model variants and configurations
BEATS_MODEL_CONFIGS = {
    'beats_base': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 768,
        'encoder_depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'tokenizer_vocab_size': 8192,
        'description': 'Base BEATs model for military vehicle detection'
    },
    
    'beats_large': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 1024,
        'encoder_depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'tokenizer_vocab_size': 8192,
        'description': 'Large BEATs model for high-accuracy deployment'
    },
    
    'beats_small': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 384,
        'encoder_depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'tokenizer_vocab_size': 4096,
        'description': 'Small BEATs model for edge deployment'
    },
    
    'beats_tiny': {
        'input_size': (128, 128),
        'patch_size': (16, 16),
        'embed_dim': 192,
        'encoder_depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'num_classes': 7,
        'tokenizer_vocab_size': 2048,
        'description': 'Tiny BEATs model for mobile deployment'
    }
}

def create_beats_model(variant: str = 'beats_base', **kwargs):
    """
    Create a BEATs model with predefined configuration.
    
    Args:
        variant: Model variant name
        **kwargs: Additional configuration parameters
        
    Returns:
        BEATsModel instance
    """
    if variant not in BEATS_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown BEATs variant: {variant}. "
            f"Available variants: {list(BEATS_MODEL_CONFIGS.keys())}"
        )
    
    # Get base configuration
    config_dict = BEATS_MODEL_CONFIGS[variant].copy()
    config_dict.update(kwargs)
    
    # Create configuration object
    config = BEATsConfig(**config_dict)
    
    # Create and return model
    return BEATsModel(config)


def load_pretrained_beats(model_name: str, num_classes: int = 7, **kwargs):
    """
    Load a pre-trained BEATs model.
    
    Args:
        model_name: Pre-trained model name
        num_classes: Number of output classes
        **kwargs: Additional configuration parameters
        
    Returns:
        Pre-trained BEATs model
    """
    # Predefined pre-trained models
    pretrained_models = {
        'beats_iter3_plus_as2m_finetuned_on_as2m_cpt2': {
            'variant': 'beats_base',
            'description': 'BEATs model pre-trained with iterative training and fine-tuned on AudioSet'
        },
        'beats_iter3_plus_as2m': {
            'variant': 'beats_base', 
            'description': 'BEATs model pre-trained with iterative training on AudioSet'
        }
    }
    
    if model_name not in pretrained_models:
        raise ValueError(f"Unknown pre-trained model: {model_name}")
    
    model_info = pretrained_models[model_name]
    variant = model_info['variant']
    
    # Create model with specified variant
    model = create_beats_model(variant, num_classes=num_classes, **kwargs)
    
    # Note: In a real implementation, you would download and load the actual weights here
    # For now, we just return the initialized model
    print(f"Created BEATs model '{model_name}' - {model_info['description']}")
    print("Note: Pre-trained weights would be loaded in a production implementation")
    
    return model


# Training utilities
def get_beats_training_config(variant: str = 'beats_base', strategy: str = 'iterative'):
    """
    Get recommended training configuration for BEATs variants.
    
    Args:
        variant: Model variant
        strategy: Training strategy ('iterative', 'standard', 'fine_tune')
        
    Returns:
        Training configuration dictionary
    """
    base_configs = {
        'beats_tiny': {
            'batch_size': 64,
            'learning_rate': 2e-4,
            'warmup_epochs': 5,
            'total_epochs': 100
        },
        'beats_small': {
            'batch_size': 48,
            'learning_rate': 1.5e-4,
            'warmup_epochs': 8,
            'total_epochs': 150
        },
        'beats_base': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'warmup_epochs': 10,
            'total_epochs': 200
        },
        'beats_large': {
            'batch_size': 16,
            'learning_rate': 5e-5,
            'warmup_epochs': 15,
            'total_epochs': 300
        }
    }
    
    config = base_configs.get(variant, base_configs['beats_base']).copy()
    
    # Adjust for training strategy
    if strategy == 'iterative':
        config['iterations'] = 3
        config['tokenizer_update_frequency'] = 10
    elif strategy == 'fine_tune':
        config['learning_rate'] *= 0.1
        config['total_epochs'] = 50
        config['freeze_tokenizer'] = True
    
    return config


# Model comparison utilities
def compare_beats_variants():
    """
    Compare different BEATs model variants.
    
    Returns:
        Dictionary with model comparison metrics
    """
    comparison = {}
    
    for variant_name, config in BEATS_MODEL_CONFIGS.items():
        # Create model to get parameter count
        model = create_beats_model(variant_name)
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        comparison[variant_name] = {
            'parameters': param_count,
            'trainable_parameters': trainable_params,
            'embed_dim': config['embed_dim'],
            'encoder_depth': config['encoder_depth'],
            'num_heads': config['num_heads'],
            'tokenizer_vocab_size': config['tokenizer_vocab_size'],
            'description': config['description']
        }
    
    return comparison


# Version and compatibility info
__version__ = "1.0.0"
__beats_paper_url__ = "https://arxiv.org/abs/2212.09058"
__compatible_torch_versions__ = [">=1.12.0"]
__supported_audio_formats__ = ["wav", "mp3", "flac", "m4a"]

# Export key constants
TOKENIZER_VOCAB_SIZES = {
    'tiny': 2048,
    'small': 4096, 
    'base': 8192,
    'large': 8192
}

RECOMMENDED_PATCH_SIZES = {
    (128, 128): (16, 16),
    (224, 224): (16, 16),
    (256, 256): (32, 32)
}