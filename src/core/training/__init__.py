"""
SereneSense Training Module

This module provides comprehensive training infrastructure for the SereneSense
military vehicle sound detection system. It includes advanced training strategies,
loss functions, optimizers, and schedulers optimized for audio transformers.

Key Features:
- Multi-model training support (AudioMAE, AST, BEATs)
- Advanced loss functions for audio classification
- Sophisticated optimization strategies
- Learning rate scheduling and warmup
- Mixed precision training support
- Distributed training capabilities
- Experiment tracking integration

Components:
    - trainer: Main training orchestrator with advanced features
    - loss_functions: Specialized loss functions for audio classification
    - optimizers: Custom optimizers and parameter group management
    - schedulers: Learning rate schedulers with warmup and decay
    - callbacks: Training callbacks for monitoring and control
    - metrics: Training metrics calculation and logging

Performance Optimizations:
- Automatic mixed precision (AMP) for faster training
- Gradient accumulation for large effective batch sizes
- Dynamic loss scaling for numerical stability
- Efficient data loading with prefetching
- Memory optimization for large models
"""

try:
    from .trainer import SereneSenseTrainer, TrainingConfig, TrainingState, ModelWrapper
except ImportError:
    SereneSenseTrainer = None
    TrainingConfig = None
    TrainingState = None
    ModelWrapper = None

# Loss functions (optional)
try:
    from .loss_functions import (
        LabelSmoothingCrossEntropy,
        FocalLoss,
        ArcFaceLoss,
        SpectralConvergenceLoss,
        MultiResolutionSTFTLoss,
        PerceptualLoss,
        MixupLoss,
        CutMixLoss,
        KnowledgeDistillationLoss,
        CombinedLoss,
        AdaptiveLossWeighting,
    )
except ImportError:
    LabelSmoothingCrossEntropy = None
    FocalLoss = None
    ArcFaceLoss = None
    SpectralConvergenceLoss = None
    MultiResolutionSTFTLoss = None
    PerceptualLoss = None
    MixupLoss = None
    CutMixLoss = None
    KnowledgeDistillationLoss = None
    CombinedLoss = None
    AdaptiveLossWeighting = None

# Optimizers (optional)
try:
    from .optimizers import (
        AdamWWithDecoupledWeightDecay,
        SAMOptimizer,
        LookaheadOptimizer,
        LayerWiseDecayParameterGroups,
        TransformerParameterGroups,
        AudioModelParameterGroups,
        OptimizerFactory,
        GradientClipping,
        ParameterStatistics,
    )
except ImportError:
    AdamWWithDecoupledWeightDecay = None
    SAMOptimizer = None
    LookaheadOptimizer = None
    LayerWiseDecayParameterGroups = None
    TransformerParameterGroups = None
    AudioModelParameterGroups = None
    OptimizerFactory = None
    GradientClipping = None
    ParameterStatistics = None

# Schedulers (optional)
try:
    from .schedulers import (
        CosineAnnealingWithWarmup,
        LinearWarmupScheduler,
        ExponentialWarmupScheduler,
        PolynomialDecayScheduler,
        OneCycleLRScheduler,
        CyclicLRWithWarmup,
        ReduceLROnPlateauWithWarmup,
        SchedulerFactory,
        WarmupWrapper,
        SchedulerState,
    )
except ImportError:
    CosineAnnealingWithWarmup = None
    LinearWarmupScheduler = None
    ExponentialWarmupScheduler = None
    PolynomialDecayScheduler = None
    OneCycleLRScheduler = None
    CyclicLRWithWarmup = None
    ReduceLROnPlateauWithWarmup = None
    SchedulerFactory = None
    WarmupWrapper = None
    SchedulerState = None

# Callbacks (optional)
try:
    from .callbacks import (
        EarlyStoppingCallback,
        ModelCheckpointCallback,
        MetricsLoggerCallback,
        LearningRateLoggerCallback,
        GradientNormCallback,
        ActivationStatisticsCallback,
        AttentionVisualizationCallback,
        WandBCallback,
        MLflowCallback,
        TensorBoardCallback,
    )
except ImportError:
    EarlyStoppingCallback = None
    ModelCheckpointCallback = None
    MetricsLoggerCallback = None
    LearningRateLoggerCallback = None
    GradientNormCallback = None
    ActivationStatisticsCallback = None
    AttentionVisualizationCallback = None
    WandBCallback = None
    MLflowCallback = None
    TensorBoardCallback = None

# Metrics (optional)
try:
    from .metrics import (
        TrainingMetrics,
        ValidationMetrics,
        AudioClassificationMetrics,
        AccuracyCalculator,
        F1ScoreCalculator,
        AUCCalculator,
        TopKAccuracyCalculator,
        SpectralMetrics,
        PerceptualMetrics,
        DiversityMetrics,
    )
except ImportError:
    TrainingMetrics = None
    ValidationMetrics = None
    AudioClassificationMetrics = None
    AccuracyCalculator = None
    F1ScoreCalculator = None
    AUCCalculator = None
    TopKAccuracyCalculator = None
    SpectralMetrics = None
    PerceptualMetrics = None
    DiversityMetrics = None

__all__ = [
    # Core training components
    "SereneSenseTrainer",
    "TrainingConfig",
    "TrainingState",
    "ModelWrapper",
    # Loss functions
    "LabelSmoothingCrossEntropy",
    "FocalLoss",
    "ArcFaceLoss",
    "SpectralConvergenceLoss",
    "MultiResolutionSTFTLoss",
    "PerceptualLoss",
    "MixupLoss",
    "CutMixLoss",
    "KnowledgeDistillationLoss",
    "CombinedLoss",
    "AdaptiveLossWeighting",
    # Optimizers
    "AdamWWithDecoupledWeightDecay",
    "SAMOptimizer",
    "LookaheadOptimizer",
    "LayerWiseDecayParameterGroups",
    "TransformerParameterGroups",
    "AudioModelParameterGroups",
    "OptimizerFactory",
    "GradientClipping",
    "ParameterStatistics",
    # Schedulers
    "CosineAnnealingWithWarmup",
    "LinearWarmupScheduler",
    "ExponentialWarmupScheduler",
    "PolynomialDecayScheduler",
    "OneCycleLRScheduler",
    "CyclicLRWithWarmup",
    "ReduceLROnPlateauWithWarmup",
    "SchedulerFactory",
    "WarmupWrapper",
    "SchedulerState",
    # Callbacks
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "MetricsLoggerCallback",
    "LearningRateLoggerCallback",
    "GradientNormCallback",
    "ActivationStatisticsCallback",
    "AttentionVisualizationCallback",
    "WandBCallback",
    "MLflowCallback",
    "TensorBoardCallback",
    # Metrics
    "TrainingMetrics",
    "ValidationMetrics",
    "AudioClassificationMetrics",
    "AccuracyCalculator",
    "F1ScoreCalculator",
    "AUCCalculator",
    "TopKAccuracyCalculator",
    "SpectralMetrics",
    "PerceptualMetrics",
    "DiversityMetrics",
]

# Training configuration presets
TRAINING_PRESETS = {
    "audioMAE_base": {
        "model_type": "audioMAE",
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine_warmup",
        "warmup_epochs": 10,
        "batch_size": 32,
        "mixed_precision": True,
        "gradient_clip_norm": 1.0,
    },
    "ast_base": {
        "model_type": "ast",
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine_warmup",
        "warmup_epochs": 5,
        "batch_size": 48,
        "mixed_precision": True,
        "gradient_clip_norm": 1.0,
    },
    "beats_base": {
        "model_type": "beats",
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine_warmup",
        "warmup_epochs": 10,
        "batch_size": 32,
        "mixed_precision": True,
        "gradient_clip_norm": 1.0,
    },
    "edge_optimized": {
        "model_type": "ast",
        "model_variant": "small",
        "optimizer": "adamw",
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine_warmup",
        "warmup_epochs": 5,
        "batch_size": 64,
        "mixed_precision": True,
        "gradient_clip_norm": 0.5,
        "quantization_aware_training": True,
    },
    "few_shot": {
        "model_type": "beats",
        "optimizer": "adamw",
        "learning_rate": 5e-5,
        "weight_decay": 1e-5,
        "scheduler": "linear_warmup",
        "warmup_epochs": 3,
        "batch_size": 16,
        "mixed_precision": True,
        "gradient_clip_norm": 0.5,
        "support_shots": 5,
        "query_shots": 15,
    },
}


def get_training_preset(preset_name: str, **overrides):
    """
    Get a training configuration preset with optional overrides.

    Args:
        preset_name: Name of the preset configuration
        **overrides: Parameters to override in the preset

    Returns:
        Training configuration dictionary
    """
    if preset_name not in TRAINING_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(TRAINING_PRESETS.keys())}"
        )

    config = TRAINING_PRESETS[preset_name].copy()
    config.update(overrides)
    return config


def create_trainer(
    model_type: str, config_path: str = None, preset: str = None, **kwargs
) -> "SereneSenseTrainer":
    """
    Create a SereneSense trainer with specified configuration.

    Args:
        model_type: Type of model ('audioMAE', 'ast', 'beats')
        config_path: Path to training configuration file
        preset: Name of training preset to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured SereneSense trainer
    """
    from ..utils.config_parser import ConfigParser

    # Load configuration
    if config_path:
        config_manager = ConfigParser()
        training_config = config_manager.load_config(config_path)
    elif preset:
        training_config = get_training_preset(preset, **kwargs)
    else:
        # Use default configuration for model type
        preset_name = f"{model_type}_base"
        training_config = get_training_preset(preset_name, **kwargs)

    # Create trainer
    return SereneSenseTrainer(TrainingConfig(**training_config))


# Performance optimization settings
PERFORMANCE_CONFIGS = {
    "memory_efficient": {
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "non_blocking": True,
    },
    "speed_optimized": {
        "mixed_precision": True,
        "compile_model": True,
        "channels_last": True,
        "dataloader_num_workers": 8,
        "prefetch_factor": 2,
    },
    "distributed": {
        "distributed_backend": "nccl",
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "ddp_bucket_cap_mb": 25,
    },
}

# Version information
__version__ = "1.0.0"
__author__ = "SereneSense Team"
__description__ = "Advanced training infrastructure for military vehicle sound detection"
