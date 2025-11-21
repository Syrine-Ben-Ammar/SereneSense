#!/usr/bin/env python3
#
# Plan:
# 1. Create comprehensive model training script for SereneSense
# 2. Support for all three models (AudioMAE, AST, BEATs)
# 3. Configurable training pipeline with YAML configs
# 4. Multi-GPU training support with distributed training
# 5. Experiment tracking with MLflow/Weights & Biases
# 6. Checkpoint management and resumable training
# 7. Comprehensive logging and progress tracking
#

"""
SereneSense Model Training Script
Trains military vehicle sound detection models with state-of-the-art architectures.

Usage:
    python scripts/train_model.py --config configs/training/audioMAE.yaml
    python scripts/train_model.py --model ast --epochs 50 --batch-size 32
    python scripts/train_model.py --resume checkpoints/audioMAE_epoch_10.pth
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.training.trainer import SereneSenseTrainer, TrainingConfig
from core.models.audioMAE.model import AudioMAE, AudioMAEConfig
try:
    from core.models.ast.model import ASTModel
except ImportError:
    ASTModel = None
try:
    from core.models.beats.model import BEATsModel
except ImportError:
    BEATsModel = None
from core.data.loaders.mad_loader import MADDataset
from core.utils.config_parser import ConfigParser
from core.utils.device_utils import get_optimal_device
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class SpectrogramResizeTransform:
    """Wrapper to resize spectrograms to fixed size"""
    def __init__(self, spec_transform, target_size=(128, 128)):
        self.spec_transform = spec_transform
        self.target_size = target_size

    def __call__(self, audio):
        import torch.nn.functional as F
        # Generate spectrogram
        spec = self.spec_transform(audio)  # [C, H, W]

        # Resize to target size
        if spec.shape[-2:] != self.target_size:
            spec = F.interpolate(
                spec.unsqueeze(0),  # [1, C, H, W]
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [C, H, W]

        return spec


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train SereneSense military vehicle detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--config", type=str, help="Path to training configuration YAML file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["audioMAE", "ast", "beats"],
        default="audioMAE",
        help="Model architecture to train",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay coefficient")

    # Data configuration
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory for datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mad", "audioset", "combined"],
        default="mad",
        help="Dataset to use for training",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument("--experiment-name", type=str, help="Name for this training experiment")
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # Resume training
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model weights")

    # Distributed training
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--local-rank", type=int, default=0, help="Local rank for distributed training"
    )

    # Optimization
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Enable mixed precision training"
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile model with torch.compile (PyTorch 2.0+)",
    )

    # Validation and testing
    parser.add_argument("--val-interval", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument(
        "--early-stopping", type=int, default=20, help="Early stopping patience (0 to disable)"
    )

    # Logging and monitoring
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow experiment tracking")

    # Device configuration
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    )

    # Debugging
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with detailed logging"
    )
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without actual training (for testing)"
    )

    return parser.parse_args()


def load_configuration(args):
    """
    Load and merge configuration from file and command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Merged configuration dictionary
    """
    # Start with default configuration
    config = {
        "model": {
            "architecture": args.model,
            "num_classes": 7,  # Standard military vehicle classes
            "input_shape": [1, 128, 128],  # Mel-spectrogram shape
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "mixed_precision": args.mixed_precision,
            "early_stopping_patience": args.early_stopping if args.early_stopping > 0 else None,
        },
        "data": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "num_workers": args.num_workers,
            "pin_memory": True,
        },
        "output": {
            "output_dir": args.output_dir,
            "save_interval": args.save_interval,
            "val_interval": args.val_interval,
        },
        "device": args.device,
        "distributed": args.distributed,
    }

    parser = ConfigParser()
    # Load configuration from file if provided
    if args.config:
        file_config = parser.load_yaml(args.config)
        # Deep merge configurations (file config takes precedence)
        config = parser.merge_configs(config, file_config)

    # Override with command line arguments (highest precedence)
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name

    return config


def create_model(config):
    """
    Create model based on configuration.

    Args:
        config: Model configuration

    Returns:
        PyTorch model instance
    """
    model_cfg = config["model"]
    arch_cfg = model_cfg.get("architecture", {})
    model_type = model_cfg.get("name", model_cfg.get("type", "audioMAE"))
    if isinstance(model_type, dict):
        model_type = model_type.get("name", "audioMAE")
    model_type = model_type.lower()

    classification_cfg = model_cfg.get("classification", {})
    num_classes = classification_cfg.get("num_classes", model_cfg.get("num_classes", 7))

    logger.info(f"Creating {model_type} model with {num_classes} classes")

    if model_type in ("audiomae", "audio_masked_autoencoder"):
        encoder_cfg = arch_cfg.get("encoder", {})
        masking_cfg = arch_cfg.get("masking", {})
        decoder_cfg = arch_cfg.get("decoder", {})
        patch_size = encoder_cfg.get("patch_size", 16)
        if isinstance(patch_size, (list, tuple)):
            patch_size = tuple(patch_size)
        img_shape = model_cfg.get("input_shape", [1, 128, 128])
        img_size = tuple(img_shape[1:]) if len(img_shape) >= 3 else (128, 128)
        audio_cfg = AudioMAEConfig(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size[0] if isinstance(patch_size, tuple) else patch_size,
            in_chans=encoder_cfg.get("in_channels", 1),
            embed_dim=encoder_cfg.get("embed_dim", 768),
            encoder_depth=encoder_cfg.get("depth", 12),
            encoder_num_heads=encoder_cfg.get("num_heads", 12),
            decoder_embed_dim=decoder_cfg.get("embed_dim", 512),
            decoder_depth=decoder_cfg.get("depth", 8),
            decoder_num_heads=decoder_cfg.get("num_heads", 16),
            mlp_ratio=encoder_cfg.get("mlp_ratio", 4.0),
            dropout=encoder_cfg.get("dropout", 0.0),
            attention_dropout=encoder_cfg.get("attention_dropout", 0.0),
            drop_path=encoder_cfg.get("drop_path", 0.1),
            mask_ratio=masking_cfg.get("mask_ratio", 0.75),
            norm_pix_loss=model_cfg.get("pretraining", {}).get("norm_pix_loss", True),
        )
        model = AudioMAE(audio_cfg)

    elif model_type == "ast":
        if ASTModel is None:
            raise ImportError("ASTModel is not available. Install or implement core.models.ast.model.")
        encoder_cfg = arch_cfg.get("encoder", {})
        patch_size = encoder_cfg.get("patch_size", 16)
        if isinstance(patch_size, (list, tuple)):
            patch_size = tuple(patch_size)
        model = ASTModel(
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=encoder_cfg.get("embed_dim", 768),
            num_heads=encoder_cfg.get("num_heads", 12),
            num_layers=encoder_cfg.get("depth", 12),
        )

    elif model_type == "beats":
        if BEATsModel is None:
            raise ImportError("BEATsModel is not available. Install or implement core.models.beats.model.")
        encoder_cfg = arch_cfg.get("encoder", {})
        model = BEATsModel(
            num_classes=num_classes,
            embed_dim=encoder_cfg.get("embed_dim", 768),
            num_heads=encoder_cfg.get("num_heads", 12),
            num_layers=encoder_cfg.get("depth", 12),
        )

    else:
        raise ValueError(f"Unknown model architecture: {model_type}")

    return model


def create_datasets(config):
    """
    Create training and validation datasets.

    Args:
        config: Data configuration

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset_type = config["data"]["dataset"]
    data_dir = config["data"].get("data_dir", "data/processed/mad")

    logger.info(f"Creating {dataset_type} datasets from {data_dir}")

    # Create spectrogram transform
    from core.data.preprocessing.spectrograms import MelSpectrogramGenerator

    # Get audio config from model config
    audio_config = config.get("model", {}).get("audio", {})
    spec_config = audio_config.get("spectrogram", {})

    mel_spec_transform = MelSpectrogramGenerator(
        sample_rate=audio_config.get("sample_rate", 16000),
        n_fft=spec_config.get("n_fft", 1024),
        hop_length=spec_config.get("hop_length", 160),
        n_mels=spec_config.get("n_mels", 128),
        fmin=spec_config.get("f_min", 50),
        fmax=spec_config.get("f_max", 8000),
        power=spec_config.get("power", 2.0),
        normalized=True,
        use_gpu=False  # Don't use GPU in dataset transform (pin_memory issue)
    )

    # Wrap with resize transform
    transform = SpectrogramResizeTransform(mel_spec_transform, target_size=(128, 128))

    if dataset_type == "mad":
        train_dataset = MADDataset(
            data_dir=data_dir,
            split="train",
            transform=transform
        )
        val_dataset = MADDataset(
            data_dir=data_dir,
            split="val",
            transform=transform
        )

    else:
        raise ValueError(f"Dataset '{dataset_type}' not supported for this training run.")

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_experiment_tracking(config, args):
    """
    Setup experiment tracking (Weights & Biases, MLflow).

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    experiment_name = (
        config.get("experiment_name")
        or f"{config['model']['architecture']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Setup Weights & Biases
    if args.wandb:
        try:
            import wandb

            wandb.init(
                project="serenesense",
                name=experiment_name,
                config=config,
                tags=[config["model"]["architecture"], config["data"]["dataset"]],
            )

            logger.info("Weights & Biases tracking initialized")

        except ImportError:
            logger.warning("Weights & Biases not available (install with: pip install wandb)")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")

    # Setup MLflow
    if args.mlflow:
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            mlflow.start_run()

            # Log configuration
            for key, value in config.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        mlflow.log_param(f"{key}.{subkey}", subvalue)
                else:
                    mlflow.log_param(key, value)

            logger.info("MLflow tracking initialized")

        except ImportError:
            logger.warning("MLflow not available (install with: pip install mlflow)")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()

    # Setup logging - always enable console output
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=args.log_level, console_output=True, file_output=True)

    logger.info("Starting SereneSense model training")
    logger.info(f"Arguments: {vars(args)}")

    # Load configuration
    config = load_configuration(args)
    logger.info(f"Configuration loaded: {json.dumps(config, indent=2, default=str)}")

    # Distributed training not supported in this lightweight run
    if args.distributed:
        raise NotImplementedError("Distributed training is currently disabled for train_model.py")

    # Determine device
    if args.device and args.device.lower() != "auto":
        device_str = args.device
    else:
        device_str = get_optimal_device(task="training")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, default=str)

    try:
        # Create datasets
        train_dataset, val_dataset = create_datasets(config)

        # Setup experiment tracking
        setup_experiment_tracking(config, args)

        # Create training configuration
        # Determine model type from config
        model_type = config["model"].get("type", config["model"].get("name", "audioMAE")).lower()
        # Map config type to trainer expected types
        if "audiomae" in model_type or "audio_masked" in model_type:
            model_type = "audiomae"
        elif "ast" in model_type or "audio_spectrogram" in model_type:
            model_type = "ast"
        elif "beats" in model_type:
            model_type = "beats"

        training_config = TrainingConfig(
            model_type=model_type,
            model_config_path=args.config,
            pretrained_path=args.pretrained,
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            save_dir=str(output_dir),
            save_every_n_epochs=config["output"]["save_interval"],
            validate_every_n_epochs=config["output"]["val_interval"],
            mixed_precision=config["training"]["mixed_precision"],
            early_stopping=False,  # Disable early stopping to train full epochs
            patience=config["training"].get("early_stopping_patience", 15),
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
        )

        # Create trainer
        trainer = SereneSenseTrainer(training_config)

        # Create data loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"].get("validation_batch_size", config["training"]["batch_size"]),
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
        )

        # Setup training with dataloaders
        trainer.setup_training(train_loader, val_loader)

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            training_config.resume_from_checkpoint = args.resume

        # Dry run mode (for testing)
        if args.dry_run:
            logger.info("Dry run mode - skipping actual training")
            return

        # Start training
        logger.info("🏋️ Starting model training...")

        # Performance profiling if requested
        if args.profile:
            logger.warning("Profiling is not yet integrated with the trainer")

        # Train the model (trainer manages the model internally)
        trainer.train()

        # Training completed
        logger.info("✅ Training completed successfully!")

        # Final model is saved automatically by trainer
        if trainer.state.best_checkpoint_path:
            logger.info(f"Best model saved to: {trainer.state.best_checkpoint_path}")
        if trainer.state.last_checkpoint_path:
            logger.info(f"Last checkpoint saved to: {trainer.state.last_checkpoint_path}")

        # Close experiment tracking
        if args.wandb:
            try:
                import wandb

                wandb.finish()
            except:
                pass

        if args.mlflow:
            try:
                import mlflow

                mlflow.end_run()
            except:
                pass

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup distributed training
        if args.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
