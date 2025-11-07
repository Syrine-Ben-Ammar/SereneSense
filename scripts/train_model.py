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
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.training.trainer import SereneSenseTrainer, TrainingConfig
from core.models.audioMAE.model import AudioMAE
from core.models.AST.model import ASTModel
from core.models.BEATs.model import BEATsModel
from core.data.loaders.mad_loader import MADDataset
from core.data.loaders.audioset_loader import AudioSetLoader
from core.utils.config_parser import ConfigParser
from core.utils.device_utils import get_optimal_device, setup_distributed_training
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


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

    # Load configuration from file if provided
    if args.config:
        file_config = ConfigParser.load_config(args.config)
        # Deep merge configurations (file config takes precedence)
        config = ConfigParser.merge_configs(config, file_config)

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
    model_type = config["model"]["architecture"].lower()
    num_classes = config["model"]["num_classes"]

    logger.info(f"Creating {model_type} model with {num_classes} classes")

    if model_type == "audioMAE":
        model = AudioMAE(
            num_classes=num_classes,
            patch_size=config["model"].get("patch_size", 16),
            embed_dim=config["model"].get("embed_dim", 768),
            num_heads=config["model"].get("num_heads", 12),
            num_layers=config["model"].get("num_layers", 12),
            masking_ratio=config["model"].get("masking_ratio", 0.75),
        )

    elif model_type == "ast":
        model = ASTModel(
            num_classes=num_classes,
            patch_size=config["model"].get("patch_size", 16),
            embed_dim=config["model"].get("embed_dim", 768),
            num_heads=config["model"].get("num_heads", 12),
            num_layers=config["model"].get("num_layers", 12),
        )

    elif model_type == "beats":
        model = BEATsModel(
            num_classes=num_classes,
            embed_dim=config["model"].get("embed_dim", 768),
            num_heads=config["model"].get("num_heads", 12),
            num_layers=config["model"].get("num_layers", 12),
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
    data_dir = config["data"]["data_dir"]

    logger.info(f"Creating {dataset_type} datasets from {data_dir}")

    if dataset_type == "mad":
        train_dataset = MADDataset(
            data_dir=data_dir, split="train", transform=config["data"].get("train_transform")
        )
        val_dataset = MADDataset(
            data_dir=data_dir, split="val", transform=config["data"].get("val_transform")
        )

    elif dataset_type == "audioset":
        loader = AudioSetLoader(data_dir)
        train_dataset = loader.get_dataset(split="train")
        val_dataset = loader.get_dataset(split="val")

    elif dataset_type == "combined":
        # Combine MAD and AudioSet datasets
        mad_train = MADDataset(data_dir, split="train")
        audioset_train = AudioSetLoader(data_dir).get_dataset(split="train")

        # Create combined dataset
        from torch.utils.data import ConcatDataset

        train_dataset = ConcatDataset([mad_train, audioset_train])

        # Use MAD validation set
        val_dataset = MADDataset(data_dir, split="val")

    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

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

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level, debug=args.debug)

    logger.info("🚀 Starting SereneSense model training")
    logger.info(f"Arguments: {vars(args)}")

    # Load configuration
    config = load_configuration(args)
    logger.info(f"Configuration loaded: {json.dumps(config, indent=2, default=str)}")

    # Setup distributed training if requested
    if args.distributed:
        setup_distributed_training(args.local_rank)
        logger.info(f"Distributed training setup complete (rank {args.local_rank})")

    # Determine device
    device = get_optimal_device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, default=str)

    try:
        # Create model
        model = create_model(config)
        logger.info(f"Model created: {model.__class__.__name__}")

        # Load pretrained weights if specified
        if args.pretrained:
            logger.info(f"Loading pretrained weights from {args.pretrained}")
            pretrained_state = torch.load(args.pretrained, map_location="cpu")
            model.load_state_dict(pretrained_state, strict=False)

        # Move model to device
        model = model.to(device)

        # Compile model if requested (PyTorch 2.0+)
        if args.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        # Setup distributed data parallel if needed
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank])
            logger.info("Model wrapped with DistributedDataParallel")

        # Create datasets
        train_dataset, val_dataset = create_datasets(config)

        # Setup experiment tracking
        setup_experiment_tracking(config, args)

        # Create training configuration
        training_config = TrainingConfig(
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            device=device,
            output_dir=str(output_dir),
            save_interval=config["output"]["save_interval"],
            val_interval=config["output"]["val_interval"],
            mixed_precision=config["training"]["mixed_precision"],
            early_stopping_patience=config["training"].get("early_stopping_patience"),
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            distributed=args.distributed,
        )

        # Create trainer
        trainer = SereneSenseTrainer(training_config)

        # Setup datasets
        trainer.setup_data(train_dataset, val_dataset)

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load model state
            if args.distributed:
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed from epoch {start_epoch}")

        # Dry run mode (for testing)
        if args.dry_run:
            logger.info("Dry run mode - skipping actual training")
            return

        # Start training
        logger.info("🏋️ Starting model training...")

        # Performance profiling if requested
        if args.profile:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(output_dir / "profiler")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                trainer.train(model, start_epoch=start_epoch, profiler=prof)
        else:
            trainer.train(model, start_epoch=start_epoch)

        # Training completed
        logger.info("✅ Training completed successfully!")

        # Save final model
        final_model_path = output_dir / "final_model.pth"
        if args.distributed:
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)

        logger.info(f"Final model saved to: {final_model_path}")

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
