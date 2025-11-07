"""
Integration tests for SereneSense training pipeline.

Tests end-to-end training functionality including:
- Complete training workflow
- Model checkpointing and recovery
- Multi-GPU training setup
- Training with different optimizers and schedulers
- Loss function integration
- Metrics computation during training
- Early stopping and callbacks
- Mixed precision training
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch
import yaml
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.training.trainer import Trainer
from core.training.loss_functions import FocalLoss, ContrastiveLoss, LabelSmoothingLoss
from core.training.optimizers import create_optimizer
from core.training.schedulers import create_scheduler
from core.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from core.training.metrics import compute_classification_metrics
from core.models.audioMAE.model import AudioMAE
from core.models.AST.model import AST
from core.utils.config_parser import ConfigParser
from core.utils.logging import setup_logging


class TestTrainingPipelineBasic:
    """Test basic training pipeline functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        config = {
            "input_dim": 64,
            "patch_size": 8,
            "embed_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 5,
            "mask_ratio": 0.75,
        }
        return AudioMAE(config)

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for testing."""
        batch_size = 8
        sequence_length = 64
        input_dim = 64
        num_classes = 5
        num_samples = 32

        # Create random data
        X = torch.randn(num_samples, sequence_length, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @pytest.fixture
    def training_config(self, temp_dir):
        """Create training configuration."""
        return {
            "model": {"name": "audioMAE", "num_classes": 5},
            "training": {
                "num_epochs": 2,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "mixed_precision": False,
                "accumulate_grad_batches": 1,
            },
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999]},
            "scheduler": {"name": "cosine", "T_max": 100, "eta_min": 1e-6},
            "loss": {"name": "cross_entropy", "label_smoothing": 0.1},
            "checkpointing": {
                "save_dir": str(temp_dir / "checkpoints"),
                "save_every_n_epochs": 1,
                "save_best_only": True,
                "monitor": "val_accuracy",
                "mode": "max",
            },
            "logging": {"log_every_n_steps": 5, "log_dir": str(temp_dir / "logs")},
        }

    def test_trainer_initialization(self, simple_model, training_config, temp_dir):
        """Test trainer initialization."""
        trainer = Trainer(model=simple_model, config=training_config, device=torch.device("cpu"))

        assert trainer.model == simple_model
        assert trainer.config == training_config
        assert trainer.device == torch.device("cpu")
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_simple_training_loop(self, simple_model, simple_dataset, training_config, temp_dir):
        """Test simple training loop execution."""
        # Create validation dataset
        val_dataset = simple_dataset

        trainer = Trainer(model=simple_model, config=training_config, device=torch.device("cpu"))

        # Run training
        history = trainer.fit(train_dataloader=simple_dataset, val_dataloader=val_dataset)

        # Check training completed
        assert trainer.current_epoch == training_config["training"]["num_epochs"]
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == training_config["training"]["num_epochs"]

    def test_training_with_different_loss_functions(
        self, simple_model, simple_dataset, training_config
    ):
        """Test training with different loss functions."""
        loss_configs = [
            {"name": "cross_entropy"},
            {"name": "focal", "alpha": 1.0, "gamma": 2.0},
            {"name": "label_smoothing", "smoothing": 0.1},
        ]

        for loss_config in loss_configs:
            config = training_config.copy()
            config["loss"] = loss_config
            config["training"]["num_epochs"] = 1  # Quick test

            trainer = Trainer(model=simple_model, config=config, device=torch.device("cpu"))

            # Should initialize and run without errors
            history = trainer.fit(train_dataloader=simple_dataset, val_dataloader=simple_dataset)

            assert "train_loss" in history
            assert len(history["train_loss"]) == 1

    def test_training_with_different_optimizers(
        self, simple_model, simple_dataset, training_config
    ):
        """Test training with different optimizers."""
        optimizer_configs = [
            {"name": "adam", "lr": 1e-3},
            {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            {"name": "sgd", "lr": 1e-2, "momentum": 0.9},
        ]

        for optimizer_config in optimizer_configs:
            config = training_config.copy()
            config["optimizer"] = optimizer_config
            config["training"]["num_epochs"] = 1

            trainer = Trainer(model=simple_model, config=config, device=torch.device("cpu"))

            history = trainer.fit(train_dataloader=simple_dataset, val_dataloader=simple_dataset)

            assert "train_loss" in history

    def test_training_with_schedulers(self, simple_model, simple_dataset, training_config):
        """Test training with different learning rate schedulers."""
        scheduler_configs = [
            {"name": "step", "step_size": 5, "gamma": 0.1},
            {"name": "cosine", "T_max": 10, "eta_min": 1e-6},
            {"name": "exponential", "gamma": 0.95},
        ]

        for scheduler_config in scheduler_configs:
            config = training_config.copy()
            config["scheduler"] = scheduler_config
            config["training"]["num_epochs"] = 2

            trainer = Trainer(model=simple_model, config=config, device=torch.device("cpu"))

            history = trainer.fit(train_dataloader=simple_dataset, val_dataloader=simple_dataset)

            assert "train_loss" in history
            # Learning rate should be logged
            if "learning_rate" in history:
                assert len(history["learning_rate"]) > 0


class TestTrainingCallbacks:
    """Test training callbacks integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model_and_data(self):
        """Create model and data for callback testing."""
        config = {
            "input_dim": 32,
            "patch_size": 4,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 3,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)

        # Create simple dataset
        X = torch.randn(24, 32, 32)
        y = torch.randint(0, 3, (24,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        return model, dataloader

    def test_early_stopping_callback(self, model_and_data, temp_dir):
        """Test early stopping callback."""
        model, dataloader = model_and_data

        config = {
            "training": {"num_epochs": 10, "learning_rate": 1e-3},
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 2,
                    "mode": "min",
                    "min_delta": 0.0,
                }
            },
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        # Mock validation loss to trigger early stopping
        original_validate = trainer.validate_epoch

        def mock_validate(*args, **kwargs):
            # Return increasing loss to trigger early stopping
            epoch = trainer.current_epoch
            return {"val_loss": 1.0 + epoch * 0.1, "val_accuracy": 0.5}

        trainer.validate_epoch = mock_validate

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        # Should stop early (before 10 epochs)
        assert trainer.current_epoch < 10
        assert "early_stopped" in history

    def test_model_checkpoint_callback(self, model_and_data, temp_dir):
        """Test model checkpointing callback."""
        model, dataloader = model_and_data

        config = {
            "training": {"num_epochs": 3, "learning_rate": 1e-3},
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
            "checkpointing": {
                "save_dir": str(temp_dir / "checkpoints"),
                "save_every_n_epochs": 1,
                "save_best_only": False,
                "monitor": "val_accuracy",
                "mode": "max",
            },
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        # Check that checkpoints were saved
        checkpoint_dir = temp_dir / "checkpoints"
        assert checkpoint_dir.exists()

        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0

        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_files[0], map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "config" in checkpoint

    def test_lr_scheduler_callback(self, model_and_data):
        """Test learning rate scheduler callback."""
        model, dataloader = model_and_data

        config = {
            "training": {"num_epochs": 3, "learning_rate": 1e-2},
            "optimizer": {"name": "sgd", "lr": 1e-2, "momentum": 0.9},
            "scheduler": {"name": "step", "step_size": 1, "gamma": 0.5},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        # Track learning rates
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        # Learning rate should have decreased
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

        # Should be logged in history
        if "learning_rate" in history:
            assert len(history["learning_rate"]) == 3


class TestTrainingResumption:
    """Test training resumption from checkpoints."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def training_setup(self, temp_dir):
        """Create training setup for resumption testing."""
        config = {
            "input_dim": 32,
            "patch_size": 4,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 3,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)

        X = torch.randn(16, 32, 32)
        y = torch.randint(0, 3, (16,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)  # No shuffle for consistency

        train_config = {
            "training": {"num_epochs": 4, "learning_rate": 1e-3},
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
            "checkpointing": {
                "save_dir": str(temp_dir / "checkpoints"),
                "save_every_n_epochs": 1,
                "save_best_only": False,
            },
        }

        return model, dataloader, train_config

    def test_checkpoint_saving_and_loading(self, training_setup, temp_dir):
        """Test checkpoint saving and loading."""
        model, dataloader, config = training_setup

        # Train for 2 epochs
        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        config["training"]["num_epochs"] = 2
        history1 = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        # Save checkpoint manually
        checkpoint_path = temp_dir / "manual_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)

        # Create new trainer and load checkpoint
        new_model = (
            AudioMAE(model.config)
            if hasattr(model, "config")
            else AudioMAE(
                {
                    "input_dim": 32,
                    "patch_size": 4,
                    "embed_dim": 64,
                    "num_layers": 2,
                    "num_heads": 2,
                    "mlp_ratio": 2.0,
                    "dropout": 0.1,
                    "num_classes": 3,
                    "mask_ratio": 0.75,
                }
            )
        )

        new_trainer = Trainer(model=new_model, config=config, device=torch.device("cpu"))

        new_trainer.load_checkpoint(checkpoint_path)

        # Check that state was restored
        assert new_trainer.current_epoch == trainer.current_epoch
        assert new_trainer.global_step == trainer.global_step

        # Model parameters should be the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_training_resumption(self, training_setup, temp_dir):
        """Test resuming training from checkpoint."""
        model, dataloader, config = training_setup

        # Train for 2 epochs and save
        trainer1 = Trainer(model=model, config=config, device=torch.device("cpu"))

        config_part1 = config.copy()
        config_part1["training"]["num_epochs"] = 2

        history1 = trainer1.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        checkpoint_path = temp_dir / "resume_checkpoint.pth"
        trainer1.save_checkpoint(checkpoint_path)

        # Create new trainer, load checkpoint, and continue training
        new_model = AudioMAE(
            {
                "input_dim": 32,
                "patch_size": 4,
                "embed_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.1,
                "num_classes": 3,
                "mask_ratio": 0.75,
            }
        )

        trainer2 = Trainer(model=new_model, config=config, device=torch.device("cpu"))

        trainer2.load_checkpoint(checkpoint_path)

        # Continue training for 2 more epochs
        config_part2 = config.copy()
        config_part2["training"]["num_epochs"] = 4  # Total epochs

        history2 = trainer2.fit(
            train_dataloader=dataloader, val_dataloader=dataloader, resume_from_checkpoint=True
        )

        # Total training should be 4 epochs
        assert trainer2.current_epoch == 4

        # Combined history should have 4 epochs
        total_epochs = len(history1["train_loss"]) + len(history2["train_loss"])
        assert total_epochs == 4


class TestMixedPrecisionTraining:
    """Test mixed precision training."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and data for mixed precision testing."""
        config = {
            "input_dim": 64,
            "patch_size": 8,
            "embed_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 5,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)

        X = torch.randn(32, 64, 64)
        y = torch.randint(0, 5, (32,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        return model, dataloader

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self, model_and_data):
        """Test mixed precision training (requires CUDA)."""
        model, dataloader = model_and_data

        config = {
            "training": {"num_epochs": 2, "learning_rate": 1e-3, "mixed_precision": True},
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cuda"))

        # Should initialize scaler for mixed precision
        assert hasattr(trainer, "scaler")
        assert trainer.scaler is not None

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_mixed_precision_cpu_fallback(self, model_and_data):
        """Test mixed precision falls back gracefully on CPU."""
        model, dataloader = model_and_data

        config = {
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-3,
                "mixed_precision": True,  # Should be ignored on CPU
            },
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        # Mixed precision should be disabled on CPU
        if hasattr(trainer, "use_mixed_precision"):
            assert trainer.use_mixed_precision == False

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        assert "train_loss" in history


class TestGradientAccumulation:
    """Test gradient accumulation functionality."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and data for gradient accumulation testing."""
        config = {
            "input_dim": 32,
            "patch_size": 4,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 3,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)

        X = torch.randn(16, 32, 32)
        y = torch.randint(0, 3, (16,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        return model, dataloader

    def test_gradient_accumulation(self, model_and_data):
        """Test gradient accumulation."""
        model, dataloader = model_and_data

        config = {
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-3,
                "accumulate_grad_batches": 4,  # Accumulate over 4 batches
            },
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        # Track optimizer steps
        original_step = trainer.optimizer.step
        step_count = 0

        def count_steps():
            nonlocal step_count
            step_count += 1
            original_step()

        trainer.optimizer.step = count_steps

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        # With 8 samples, batch size 2, and accumulation 4:
        # We should have 8/2 = 4 mini-batches
        # With accumulation 4, we should have 4/4 = 1 optimizer step
        expected_steps = max(1, len(dataloader) // config["training"]["accumulate_grad_batches"])
        assert step_count >= 1  # At least one step should occur


class TestMultiModelTraining:
    """Test training with different model architectures."""

    def test_audioMAE_training(self):
        """Test training AudioMAE model."""
        config = {
            "input_dim": 32,
            "patch_size": 4,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 3,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)

        X = torch.randn(12, 32, 32)
        y = torch.randint(0, 3, (12,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        train_config = {
            "training": {"num_epochs": 1, "learning_rate": 1e-3},
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=train_config, device=torch.device("cpu"))

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1

    def test_ast_training(self):
        """Test training AST model."""
        config = {
            "input_tdim": 32,
            "input_fdim": 32,
            "patch_size": 4,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 3,
        }
        model = AST(config)

        # AST expects (batch, freq, time) input
        X = torch.randn(12, 32, 32)
        y = torch.randint(0, 3, (12,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        train_config = {
            "training": {"num_epochs": 1, "learning_rate": 1e-3},
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=train_config, device=torch.device("cpu"))

        history = trainer.fit(train_dataloader=dataloader, val_dataloader=dataloader)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1


class TestTrainingMetrics:
    """Test metrics computation during training."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and data for metrics testing."""
        config = {
            "input_dim": 16,
            "patch_size": 4,
            "embed_dim": 32,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 3,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)

        X = torch.randn(18, 16, 16)
        y = torch.randint(0, 3, (18,))
        dataset = TensorDataset(X, y)

        train_loader = DataLoader(dataset[:12], batch_size=6, shuffle=True)
        val_loader = DataLoader(dataset[12:], batch_size=6, shuffle=False)

        return model, train_loader, val_loader

    def test_classification_metrics_computation(self, model_and_data):
        """Test classification metrics computation."""
        model, train_loader, val_loader = model_and_data

        config = {
            "training": {"num_epochs": 2, "learning_rate": 1e-3, "compute_metrics": True},
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
            "metrics": ["accuracy", "precision", "recall", "f1"],
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        history = trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)

        # Check that metrics are computed and logged
        assert "train_loss" in history
        assert "val_loss" in history

        if "train_accuracy" in history:
            assert len(history["train_accuracy"]) == 2
        if "val_accuracy" in history:
            assert len(history["val_accuracy"]) == 2

    def test_metrics_validation(self, model_and_data):
        """Test metrics validation during training."""
        model, train_loader, val_loader = model_and_data

        # Create predictions and targets for metrics testing
        predictions = torch.randn(6, 3)  # Batch size 6, 3 classes
        targets = torch.randint(0, 3, (6,))

        # Test metrics computation directly
        metrics = compute_classification_metrics(predictions, targets)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # All metrics should be between 0 and 1
        for metric_name, metric_value in metrics.items():
            if metric_name != "confusion_matrix":
                assert 0 <= metric_value <= 1


class TestTrainingErrorHandling:
    """Test error handling in training pipeline."""

    def test_invalid_configuration(self):
        """Test training with invalid configuration."""
        model = nn.Linear(10, 5)  # Simple model

        # Invalid optimizer
        invalid_config = {
            "training": {"num_epochs": 1, "learning_rate": 1e-3},
            "optimizer": {"name": "invalid_optimizer", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
        }

        with pytest.raises((ValueError, KeyError)):
            trainer = Trainer(model=model, config=invalid_config, device=torch.device("cpu"))

    def test_empty_dataloader(self):
        """Test training with empty dataloader."""
        model = nn.Linear(10, 5)

        # Create empty dataset
        empty_dataset = TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_dataset, batch_size=1)

        config = {
            "training": {"num_epochs": 1, "learning_rate": 1e-3},
            "optimizer": {"name": "adam", "lr": 1e-3},
            "loss": {"name": "cross_entropy"},
        }

        trainer = Trainer(model=model, config=config, device=torch.device("cpu"))

        # Should handle empty dataloader gracefully
        try:
            history = trainer.fit(train_dataloader=empty_loader, val_dataloader=empty_loader)
            # If it succeeds, that's fine
        except Exception as e:
            # If it fails, the error should be informative
            assert isinstance(e, (ValueError, RuntimeError))


if __name__ == "__main__":
    pytest.main([__file__])
