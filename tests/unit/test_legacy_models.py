# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Unit Tests for Legacy Models
=============================
Test CNN and CRNN model implementations.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelConfig
from core.models.legacy.legacy_config import LegacyModelType, CNNConfig, CRNNConfig


class TestCNNMFCCModel:
    """Test CNN MFCC model."""

    @pytest.fixture
    def model(self):
        """Create a CNN model for testing."""
        config = LegacyModelConfig(
            model_type=LegacyModelType.CNN,
            device="cpu",
        )
        return CNNMFCCModel(config)

    @pytest.fixture
    def dummy_input(self):
        """Create dummy input."""
        return torch.randn(4, 3, 40, 92)  # (batch, channels, freq, time)

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert model.model_name == "cnn_mfcc"

    def test_parameter_count(self, model):
        """Test parameter count is reasonable."""
        num_params = model.count_parameters()
        assert 200000 < num_params < 300000, f"Expected ~242K params, got {num_params}"

    def test_forward_pass(self, model, dummy_input):
        """Test forward pass."""
        output = model(dummy_input)
        assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

    def test_output_range(self, model, dummy_input):
        """Test output is reasonable (logits)."""
        output = model(dummy_input)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_feature_extraction(self, model, dummy_input):
        """Test feature extraction."""
        features = model.extract_features(dummy_input)
        assert features.shape[0] == 4, "Batch size mismatch"
        assert features.shape[1] == 192, "Feature dimension should be 192 (filters)"

    def test_predict(self, model, dummy_input):
        """Test prediction (argmax)."""
        predictions = model.predict(dummy_input)
        assert predictions.shape == (4,), f"Expected (4,), got {predictions.shape}"
        assert torch.all((predictions >= 0) & (predictions < 7)), "Invalid class indices"

    def test_predict_proba(self, model, dummy_input):
        """Test probability prediction."""
        proba = model.predict_proba(dummy_input)
        assert proba.shape == (4, 7), f"Expected (4, 7), got {proba.shape}"
        assert torch.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"
        assert torch.allclose(proba.sum(dim=1), torch.ones(4)), "Probabilities don't sum to 1"

    def test_with_features(self, model, dummy_input):
        """Test forward pass with feature extraction."""
        output = model(dummy_input, return_features=True)
        assert hasattr(output, "logits"), "Missing logits"
        assert hasattr(output, "predictions"), "Missing predictions"
        assert hasattr(output, "probabilities"), "Missing probabilities"
        assert hasattr(output, "features"), "Missing features"

    def test_gradient_flow(self, model, dummy_input):
        """Test gradients flow through model."""
        dummy_input.requires_grad = True
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        assert dummy_input.grad is not None, "Gradient not computed"
        assert dummy_input.grad.abs().max() > 0, "Gradient is zero"

    def test_eval_mode(self, model):
        """Test evaluation mode."""
        model.eval_mode()
        assert not model.training, "Model should be in eval mode"

    def test_train_mode(self, model):
        """Test training mode."""
        model.train_mode()
        assert model.training, "Model should be in train mode"

    def test_device_movement(self, model):
        """Test moving model to different devices."""
        if torch.cuda.is_available():
            model.to_device("cuda")
            assert str(model.device) == "cuda:0"

        model.to_device("cpu")
        assert str(model.device) == "cpu"


class TestCRNNMFCCModel:
    """Test CRNN MFCC model."""

    @pytest.fixture
    def model(self):
        """Create a CRNN model for testing."""
        config = LegacyModelConfig(
            model_type=LegacyModelType.CRNN,
            device="cpu",
        )
        return CRNNMFCCModel(config)

    @pytest.fixture
    def dummy_input(self):
        """Create dummy input."""
        return torch.randn(4, 3, 40, 124)  # (batch, channels, freq, time)

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert model.model_name == "crnn_mfcc"

    def test_parameter_count(self, model):
        """Test parameter count is reasonable."""
        num_params = model.count_parameters()
        # CRNN has ~1.5M parameters
        assert 1400000 < num_params < 1600000, f"Expected ~1.5M params, got {num_params}"

    def test_forward_pass(self, model, dummy_input):
        """Test forward pass."""
        output = model(dummy_input)
        assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

    def test_output_range(self, model, dummy_input):
        """Test output is reasonable."""
        output = model(dummy_input)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_feature_extraction(self, model, dummy_input):
        """Test feature extraction."""
        features = model.extract_features(dummy_input)
        assert features.shape[0] == 4, "Batch size mismatch"
        # Features from avg + max pooling of LSTM output: 64*2*2 = 256
        assert features.shape[1] == 256, f"Expected 256 features, got {features.shape[1]}"

    def test_predict(self, model, dummy_input):
        """Test prediction."""
        predictions = model.predict(dummy_input)
        assert predictions.shape == (4,), f"Expected (4,), got {predictions.shape}"
        assert torch.all((predictions >= 0) & (predictions < 7)), "Invalid class indices"

    def test_predict_proba(self, model, dummy_input):
        """Test probability prediction."""
        proba = model.predict_proba(dummy_input)
        assert proba.shape == (4, 7)
        assert torch.all((proba >= 0) & (proba <= 1))
        assert torch.allclose(proba.sum(dim=1), torch.ones(4))

    def test_with_features(self, model, dummy_input):
        """Test forward pass with feature extraction."""
        output = model(dummy_input, return_features=True)
        assert hasattr(output, "logits")
        assert hasattr(output, "predictions")
        assert hasattr(output, "probabilities")
        assert hasattr(output, "features")

    def test_lstm_processing(self, model, dummy_input):
        """Test that LSTM is properly processing sequential data."""
        features = model.extract_features(dummy_input)
        # LSTM should output meaningful features
        assert not torch.allclose(features[0], features[1]), "Features too similar"

    def test_bidirectional_lstm(self, model):
        """Test bidirectional LSTM."""
        # Create input
        x = torch.randn(2, 3, 40, 124)
        output = model(x, return_features=True)

        # Features should be 256-dimensional (128*2 from bi-LSTM * 2 for pooling)
        assert output.features.shape[1] == 256


class TestLegacyModelConfig:
    """Test configuration classes."""

    def test_cnn_config_creation(self):
        """Test CNN config creation."""
        config = LegacyModelConfig(model_type=LegacyModelType.CNN)
        assert config.model_type == LegacyModelType.CNN

    def test_crnn_config_creation(self):
        """Test CRNN config creation."""
        config = LegacyModelConfig(model_type=LegacyModelType.CRNN)
        assert config.model_type == LegacyModelType.CRNN

    def test_config_validation(self):
        """Test config validation."""
        config = LegacyModelConfig()
        # Should not raise any errors
        assert config is not None

    def test_active_arch_config_cnn(self):
        """Test getting active architecture config for CNN."""
        config = LegacyModelConfig(model_type=LegacyModelType.CNN)
        arch_config = config.active_arch_config
        assert isinstance(arch_config, CNNConfig)

    def test_active_arch_config_crnn(self):
        """Test getting active architecture config for CRNN."""
        config = LegacyModelConfig(model_type=LegacyModelType.CRNN)
        arch_config = config.active_arch_config
        assert isinstance(arch_config, CRNNConfig)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLegacyModelsCUDA:
    """Test legacy models on CUDA (if available)."""

    def test_cnn_on_cuda(self):
        """Test CNN on CUDA."""
        config = LegacyModelConfig(
            model_type=LegacyModelType.CNN,
            device="cuda",
        )
        model = CNNMFCCModel(config)
        x = torch.randn(2, 3, 40, 92, device="cuda")

        output = model(x)
        assert output.device.type == "cuda"

    def test_crnn_on_cuda(self):
        """Test CRNN on CUDA."""
        config = LegacyModelConfig(
            model_type=LegacyModelType.CRNN,
            device="cuda",
        )
        model = CRNNMFCCModel(config)
        x = torch.randn(2, 3, 40, 124, device="cuda")

        output = model(x)
        assert output.device.type == "cuda"
