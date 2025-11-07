"""
Unit tests for SereneSense model implementations.

Tests all model architectures (AudioMAE, AST, BEATs) for:
- Correct initialization
- Forward pass functionality
- Output shape validation
- Parameter counting
- Device placement
- Memory usage
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.models.base_model import BaseAudioModel
from core.models.audioMAE.model import AudioMAE
from core.models.AST.model import AST
from core.models.BEATs.model import BEATs
from core.utils.config_parser import ConfigParser


class TestBaseAudioModel:
    """Test the base audio model abstract class."""

    def test_base_model_is_abstract(self):
        """Test that BaseAudioModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAudioModel()

    def test_base_model_interface(self):
        """Test that BaseAudioModel defines the required interface."""
        assert hasattr(BaseAudioModel, "forward")
        assert hasattr(BaseAudioModel, "get_feature_dim")
        assert hasattr(BaseAudioModel, "load_pretrained")


class TestAudioMAE:
    """Test AudioMAE model implementation."""

    @pytest.fixture
    def audioMAE_config(self):
        """AudioMAE configuration for testing."""
        return {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
            "mask_ratio": 0.75,
        }

    @pytest.fixture
    def audioMAE_model(self, audioMAE_config):
        """Create AudioMAE model for testing."""
        return AudioMAE(audioMAE_config)

    def test_audioMAE_initialization(self, audioMAE_model, audioMAE_config):
        """Test AudioMAE model initialization."""
        assert isinstance(audioMAE_model, AudioMAE)
        assert isinstance(audioMAE_model, BaseAudioModel)
        assert audioMAE_model.num_classes == audioMAE_config["num_classes"]
        assert audioMAE_model.embed_dim == audioMAE_config["embed_dim"]

    def test_audioMAE_forward_pass(self, audioMAE_model):
        """Test AudioMAE forward pass."""
        batch_size = 2
        time_steps = 1024
        input_dim = 128

        # Create test input (batch_size, time_steps, input_dim)
        x = torch.randn(batch_size, time_steps, input_dim)

        # Forward pass
        output = audioMAE_model(x)

        # Check output shape
        assert output.shape == (batch_size, 10)  # num_classes = 10
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_audioMAE_masked_forward(self, audioMAE_model):
        """Test AudioMAE forward pass with masking."""
        batch_size = 2
        time_steps = 1024
        input_dim = 128

        x = torch.randn(batch_size, time_steps, input_dim)

        # Test with mask_ratio
        masked_output, mask, ids_restore = audioMAE_model.forward_encoder(x, mask_ratio=0.75)

        # Check output shapes
        assert masked_output.shape[0] == batch_size
        assert mask.shape[0] == batch_size
        assert ids_restore.shape[0] == batch_size

    def test_audioMAE_parameter_count(self, audioMAE_model):
        """Test AudioMAE parameter count."""
        total_params = sum(p.numel() for p in audioMAE_model.parameters())
        trainable_params = sum(p.numel() for p in audioMAE_model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable by default
        assert total_params > 1e6  # Should have at least 1M parameters

    def test_audioMAE_device_placement(self, audioMAE_model):
        """Test AudioMAE device placement."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        audioMAE_model = audioMAE_model.to(device)

        # Check model is on correct device
        for param in audioMAE_model.parameters():
            assert param.device == device

        # Test forward pass on device
        x = torch.randn(1, 1024, 128).to(device)
        output = audioMAE_model(x)
        assert output.device == device

    def test_audioMAE_gradient_flow(self, audioMAE_model):
        """Test gradient flow through AudioMAE."""
        x = torch.randn(1, 1024, 128, requires_grad=True)
        target = torch.randint(0, 10, (1,))

        # Forward pass
        output = audioMAE_model(x)
        loss = nn.CrossEntropyLoss()(output, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in audioMAE_model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAST:
    """Test Audio Spectrogram Transformer implementation."""

    @pytest.fixture
    def ast_config(self):
        """AST configuration for testing."""
        return {
            "input_tdim": 1024,
            "input_fdim": 128,
            "patch_size": 16,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
        }

    @pytest.fixture
    def ast_model(self, ast_config):
        """Create AST model for testing."""
        return AST(ast_config)

    def test_ast_initialization(self, ast_model, ast_config):
        """Test AST model initialization."""
        assert isinstance(ast_model, AST)
        assert isinstance(ast_model, BaseAudioModel)
        assert ast_model.num_classes == ast_config["num_classes"]
        assert ast_model.embed_dim == ast_config["embed_dim"]

    def test_ast_forward_pass(self, ast_model):
        """Test AST forward pass."""
        batch_size = 2
        time_dim = 1024
        freq_dim = 128

        # Create test input (batch_size, freq_dim, time_dim)
        x = torch.randn(batch_size, freq_dim, time_dim)

        # Forward pass
        output = ast_model(x)

        # Check output shape
        assert output.shape == (batch_size, 10)  # num_classes = 10
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_ast_patch_embedding(self, ast_model):
        """Test AST patch embedding."""
        batch_size = 2
        time_dim = 1024
        freq_dim = 128

        x = torch.randn(batch_size, freq_dim, time_dim)

        # Test patch embedding
        patches = ast_model.patch_embed(x)

        # Check patch dimensions
        expected_patches = (freq_dim // 16) * (time_dim // 16)
        assert patches.shape == (batch_size, expected_patches, 768)  # embed_dim = 768

    def test_ast_positional_encoding(self, ast_model):
        """Test AST positional encoding."""
        # Test that positional encoding is properly added
        assert hasattr(ast_model, "pos_embed")
        assert ast_model.pos_embed.requires_grad

    def test_ast_attention_weights(self, ast_model):
        """Test AST attention mechanism."""
        batch_size = 1
        x = torch.randn(batch_size, 128, 1024)

        # Forward pass with attention weights
        ast_model.eval()
        with torch.no_grad():
            output = ast_model(x)

        # Check that attention layers exist
        for layer in ast_model.transformer.layers:
            assert hasattr(layer, "self_attn")


class TestBEATs:
    """Test BEATs model implementation."""

    @pytest.fixture
    def beats_config(self):
        """BEATs configuration for testing."""
        return {
            "input_dim": 768,
            "encoder_layers": 12,
            "encoder_embed_dim": 768,
            "encoder_ffn_embed_dim": 3072,
            "encoder_attention_heads": 12,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "num_classes": 10,
            "predictor_dropout": 0.1,
            "predictor_embed_dim": 384,
        }

    @pytest.fixture
    def beats_model(self, beats_config):
        """Create BEATs model for testing."""
        return BEATs(beats_config)

    def test_beats_initialization(self, beats_model, beats_config):
        """Test BEATs model initialization."""
        assert isinstance(beats_model, BEATs)
        assert isinstance(beats_model, BaseAudioModel)
        assert beats_model.num_classes == beats_config["num_classes"]
        assert beats_model.encoder_embed_dim == beats_config["encoder_embed_dim"]

    def test_beats_forward_pass(self, beats_model):
        """Test BEATs forward pass."""
        batch_size = 2
        sequence_length = 1000
        input_dim = 768

        # Create test input (batch_size, sequence_length, input_dim)
        x = torch.randn(batch_size, sequence_length, input_dim)

        # Forward pass
        output = beats_model(x)

        # Check output shape
        assert output.shape == (batch_size, 10)  # num_classes = 10
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_beats_encoder(self, beats_model):
        """Test BEATs encoder."""
        batch_size = 2
        sequence_length = 1000
        input_dim = 768

        x = torch.randn(batch_size, sequence_length, input_dim)

        # Test encoder forward pass
        encoder_output = beats_model.encoder(x)

        # Check encoder output shape
        assert encoder_output.shape == (batch_size, sequence_length, 768)

    def test_beats_predictor(self, beats_model):
        """Test BEATs predictor head."""
        batch_size = 2
        sequence_length = 1000
        embed_dim = 768

        # Test predictor
        x = torch.randn(batch_size, sequence_length, embed_dim)

        # Get averaged representation (similar to what happens in forward)
        x_avg = x.mean(dim=1)  # Average over sequence length
        prediction = beats_model.predictor(x_avg)

        assert prediction.shape == (batch_size, 10)  # num_classes = 10


class TestModelComparison:
    """Test comparative functionality across models."""

    @pytest.fixture
    def all_models(self):
        """Create instances of all models for comparison."""
        audioMAE_config = {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
            "mask_ratio": 0.75,
        }

        ast_config = {
            "input_tdim": 1024,
            "input_fdim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
        }

        beats_config = {
            "input_dim": 256,
            "encoder_layers": 6,
            "encoder_embed_dim": 256,
            "encoder_ffn_embed_dim": 1024,
            "encoder_attention_heads": 8,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "num_classes": 10,
            "predictor_dropout": 0.1,
            "predictor_embed_dim": 128,
        }

        return {
            "audioMAE": AudioMAE(audioMAE_config),
            "ast": AST(ast_config),
            "beats": BEATs(beats_config),
        }

    def test_all_models_inherit_base_model(self, all_models):
        """Test that all models inherit from BaseAudioModel."""
        for model_name, model in all_models.items():
            assert isinstance(
                model, BaseAudioModel
            ), f"{model_name} does not inherit from BaseAudioModel"

    def test_all_models_have_required_methods(self, all_models):
        """Test that all models implement required methods."""
        required_methods = ["forward", "get_feature_dim"]

        for model_name, model in all_models.items():
            for method in required_methods:
                assert hasattr(model, method), f"{model_name} missing method {method}"
                assert callable(getattr(model, method)), f"{model_name}.{method} is not callable"

    def test_all_models_output_correct_shape(self, all_models):
        """Test that all models output correct classification shape."""
        batch_size = 2
        num_classes = 10

        for model_name, model in all_models.items():
            model.eval()

            # Create appropriate input for each model
            if model_name == "audioMAE":
                x = torch.randn(batch_size, 1024, 128)
            elif model_name == "ast":
                x = torch.randn(batch_size, 128, 1024)
            elif model_name == "beats":
                x = torch.randn(batch_size, 1000, 256)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (
                batch_size,
                num_classes,
            ), f"{model_name} output shape {output.shape} != expected {(batch_size, num_classes)}"

    def test_all_models_parameter_efficiency(self, all_models):
        """Test parameter efficiency of models."""
        param_counts = {}

        for model_name, model in all_models.items():
            total_params = sum(p.numel() for p in model.parameters())
            param_counts[model_name] = total_params

            # Each model should have reasonable number of parameters
            assert (
                10000 < total_params < 100000000
            ), f"{model_name} has unreasonable parameter count: {total_params}"

        # Log parameter counts for comparison
        print("\nModel Parameter Counts:")
        for model_name, count in param_counts.items():
            print(f"{model_name}: {count:,} parameters")


class TestModelUtilities:
    """Test model utility functions."""

    def test_model_save_load(self):
        """Test model saving and loading."""
        config = {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
            "mask_ratio": 0.75,
        }

        # Create and save model
        original_model = AudioMAE(config)
        state_dict = original_model.state_dict()

        # Create new model and load state
        loaded_model = AudioMAE(config)
        loaded_model.load_state_dict(state_dict)

        # Test that parameters are identical
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_model_eval_mode(self):
        """Test model evaluation mode."""
        config = {
            "input_tdim": 1024,
            "input_fdim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
        }

        model = AST(config)

        # Test train mode
        model.train()
        assert model.training

        # Test eval mode
        model.eval()
        assert not model.training

        # Test consistent outputs in eval mode
        x = torch.randn(1, 128, 1024)
        model.eval()
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2, atol=1e-6)


# Performance and edge case tests
class TestModelEdgeCases:
    """Test model behavior with edge cases."""

    def test_empty_input(self):
        """Test model behavior with empty input."""
        config = {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
            "mask_ratio": 0.75,
        }

        model = AudioMAE(config)

        # Test with minimum valid input
        x = torch.randn(1, 16, 128)  # Minimum size for patch_size=16

        try:
            output = model(x)
            assert output.shape == (1, 10)
        except Exception as e:
            pytest.fail(f"Model failed with minimum input: {e}")

    def test_large_batch_size(self):
        """Test model with large batch size."""
        config = {
            "input_tdim": 1024,
            "input_fdim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "num_classes": 10,
        }

        model = AST(config)

        # Test with large batch
        large_batch_size = 32
        x = torch.randn(large_batch_size, 128, 1024)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (large_batch_size, 10)

    def test_extreme_values(self):
        """Test model with extreme input values."""
        config = {
            "input_dim": 256,
            "encoder_layers": 6,
            "encoder_embed_dim": 256,
            "encoder_ffn_embed_dim": 1024,
            "encoder_attention_heads": 8,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "num_classes": 10,
            "predictor_dropout": 0.1,
            "predictor_embed_dim": 128,
        }

        model = BEATs(config)
        model.eval()

        # Test with extreme values
        test_cases = [
            torch.zeros(1, 1000, 256),  # All zeros
            torch.ones(1, 1000, 256) * 1000,  # Large positive values
            torch.ones(1, 1000, 256) * -1000,  # Large negative values
        ]

        for x in test_cases:
            with torch.no_grad():
                output = model(x)
                assert not torch.isnan(output).any()
                assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
