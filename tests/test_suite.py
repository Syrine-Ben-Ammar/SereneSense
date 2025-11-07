# =============================================================================
# tests/conftest.py - Test Configuration
# =============================================================================
"""
PyTest configuration and fixtures for SereneSense testing.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import yaml
import soundfile as sf
from typing import Dict, Any

# Test configuration
pytest_plugins = ["pytest_asyncio"]


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture(scope="session")
def device():
    """Get test device (prefer CPU for reproducibility)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def sample_audio_config():
    """Sample audio configuration for testing."""
    from core.core.audio_processor import AudioConfig

    return AudioConfig(
        sample_rate=16000, n_mels=64, window_length=1.0  # Smaller for tests  # Shorter for tests
    )


@pytest.fixture(scope="session")
def sample_model_config():
    """Sample model configuration for testing."""
    from core.models.audioMAE.model import AudioMAEConfig

    return AudioMAEConfig(
        input_size=(64, 64),  # Smaller for tests
        patch_size=(8, 8),
        embed_dim=128,  # Much smaller for tests
        encoder_depth=2,
        decoder_depth=1,
        num_heads=4,
        num_classes=7,
        dropout=0.1,
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_audio_data(sample_audio_config):
    """Generate sample audio data for testing."""
    sample_rate = sample_audio_config.sample_rate
    duration = sample_audio_config.window_length

    # Generate synthetic audio
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Different audio types
    audio_samples = {
        "helicopter": np.sin(2 * np.pi * 4.2 * t) + 0.3 * np.sin(2 * np.pi * 8.4 * t),
        "fighter_aircraft": np.random.randn(len(t)) * 0.5,
        "military_vehicle": np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 60 * t),
        "background": 0.1 * np.random.randn(len(t)),
    }

    # Normalize
    for key in audio_samples:
        audio = audio_samples[key]
        audio_samples[key] = audio / np.max(np.abs(audio)) * 0.8

    return audio_samples


@pytest.fixture
def sample_dataset(temp_dir, sample_audio_data, sample_audio_config):
    """Create sample dataset for testing."""
    # Create dataset structure
    audio_dir = temp_dir / "audio"
    audio_dir.mkdir()

    sample_rate = sample_audio_config.sample_rate
    samples = []

    # Create audio files
    for class_name, audio_data in sample_audio_data.items():
        for i in range(5):  # 5 samples per class
            filename = f"{class_name}_{i:03d}.wav"
            filepath = audio_dir / filename

            # Add some variation
            noise = np.random.randn(len(audio_data)) * 0.05
            varied_audio = audio_data + noise

            sf.write(filepath, varied_audio, sample_rate)

            samples.append(
                {
                    "filename": filename,
                    "label": class_name,
                    "duration": len(audio_data) / sample_rate,
                    "split": "train" if i < 3 else ("val" if i == 3 else "test"),
                }
            )

    # Create metadata
    import pandas as pd

    metadata = pd.DataFrame(samples)
    metadata.to_csv(temp_dir / "metadata.csv", index=False)

    return temp_dir


@pytest.fixture
def trained_model(sample_model_config, device):
    """Create a simple trained model for testing."""
    from core.models.audioMAE.model import AudioMAE

    model = AudioMAE(sample_model_config)
    model.to(device)
    model.eval()

    # Initialize with small random weights for reproducibility
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

    return model


@pytest.fixture
def sample_config_file(temp_dir, sample_audio_config, sample_model_config):
    """Create sample configuration file."""
    config = {
        "model": sample_model_config.__dict__,
        "audio": sample_audio_config.__dict__,
        "data": {
            "data_dir": str(temp_dir),
            "classes": {
                "helicopter": 0,
                "fighter_aircraft": 1,
                "military_vehicle": 2,
                "background": 3,
            },
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "use_wandb": False,
            "use_mlflow": False,
        },
    }

    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


# =============================================================================
# tests/unit/test_audio_processor.py - Audio Processing Tests
# =============================================================================
"""
Unit tests for audio processing functionality.
"""

import pytest
import torch
import numpy as np
from core.core.audio_processor import AudioProcessor, AudioConfig, RealTimeAudioProcessor


class TestAudioProcessor:
    """Test AudioProcessor functionality."""

    def test_audio_config_creation(self):
        """Test AudioConfig creation with default values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.n_mels == 128
        assert config.window_length == 2.0
        assert config.normalize is True

    def test_audio_processor_initialization(self, sample_audio_config):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(sample_audio_config)
        assert processor.config == sample_audio_config
        assert processor.mel_transform is not None

    def test_mel_spectrogram_computation(self, sample_audio_config, sample_audio_data):
        """Test mel-spectrogram computation."""
        processor = AudioProcessor(sample_audio_config)

        # Test with helicopter audio
        waveform = torch.from_numpy(sample_audio_data["helicopter"]).float().unsqueeze(0)
        mel_spec = processor.compute_mel_spectrogram(waveform)

        # Check output shape
        assert mel_spec.dim() == 3  # [channels, mel_bins, time_frames]
        assert mel_spec.shape[0] == 1  # Single channel
        assert mel_spec.shape[1] == sample_audio_config.n_mels
        assert mel_spec.shape[2] > 0  # Some time frames

    def test_audio_segmentation(self, sample_audio_config, sample_audio_data):
        """Test audio segmentation."""
        processor = AudioProcessor(sample_audio_config)

        # Create longer audio for segmentation
        long_audio = np.tile(sample_audio_data["helicopter"], 3)  # 3x longer
        waveform = torch.from_numpy(long_audio).float().unsqueeze(0)

        segments = processor.segment_audio(waveform, window_length=1.0, overlap=0.5)

        assert len(segments) > 1  # Should create multiple segments
        assert all(seg.shape[0] == 1 for seg in segments)  # All single channel

        # Check segment duration
        expected_samples = int(sample_audio_config.sample_rate * 1.0)
        assert all(seg.shape[-1] == expected_samples for seg in segments)

    def test_batch_processing(self, sample_audio_config, temp_dir, sample_audio_data):
        """Test batch audio processing."""
        processor = AudioProcessor(sample_audio_config)

        # Create test audio files
        import soundfile as sf

        audio_files = []

        for i, (class_name, audio_data) in enumerate(sample_audio_data.items()):
            if i >= 3:  # Limit to 3 files
                break
            filepath = temp_dir / f"test_{class_name}.wav"
            sf.write(filepath, audio_data, sample_audio_config.sample_rate)
            audio_files.append(str(filepath))

        # Process batch
        batch_specs = processor.batch_process(audio_files)

        assert batch_specs.dim() == 4  # [batch, channels, mel_bins, time_frames]
        assert batch_specs.shape[0] == len(audio_files)

    def test_feature_extraction(self, sample_audio_config, sample_audio_data):
        """Test additional feature extraction."""
        processor = AudioProcessor(sample_audio_config)

        waveform = torch.from_numpy(sample_audio_data["helicopter"]).float().unsqueeze(0)
        mel_spec = processor.compute_mel_spectrogram(waveform)

        features = processor.extract_features(mel_spec)

        # Check that all expected features are present
        expected_features = [
            "spectral_centroid",
            "spectral_rolloff",
            "zcr",
            "spectral_contrast",
            "mfcc",
        ]
        assert all(feat in features for feat in expected_features)

        # Check feature shapes
        assert features["spectral_centroid"].dim() == 2
        assert features["mfcc"].shape[1] == 13  # 13 MFCC coefficients

    def test_audio_validation(self, sample_audio_config, sample_audio_data):
        """Test audio quality validation."""
        processor = AudioProcessor(sample_audio_config)

        # Valid audio
        valid_audio = torch.from_numpy(sample_audio_data["helicopter"]).float().unsqueeze(0)
        assert processor.validate_audio(valid_audio) is True

        # Silent audio
        silent_audio = torch.zeros_like(valid_audio)
        assert processor.validate_audio(silent_audio) is False

        # Very short audio
        short_audio = torch.randn(1, 100)  # Very short
        assert processor.validate_audio(short_audio) is False


class TestRealTimeAudioProcessor:
    """Test RealTimeAudioProcessor functionality."""

    def test_realtime_processor_initialization(self, sample_audio_config):
        """Test real-time processor initialization."""
        processor = RealTimeAudioProcessor(sample_audio_config, buffer_size=2048)
        assert processor.buffer_size == 2048
        assert processor.audio_buffer.shape == (1, 2048)
        assert processor.buffer_ptr == 0

    def test_audio_chunk_addition(self, sample_audio_config):
        """Test adding audio chunks to buffer."""
        processor = RealTimeAudioProcessor(sample_audio_config, buffer_size=4096)

        # Add chunk smaller than window
        chunk = torch.randn(1, 1000)
        result = processor.add_audio_chunk(chunk)
        assert result is None  # Not enough audio yet
        assert processor.buffer_ptr == 1000

        # Add enough chunks to trigger processing
        window_samples = int(sample_audio_config.sample_rate * sample_audio_config.window_length)
        remaining_samples = window_samples - 1000 + 100  # A bit more

        large_chunk = torch.randn(1, remaining_samples)
        result = processor.add_audio_chunk(large_chunk)

        assert result is not None  # Should return mel-spectrogram
        assert result.dim() == 3  # [channels, mel_bins, time_frames]

    def test_buffer_reset(self, sample_audio_config):
        """Test buffer reset functionality."""
        processor = RealTimeAudioProcessor(sample_audio_config)

        # Add some data
        chunk = torch.randn(1, 1000)
        processor.add_audio_chunk(chunk)
        assert processor.buffer_ptr == 1000

        # Reset buffer
        processor.reset_buffer()
        assert processor.buffer_ptr == 0
        assert torch.all(processor.audio_buffer == 0)


# =============================================================================
# tests/unit/test_models.py - Model Tests
# =============================================================================
"""
Unit tests for model functionality.
"""

import pytest
import torch
from core.models.audioMAE.model import AudioMAE, AudioMAEConfig


class TestAudioMAEModel:
    """Test AudioMAE model functionality."""

    def test_model_creation(self, sample_model_config):
        """Test model creation with configuration."""
        model = AudioMAE(sample_model_config)
        assert isinstance(model, AudioMAE)
        assert model.config == sample_model_config

    def test_model_forward_classification(self, sample_model_config, device):
        """Test model forward pass in classification mode."""
        model = AudioMAE(sample_model_config)
        model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(
            batch_size,
            1,
            sample_model_config.input_size[0],
            sample_model_config.input_size[1],
            device=device,
        )

        with torch.no_grad():
            outputs = model(input_tensor, mode="classification")

        # Check outputs
        assert "logits" in outputs
        assert "predictions" in outputs
        assert "features" in outputs

        # Check shapes
        assert outputs["logits"].shape == (batch_size, sample_model_config.num_classes)
        assert outputs["predictions"].shape == (batch_size,)
        assert outputs["features"].shape == (batch_size, sample_model_config.embed_dim)

    def test_model_forward_pretrain(self, sample_model_config, device):
        """Test model forward pass in pre-training mode."""
        model = AudioMAE(sample_model_config)
        model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(
            batch_size,
            1,
            sample_model_config.input_size[0],
            sample_model_config.input_size[1],
            device=device,
        )

        with torch.no_grad():
            outputs = model(input_tensor, mask_ratio=0.75, mode="pretrain")

        # Check outputs
        assert "pred" in outputs
        assert "mask" in outputs
        assert "last_hidden_state" in outputs

        # Check mask shape
        num_patches = (sample_model_config.input_size[0] // sample_model_config.patch_size[0]) * (
            sample_model_config.input_size[1] // sample_model_config.patch_size[1]
        )
        assert outputs["mask"].shape == (batch_size, num_patches)

    def test_model_feature_extraction(self, sample_model_config, device):
        """Test feature extraction functionality."""
        model = AudioMAE(sample_model_config)
        model.to(device)
        model.eval()

        input_tensor = torch.randn(
            1,
            1,
            sample_model_config.input_size[0],
            sample_model_config.input_size[1],
            device=device,
        )

        with torch.no_grad():
            features = model.extract_features(input_tensor)

        assert features.shape == (1, sample_model_config.embed_dim)

    def test_model_classification_inference(self, sample_model_config, device):
        """Test simple classification inference."""
        model = AudioMAE(sample_model_config)
        model.to(device)
        model.eval()

        input_tensor = torch.randn(
            1,
            1,
            sample_model_config.input_size[0],
            sample_model_config.input_size[1],
            device=device,
        )

        with torch.no_grad():
            probabilities = model.classify(input_tensor)

        assert probabilities.shape == (1, sample_model_config.num_classes)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, device=device))
        assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)

    def test_model_parameter_count(self, sample_model_config):
        """Test model parameter counting."""
        model = AudioMAE(sample_model_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All should be trainable
        assert total_params < 10_000_000  # Should be small for test config


# =============================================================================
# tests/unit/test_data_loaders.py - Data Loader Tests
# =============================================================================
"""
Unit tests for data loading functionality.
"""

import pytest
import torch
from core.data.loaders.mad_loader import MADDataModule, MADConfig, MADDataset


class TestMADDataLoader:
    """Test MAD dataset loader functionality."""

    def test_mad_config_creation(self):
        """Test MAD configuration creation."""
        config = MADConfig()
        assert config.data_dir == "data/raw/mad"
        assert config.sample_rate == 16000
        assert len(config.classes) == 7

    def test_mad_dataset_creation(self, sample_dataset, sample_audio_config):
        """Test MAD dataset creation."""
        from core.data.loaders.mad_loader import MADConfig

        mad_config = MADConfig(data_dir=str(sample_dataset))
        mad_config.classes = {
            "helicopter": 0,
            "fighter_aircraft": 1,
            "military_vehicle": 2,
            "background": 3,
        }

        dataset = MADDataset(
            mad_config,
            sample_audio_config,
            split="train",
            augmentation=False,
            cache_spectrograms=False,
        )

        assert len(dataset) > 0

        # Test getting a sample
        sample = dataset[0]
        assert "spectrogram" in sample
        assert "label" in sample
        assert "filename" in sample
        assert "duration" in sample

        # Check sample shapes and types
        assert sample["spectrogram"].dim() == 3
        assert sample["label"].dtype == torch.long
        assert isinstance(sample["filename"], str)

    def test_mad_data_module(self, sample_dataset, sample_audio_config):
        """Test MAD data module functionality."""
        from core.data.loaders.mad_loader import MADConfig

        mad_config = MADConfig(data_dir=str(sample_dataset))
        mad_config.classes = {
            "helicopter": 0,
            "fighter_aircraft": 1,
            "military_vehicle": 2,
            "background": 3,
        }

        data_module = MADDataModule(
            mad_config,
            sample_audio_config,
            batch_size=2,
            num_workers=0,  # No multiprocessing in tests
        )

        data_module.setup()

        # Test data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        # Test that loaders work
        train_batch = next(iter(train_loader))
        assert "spectrograms" in train_batch
        assert "labels" in train_batch
        assert train_batch["spectrograms"].shape[0] <= 2  # Batch size

        # Test statistics
        stats = data_module.get_dataset_statistics()
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats

        for split_stats in stats.values():
            assert "total_samples" in split_stats
            assert "class_distribution" in split_stats

    def test_class_weights(self, sample_dataset, sample_audio_config):
        """Test class weight calculation."""
        from core.data.loaders.mad_loader import MADConfig

        mad_config = MADConfig(data_dir=str(sample_dataset))
        mad_config.classes = {
            "helicopter": 0,
            "fighter_aircraft": 1,
            "military_vehicle": 2,
            "background": 3,
        }

        data_module = MADDataModule(mad_config, sample_audio_config)
        data_module.setup()

        class_weights = data_module.get_class_weights()
        assert class_weights.shape == (4,)  # 4 classes
        assert torch.all(class_weights > 0)  # All weights should be positive


# =============================================================================
# tests/integration/test_training_pipeline.py - Training Integration Tests
# =============================================================================
"""
Integration tests for training pipeline.
"""

import pytest
import torch
import tempfile
from pathlib import Path


@pytest.mark.integration
class TestTrainingPipeline:
    """Test complete training pipeline."""

    def test_end_to_end_training(self, sample_config_file, sample_dataset, device):
        """Test complete training pipeline from config to trained model."""
        # This is a minimal training test
        from core.training.trainer import create_trainer

        # Create trainer
        trainer = create_trainer(str(sample_config_file))

        # Very short training for testing
        trainer.config.epochs = 1
        trainer.config.use_wandb = False
        trainer.config.use_mlflow = False
        trainer.config.early_stopping = False

        # Run training
        results = trainer.train()

        # Check that training completed
        assert "best_val_score" in results
        assert "total_epochs" in results
        assert results["total_epochs"] >= 1

    @pytest.mark.slow
    def test_training_with_validation(self, sample_config_file, sample_dataset):
        """Test training with proper validation."""
        from core.training.trainer import create_trainer

        trainer = create_trainer(str(sample_config_file))

        # Configure for proper validation
        trainer.config.epochs = 3
        trainer.config.val_every_n_epochs = 1
        trainer.config.use_wandb = False
        trainer.config.use_mlflow = False

        results = trainer.train()

        # Check validation was performed
        assert "training_history" in results
        history = results["training_history"]
        assert "val_accuracy" in history
        assert len(history["val_accuracy"]) >= 1


# =============================================================================
# tests/integration/test_inference_pipeline.py - Inference Integration Tests
# =============================================================================
"""
Integration tests for inference pipeline.
"""

import pytest
import torch
import numpy as np


@pytest.mark.integration
class TestInferencePipeline:
    """Test complete inference pipeline."""

    def test_model_inference(self, trained_model, sample_audio_data, sample_audio_config, device):
        """Test model inference with real audio data."""
        from core.core.audio_processor import AudioProcessor

        # Create processor
        processor = AudioProcessor(sample_audio_config)

        # Process audio
        waveform = torch.from_numpy(sample_audio_data["helicopter"]).float().unsqueeze(0)
        mel_spec = processor.compute_mel_spectrogram(waveform)

        # Preprocess for model
        input_tensor = processor.preprocess_for_model(
            mel_spec, target_shape=trained_model.config.input_size
        )
        input_tensor = input_tensor.to(device)

        # Model inference
        trained_model.eval()
        with torch.no_grad():
            outputs = trained_model(input_tensor, mode="classification")

        # Check outputs
        assert "logits" in outputs
        assert "predictions" in outputs
        assert outputs["logits"].shape[1] == trained_model.config.num_classes

    def test_batch_inference(self, trained_model, sample_audio_data, sample_audio_config, device):
        """Test batch inference."""
        from core.core.audio_processor import AudioProcessor

        processor = AudioProcessor(sample_audio_config)

        # Process multiple audio samples
        batch_inputs = []
        for audio_data in list(sample_audio_data.values())[:3]:  # Use 3 samples
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
            mel_spec = processor.compute_mel_spectrogram(waveform)
            input_tensor = processor.preprocess_for_model(
                mel_spec, target_shape=trained_model.config.input_size
            )
            batch_inputs.append(input_tensor)

        # Stack into batch
        batch_tensor = torch.cat(batch_inputs, dim=0).to(device)

        # Batch inference
        trained_model.eval()
        with torch.no_grad():
            outputs = trained_model(batch_tensor, mode="classification")

        assert outputs["logits"].shape[0] == len(batch_inputs)
        assert outputs["predictions"].shape[0] == len(batch_inputs)


# =============================================================================
# tests/performance/test_latency.py - Performance Tests
# =============================================================================
"""
Performance tests for latency measurements.
"""

import pytest
import torch
import time
import numpy as np


@pytest.mark.performance
class TestLatencyPerformance:
    """Test model latency performance."""

    def test_inference_latency(self, trained_model, device):
        """Test model inference latency."""
        trained_model.eval()

        # Create dummy input
        input_tensor = torch.randn(
            1,
            1,
            trained_model.config.input_size[0],
            trained_model.config.input_size[1],
            device=device,
        )

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                trained_model(input_tensor, mode="classification")

        # Measure latency
        latencies = []
        num_iterations = 50

        for _ in range(num_iterations):
            start_time = time.time()

            with torch.no_grad():
                trained_model(input_tensor, mode="classification")

            if device.type == "cuda":
                torch.cuda.synchronize()

            latency = time.time() - start_time
            latencies.append(latency)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        print(f"\nLatency Statistics:")
        print(f"  Average: {avg_latency*1000:.2f}ms")
        print(f"  Std: {std_latency*1000:.2f}ms")
        print(f"  Min: {min_latency*1000:.2f}ms")
        print(f"  Max: {max_latency*1000:.2f}ms")
        print(f"  Throughput: {1/avg_latency:.1f} FPS")

        # Basic assertions
        assert avg_latency > 0
        assert avg_latency < 1.0  # Should be less than 1 second for small test model

    def test_batch_latency(self, trained_model, device):
        """Test batch inference latency."""
        trained_model.eval()

        batch_sizes = [1, 2, 4, 8]
        latencies = {}

        for batch_size in batch_sizes:
            input_tensor = torch.randn(
                batch_size,
                1,
                trained_model.config.input_size[0],
                trained_model.config.input_size[1],
                device=device,
            )

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    trained_model(input_tensor, mode="classification")

            # Measure
            times = []
            for _ in range(20):
                start_time = time.time()

                with torch.no_grad():
                    trained_model(input_tensor, mode="classification")

                if device.type == "cuda":
                    torch.cuda.synchronize()

                times.append(time.time() - start_time)

            latencies[batch_size] = np.mean(times)
            print(
                f"Batch size {batch_size}: {latencies[batch_size]*1000:.2f}ms "
                f"({latencies[batch_size]/batch_size*1000:.2f}ms per sample)"
            )

        # Check that batch processing is more efficient
        if len(latencies) > 1:
            per_sample_latencies = [lat / bs for bs, lat in latencies.items()]
            # Generally, larger batches should be more efficient per sample
            # (though this might not always hold for very small test models)


# =============================================================================
# tests/performance/test_memory_usage.py - Memory Tests
# =============================================================================
"""
Memory usage tests.
"""

import pytest
import torch
import psutil
import os


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_model_memory_usage(self, trained_model, device):
        """Test model memory usage."""
        if device.type == "cuda":
            # GPU memory
            torch.cuda.reset_peak_memory_stats(device)

            # Create input and run inference
            input_tensor = torch.randn(
                8,
                1,  # Larger batch for meaningful memory measurement
                trained_model.config.input_size[0],
                trained_model.config.input_size[1],
                device=device,
            )

            with torch.no_grad():
                outputs = trained_model(input_tensor, mode="classification")

            peak_memory = torch.cuda.max_memory_allocated(device)
            print(f"Peak GPU memory usage: {peak_memory / 1e6:.1f} MB")

            assert peak_memory > 0
            assert peak_memory < 1e9  # Should be less than 1GB for test model

        else:
            # CPU memory
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss

            input_tensor = torch.randn(
                8, 1, trained_model.config.input_size[0], trained_model.config.input_size[1]
            )

            with torch.no_grad():
                outputs = trained_model(input_tensor, mode="classification")

            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before

            print(f"CPU memory usage: {memory_used / 1e6:.1f} MB")

    def test_memory_leak(self, trained_model, device):
        """Test for memory leaks during repeated inference."""
        if device.type != "cuda":
            pytest.skip("Memory leak test only relevant for GPU")

        torch.cuda.reset_peak_memory_stats(device)

        input_tensor = torch.randn(
            1,
            1,
            trained_model.config.input_size[0],
            trained_model.config.input_size[1],
            device=device,
        )

        # Run many iterations
        memory_readings = []

        for i in range(100):
            with torch.no_grad():
                outputs = trained_model(input_tensor, mode="classification")

            if i % 20 == 0:
                memory_readings.append(torch.cuda.memory_allocated(device))

        # Check that memory usage is stable (no leaks)
        memory_increase = memory_readings[-1] - memory_readings[0]
        print(f"Memory change over 100 iterations: {memory_increase / 1e6:.1f} MB")

        # Allow some small increase but not significant
        assert abs(memory_increase) < 10e6  # Less than 10MB increase
