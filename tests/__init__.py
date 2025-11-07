"""
SereneSense Test Suite

This module provides comprehensive testing for the SereneSense military vehicle sound detection system.
Includes unit tests, integration tests, and performance tests to ensure reliability and performance.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: End-to-end pipeline testing  
- Performance Tests: Latency, throughput, and memory benchmarks
- Fixtures: Shared test data and configurations

All tests follow pytest conventions and can be run with:
    pytest tests/
    pytest tests/unit/
    pytest tests/integration/
    pytest tests/performance/
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_CONFIG_DIR = TEST_DATA_DIR / "test_configs"


# Common test utilities
def get_test_device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_audio_sample(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a test audio sample."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a mix of sine waves to simulate audio
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        + 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 note
        + 0.1 * np.random.randn(len(t))  # Noise
    )
    return audio.astype(np.float32)


def get_test_config() -> Dict[str, Any]:
    """Get a basic test configuration."""
    return {
        "audio_processor": {
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512,
            "window_length": 1024,
        },
        "model": {"num_classes": 10, "hidden_dim": 256, "dropout": 0.1},
        "training": {"batch_size": 4, "learning_rate": 1e-4, "num_epochs": 1},
    }


# Pytest markers for test categorization
pytestmark = [
    pytest.mark.serenesense,
]

# Test fixtures are defined in conftest.py files in each test directory
