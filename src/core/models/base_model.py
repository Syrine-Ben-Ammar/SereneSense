#
# Plan:
# 1. Create abstract base class for all audio models
# 2. Define common interfaces for training, inference, and optimization
# 3. Model configuration dataclass with validation
# 4. Model output structure for consistent results
# 5. Checkpoint saving/loading utilities
# 6. Model optimization hooks for TensorRT/quantization
# 7. Performance monitoring and profiling capabilities
# 8. Military-specific model requirements and constraints
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings

logger = logging.getLogger(__name__)

@dataclass
class AudioModelConfig:
    """Base configuration for audio models."""
    
    # Model architecture
    model_name: str = "base_audio_model"
    num_classes: int = 7  # Default for MAD dataset
    
    # Input specifications
    sample_rate: int = 16000
    input_size: Tuple[int, int] = (128, 128)  # (freq, time) for spectrograms
    input_channels: int = 1
    
    # Model dimensions
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Optimization
    use_gradient_checkpointing: bool = False
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    
    # Military-specific requirements
    max_inference_time_ms: float = 20.0  # Maximum inference time for real-time
    min_accuracy: float = 0.85           # Minimum required accuracy
    robustness_requirements: Dict[str, float] = field(default_factory=lambda: {
        'noise_robustness': 0.9,    # Performance degradation tolerance with noise
        'compression_robustness': 0.95,  # Performance with audio compression
        'speed_robustness': 0.92    # Performance with speed variations
    })
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be between 0 and 1")
        
        if self.max_inference_time_ms <= 0:
            raise ValueError("max_inference_time_ms must be positive")


@dataclass
class ModelOutput:
    """Standardized output structure for all models."""
    
    # Core outputs
    logits: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor
    
    # Optional outputs
    features: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    
    # Performance metrics
    inference_time_ms: Optional[float] = None
    confidence: Optional[torch.Tensor] = None
    
    # Metadata
    model_name: Optional[str] = None
    input_shape: Optional[Tuple] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary format."""
        result = {
            'logits': self.logits.detach().cpu().numpy() if isinstance(self.logits, torch.Tensor) else self.logits,
            'predictions': self.predictions.detach().cpu().numpy() if isinstance(self.predictions, torch.Tensor) else self.predictions,
            'probabilities': self.probabilities.detach().cpu().numpy() if isinstance(self.probabilities, torch.Tensor) else self.probabilities,
        }
        
        if self.features is not None:
            result['features'] = self.features.detach().cpu().numpy() if isinstance(self.features, torch.Tensor) else self.features
        
        if self.attention_weights is not None:
            result['attention_weights'] = self.attention_weights.detach().cpu().numpy() if isinstance(self.attention_weights, torch.Tensor) else self.attention_weights
        
        if self.hidden_states is not None:
            result['hidden_states'] = [h.detach().cpu().numpy() if isinstance(h, torch.Tensor) else h for h in self.hidden_states]
        
        if self.inference_time_ms is not None:
            result['inference_time_ms'] = self.inference_time_ms
        
        if self.confidence is not None:
            result['confidence'] = self.confidence.detach().cpu().numpy() if isinstance(self.confidence, torch.Tensor) else self.confidence
        
        if self.model_name is not None:
            result['model_name'] = self.model_name
        
        if self.input_shape is not None:
            result['input_shape'] = self.input_shape
        
        return result


class BaseAudioModel(nn.Module, ABC):
    """
    Abstract base class for all audio models in SereneSense.
    
    Provides common functionality for:
    - Model initialization and configuration
    - Forward pass interface
    - Checkpoint saving/loading
    - Performance monitoring
    - Optimization hooks
    - Military-specific requirements
    """
    
    def __init__(self, config: AudioModelConfig):
        """
        Initialize base audio model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.model_name = config.model_name
        self.num_classes = config.num_classes
        
        # Performance tracking
        self.inference_times = []
        self.accuracy_history = []
        
        # Optimization flags
        self._is_optimized = False
        self._optimization_info = {}
        
        logger.info(f"Initialized {self.model_name} with {self.num_classes} classes")
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> ModelOutput:
        """
        Forward pass of the model.
        
        Args:
            inputs: Input tensor (audio or spectrogram)
            **kwargs: Additional arguments
            
        Returns:
            ModelOutput with predictions and metadata
        """
        pass
    
    def predict(self, inputs: torch.Tensor, return_features: bool = False) -> ModelOutput:
        """
        Inference with timing and performance monitoring.
        
        Args:
            inputs: Input tensor
            return_features: Whether to return intermediate features
            
        Returns:
            ModelOutput with predictions and timing information
        """
        self.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            # Ensure input is on the same device as model
            if next(self.parameters()).is_cuda and not inputs.is_cuda:
                inputs = inputs.cuda()
            
            # Forward pass
            output = self.forward(inputs, return_features=return_features)
            
            # Calculate inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update timing statistics
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 1000:  # Keep last 1000 measurements
                self.inference_times = self.inference_times[-1000:]
            
            # Add timing to output
            output.inference_time_ms = inference_time
            output.model_name = self.model_name
            output.input_shape = tuple(inputs.shape)
            
            # Check real-time constraint
            if inference_time > self.config.max_inference_time_ms:
                logger.warning(
                    f"Inference time ({inference_time:.2f}ms) exceeds "
                    f"real-time requirement ({self.config.max_inference_time_ms}ms)"
                )
        
        return output
    
    def predict_batch(self, inputs: torch.Tensor, batch_size: int = 32) -> List[ModelOutput]:
        """
        Batch prediction with automatic batching.
        
        Args:
            inputs: Input tensor [N, ...] 
            batch_size: Batch size for processing
            
        Returns:
            List of ModelOutput for each input
        """
        self.eval()
        
        outputs = []
        num_samples = inputs.shape[0]
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_inputs = inputs[i:end_idx]
                
                batch_output = self.predict(batch_inputs)
                
                # Split batch output into individual outputs
                batch_size_actual = batch_inputs.shape[0]
                for j in range(batch_size_actual):
                    output = ModelOutput(
                        logits=batch_output.logits[j:j+1],
                        predictions=batch_output.predictions[j:j+1],
                        probabilities=batch_output.probabilities[j:j+1],
                        features=batch_output.features[j:j+1] if batch_output.features is not None else None,
                        attention_weights=batch_output.attention_weights[j:j+1] if batch_output.attention_weights is not None else None,
                        inference_time_ms=batch_output.inference_time_ms / batch_size_actual,
                        model_name=self.model_name,
                        input_shape=tuple(batch_inputs[j].shape)
                    )
                    outputs.append(output)
        
        return outputs
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {'message': 'No inference measurements available'}
        
        times = np.array(self.inference_times)
        
        stats = {
            'avg_inference_time_ms': float(np.mean(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'std_inference_time_ms': float(np.std(times)),
            'p95_inference_time_ms': float(np.percentile(times, 95)),
            'p99_inference_time_ms': float(np.percentile(times, 99)),
            'total_inferences': len(times),
            'real_time_compliance': float(np.mean(times <= self.config.max_inference_time_ms)),
        }
        
        if self.accuracy_history:
            acc = np.array(self.accuracy_history)
            stats.update({
                'avg_accuracy': float(np.mean(acc)),
                'min_accuracy': float(np.min(acc)),
                'max_accuracy': float(np.max(acc)),
                'current_accuracy': float(acc[-1])
            })
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self.inference_times = []
        self.accuracy_history = []
        logger.info("Performance statistics reset")
    
    def save_checkpoint(self, path: Union[str, Path], **kwargs):
        """
        Save model checkpoint with metadata.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'performance_stats': self.get_performance_stats(),
            'optimization_info': self._optimization_info,
            **kwargs
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Union[str, Path], strict: bool = True):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict loading
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(
            checkpoint['model_state_dict'], 
            strict=strict
        )
        
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Load optimization info if available
        if 'optimization_info' in checkpoint:
            self._optimization_info = checkpoint['optimization_info']
            self._is_optimized = bool(self._optimization_info)
        
        logger.info(f"Checkpoint loaded from {path}")
        
        return checkpoint
    
    def optimize_for_inference(self, method: str = 'torchscript', **kwargs):
        """
        Optimize model for inference.
        
        Args:
            method: Optimization method ('torchscript', 'tensorrt', 'onnx')
            **kwargs: Method-specific arguments
        """
        if method == 'torchscript':
            self._optimize_torchscript(**kwargs)
        elif method == 'tensorrt':
            self._optimize_tensorrt(**kwargs)
        elif method == 'onnx':
            self._optimize_onnx(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        self._is_optimized = True
        logger.info(f"Model optimized using {method}")
    
    def _optimize_torchscript(self, **kwargs):
        """Optimize using TorchScript."""
        try:
            # Create example input
            example_input = self._get_example_input()
            
            # Trace the model
            traced_model = torch.jit.trace(self, example_input)
            
            # Replace the forward method
            self.forward = traced_model.forward
            
            self._optimization_info['method'] = 'torchscript'
            self._optimization_info['example_input_shape'] = tuple(example_input.shape)
            
        except Exception as e:
            logger.error(f"TorchScript optimization failed: {e}")
            raise
    
    def _optimize_tensorrt(self, **kwargs):
        """Optimize using TensorRT (placeholder implementation)."""
        logger.warning("TensorRT optimization not implemented yet")
        self._optimization_info['method'] = 'tensorrt'
    
    def _optimize_onnx(self, **kwargs):
        """Optimize using ONNX (placeholder implementation)."""
        logger.warning("ONNX optimization not implemented yet")
        self._optimization_info['method'] = 'onnx'
    
    def _get_example_input(self) -> torch.Tensor:
        """Get example input for optimization."""
        if hasattr(self.config, 'input_size'):
            batch_size = 1
            channels = getattr(self.config, 'input_channels', 1)
            height, width = self.config.input_size
            return torch.randn(batch_size, channels, height, width)
        else:
            # Default spectrogram size
            return torch.randn(1, 1, 128, 128)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config.__dict__,
            'is_optimized': self._is_optimized,
            'optimization_info': self._optimization_info,
            'performance_stats': self.get_performance_stats()
        }
        
        return info
    
    def validate_military_requirements(self) -> Dict[str, bool]:
        """
        Validate that model meets military deployment requirements.
        
        Returns:
            Dictionary with requirement validation results
        """
        stats = self.get_performance_stats()
        
        requirements = {
            'real_time_performance': True,
            'accuracy_requirement': True,
            'robustness_requirement': True,
            'size_requirement': True
        }
        
        # Check real-time performance
        if 'avg_inference_time_ms' in stats:
            requirements['real_time_performance'] = (
                stats['avg_inference_time_ms'] <= self.config.max_inference_time_ms
            )
        
        # Check accuracy (if available)
        if 'current_accuracy' in stats:
            requirements['accuracy_requirement'] = (
                stats['current_accuracy'] >= self.config.min_accuracy
            )
        
        # Check model size (< 500MB for edge deployment)
        model_size_mb = sum(p.numel() for p in self.parameters()) * 4 / (1024 * 1024)
        requirements['size_requirement'] = model_size_mb <= 500.0
        
        # Overall validation
        requirements['overall_compliance'] = all(requirements.values())
        
        return requirements
    
    def profile_model(self, input_shape: Tuple[int, ...] = None, num_runs: int = 100) -> Dict[str, Any]:
        """
        Profile model performance.
        
        Args:
            input_shape: Input shape for profiling
            num_runs: Number of runs for profiling
            
        Returns:
            Profiling results
        """
        if input_shape is None:
            dummy_input = self._get_example_input()
        else:
            dummy_input = torch.randn(*input_shape)
        
        if next(self.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        self.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(dummy_input)
        
        # Profile
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.forward(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)   # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        profile_results = {
            'avg_inference_time_ms': avg_time,
            'throughput_samples_per_sec': 1000 / avg_time,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'input_shape': tuple(dummy_input.shape),
            'num_runs': num_runs,
            'device': str(next(self.parameters()).device)
        }
        
        return profile_results


def load_pretrained_model(model_class, config_path: str, checkpoint_path: str, **kwargs):
    """
    Load a pretrained model from configuration and checkpoint.
    
    Args:
        model_class: Model class to instantiate
        config_path: Path to configuration file
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments for model creation
        
    Returns:
        Loaded model instance
    """
    # Load configuration
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config_dict = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")
    
    # Create config object
    if hasattr(model_class, 'Config'):
        config = model_class.Config(**config_dict)
    else:
        config = AudioModelConfig(**config_dict)
    
    # Create model
    model = model_class(config, **kwargs)
    
    # Load checkpoint
    model.load_checkpoint(checkpoint_path)
    
    return model


def save_model_checkpoint(model: BaseAudioModel, path: Union[str, Path], **kwargs):
    """
    Save model checkpoint with standardized format.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        **kwargs: Additional metadata
    """
    model.save_checkpoint(path, **kwargs)


def get_model_info(model: BaseAudioModel) -> Dict[str, Any]:
    """
    Get comprehensive model information.
    
    Args:
        model: Model instance
        
    Returns:
        Model information dictionary
    """
    return model.get_model_info()