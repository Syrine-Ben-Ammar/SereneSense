"""
Model Manager for SereneSense

This module provides comprehensive model management capabilities for the SereneSense
military vehicle sound detection system.

Features:
- Model loading and initialization from checkpoints
- Multi-model support (AudioMAE, AST, BEATs)
- Model optimization (quantization, compilation, TensorRT)
- Device management and model placement
- Model ensembling and fusion
- Inference pipeline management
- Model versioning and caching
- Performance monitoring and profiling

Example:
    >>> from core.core.model_manager import ModelManager
    >>> 
    >>> # Initialize manager
    >>> config = {"device": "cuda", "precision": "fp16"}
    >>> manager = ModelManager(config)
    >>> 
    >>> # Load model
    >>> model = manager.load_model("audioMAE", "path/to/checkpoint.pth")
    >>> 
    >>> # Make predictions
    >>> predictions = manager.predict(model, features)
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import logging
import warnings
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Third-party imports
try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type definitions
ModelInput = Union[torch.Tensor, Dict[str, torch.Tensor]]
ModelOutput = Union[torch.Tensor, Dict[str, torch.Tensor]]
DeviceType = Union[str, torch.device]


@dataclass
class ModelInfo:
    """Container for model information and metadata."""

    name: str
    type: str
    version: str
    path: str
    device: str
    precision: str
    parameters: int
    memory_usage_mb: float
    inference_time_ms: float
    accuracy: Optional[float] = None
    checksum: Optional[str] = None
    created_at: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResult:
    """Container for inference results."""

    predictions: torch.Tensor
    probabilities: torch.Tensor
    confidence: float
    inference_time_ms: float
    model_name: str
    preprocessing_time_ms: Optional[float] = None
    postprocessing_time_ms: Optional[float] = None


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    @abstractmethod
    def predict(self, inputs: ModelInput) -> ModelOutput:
        """Make predictions using the model."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        pass

    @abstractmethod
    def optimize(self, optimization_config: Dict[str, Any]) -> None:
        """Optimize model for inference."""
        pass


class PyTorchModelWrapper(BaseModelWrapper):
    """Wrapper for PyTorch models."""

    def __init__(self, model: nn.Module, model_info: ModelInfo, config: Dict[str, Any]):
        """
        Initialize PyTorch model wrapper.

        Args:
            model: PyTorch model
            model_info: Model information
            config: Configuration dictionary
        """
        self.model = model
        self.model_info = model_info
        self.config = config

        # Set model to evaluation mode
        self.model.eval()

        # Apply optimizations
        self._apply_torch_optimizations()

    def _apply_torch_optimizations(self) -> None:
        """Apply PyTorch-specific optimizations."""

        # JIT compilation
        if self.config.get("jit_compile", False):
            try:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 1, 128, 1024).to(self.model_info.device)
                self.model = torch.jit.trace(self.model, dummy_input)
                logger.info("Model compiled with TorchScript")
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}")

        # Channels last memory format (for CNN models)
        if self.config.get("channels_last", False):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                logger.info("Model converted to channels_last format")
            except Exception as e:
                logger.warning(f"Channels last conversion failed: {e}")

    def predict(self, inputs: ModelInput) -> ModelOutput:
        """Make predictions using PyTorch model."""
        with torch.no_grad():
            return self.model(inputs)

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return self.model_info

    def optimize(self, optimization_config: Dict[str, Any]) -> None:
        """Optimize PyTorch model."""
        # Quantization
        if optimization_config.get("quantize", False):
            self._apply_quantization(optimization_config.get("quantization", {}))

        # Pruning
        if optimization_config.get("prune", False):
            self._apply_pruning(optimization_config.get("pruning", {}))

    def _apply_quantization(self, quant_config: Dict[str, Any]) -> None:
        """Apply quantization to model."""
        try:
            method = quant_config.get("method", "dynamic")

            if method == "dynamic":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization")

            elif method == "static":
                # Static quantization requires calibration data
                logger.warning("Static quantization not implemented")

        except Exception as e:
            logger.error(f"Quantization failed: {e}")

    def _apply_pruning(self, prune_config: Dict[str, Any]) -> None:
        """Apply pruning to model."""
        try:
            import torch.nn.utils.prune as prune

            sparsity = prune_config.get("sparsity", 0.2)

            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name="weight", amount=sparsity)

            logger.info(f"Applied pruning with {sparsity:.1%} sparsity")

        except ImportError:
            logger.error("Pruning requires PyTorch >= 1.4")
        except Exception as e:
            logger.error(f"Pruning failed: {e}")


class ONNXModelWrapper(BaseModelWrapper):
    """Wrapper for ONNX models."""

    def __init__(self, model_path: str, model_info: ModelInfo, config: Dict[str, Any]):
        """
        Initialize ONNX model wrapper.

        Args:
            model_path: Path to ONNX model
            model_info: Model information
            config: Configuration dictionary
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available")

        self.model_path = model_path
        self.model_info = model_info
        self.config = config

        # Create ONNX Runtime session
        self._create_session()

    def _create_session(self) -> None:
        """Create ONNX Runtime inference session."""
        providers = []

        # Configure execution providers
        if "cuda" in self.model_info.device and ort.get_device() == "GPU":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Intra-op threads
        if "cpu" in self.model_info.device:
            sess_options.intra_op_num_threads = self.config.get("num_threads", 4)

        self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)

        logger.info(f"ONNX session created with providers: {self.session.get_providers()}")

    def predict(self, inputs: ModelInput) -> ModelOutput:
        """Make predictions using ONNX model."""
        # Convert inputs to numpy
        if isinstance(inputs, torch.Tensor):
            input_dict = {self.session.get_inputs()[0].name: inputs.cpu().numpy()}
        else:
            input_dict = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

        # Run inference
        outputs = self.session.run(None, input_dict)

        # Convert outputs back to torch tensors
        if len(outputs) == 1:
            return torch.from_numpy(outputs[0])
        else:
            return {f"output_{i}": torch.from_numpy(out) for i, out in enumerate(outputs)}

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return self.model_info

    def optimize(self, optimization_config: Dict[str, Any]) -> None:
        """Optimize ONNX model."""
        logger.info("ONNX model optimization not implemented")


class TensorRTModelWrapper(BaseModelWrapper):
    """Wrapper for TensorRT models."""

    def __init__(self, engine_path: str, model_info: ModelInfo, config: Dict[str, Any]):
        """
        Initialize TensorRT model wrapper.

        Args:
            engine_path: Path to TensorRT engine
            model_info: Model information
            config: Configuration dictionary
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.engine_path = engine_path
        self.model_info = model_info
        self.config = config

        # Load TensorRT engine
        self._load_engine()

    def _load_engine(self) -> None:
        """Load TensorRT engine."""
        import pycuda.driver as cuda
        import pycuda.autoinit

        # Create logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

        logger.info(f"TensorRT engine loaded from {self.engine_path}")

    def _allocate_buffers(self) -> None:
        """Allocate GPU and CPU buffers."""
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def predict(self, inputs: ModelInput) -> ModelOutput:
        """Make predictions using TensorRT model."""
        import pycuda.driver as cuda

        # Copy input data to GPU
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.cpu().numpy()
        else:
            # Assume first input for simplicity
            inputs_np = list(inputs.values())[0].cpu().numpy()

        np.copyto(self.inputs[0]["host"], inputs_np.ravel())
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])

        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Copy output data to CPU
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])

        # Convert to tensor
        output_shape = self.context.get_binding_shape(1)  # Assume single output
        output = self.outputs[0]["host"].reshape(output_shape)

        return torch.from_numpy(output)

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return self.model_info

    def optimize(self, optimization_config: Dict[str, Any]) -> None:
        """TensorRT models are already optimized."""
        logger.info("TensorRT model is already optimized")


class ModelEnsemble:
    """Ensemble of multiple models for improved performance."""

    def __init__(
        self,
        models: List[BaseModelWrapper],
        weights: Optional[List[float]] = None,
        aggregation: str = "average",
    ):
        """
        Initialize model ensemble.

        Args:
            models: List of model wrappers
            weights: Optional weights for ensemble (equal weights if None)
            aggregation: Aggregation method ('average', 'weighted_average', 'voting')
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.aggregation = aggregation

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        logger.info(f"Model ensemble created with {len(models)} models")

    def predict(self, inputs: ModelInput) -> InferenceResult:
        """
        Make ensemble predictions.

        Args:
            inputs: Model inputs

        Returns:
            Ensemble inference result
        """
        start_time = time.time()

        predictions = []
        probabilities = []

        # Get predictions from each model
        for i, model in enumerate(self.models):
            pred = model.predict(inputs)

            # Handle different output formats
            if isinstance(pred, torch.Tensor):
                logits = pred
            else:
                logits = pred.get("logits", pred.get("output", list(pred.values())[0]))

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            predictions.append(logits)
            probabilities.append(probs)

        # Aggregate predictions
        if self.aggregation == "average":
            ensemble_probs = torch.mean(torch.stack(probabilities), dim=0)
        elif self.aggregation == "weighted_average":
            weighted_probs = [prob * weight for prob, weight in zip(probabilities, self.weights)]
            ensemble_probs = torch.sum(torch.stack(weighted_probs), dim=0)
        elif self.aggregation == "voting":
            # Hard voting based on argmax
            votes = torch.stack([torch.argmax(prob, dim=-1) for prob in probabilities])
            ensemble_pred = torch.mode(votes, dim=0)[0]
            ensemble_probs = torch.zeros_like(probabilities[0])
            ensemble_probs.scatter_(-1, ensemble_pred.unsqueeze(-1), 1.0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        # Get final prediction and confidence
        final_pred = torch.argmax(ensemble_probs, dim=-1)
        confidence = torch.max(ensemble_probs, dim=-1)[0].item()

        inference_time = (time.time() - start_time) * 1000

        return InferenceResult(
            predictions=final_pred,
            probabilities=ensemble_probs,
            confidence=confidence,
            inference_time_ms=inference_time,
            model_name=f"ensemble_{len(self.models)}_models",
        )


class ModelManager:
    """
    Comprehensive model management system for SereneSense.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Device configuration
        self.device = self._setup_device(config.get("device", "auto"))
        self.precision = config.get("precision", "fp32")

        # Model storage
        self.models: Dict[str, BaseModelWrapper] = {}
        self.model_cache: Dict[str, BaseModelWrapper] = {}

        # Cache configuration
        self.enable_cache = config.get("cache", {}).get("enabled", True)
        self.max_cache_size = config.get("cache", {}).get("max_size", "2GB")

        # Performance monitoring
        self.performance_stats: Dict[str, List[float]] = {}

        logger.info(
            f"ModelManager initialized - Device: {self.device}, " f"Precision: {self.precision}"
        )

    def _setup_device(self, device_spec: str) -> torch.device:
        """
        Setup and validate device configuration.

        Args:
            device_spec: Device specification

        Returns:
            Configured torch device
        """
        if device_spec == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_spec)

        # Validate device
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")

        return device

    def load_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str = "pytorch",
        config_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> BaseModelWrapper:
        """
        Load a model from checkpoint or saved file.

        Args:
            model_name: Name identifier for the model
            model_path: Path to model file
            model_type: Type of model ("pytorch", "onnx", "tensorrt")
            config_path: Optional path to model configuration
            force_reload: Force reload even if cached

        Returns:
            Loaded model wrapper
        """
        # Check cache first
        cache_key = f"{model_name}_{model_path}"
        if not force_reload and cache_key in self.model_cache:
            logger.info(f"Loading {model_name} from cache")
            return self.model_cache[cache_key]

        logger.info(f"Loading model {model_name} from {model_path}")

        start_time = time.time()

        try:
            if model_type == "pytorch":
                model_wrapper = self._load_pytorch_model(model_name, model_path, config_path)
            elif model_type == "onnx":
                model_wrapper = self._load_onnx_model(model_name, model_path, config_path)
            elif model_type == "tensorrt":
                model_wrapper = self._load_tensorrt_model(model_name, model_path, config_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model {model_name} loaded in {load_time:.2f}ms")

            # Cache model
            if self.enable_cache:
                self.model_cache[cache_key] = model_wrapper

            # Store in active models
            self.models[model_name] = model_wrapper

            return model_wrapper

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _load_pytorch_model(
        self, model_name: str, model_path: str, config_path: Optional[str] = None
    ) -> PyTorchModelWrapper:
        """Load PyTorch model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model architecture info
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        elif config_path:
            with open(config_path, "r") as f:
                import yaml

                model_config = yaml.safe_load(f)
        else:
            raise ValueError("Model config not found in checkpoint or config_path")

        # Initialize model based on type
        model = self._create_model_from_config(model_config)

        # Load state dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Handle DataParallel models
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key[7:]: value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)

        # Move to device and set precision
        model = model.to(self.device)
        if self.precision == "fp16":
            model = model.half()

        # Create model info
        model_info = ModelInfo(
            name=model_name,
            type=model_config.get("name", "unknown"),
            version=checkpoint.get("version", "unknown"),
            path=model_path,
            device=str(self.device),
            precision=self.precision,
            parameters=sum(p.numel() for p in model.parameters()),
            memory_usage_mb=self._estimate_memory_usage(model),
            inference_time_ms=0.0,  # Will be updated during inference
            accuracy=checkpoint.get("accuracy", None),
            config=model_config,
        )

        return PyTorchModelWrapper(model, model_info, self.config)

    def _load_onnx_model(
        self, model_name: str, model_path: str, config_path: Optional[str] = None
    ) -> ONNXModelWrapper:
        """Load ONNX model."""
        # Create model info
        model_info = ModelInfo(
            name=model_name,
            type="onnx",
            version="unknown",
            path=model_path,
            device=str(self.device),
            precision=self.precision,
            parameters=0,  # ONNX doesn't easily expose parameter count
            memory_usage_mb=0.0,
            inference_time_ms=0.0,
        )

        return ONNXModelWrapper(model_path, model_info, self.config)

    def _load_tensorrt_model(
        self, model_name: str, model_path: str, config_path: Optional[str] = None
    ) -> TensorRTModelWrapper:
        """Load TensorRT model."""
        # Create model info
        model_info = ModelInfo(
            name=model_name,
            type="tensorrt",
            version="unknown",
            path=model_path,
            device=str(self.device),
            precision=self.precision,
            parameters=0,
            memory_usage_mb=0.0,
            inference_time_ms=0.0,
        )

        return TensorRTModelWrapper(model_path, model_info, self.config)

    def _create_model_from_config(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create model instance from configuration."""
        model_type = model_config.get("name", "").lower()

        if "audiomae" in model_type:
            from ..models.audioMAE.model import AudioMAE

            return AudioMAE(model_config)
        elif "ast" in model_type:
            from ..models.ast.model import AudioSpectrogramTransformer

            return AudioSpectrogramTransformer(model_config)
        elif "beats" in model_type:
            from ..models.beats.model import BEATs

            return BEATs(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024**2)

    def predict(
        self,
        model: Union[str, BaseModelWrapper],
        inputs: ModelInput,
        return_probabilities: bool = True,
        apply_softmax: bool = True,
    ) -> InferenceResult:
        """
        Make predictions using a loaded model.

        Args:
            model: Model name or wrapper instance
            inputs: Model inputs
            return_probabilities: Whether to return probabilities
            apply_softmax: Whether to apply softmax to outputs

        Returns:
            Inference results
        """
        # Get model wrapper
        if isinstance(model, str):
            if model not in self.models:
                raise ValueError(f"Model {model} not loaded")
            model_wrapper = self.models[model]
        else:
            model_wrapper = model

        # Prepare inputs
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            if self.precision == "fp16":
                inputs = inputs.half()

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            outputs = model_wrapper.predict(inputs)

        inference_time = (time.time() - start_time) * 1000

        # Process outputs
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.get("logits", outputs.get("output", list(outputs.values())[0]))

        # Get predictions and probabilities
        if apply_softmax:
            probabilities = torch.softmax(logits, dim=-1)
        else:
            probabilities = logits

        predictions = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0].item()

        # Update performance stats
        model_name = model_wrapper.get_model_info().name
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = []
        self.performance_stats[model_name].append(inference_time)

        return InferenceResult(
            predictions=predictions,
            probabilities=probabilities if return_probabilities else None,
            confidence=confidence,
            inference_time_ms=inference_time,
            model_name=model_name,
        )

    def create_ensemble(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        aggregation: str = "average",
    ) -> ModelEnsemble:
        """
        Create model ensemble from loaded models.

        Args:
            model_names: List of model names to ensemble
            weights: Optional ensemble weights
            aggregation: Aggregation method

        Returns:
            Model ensemble
        """
        models = []
        for name in model_names:
            if name not in self.models:
                raise ValueError(f"Model {name} not loaded")
            models.append(self.models[name])

        return ModelEnsemble(models, weights, aggregation)

    def optimize_model(self, model_name: str, optimization_config: Dict[str, Any]) -> None:
        """
        Optimize a loaded model.

        Args:
            model_name: Name of model to optimize
            optimization_config: Optimization configuration
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        logger.info(f"Optimizing model {model_name}")
        self.models[model_name].optimize(optimization_config)

    def export_model(
        self,
        model_name: str,
        export_path: str,
        format: str = "onnx",
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """
        Export model to specified format.

        Args:
            model_name: Name of model to export
            export_path: Export file path
            format: Export format ("onnx", "torchscript", "tensorrt")
            input_shape: Input shape for tracing
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model_wrapper = self.models[model_name]

        if not isinstance(model_wrapper, PyTorchModelWrapper):
            raise ValueError("Only PyTorch models can be exported")

        model = model_wrapper.model

        if format == "onnx":
            self._export_to_onnx(model, export_path, input_shape)
        elif format == "torchscript":
            self._export_to_torchscript(model, export_path, input_shape)
        elif format == "tensorrt":
            self._export_to_tensorrt(model, export_path, input_shape)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Model {model_name} exported to {export_path}")

    def _export_to_onnx(
        self, model: nn.Module, export_path: str, input_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Export model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available")

        # Create dummy input
        if input_shape is None:
            input_shape = (1, 1, 128, 1024)  # Default for audio spectrograms

        dummy_input = torch.randn(input_shape).to(self.device)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def _export_to_torchscript(
        self, model: nn.Module, export_path: str, input_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Export model to TorchScript format."""
        if input_shape is None:
            input_shape = (1, 1, 128, 1024)

        dummy_input = torch.randn(input_shape).to(self.device)

        # Trace model
        traced_model = torch.jit.trace(model, dummy_input)

        # Save traced model
        traced_model.save(export_path)

    def _export_to_tensorrt(
        self, model: nn.Module, export_path: str, input_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Export model to TensorRT format."""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        # First export to ONNX
        onnx_path = export_path.replace(".trt", ".onnx")
        self._export_to_onnx(model, onnx_path, input_shape)

        # Then convert ONNX to TensorRT
        # This would require additional TensorRT conversion code
        logger.warning("TensorRT export not fully implemented")

    def get_model_info(self, model_name: str) -> ModelInfo:
        """
        Get information about a loaded model.

        Args:
            model_name: Name of model

        Returns:
            Model information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        return self.models[model_name].get_model_info()

    def list_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.

        Args:
            model_name: Name of model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Model {model_name} unloaded")

        # Also remove from cache
        cache_keys_to_remove = [
            key for key in self.model_cache.keys() if key.startswith(model_name)
        ]
        for key in cache_keys_to_remove:
            del self.model_cache[key]

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for all models.

        Returns:
            Dictionary of performance stats per model
        """
        stats = {}

        for model_name, times in self.performance_stats.items():
            if times:
                stats[model_name] = {
                    "mean_inference_time_ms": np.mean(times),
                    "std_inference_time_ms": np.std(times),
                    "min_inference_time_ms": np.min(times),
                    "max_inference_time_ms": np.max(times),
                    "total_inferences": len(times),
                }

        return stats

    def clear_cache(self) -> None:
        """Clear model cache."""
        self.model_cache.clear()
        logger.info("Model cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory usage information
        """
        memory_info = {}

        if self.device.type == "cuda":
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

        # Estimate model memory usage
        total_model_memory = sum(
            model.get_model_info().memory_usage_mb for model in self.models.values()
        )
        memory_info["models_memory_mb"] = total_model_memory

        return memory_info


# Convenience functions
def load_model_simple(
    model_path: str, model_type: str = "audioMAE", device: str = "auto"
) -> BaseModelWrapper:
    """
    Simple model loading function.

    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        device: Device to load model on

    Returns:
        Loaded model wrapper
    """
    config = {"device": device, "precision": "fp32"}
    manager = ModelManager(config)

    return manager.load_model("default", model_path, "pytorch")
