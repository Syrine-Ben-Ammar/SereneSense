#
# Plan:
# 1. Implement advanced optimizers (AdamW with decoupled weight decay, SAM, Lookahead)
# 2. Create parameter group builders for layer-wise learning rates
# 3. Add TransformerParameterGroups for transformer-specific optimization
# 4. Implement AudioModelParameterGroups for audio model optimization
# 5. Create OptimizerFactory for easy optimizer creation
# 6. Add GradientClipping utilities for training stability
# 7. Implement ParameterStatistics for monitoring training dynamics
# 8. Support for advanced optimization techniques (gradient accumulation, etc.)
#

"""
Advanced Optimizers for SereneSense Training

This module implements sophisticated optimization strategies specifically
designed for training large audio transformer models with improved
convergence and generalization.

Key Features:
- Advanced optimizers (SAM, Lookahead, AdamW variants)
- Layer-wise learning rate scheduling
- Transformer-specific parameter grouping
- Gradient clipping and normalization
- Parameter statistics and monitoring
- Memory-efficient optimization strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AdamWWithDecoupledWeightDecay(Optimizer):
    """
    AdamW optimizer with properly decoupled weight decay.

    Implements the AdamW algorithm with decoupled weight decay as described
    in "Decoupled Weight Decay Regularization" paper, which often performs
    better than standard Adam with L2 regularization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterable of parameters or parameter groups
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient
            amsgrad: Whether to use AMSGrad variant
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsgrad"]:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Decoupled weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class SAMOptimizer(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    Seeks parameters that lie in neighborhoods having uniformly low loss,
    improving generalization by finding flatter minima.
    """

    def __init__(self, base_optimizer: Optimizer, rho: float = 0.05, adaptive: bool = False):
        """
        Initialize SAM optimizer.

        Args:
            base_optimizer: Base optimizer (e.g., SGD, Adam)
            rho: Neighborhood size for sharpness measurement
            adaptive: Whether to use adaptive neighborhood size
        """
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive

        # Store original parameter values
        self.param_groups = self.base_optimizer.param_groups
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: ascent to find adversarial weights"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Store original parameters
                self.state[p]["old_p"] = p.data.clone()

                # Adaptive neighborhood size
                if self.adaptive:
                    scale = self.rho / (grad_norm + 1e-12) * p.grad.norm()

                # Ascent step
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: descent with the base optimizer"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameters
                p.data = self.state[p]["old_p"]

        # Apply base optimizer step
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Complete SAM step (both ascent and descent)"""
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # Enable grad for closure

        # First step: ascent
        self.first_step(zero_grad=True)

        # Re-evaluate loss at adversarial point
        closure()

        # Second step: descent
        self.second_step()

    def _grad_norm(self):
        """Compute gradient norm across all parameters"""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(
                        dtype=torch.float32
                    )
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        ).to(shared_device)
        return norm

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.base_optimizer.load_state_dict(state_dict)

    def state_dict(self):
        """Return state dict"""
        return self.base_optimizer.state_dict()

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients"""
        self.base_optimizer.zero_grad(set_to_none)


class LookaheadOptimizer(Optimizer):
    """
    Lookahead optimizer wrapper.

    Maintains two sets of weights: fast weights updated by the base optimizer
    and slow weights updated by lookahead. Improves convergence stability.
    """

    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        """
        Initialize Lookahead optimizer.

        Args:
            base_optimizer: Base optimizer for fast weights
            k: Number of fast weight updates before slow weight update
            alpha: Interpolation factor for slow weight update
        """
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha

        self.param_groups = self.base_optimizer.param_groups
        self.state = defaultdict(dict)

        # Initialize slow weights
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["slow_weights"] = p.data.clone()
                param_state["step_count"] = 0

    def step(self, closure=None):
        """Perform optimization step"""
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["step_count"] += 1

                # Update slow weights every k steps
                if param_state["step_count"] % self.k == 0:
                    slow_weights = param_state["slow_weights"]
                    # Interpolate between slow and fast weights
                    slow_weights.add_(p.data - slow_weights, alpha=self.alpha)
                    p.data.copy_(slow_weights)

        return loss

    def state_dict(self):
        """Return state dict"""
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()
        }
        fast_state_dict["slow_state"] = slow_state
        return fast_state_dict

    def load_state_dict(self, state_dict):
        """Load state dict"""
        slow_state_dict = state_dict.pop("slow_state", {})
        self.base_optimizer.load_state_dict(state_dict)

        # Restore slow state
        for k, v in slow_state_dict.items():
            self.state[k] = v


class LayerWiseDecayParameterGroups:
    """
    Creates parameter groups with layer-wise learning rate decay.

    Assigns different learning rates to different layers of the model,
    typically with earlier layers having lower learning rates.
    """

    def __init__(self, model: nn.Module, base_lr: float, decay_factor: float = 0.8):
        """
        Initialize layer-wise parameter groups.

        Args:
            model: Neural network model
            base_lr: Base learning rate
            decay_factor: Decay factor for each layer
        """
        self.model = model
        self.base_lr = base_lr
        self.decay_factor = decay_factor

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups with layer-wise learning rates"""
        parameter_groups = []

        # Count total number of layers
        total_layers = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.TransformerEncoderLayer)):
                total_layers += 1

        # Create parameter groups
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.TransformerEncoderLayer)):
                # Calculate learning rate for this layer
                layer_lr = self.base_lr * (self.decay_factor ** (total_layers - layer_idx - 1))

                # Get parameters for this layer
                layer_params = list(module.parameters())
                if layer_params:
                    parameter_groups.append({"params": layer_params, "lr": layer_lr, "name": name})

                layer_idx += 1

        return parameter_groups


class TransformerParameterGroups:
    """
    Creates parameter groups optimized for transformer architectures.

    Groups parameters by type (embeddings, attention, feedforward, etc.)
    with appropriate learning rates for each component.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        embedding_lr_mult: float = 0.1,
        attention_lr_mult: float = 1.0,
        feedforward_lr_mult: float = 1.0,
        head_lr_mult: float = 10.0,
    ):
        """
        Initialize transformer parameter groups.

        Args:
            model: Transformer model
            base_lr: Base learning rate
            embedding_lr_mult: Learning rate multiplier for embeddings
            attention_lr_mult: Learning rate multiplier for attention layers
            feedforward_lr_mult: Learning rate multiplier for feedforward layers
            head_lr_mult: Learning rate multiplier for classification head
        """
        self.model = model
        self.base_lr = base_lr
        self.embedding_lr_mult = embedding_lr_mult
        self.attention_lr_mult = attention_lr_mult
        self.feedforward_lr_mult = feedforward_lr_mult
        self.head_lr_mult = head_lr_mult

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create transformer-specific parameter groups"""
        parameter_groups = []

        # Collect parameters by type
        embedding_params = []
        attention_params = []
        feedforward_params = []
        head_params = []
        other_params = []

        for name, module in self.model.named_modules():
            module_params = list(module.parameters())

            if "embedding" in name.lower() or "pos_embed" in name.lower():
                embedding_params.extend(module_params)
            elif "attention" in name.lower() or "attn" in name.lower():
                attention_params.extend(module_params)
            elif "feedforward" in name.lower() or "mlp" in name.lower() or "ffn" in name.lower():
                feedforward_params.extend(module_params)
            elif "head" in name.lower() or "classifier" in name.lower():
                head_params.extend(module_params)
            else:
                other_params.extend(module_params)

        # Create parameter groups
        if embedding_params:
            parameter_groups.append(
                {
                    "params": embedding_params,
                    "lr": self.base_lr * self.embedding_lr_mult,
                    "name": "embeddings",
                }
            )

        if attention_params:
            parameter_groups.append(
                {
                    "params": attention_params,
                    "lr": self.base_lr * self.attention_lr_mult,
                    "name": "attention",
                }
            )

        if feedforward_params:
            parameter_groups.append(
                {
                    "params": feedforward_params,
                    "lr": self.base_lr * self.feedforward_lr_mult,
                    "name": "feedforward",
                }
            )

        if head_params:
            parameter_groups.append(
                {
                    "params": head_params,
                    "lr": self.base_lr * self.head_lr_mult,
                    "name": "classification_head",
                }
            )

        if other_params:
            parameter_groups.append({"params": other_params, "lr": self.base_lr, "name": "other"})

        return parameter_groups


class AudioModelParameterGroups:
    """
    Creates parameter groups specifically for audio models.

    Optimizes parameter grouping for audio transformer architectures
    like AudioMAE, AST, and BEATs with audio-specific considerations.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        patch_embed_lr_mult: float = 0.1,
        tokenizer_lr_mult: float = 0.5,
        encoder_lr_mult: float = 1.0,
        decoder_lr_mult: float = 1.5,
    ):
        """
        Initialize audio model parameter groups.

        Args:
            model: Audio model (AudioMAE, AST, BEATs)
            base_lr: Base learning rate
            patch_embed_lr_mult: Multiplier for patch embedding parameters
            tokenizer_lr_mult: Multiplier for tokenizer parameters
            encoder_lr_mult: Multiplier for encoder parameters
            decoder_lr_mult: Multiplier for decoder parameters
        """
        self.model = model
        self.base_lr = base_lr
        self.patch_embed_lr_mult = patch_embed_lr_mult
        self.tokenizer_lr_mult = tokenizer_lr_mult
        self.encoder_lr_mult = encoder_lr_mult
        self.decoder_lr_mult = decoder_lr_mult

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create audio model-specific parameter groups"""
        parameter_groups = []
        handled_params = set()

        # Patch embedding parameters
        patch_embed_params = []
        for name, module in self.model.named_modules():
            if "patch_embed" in name or "patch_projection" in name:
                for param in module.parameters():
                    if id(param) not in handled_params:
                        patch_embed_params.append(param)
                        handled_params.add(id(param))

        if patch_embed_params:
            parameter_groups.append(
                {
                    "params": patch_embed_params,
                    "lr": self.base_lr * self.patch_embed_lr_mult,
                    "name": "patch_embedding",
                }
            )

        # Tokenizer parameters (for BEATs)
        tokenizer_params = []
        for name, module in self.model.named_modules():
            if "tokenizer" in name or "codebook" in name:
                for param in module.parameters():
                    if id(param) not in handled_params:
                        tokenizer_params.append(param)
                        handled_params.add(id(param))

        if tokenizer_params:
            parameter_groups.append(
                {
                    "params": tokenizer_params,
                    "lr": self.base_lr * self.tokenizer_lr_mult,
                    "name": "tokenizer",
                }
            )

        # Encoder parameters
        encoder_params = []
        for name, module in self.model.named_modules():
            if "encoder" in name and "decoder" not in name:
                for param in module.parameters():
                    if id(param) not in handled_params:
                        encoder_params.append(param)
                        handled_params.add(id(param))

        if encoder_params:
            parameter_groups.append(
                {
                    "params": encoder_params,
                    "lr": self.base_lr * self.encoder_lr_mult,
                    "name": "encoder",
                }
            )

        # Decoder parameters (for AudioMAE)
        decoder_params = []
        for name, module in self.model.named_modules():
            if "decoder" in name:
                for param in module.parameters():
                    if id(param) not in handled_params:
                        decoder_params.append(param)
                        handled_params.add(id(param))

        if decoder_params:
            parameter_groups.append(
                {
                    "params": decoder_params,
                    "lr": self.base_lr * self.decoder_lr_mult,
                    "name": "decoder",
                }
            )

        # Remaining parameters
        other_params = []
        for param in self.model.parameters():
            if id(param) not in handled_params:
                other_params.append(param)

        if other_params:
            parameter_groups.append({"params": other_params, "lr": self.base_lr, "name": "other"})

        return parameter_groups


class OptimizerFactory:
    """
    Factory for creating optimizers with various configurations.

    Provides a unified interface for creating different types of optimizers
    with appropriate parameter grouping strategies.
    """

    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        parameters: Union[List[Dict[str, Any]], List[torch.nn.Parameter]],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> Optimizer:
        """
        Create optimizer with specified configuration.

        Args:
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'sam', 'lookahead')
            parameters: Model parameters or parameter groups
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Configured optimizer
        """
        optimizer_type = optimizer_type.lower()

        if optimizer_type == "adam":
            return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs)

        elif optimizer_type == "adamw":
            return AdamWWithDecoupledWeightDecay(
                parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs
            )

        elif optimizer_type == "sgd":
            return optim.SGD(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get("momentum", 0.9),
                **{k: v for k, v in kwargs.items() if k != "momentum"},
            )

        elif optimizer_type == "sam":
            base_optimizer_type = kwargs.pop("base_optimizer", "sgd")
            base_optimizer = OptimizerFactory.create_optimizer(
                base_optimizer_type, parameters, learning_rate, weight_decay, **kwargs
            )
            return SAMOptimizer(base_optimizer, **kwargs)

        elif optimizer_type == "lookahead":
            base_optimizer_type = kwargs.pop("base_optimizer", "adamw")
            base_optimizer = OptimizerFactory.create_optimizer(
                base_optimizer_type, parameters, learning_rate, weight_decay, **kwargs
            )
            return LookaheadOptimizer(base_optimizer, **kwargs)

        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class GradientClipping:
    """
    Utilities for gradient clipping and normalization.

    Provides various gradient clipping strategies to improve training stability
    and prevent gradient explosion in large models.
    """

    @staticmethod
    def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0) -> float:
        """
        Clip gradients by norm.

        Args:
            parameters: Model parameters
            max_norm: Maximum norm allowed
            norm_type: Type of norm to compute

        Returns:
            Total norm of parameters
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    @staticmethod
    def clip_grad_value(parameters, clip_value: float):
        """
        Clip gradients by value.

        Args:
            parameters: Model parameters
            clip_value: Maximum absolute value allowed
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    @staticmethod
    def adaptive_grad_clip(parameters, clip_factor: float = 0.01, eps: float = 1e-3) -> float:
        """
        Adaptive gradient clipping based on parameter norms.

        Args:
            parameters: Model parameters
            clip_factor: Clipping factor relative to parameter norm
            eps: Small constant for numerical stability

        Returns:
            Clipping ratio applied
        """
        param_norm = 0.0
        grad_norm = 0.0

        for p in parameters:
            if p.grad is not None:
                param_norm += p.data.norm(dtype=torch.float32).item() ** 2
                grad_norm += p.grad.data.norm(dtype=torch.float32).item() ** 2

        param_norm = param_norm**0.5
        grad_norm = grad_norm**0.5

        # Compute adaptive clipping threshold
        clip_threshold = clip_factor * param_norm / (grad_norm + eps)
        clip_threshold = min(clip_threshold, 1.0)

        # Apply clipping
        if clip_threshold < 1.0:
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_threshold)

        return clip_threshold


class ParameterStatistics:
    """
    Utilities for monitoring parameter and gradient statistics.

    Tracks various statistics during training to help diagnose
    optimization issues and monitor training dynamics.
    """

    def __init__(self):
        self.statistics = defaultdict(list)

    def compute_statistics(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute parameter and gradient statistics.

        Args:
            model: Neural network model

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Parameter statistics
        param_norm = 0.0
        param_count = 0

        # Gradient statistics
        grad_norm = 0.0
        grad_count = 0
        grad_max = 0.0
        grad_min = float("inf")

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Parameter statistics
                param_norm += param.data.norm(dtype=torch.float32).item() ** 2
                param_count += param.numel()

                # Gradient statistics
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(dtype=torch.float32).item() ** 2
                    grad_count += param.grad.numel()
                    grad_max = max(grad_max, param.grad.data.abs().max().item())
                    grad_min = min(grad_min, param.grad.data.abs().min().item())

        # Compile statistics
        stats["param_norm"] = param_norm**0.5
        stats["param_count"] = param_count
        stats["param_norm_avg"] = (param_norm**0.5) / param_count if param_count > 0 else 0.0

        if grad_count > 0:
            stats["grad_norm"] = grad_norm**0.5
            stats["grad_count"] = grad_count
            stats["grad_norm_avg"] = (grad_norm**0.5) / grad_count
            stats["grad_max"] = grad_max
            stats["grad_min"] = grad_min if grad_min != float("inf") else 0.0
            stats["grad_ratio"] = (grad_norm**0.5) / (param_norm**0.5) if param_norm > 0 else 0.0
        else:
            stats.update(
                {
                    "grad_norm": 0.0,
                    "grad_count": 0,
                    "grad_norm_avg": 0.0,
                    "grad_max": 0.0,
                    "grad_min": 0.0,
                    "grad_ratio": 0.0,
                }
            )

        # Store statistics history
        for key, value in stats.items():
            self.statistics[key].append(value)

        return stats

    def get_statistics_history(self) -> Dict[str, List[float]]:
        """Get complete statistics history"""
        return dict(self.statistics)

    def reset_statistics(self):
        """Reset statistics history"""
        self.statistics.clear()
