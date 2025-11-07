#
# Plan:
# 1. Implement advanced learning rate schedulers with warmup support
# 2. Create CosineAnnealingWithWarmup for transformer training
# 3. Add LinearWarmupScheduler and ExponentialWarmupScheduler
# 4. Implement PolynomialDecayScheduler for gradual learning rate reduction
# 5. Create OneCycleLRScheduler for super-convergence training
# 6. Add CyclicLRWithWarmup for cyclic learning rate strategies
# 7. Implement ReduceLROnPlateauWithWarmup for adaptive scheduling
# 8. Create SchedulerFactory and WarmupWrapper utilities
#

"""
Learning Rate Schedulers for SereneSense Training

This module implements sophisticated learning rate scheduling strategies
optimized for training large audio transformer models with improved
convergence and generalization performance.

Key Features:
- Warmup-enabled schedulers for transformer training
- Cosine annealing with restart capabilities
- Cyclic learning rate strategies
- Polynomial decay scheduling
- Adaptive scheduling based on validation metrics
- OneCycle learning rate for super-convergence
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import logging

logger = logging.getLogger(__name__)


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.

    Combines linear warmup with cosine annealing decay, which is highly
    effective for training transformer models from scratch.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Initialize cosine annealing with warmup scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            warmup_start_lr: Learning rate at start of warmup
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            warmup_ratio = self.last_epoch / self.warmup_steps
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.warmup_start_lr + warmup_ratio * (base_lr - self.warmup_start_lr)
                lrs.append(lr)
            return lrs
        else:
            # Cosine annealing phase
            cosine_steps = self.total_steps - self.warmup_steps
            current_cosine_step = self.last_epoch - self.warmup_steps

            lrs = []
            for base_lr in self.base_lrs:
                cosine_ratio = 0.5 * (1 + math.cos(math.pi * current_cosine_step / cosine_steps))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine_ratio
                lrs.append(lr)
            return lrs


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup learning rate scheduler.

    Linearly increases learning rate from a small value to the target
    learning rate over a specified number of steps.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Initialize linear warmup scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            warmup_start_lr: Learning rate at start of warmup
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_ratio = self.last_epoch / self.warmup_steps
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.warmup_start_lr + warmup_ratio * (base_lr - self.warmup_start_lr)
                lrs.append(lr)
            return lrs
        else:
            # Constant phase after warmup
            return self.base_lrs


class ExponentialWarmupScheduler(_LRScheduler):
    """
    Exponential warmup learning rate scheduler.

    Exponentially increases learning rate to the target value,
    providing a smoother warmup curve than linear warmup.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        warmup_start_lr: float = 1e-6,
        gamma: float = 0.99,
        last_epoch: int = -1,
    ):
        """
        Initialize exponential warmup scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            warmup_start_lr: Learning rate at start of warmup
            gamma: Decay factor for exponential decay after warmup
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.gamma = gamma

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Exponential warmup phase
            warmup_ratio = self.last_epoch / self.warmup_steps
            lrs = []
            for base_lr in self.base_lrs:
                # Exponential interpolation
                log_start = math.log(self.warmup_start_lr)
                log_end = math.log(base_lr)
                log_lr = log_start + warmup_ratio * (log_end - log_start)
                lr = math.exp(log_lr)
                lrs.append(lr)
            return lrs
        else:
            # Exponential decay phase
            decay_steps = self.last_epoch - self.warmup_steps
            lrs = []
            for base_lr in self.base_lrs:
                lr = base_lr * (self.gamma**decay_steps)
                lrs.append(lr)
            return lrs


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial decay learning rate scheduler with warmup.

    Applies polynomial decay to the learning rate after warmup,
    providing smooth and controllable decay curves.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Initialize polynomial decay scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            power: Power of polynomial decay
            min_lr: Minimum learning rate
            warmup_start_lr: Learning rate at start of warmup
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            warmup_ratio = self.last_epoch / self.warmup_steps
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.warmup_start_lr + warmup_ratio * (base_lr - self.warmup_start_lr)
                lrs.append(lr)
            return lrs
        else:
            # Polynomial decay phase
            decay_steps = self.total_steps - self.warmup_steps
            current_decay_step = self.last_epoch - self.warmup_steps

            lrs = []
            for base_lr in self.base_lrs:
                decay_ratio = 1 - (current_decay_step / decay_steps)
                decay_ratio = max(0, decay_ratio) ** self.power
                lr = self.min_lr + (base_lr - self.min_lr) * decay_ratio
                lrs.append(lr)
            return lrs


class OneCycleLRScheduler(_LRScheduler):
    """
    OneCycle learning rate scheduler for super-convergence.

    Implements the OneCycle learning rate policy that can lead to
    super-convergence, enabling faster training with better results.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        """
        Initialize OneCycle learning rate scheduler.

        Args:
            optimizer: Wrapped optimizer
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing learning rate
            anneal_strategy: Annealing strategy ('cos' or 'linear')
            div_factor: Determines initial learning rate (max_lr / div_factor)
            final_div_factor: Determines minimum learning rate
            last_epoch: The index of the last epoch
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Calculate phase boundaries
        self.step_up_size = int(self.pct_start * self.total_steps)
        self.step_down_size = self.total_steps - self.step_up_size

        # Calculate learning rate bounds
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step"""
        lrs = []

        for base_lr in self.base_lrs:
            if self.last_epoch <= self.step_up_size:
                # Ascending phase
                pct = self.last_epoch / self.step_up_size
                if self.anneal_strategy == "cos":
                    lr = self.initial_lr + (self.max_lr - self.initial_lr) * 0.5 * (
                        1 + math.cos(math.pi + math.pi * pct)
                    )
                else:  # linear
                    lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
            else:
                # Descending phase
                pct = (self.last_epoch - self.step_up_size) / self.step_down_size
                if self.anneal_strategy == "cos":
                    lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                        1 + math.cos(math.pi * pct)
                    )
                else:  # linear
                    lr = self.max_lr - (self.max_lr - self.min_lr) * pct

            lrs.append(lr)

        return lrs


class CyclicLRWithWarmup(_LRScheduler):
    """
    Cyclic learning rate scheduler with warmup.

    Combines warmup with cyclic learning rate policy, which can help
    escape local minima and improve convergence.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        warmup_steps: int = 0,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Initialize cyclic learning rate with warmup scheduler.

        Args:
            optimizer: Wrapped optimizer
            base_lr: Lower boundary of learning rate cycle
            max_lr: Upper boundary of learning rate cycle
            step_size_up: Number of steps in increasing half of cycle
            step_size_down: Number of steps in decreasing half of cycle
            mode: Cyclic mode ('triangular', 'triangular2', 'exp_range')
            gamma: Constant for 'exp_range' mode
            warmup_steps: Number of warmup steps
            warmup_start_lr: Learning rate at start of warmup
            last_epoch: The index of the last epoch
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

        self.cycle_size = self.step_size_up + self.step_size_down

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_ratio = self.last_epoch / self.warmup_steps
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.warmup_start_lr + warmup_ratio * (self.base_lr - self.warmup_start_lr)
                lrs.append(lr)
            return lrs
        else:
            # Cyclic phase
            cycle_epoch = (self.last_epoch - self.warmup_steps) % self.cycle_size
            cycle_number = (self.last_epoch - self.warmup_steps) // self.cycle_size

            if cycle_epoch <= self.step_size_up:
                # Ascending phase
                x = cycle_epoch / self.step_size_up
            else:
                # Descending phase
                x = (self.cycle_size - cycle_epoch) / self.step_size_down

            # Apply mode-specific scaling
            if self.mode == "triangular":
                scale = 1.0
            elif self.mode == "triangular2":
                scale = 1.0 / (2.0**cycle_number)
            elif self.mode == "exp_range":
                scale = self.gamma ** (self.last_epoch - self.warmup_steps)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            lrs = []
            for base_lr in self.base_lrs:
                lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, x) * scale
                lrs.append(lr)
            return lrs


class ReduceLROnPlateauWithWarmup:
    """
    Reduce learning rate on plateau with warmup support.

    Combines warmup with adaptive learning rate reduction based on
    validation metrics, providing both stability and adaptivity.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        warmup_steps: int = 0,
        warmup_start_lr: float = 1e-6,
    ):
        """
        Initialize ReduceLROnPlateau with warmup scheduler.

        Args:
            optimizer: Wrapped optimizer
            mode: 'min' or 'max' for metric optimization direction
            factor: Factor to reduce learning rate by
            patience: Number of epochs with no improvement after which LR is reduced
            threshold: Threshold for measuring new optimum
            threshold_mode: 'rel' or 'abs' threshold mode
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Minimum learning rate
            eps: Minimal decay applied to LR
            warmup_steps: Number of warmup steps
            warmup_start_lr: Learning rate at start of warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0

        # Store initial learning rates for warmup
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Initialize ReduceLROnPlateau scheduler
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps
        )

    def step(self, metric: Optional[float] = None):
        """
        Update learning rate.

        Args:
            metric: Metric value for plateau detection (only used after warmup)
        """
        if self.current_step < self.warmup_steps:
            # Warmup phase
            warmup_ratio = self.current_step / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                base_lr = self.base_lrs[i]
                lr = self.warmup_start_lr + warmup_ratio * (base_lr - self.warmup_start_lr)
                group["lr"] = lr
        else:
            # Plateau reduction phase
            if metric is not None:
                self.plateau_scheduler.step(metric)

        self.current_step += 1

    def state_dict(self):
        """Return state dict"""
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]
        self.plateau_scheduler.load_state_dict(state_dict["plateau_scheduler"])


class WarmupWrapper:
    """
    Generic warmup wrapper for any PyTorch scheduler.

    Adds warmup capability to existing PyTorch learning rate schedulers.
    """

    def __init__(
        self,
        scheduler: _LRScheduler,
        warmup_steps: int,
        warmup_start_lr: float = 1e-6,
        warmup_mode: str = "linear",
    ):
        """
        Initialize warmup wrapper.

        Args:
            scheduler: Base PyTorch scheduler to wrap
            warmup_steps: Number of warmup steps
            warmup_start_lr: Learning rate at start of warmup
            warmup_mode: Warmup mode ('linear' or 'exponential')
        """
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_mode = warmup_mode
        self.current_step = 0

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in scheduler.optimizer.param_groups]

    def step(self, *args, **kwargs):
        """Update learning rate"""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            if self.warmup_mode == "linear":
                warmup_ratio = self.current_step / self.warmup_steps
                for i, group in enumerate(self.scheduler.optimizer.param_groups):
                    base_lr = self.base_lrs[i]
                    lr = self.warmup_start_lr + warmup_ratio * (base_lr - self.warmup_start_lr)
                    group["lr"] = lr
            elif self.warmup_mode == "exponential":
                warmup_ratio = self.current_step / self.warmup_steps
                for i, group in enumerate(self.scheduler.optimizer.param_groups):
                    base_lr = self.base_lrs[i]
                    log_start = math.log(self.warmup_start_lr)
                    log_end = math.log(base_lr)
                    log_lr = log_start + warmup_ratio * (log_end - log_start)
                    lr = math.exp(log_lr)
                    group["lr"] = lr
        else:
            # Use base scheduler
            self.scheduler.step(*args, **kwargs)

        self.current_step += 1

    def state_dict(self):
        """Return state dict"""
        return {
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]


class SchedulerState:
    """
    Utilities for managing scheduler state and diagnostics.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.lr_history = []

    def log_lr(self):
        """Log current learning rates"""
        if hasattr(self.scheduler, "optimizer"):
            current_lrs = [group["lr"] for group in self.scheduler.optimizer.param_groups]
            self.lr_history.append(current_lrs)
            return current_lrs
        return []

    def get_lr_history(self) -> List[List[float]]:
        """Get learning rate history"""
        return self.lr_history

    def plot_lr_schedule(self, steps: int = 1000):
        """
        Simulate and plot learning rate schedule.

        Args:
            steps: Number of steps to simulate
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return

        # Create dummy optimizer and scheduler for simulation
        dummy_param = torch.nn.Parameter(torch.randn(1))
        dummy_optimizer = torch.optim.SGD([dummy_param], lr=1.0)

        # Create scheduler of same type with same parameters
        scheduler_class = type(self.scheduler)
        if hasattr(self.scheduler, "__dict__"):
            scheduler_kwargs = {
                k: v
                for k, v in self.scheduler.__dict__.items()
                if not k.startswith("_") and k != "optimizer"
            }
            test_scheduler = scheduler_class(dummy_optimizer, **scheduler_kwargs)
        else:
            logger.warning("Cannot simulate scheduler - missing parameters")
            return

        # Simulate learning rate schedule
        lrs = []
        for step in range(steps):
            lrs.append(dummy_optimizer.param_groups[0]["lr"])
            test_scheduler.step()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title(f"{scheduler_class.__name__} Schedule")
        plt.grid(True)
        plt.show()


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers.

    Provides a unified interface for creating different types of schedulers
    with warmup and other advanced features.
    """

    @staticmethod
    def create_scheduler(
        scheduler_type: str,
        optimizer: optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_start_lr: float = 1e-6,
        **kwargs,
    ) -> Union[_LRScheduler, Any]:
        """
        Create scheduler with specified configuration.

        Args:
            scheduler_type: Type of scheduler
            optimizer: Optimizer to schedule
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            warmup_start_lr: Learning rate at start of warmup
            **kwargs: Additional scheduler-specific arguments

        Returns:
            Configured learning rate scheduler
        """
        scheduler_type = scheduler_type.lower()

        if scheduler_type == "cosine_warmup":
            return CosineAnnealingWithWarmup(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_start_lr=warmup_start_lr,
                **kwargs,
            )

        elif scheduler_type == "linear_warmup":
            return LinearWarmupScheduler(
                optimizer, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, **kwargs
            )

        elif scheduler_type == "exponential_warmup":
            return ExponentialWarmupScheduler(
                optimizer, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, **kwargs
            )

        elif scheduler_type == "polynomial":
            return PolynomialDecayScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_start_lr=warmup_start_lr,
                **kwargs,
            )

        elif scheduler_type == "onecycle":
            max_lr = kwargs.pop("max_lr", optimizer.param_groups[0]["lr"])
            return OneCycleLRScheduler(optimizer, max_lr=max_lr, total_steps=total_steps, **kwargs)

        elif scheduler_type == "cyclic_warmup":
            base_lr = kwargs.pop("base_lr", optimizer.param_groups[0]["lr"] * 0.1)
            max_lr = kwargs.pop("max_lr", optimizer.param_groups[0]["lr"])
            step_size_up = kwargs.pop("step_size_up", total_steps // 20)

            return CyclicLRWithWarmup(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                **kwargs,
            )

        elif scheduler_type == "plateau_warmup":
            return ReduceLROnPlateauWithWarmup(
                optimizer, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, **kwargs
            )

        elif scheduler_type == "constant":
            # Constant learning rate with optional warmup
            if warmup_steps > 0:
                constant_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
                return WarmupWrapper(
                    constant_scheduler, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr
                )
            else:
                return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    @staticmethod
    def get_recommended_scheduler(
        model_type: str, optimizer: optim.Optimizer, total_steps: int, **kwargs
    ) -> Union[_LRScheduler, Any]:
        """
        Get recommended scheduler for specific model type.

        Args:
            model_type: Type of model ('transformer', 'cnn', 'hybrid')
            optimizer: Optimizer to schedule
            total_steps: Total number of training steps
            **kwargs: Additional configuration

        Returns:
            Recommended scheduler for the model type
        """
        if model_type.lower() in ["transformer", "audiomae", "ast", "beats"]:
            # Transformers benefit from cosine annealing with warmup
            warmup_steps = kwargs.get("warmup_steps", total_steps // 10)
            return SchedulerFactory.create_scheduler(
                "cosine_warmup", optimizer, total_steps, warmup_steps=warmup_steps, **kwargs
            )

        elif model_type.lower() in ["cnn", "resnet", "efficientnet"]:
            # CNNs often work well with step decay or polynomial decay
            warmup_steps = kwargs.get("warmup_steps", total_steps // 20)
            return SchedulerFactory.create_scheduler(
                "polynomial", optimizer, total_steps, warmup_steps=warmup_steps, power=0.9, **kwargs
            )

        elif model_type.lower() in ["hybrid", "mixed"]:
            # Hybrid models can benefit from cyclic learning rates
            warmup_steps = kwargs.get("warmup_steps", total_steps // 15)
            return SchedulerFactory.create_scheduler(
                "cyclic_warmup", optimizer, total_steps, warmup_steps=warmup_steps, **kwargs
            )

        else:
            # Default to cosine annealing with warmup
            warmup_steps = kwargs.get("warmup_steps", total_steps // 10)
            return SchedulerFactory.create_scheduler(
                "cosine_warmup", optimizer, total_steps, warmup_steps=warmup_steps, **kwargs
            )
