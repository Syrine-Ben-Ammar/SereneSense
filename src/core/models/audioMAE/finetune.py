#
# Plan:
# 1. Implement AudioMAE fine-tuner for downstream classification
# 2. Classification head with dropout and label smoothing
# 3. Layer freezing strategies for transfer learning
# 4. Advanced optimization techniques (layer decay, mixup, cutmix)
# 5. Military-specific fine-tuning configurations
# 6. Performance monitoring and early stopping
# 7. Model compression and quantization hooks
# 8. Real-time inference optimization
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import math
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

from .model import AudioMAE, AudioMAEConfig
from ...utils.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

@dataclass
class FinetuneConfig:
    """Configuration for AudioMAE fine-tuning."""
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    max_epochs: int = 100
    
    # Layer-wise learning rate decay
    layer_decay: float = 0.75
    use_layer_decay: bool = True
    
    # Regularization
    dropout: float = 0.1
    drop_path: float = 0.1
    label_smoothing: float = 0.1
    
    # Data augmentation
    mixup: float = 0.8
    cutmix: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = 'batch'
    
    # Optimization
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    lr_scheduler: str = 'cosine'  # 'cosine', 'linear', 'step'
    min_lr: float = 1e-6
    
    # Mixed precision training
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # Transfer learning
    freeze_encoder_epochs: int = 0  # Number of epochs to freeze encoder
    freeze_patch_embed: bool = False
    freeze_pos_embed: bool = False
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Validation and logging
    val_frequency: int = 1  # Validate every N epochs
    save_frequency: int = 10  # Save checkpoint every N epochs
    log_frequency: int = 50  # Log every N steps
    
    # Model compression
    use_knowledge_distillation: bool = False
    distillation_alpha: float = 0.5
    distillation_temperature: float = 4.0
    
    # Military-specific optimizations
    target_accuracy: float = 0.91  # Target accuracy for MAD dataset
    max_inference_time_ms: float = 10.0  # Maximum inference time
    enable_quantization_aware_training: bool = False


class ClassificationHead(nn.Module):
    """Classification head for AudioMAE fine-tuning."""
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_norm: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        if use_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        if isinstance(self.classifier, nn.Linear):
            # Truncated normal initialization
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of classification head.
        
        Args:
            x: Input features [B, embed_dim]
            
        Returns:
            Classification logits [B, num_classes]
        """
        x = self.norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class MixupCutmix:
    """Mixup and Cutmix data augmentation for spectrograms."""
    
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = 'batch'
    ):
        """
        Initialize Mixup/Cutmix augmentation.
        
        Args:
            mixup_alpha: Mixup interpolation strength
            cutmix_alpha: Cutmix interpolation strength  
            prob: Probability of applying augmentation
            switch_prob: Probability of switching between mixup and cutmix
            mode: Augmentation mode ('batch', 'pair')
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
    
    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup or cutmix augmentation.
        
        Args:
            x: Input tensor [B, C, H, W]
            target: Target labels [B] or [B, num_classes]
            
        Returns:
            Augmented input and mixed targets
        """
        if np.random.rand() > self.prob:
            return x, target
        
        # Choose between mixup and cutmix
        if np.random.rand() < self.switch_prob:
            return self._mixup(x, target)
        else:
            return self._cutmix(x, target)
    
    def _mixup(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation."""
        batch_size = x.shape[0]
        
        # Sample lambda
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Mix targets
        if target.dim() == 1:  # Integer labels
            mixed_target = {
                'target_a': target,
                'target_b': target[index],
                'lam': lam
            }
        else:  # One-hot labels
            mixed_target = lam * target + (1 - lam) * target[index]
        
        return mixed_x, mixed_target
    
    def _cutmix(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix augmentation."""
        batch_size = x.shape[0]
        
        # Sample lambda
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Generate random box
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Clamp box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Mix targets
        if target.dim() == 1:  # Integer labels
            mixed_target = {
                'target_a': target,
                'target_b': target[index],
                'lam': lam
            }
        else:  # One-hot labels
            mixed_target = lam * target + (1 - lam) * target[index]
        
        return mixed_x, mixed_target


class AudioMAEFinetuner:
    """
    AudioMAE fine-tuner for downstream classification tasks.
    
    Implements advanced fine-tuning techniques optimized for military
    vehicle sound detection with 91%+ accuracy targets.
    """
    
    def __init__(
        self,
        model: AudioMAE,
        config: FinetuneConfig,
        device: torch.device = None
    ):
        """
        Initialize AudioMAE fine-tuner.
        
        Args:
            model: AudioMAE model to fine-tune
            config: Fine-tuning configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Add classification head if not present
        if self.model.classification_head is None:
            self.model.classification_head = ClassificationHead(
                embed_dim=self.model.config.embed_dim,
                num_classes=self.model.config.num_classes,
                dropout=config.dropout
            ).to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Data augmentation
        self.mixup_cutmix = MixupCutmix(
            mixup_alpha=config.mixup,
            cutmix_alpha=config.cutmix,
            prob=config.mixup_prob,
            switch_prob=config.mixup_switch_prob,
            mode=config.mixup_mode
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        
        # Early stopping
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        logger.info(f"AudioMAE fine-tuner initialized on {self.device}")
        logger.info(f"Target accuracy: {config.target_accuracy:.1%}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with layer-wise learning rate decay."""
        if self.config.use_layer_decay:
            param_groups = self._get_layer_wise_param_groups()
        else:
            param_groups = self._get_param_groups()
        
        if self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _get_layer_wise_param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with layer-wise learning rate decay."""
        param_groups = []
        
        # Classification head gets full learning rate
        if hasattr(self.model, 'classification_head') and self.model.classification_head:
            param_groups.append({
                'params': list(self.model.classification_head.parameters()),
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
        
        # Encoder layers get decaying learning rates
        num_layers = self.model.config.encoder_depth
        
        for i, block in enumerate(self.model.encoder.blocks):
            layer_id = i
            lr_scale = self.config.layer_decay ** (num_layers - layer_id - 1)
            
            param_groups.append({
                'params': list(block.parameters()),
                'lr': self.config.learning_rate * lr_scale,
                'weight_decay': self.config.weight_decay
            })
        
        # Patch embedding and positional encoding
        remaining_params = []
        
        # Patch embedding
        if not self.config.freeze_patch_embed:
            remaining_params.extend(list(self.model.encoder.patch_embed.parameters()))
        
        # Positional encoding
        if not self.config.freeze_pos_embed:
            remaining_params.extend(list(self.model.encoder.pos_embed.parameters()))
        
        # CLS token
        if hasattr(self.model.encoder, 'cls_token') and self.model.encoder.cls_token is not None:
            remaining_params.append(self.model.encoder.cls_token)
        
        # Layer norm
        remaining_params.extend(list(self.model.encoder.norm.parameters()))
        
        if remaining_params:
            lr_scale = self.config.layer_decay ** num_layers
            param_groups.append({
                'params': remaining_params,
                'lr': self.config.learning_rate * lr_scale,
                'weight_decay': self.config.weight_decay
            })
        
        return param_groups
    
    def _get_param_groups(self) -> List[Dict[str, Any]]:
        """Get standard parameter groups without layer decay."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for bias, layer norm, and positional embeddings
            if any(nd in name for nd in ['bias', 'norm', 'pos_embed', 'cls_token']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return param_groups
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == 'linear':
            def lr_lambda(epoch):
                if epoch < self.config.warmup_epochs:
                    return epoch / self.config.warmup_epochs
                else:
                    return max(
                        self.config.min_lr / self.config.learning_rate,
                        1.0 - (epoch - self.config.warmup_epochs) / (self.config.max_epochs - self.config.warmup_epochs)
                    )
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.config.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def freeze_encoder(self, freeze: bool = True):
        """Freeze or unfreeze encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = not freeze
        
        # Always keep classification head trainable
        if hasattr(self.model, 'classification_head') and self.model.classification_head:
            for param in self.model.classification_head.parameters():
                param.requires_grad = True
        
        logger.info(f"Encoder {'frozen' if freeze else 'unfrozen'}")
    
    def _mixup_criterion(self, pred: torch.Tensor, target_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for mixup/cutmix targets."""
        target_a = target_dict['target_a']
        target_b = target_dict['target_b']
        lam = target_dict['lam']
        
        loss_a = self.criterion(pred, target_a)
        loss_b = self.criterion(pred, target_b)
        
        return lam * loss_a + (1 - lam) * loss_b
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary with loss and accuracy values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Apply mixup/cutmix
        if self.config.mixup > 0 or self.config.cutmix > 0:
            inputs, targets = self.mixup_cutmix(inputs, targets)
        
        if self.config.use_mixed_precision:
            with autocast():
                # Forward pass
                output = self.model(inputs)
                
                # Compute loss
                if isinstance(targets, dict):  # Mixup/cutmix targets
                    loss = self._mixup_criterion(output.logits, targets)
                else:
                    loss = self.criterion(output.logits, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            # Forward pass
            output = self.model(inputs)
            
            # Compute loss
            if isinstance(targets, dict):  # Mixup/cutmix targets
                loss = self._mixup_criterion(output.logits, targets)
            else:
                loss = self.criterion(output.logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            if isinstance(targets, dict):
                # For mixup, use original targets for accuracy calculation
                accuracy = (output.predictions == targets['target_a']).float().mean().item()
            else:
                accuracy = (output.predictions == targets).float().mean().item()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation step.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = self.model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                # Compute loss
                loss = self.criterion(output.logits, targets)
                total_loss += loss.item()
                
                # Collect predictions and targets
                all_predictions.extend(output.predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        avg_loss = total_loss / len(val_loader)
        avg_inference_time = np.mean(inference_times)
        
        # Check real-time requirement
        real_time_compliance = avg_inference_time <= self.config.max_inference_time_ms
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'avg_inference_time_ms': avg_inference_time,
            'real_time_compliance': real_time_compliance
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
        resume_from: str = None
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting fine-tuning for {self.config.max_epochs} epochs")
        logger.info(f"Target accuracy: {self.config.target_accuracy:.1%}")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Freeze encoder for initial epochs if specified
            if epoch < self.config.freeze_encoder_epochs:
                self.freeze_encoder(True)
            else:
                self.freeze_encoder(False)
            
            # Training epoch
            epoch_losses = []
            epoch_accuracies = []
            epoch_start_time = time.time()
            
            for step, batch in enumerate(train_loader):
                # Training step
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                epoch_accuracies.append(metrics['accuracy'])
                
                # Logging
                if step % self.config.log_frequency == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch}, Step {step}: "
                        f"Loss = {metrics['loss']:.6f}, "
                        f"Acc = {metrics['accuracy']:.4f}, "
                        f"LR = {current_lr:.2e}"
                    )
                    self.learning_rates.append(current_lr)
            
            # Epoch statistics
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_accuracy = np.mean(epoch_accuracies)
            self.train_losses.append(avg_epoch_loss)
            self.train_accuracies.append(avg_epoch_accuracy)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch} completed: "
                f"Train Loss = {avg_epoch_loss:.6f}, "
                f"Train Acc = {avg_epoch_accuracy:.4f}, "
                f"Time = {epoch_time:.2f}s"
            )
            
            # Validation
            if epoch % self.config.val_frequency == 0:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['val_loss'])
                self.val_accuracies.append(val_metrics['val_accuracy'])
                
                logger.info(
                    f"Validation: "
                    f"Loss = {val_metrics['val_loss']:.6f}, "
                    f"Acc = {val_metrics['val_accuracy']:.4f}, "
                    f"F1 = {val_metrics['val_f1']:.4f}, "
                    f"Inference = {val_metrics['avg_inference_time_ms']:.2f}ms"
                )
                
                # Check if target accuracy reached
                if val_metrics['val_accuracy'] >= self.config.target_accuracy:
                    logger.info(
                        f"🎯 Target accuracy {self.config.target_accuracy:.1%} reached! "
                        f"Current: {val_metrics['val_accuracy']:.1%}"
                    )
                
                # Save best model
                if val_metrics['val_accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['val_accuracy']
                    self.save_checkpoint(
                        checkpoint_dir / f"best_model_acc_{val_metrics['val_accuracy']:.4f}_epoch_{epoch}.pt",
                        is_best=True,
                        val_metrics=val_metrics
                    )
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {self.config.early_stopping_patience} epochs without improvement")
                    break
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                )
        
        logger.info("Fine-tuning completed!")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")
        
        # Save final model
        self.save_checkpoint(
            checkpoint_dir / "final_model.pt",
            is_final=True
        )
        
        # Plot training curves
        self.plot_training_curves(checkpoint_dir)
    
    def plot_training_curves(self, save_dir: Path):
        """Plot training and validation curves."""
        if not self.train_losses:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.train_losses))
        val_epochs = range(0, len(self.val_losses) * self.config.val_frequency, self.config.val_frequency)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, label='Train Loss', color='blue')
        if self.val_losses:
            ax1.plot(val_epochs, self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, label='Train Acc', color='blue')
        if self.val_accuracies:
            ax2.plot(val_epochs, self.val_accuracies, label='Val Acc', color='red')
        ax2.axhline(y=self.config.target_accuracy, color='green', linestyle='--', label='Target Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        if self.learning_rates:
            steps = range(len(self.learning_rates))
            ax3.plot(steps, self.learning_rates, color='purple')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True)
        
        # Best accuracy indicator
        if self.val_accuracies:
            best_epoch = val_epochs[np.argmax(self.val_accuracies)]
            ax4.bar(['Best Val Acc', 'Target Acc'], [self.best_accuracy, self.config.target_accuracy], 
                   color=['blue', 'green'], alpha=0.7)
            ax4.set_ylabel('Accuracy')
            ax4.set_title(f'Best Validation Accuracy: {self.best_accuracy:.4f}')
            ax4.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, path: Union[str, Path], **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'model_config': self.model.config.__dict__,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            **kwargs
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.lr_scheduler and checkpoint.get('lr_scheduler_state_dict'):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logger.info(f"Checkpoint loaded: {path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
        logger.info(f"Best accuracy: {self.best_accuracy:.4f}")


def create_classification_head(
    embed_dim: int,
    num_classes: int,
    dropout: float = 0.1,
    use_norm: bool = True
) -> ClassificationHead:
    """
    Create classification head for AudioMAE fine-tuning.
    
    Args:
        embed_dim: Input embedding dimension
        num_classes: Number of output classes
        dropout: Dropout probability
        use_norm: Whether to use layer normalization
        
    Returns:
        ClassificationHead instance
    """
    return ClassificationHead(embed_dim, num_classes, dropout, use_norm)


def freeze_encoder_layers(
    model: AudioMAE,
    freeze_patch_embed: bool = False,
    freeze_pos_embed: bool = False,
    freeze_blocks: bool = True
):
    """
    Freeze specific encoder layers for transfer learning.
    
    Args:
        model: AudioMAE model
        freeze_patch_embed: Whether to freeze patch embedding
        freeze_pos_embed: Whether to freeze positional embedding
        freeze_blocks: Whether to freeze transformer blocks
    """
    # Freeze patch embedding
    if freeze_patch_embed:
        for param in model.encoder.patch_embed.parameters():
            param.requires_grad = False
    
    # Freeze positional embedding
    if freeze_pos_embed:
        if hasattr(model.encoder, 'pos_embed'):
            for param in model.encoder.pos_embed.parameters():
                param.requires_grad = False
    
    # Freeze transformer blocks
    if freeze_blocks:
        for block in model.encoder.blocks:
            for param in block.parameters():
                param.requires_grad = False
    
    # Always keep classification head trainable
    if hasattr(model, 'classification_head') and model.classification_head:
        for param in model.classification_head.parameters():
            param.requires_grad = True
    
    logger.info("Encoder layers frozen for transfer learning")