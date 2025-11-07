#
# Plan:
# 1. Implement AudioMAE pretrainer for self-supervised learning
# 2. Masked spectrogram dataset for pretraining
# 3. Optimized training loop with mixed precision
# 4. Advanced masking strategies (random, block, frequency-aware)
# 5. Learning rate scheduling and warmup
# 6. Checkpointing and model versioning
# 7. Distributed training support
# 8. Performance monitoring and visualization
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

from .model import AudioMAE, AudioMAEConfig

logger = logging.getLogger(__name__)

@dataclass
class PretrainConfig:
    """Configuration for AudioMAE pretraining."""
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 40
    max_epochs: int = 800
    
    # Masking strategy
    mask_ratio: float = 0.75
    masking_strategy: str = 'random'  # 'random', 'block', 'frequency_aware'
    block_size: int = 4  # For block masking
    
    # Loss configuration
    norm_pix_loss: bool = True
    reconstruction_weight: float = 1.0
    
    # Optimization
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    lr_scheduler: str = 'cosine'  # 'cosine', 'linear', 'step'
    min_lr: float = 0.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # Validation
    val_frequency: int = 10  # Validate every N epochs
    save_frequency: int = 50  # Save checkpoint every N epochs
    
    # Data augmentation (applied before masking)
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Logging and monitoring
    log_frequency: int = 100  # Log every N steps
    plot_reconstructions: bool = True
    save_attention_maps: bool = False
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1


class MaskedSpectrogramDataset(Dataset):
    """Dataset for masked spectrogram pretraining."""
    
    def __init__(
        self,
        spectrograms: List[torch.Tensor],
        transform: Optional[Callable] = None,
        augment_prob: float = 0.5
    ):
        """
        Initialize masked spectrogram dataset.
        
        Args:
            spectrograms: List of spectrogram tensors
            transform: Optional transform to apply
            augment_prob: Probability of applying augmentation
        """
        self.spectrograms = spectrograms
        self.transform = transform
        self.augment_prob = augment_prob
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get spectrogram sample."""
        spectrogram = self.spectrograms[idx]
        
        # Apply augmentation
        if self.transform and np.random.random() < self.augment_prob:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram


class AudioMAEPretrainer:
    """
    AudioMAE pretrainer for self-supervised learning on spectrograms.
    
    Implements masked autoencoder pretraining with advanced masking strategies,
    optimized for military audio datasets with robust feature learning.
    """
    
    def __init__(
        self,
        model: AudioMAE,
        config: PretrainConfig,
        device: torch.device = None
    ):
        """
        Initialize AudioMAE pretrainer.
        
        Args:
            model: AudioMAE model to train
            config: Pretraining configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"AudioMAE pretrainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        # Separate weight decay for different parameter groups
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
    
    def _get_param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with different weight decay settings."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for bias, layer norm, and positional embeddings
            if any(nd in name for nd in ['bias', 'norm', 'pos_embed', 'cls_token', 'mask_token']):
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
    
    def generate_masks(self, batch_size: int, num_patches: int, strategy: str = None) -> List[torch.Tensor]:
        """
        Generate mask indices for batch.
        
        Args:
            batch_size: Batch size
            num_patches: Number of patches per sample
            strategy: Masking strategy ('random', 'block', 'frequency_aware')
            
        Returns:
            List of mask indices for each sample
        """
        strategy = strategy or self.config.masking_strategy
        mask_ratio = self.config.mask_ratio
        num_masked = int(num_patches * mask_ratio)
        
        mask_indices = []
        
        for _ in range(batch_size):
            if strategy == 'random':
                # Random masking
                indices = torch.randperm(num_patches)[:num_masked]
            
            elif strategy == 'block':
                # Block masking (spatial coherence)
                indices = self._generate_block_mask(num_patches, num_masked)
            
            elif strategy == 'frequency_aware':
                # Frequency-aware masking (preserve some frequency bands)
                indices = self._generate_frequency_aware_mask(num_patches, num_masked)
            
            else:
                raise ValueError(f"Unknown masking strategy: {strategy}")
            
            mask_indices.append(indices.to(self.device))
        
        return mask_indices
    
    def _generate_block_mask(self, num_patches: int, num_masked: int) -> torch.Tensor:
        """Generate block-based mask indices."""
        # Assume square grid of patches
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            # Fall back to random if not square
            return torch.randperm(num_patches)[:num_masked]
        
        # Generate block centers
        block_size = self.config.block_size
        num_blocks = max(1, num_masked // (block_size * block_size))
        
        mask_indices = []
        
        for _ in range(num_blocks):
            # Random block center
            center_h = torch.randint(0, grid_size, (1,)).item()
            center_w = torch.randint(0, grid_size, (1,)).item()
            
            # Generate block indices
            for dh in range(-block_size//2, block_size//2 + 1):
                for dw in range(-block_size//2, block_size//2 + 1):
                    h = center_h + dh
                    w = center_w + dw
                    
                    if 0 <= h < grid_size and 0 <= w < grid_size:
                        idx = h * grid_size + w
                        mask_indices.append(idx)
        
        # Convert to tensor and trim to desired length
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
        mask_indices = mask_indices[:num_masked]
        
        # Fill remaining with random if needed
        if len(mask_indices) < num_masked:
            remaining = num_masked - len(mask_indices)
            all_indices = torch.arange(num_patches)
            available = all_indices[~torch.isin(all_indices, mask_indices)]
            additional = available[torch.randperm(len(available))[:remaining]]
            mask_indices = torch.cat([mask_indices, additional])
        
        return mask_indices
    
    def _generate_frequency_aware_mask(self, num_patches: int, num_masked: int) -> torch.Tensor:
        """Generate frequency-aware mask indices."""
        # Assume patches are arranged in (freq, time) grid
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            return torch.randperm(num_patches)[:num_masked]
        
        # Preserve some low-frequency information (bottom of spectrogram)
        preserve_ratio = 0.2  # Preserve 20% of low frequencies
        preserve_freq_bands = max(1, int(grid_size * preserve_ratio))
        
        # Generate candidate indices (excluding preserved bands)
        candidate_indices = []
        for h in range(preserve_freq_bands, grid_size):
            for w in range(grid_size):
                candidate_indices.append(h * grid_size + w)
        
        candidate_indices = torch.tensor(candidate_indices, dtype=torch.long)
        
        # Random selection from candidates
        if len(candidate_indices) >= num_masked:
            selected = torch.randperm(len(candidate_indices))[:num_masked]
            mask_indices = candidate_indices[selected]
        else:
            # If not enough candidates, add some from preserved region
            mask_indices = candidate_indices
            remaining = num_masked - len(mask_indices)
            
            preserved_indices = []
            for h in range(preserve_freq_bands):
                for w in range(grid_size):
                    preserved_indices.append(h * grid_size + w)
            
            preserved_indices = torch.tensor(preserved_indices, dtype=torch.long)
            additional = preserved_indices[torch.randperm(len(preserved_indices))[:remaining]]
            mask_indices = torch.cat([mask_indices, additional])
        
        return mask_indices
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of spectrograms [B, C, H, W]
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Generate masks
        mask_indices = self.generate_masks(batch_size, self.model.config.num_patches)
        
        if self.config.use_mixed_precision:
            with autocast():
                # Forward pass
                output = self.model(batch, mask_ratio=self.config.mask_ratio)
                
                # Compute reconstruction loss
                reconstruction_loss = self.model.compute_reconstruction_loss(
                    batch, output.logits, mask_indices
                )
                
                total_loss = reconstruction_loss * self.config.reconstruction_weight
            
            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()
            
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
            output = self.model(batch, mask_ratio=self.config.mask_ratio)
            
            # Compute reconstruction loss
            reconstruction_loss = self.model.compute_reconstruction_loss(
                batch, output.logits, mask_indices
            )
            
            total_loss = reconstruction_loss * self.config.reconstruction_weight
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item()
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
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                
                # Generate masks
                mask_indices = self.generate_masks(batch_size, self.model.config.num_patches)
                
                # Forward pass
                output = self.model(batch, mask_ratio=self.config.mask_ratio)
                
                # Compute loss
                reconstruction_loss = self.model.compute_reconstruction_loss(
                    batch, output.logits, mask_indices
                )
                
                total_loss += reconstruction_loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_reconstruction_loss = total_reconstruction_loss / max(num_batches, 1)
        
        return {
            'val_loss': avg_loss,
            'val_reconstruction_loss': avg_reconstruction_loss
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        resume_from: str = None
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting pretraining for {self.config.max_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            epoch_losses = []
            epoch_start_time = time.time()
            
            for step, batch in enumerate(train_loader):
                # Training step
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # Logging
                if step % self.config.log_frequency == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch}, Step {step}: "
                        f"Loss = {losses['total_loss']:.6f}, "
                        f"LR = {current_lr:.2e}"
                    )
                    self.learning_rates.append(current_lr)
            
            # Epoch statistics
            avg_epoch_loss = np.mean([l['total_loss'] for l in epoch_losses])
            self.train_losses.append(avg_epoch_loss)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch} completed: "
                f"Avg Loss = {avg_epoch_loss:.6f}, "
                f"Time = {epoch_time:.2f}s"
            )
            
            # Validation
            if val_loader and epoch % self.config.val_frequency == 0:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['val_loss'])
                
                logger.info(
                    f"Validation: Loss = {val_metrics['val_loss']:.6f}"
                )
                
                # Save best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint(
                        checkpoint_dir / f"best_model_epoch_{epoch}.pt",
                        is_best=True
                    )
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                )
            
            # Plot reconstructions
            if self.config.plot_reconstructions and epoch % (self.config.val_frequency * 2) == 0:
                self.plot_reconstructions(train_loader, checkpoint_dir, epoch)
        
        logger.info("Pretraining completed!")
        
        # Save final model
        self.save_checkpoint(
            checkpoint_dir / "final_model.pt",
            is_final=True
        )
    
    def plot_reconstructions(self, data_loader: DataLoader, save_dir: Path, epoch: int):
        """Plot sample reconstructions for visualization."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch
            batch = next(iter(data_loader)).to(self.device)
            batch_size = min(4, batch.shape[0])  # Plot up to 4 samples
            
            # Generate masks
            mask_indices = self.generate_masks(batch_size, self.model.config.num_patches)
            
            # Forward pass
            output = self.model(batch[:batch_size], mask_ratio=self.config.mask_ratio)
            
            # Convert reconstruction to images
            reconstructions = self._reconstruction_to_images(output.logits[:batch_size])
            
            # Plot
            fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
            if batch_size == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(batch_size):
                # Original
                axes[i, 0].imshow(batch[i, 0].cpu().numpy(), aspect='auto', origin='lower')
                axes[i, 0].set_title('Original')
                axes[i, 0].axis('off')
                
                # Masked
                masked_img = self._apply_mask_to_image(batch[i, 0], mask_indices[i])
                axes[i, 1].imshow(masked_img.cpu().numpy(), aspect='auto', origin='lower')
                axes[i, 1].set_title('Masked')
                axes[i, 1].axis('off')
                
                # Reconstruction
                axes[i, 2].imshow(reconstructions[i, 0].cpu().numpy(), aspect='auto', origin='lower')
                axes[i, 2].set_title('Reconstruction')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / f"reconstructions_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        self.model.train()
    
    def _reconstruction_to_images(self, reconstructions: torch.Tensor) -> torch.Tensor:
        """Convert patch reconstructions back to images."""
        B, N, patch_dim = reconstructions.shape
        
        # Patch parameters
        patch_size = self.model.config.patch_size
        img_size = self.model.config.img_size
        grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size
        
        # Reshape patches
        patches = reconstructions.view(B, grid_h, grid_w, 1, patch_size, patch_size)
        
        # Reconstruct images
        images = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = images.view(B, 1, img_size[0], img_size[1])
        
        return images
    
    def _apply_mask_to_image(self, image: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Apply mask to image for visualization."""
        masked_image = image.clone()
        
        # Patch parameters
        patch_size = self.model.config.patch_size
        img_size = self.model.config.img_size
        grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size
        
        # Apply mask
        for idx in mask_indices:
            h_idx = idx // grid_w
            w_idx = idx % grid_w
            
            h_start, h_end = h_idx * patch_size, (h_idx + 1) * patch_size
            w_start, w_end = w_idx * patch_size, (w_idx + 1) * patch_size
            
            masked_image[h_start:h_end, w_start:w_end] = 0  # Set to black
        
        return masked_image
    
    def save_checkpoint(self, path: Union[str, Path], **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'model_config': self.model.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
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
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logger.info(f"Checkpoint loaded: {path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")


def create_pretraining_dataset(
    data_loaders: List[DataLoader],
    max_samples: Optional[int] = None,
    transform: Optional[Callable] = None
) -> MaskedSpectrogramDataset:
    """
    Create pretraining dataset from multiple data loaders.
    
    Args:
        data_loaders: List of data loaders to combine
        max_samples: Maximum number of samples to use
        transform: Optional transform to apply
        
    Returns:
        MaskedSpectrogramDataset for pretraining
    """
    all_spectrograms = []
    
    for loader in data_loaders:
        for batch in loader:
            if isinstance(batch, dict):
                spectrograms = batch.get('spectrogram', batch.get('audio'))
            else:
                spectrograms = batch
            
            for spec in spectrograms:
                all_spectrograms.append(spec)
                
                if max_samples and len(all_spectrograms) >= max_samples:
                    break
            
            if max_samples and len(all_spectrograms) >= max_samples:
                break
        
        if max_samples and len(all_spectrograms) >= max_samples:
            break
    
    return MaskedSpectrogramDataset(all_spectrograms, transform)


def create_pretrainer_from_config(
    config_path: str,
    model_config_path: str = None,
    device: torch.device = None
) -> AudioMAEPretrainer:
    """
    Create AudioMAE pretrainer from configuration files.
    
    Args:
        config_path: Path to pretraining configuration
        model_config_path: Path to model configuration (optional)
        device: Training device
        
    Returns:
        AudioMAEPretrainer instance
    """
    # Load pretraining config
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            pretrain_config_dict = json.load(f)
        else:
            import yaml
            pretrain_config_dict = yaml.safe_load(f)
    
    pretrain_config = PretrainConfig(**pretrain_config_dict)
    
    # Load model config
    if model_config_path:
        with open(model_config_path, 'r') as f:
            if model_config_path.endswith('.json'):
                model_config_dict = json.load(f)
            else:
                import yaml
                model_config_dict = yaml.safe_load(f)
        model_config = AudioMAEConfig(**model_config_dict)
    else:
        model_config = AudioMAEConfig()
    
    # Create model
    model = AudioMAE(model_config)
    
    # Create pretrainer
    pretrainer = AudioMAEPretrainer(model, pretrain_config, device)
    
    return pretrainer