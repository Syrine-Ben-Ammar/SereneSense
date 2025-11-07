#
# Plan:
# 1. Import necessary libraries: torch, torch.nn, typing, math
# 2. Create configuration dataclass for AST attention parameters
# 3. Implement core attention components:
#    a) MultiHeadSelfAttention class:
#       - Scaled dot-product attention optimized for audio spectrograms
#       - Support for different attention patterns (global, local, sparse)
#       - Efficient computation for patch-based inputs
#       - Dropout and normalization for regularization
#    b) AudioSpectrogramAttention class:
#       - Audio-specific attention modifications
#       - Frequency-time dimension awareness
#       - Patch relationship modeling
#       - Attention mask generation for variable-length audio
#    c) CrossAttention class for hierarchical processing
#    d) AttentionBlock class combining attention with feed-forward
# 4. Audio-specific optimizations:
#    - Frequency-aware positional encoding
#    - Time-domain attention patterns
#    - Efficient memory usage for long audio sequences
#    - Support for both classification and reconstruction tasks
# 5. Integration with AST architecture requirements
# 6. Performance optimizations for edge deployment
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ASTAttentionConfig:
    """Configuration for AST attention mechanisms"""
    
    # Attention parameters
    embed_dim: int = 768
    num_heads: int = 12
    head_dim: Optional[int] = None  # Will be calculated as embed_dim // num_heads
    
    # Dropout
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    
    # Audio-specific parameters
    input_tdim: int = 1024  # Time dimension
    input_fdim: int = 128   # Frequency dimension
    patch_size: int = 16    # Patch size for both dimensions
    
    # Attention patterns
    use_sparse_attention: bool = False
    sparse_attention_pattern: str = "local"  # local, stride, random
    attention_window_size: int = 64
    
    # Performance optimizations
    use_flash_attention: bool = False
    use_gradient_checkpointing: bool = False
    
    # Scaling and normalization
    scale_factor: Optional[float] = None  # Will be calculated
    use_bias: bool = True
    
    def __post_init__(self):
        if self.head_dim is None:
            assert self.embed_dim % self.num_heads == 0, \
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            self.head_dim = self.embed_dim // self.num_heads
        
        if self.scale_factor is None:
            self.scale_factor = self.head_dim ** -0.5


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism optimized for Audio Spectrogram Transformer.
    
    Supports various attention patterns and optimizations for audio processing:
    - Standard global attention for full sequence modeling
    - Sparse attention patterns for efficiency with long audio sequences
    - Audio-aware attention scaling and normalization
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.scale_factor
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=config.use_bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.use_bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.proj_dropout = nn.Dropout(config.projection_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.config.use_bias:
            nn.init.constant_(self.qkv.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask [batch_size, seq_len, seq_len]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
            Optionally attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        if self.config.use_sparse_attention:
            attn_output, attn_weights = self._sparse_attention(q, k, v, attention_mask)
        else:
            attn_output, attn_weights = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)
        
        if return_attention_weights:
            return output, attn_weights
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention"""
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                # Expand mask for all heads
                attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights
    
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse attention for efficient processing of long sequences"""
        
        B, H, N, D = q.shape
        
        if self.config.sparse_attention_pattern == "local":
            return self._local_attention(q, k, v, attention_mask)
        elif self.config.sparse_attention_pattern == "stride":
            return self._strided_attention(q, k, v, attention_mask)
        else:
            # Fallback to standard attention
            return self._standard_attention(q, k, v, attention_mask)
    
    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local attention within sliding windows"""
        
        B, H, N, D = q.shape
        window_size = self.config.attention_window_size
        
        # Pad sequence if necessary
        pad_size = window_size - (N % window_size) if N % window_size != 0 else 0
        if pad_size > 0:
            q = F.pad(q, (0, 0, 0, pad_size))
            k = F.pad(k, (0, 0, 0, pad_size))
            v = F.pad(v, (0, 0, 0, pad_size))
            N_padded = N + pad_size
        else:
            N_padded = N
        
        # Reshape for windowed attention
        num_windows = N_padded // window_size
        q_windowed = q.reshape(B, H, num_windows, window_size, D)
        k_windowed = k.reshape(B, H, num_windows, window_size, D)
        v_windowed = v.reshape(B, H, num_windows, window_size, D)
        
        # Compute attention within each window
        attn_scores = torch.matmul(q_windowed, k_windowed.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v_windowed)
        
        # Reshape back to original format
        attn_output = attn_output.reshape(B, H, N_padded, D)
        attn_weights = attn_weights.reshape(B, H, N_padded, window_size)
        
        # Remove padding if it was added
        if pad_size > 0:
            attn_output = attn_output[:, :, :N]
            attn_weights = attn_weights[:, :, :N]
        
        return attn_output, attn_weights
    
    def _strided_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strided attention pattern for long sequences"""
        
        B, H, N, D = q.shape
        stride = max(1, N // self.config.attention_window_size)
        
        # Select strided positions
        indices = torch.arange(0, N, stride, device=q.device)
        
        # Extract strided keys and values
        k_strided = k[:, :, indices]  # [B, H, reduced_N, D]
        v_strided = v[:, :, indices]
        
        # Compute attention with reduced key-value pairs
        attn_scores = torch.matmul(q, k_strided.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v_strided)
        
        return attn_output, attn_weights


class AudioSpectrogramAttention(nn.Module):
    """
    Audio-specific attention mechanism that considers frequency-time relationships
    in spectrograms and patch-based audio processing.
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        # Core attention mechanism
        self.attention = MultiHeadSelfAttention(config)
        
        # Audio-specific components
        self.freq_time_encoding = FrequencyTimeEncoding(config)
        self.patch_relationship_modeling = PatchRelationshipModeling(config)
        
        # Normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        freq_pos: Optional[torch.Tensor] = None,
        time_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with audio-specific processing.
        
        Args:
            x: Input tensor [batch_size, num_patches, embed_dim]
            freq_pos: Frequency position encodings
            time_pos: Time position encodings
            attention_mask: Attention mask for variable-length audio
        """
        
        # Apply frequency-time encoding
        if freq_pos is not None and time_pos is not None:
            x = self.freq_time_encoding(x, freq_pos, time_pos)
        
        # Model patch relationships
        x = self.patch_relationship_modeling(x)
        
        # Apply attention
        attn_output = self.attention(x, attention_mask=attention_mask)
        
        # Residual connection and normalization
        output = self.norm(x + attn_output)
        
        return output


class FrequencyTimeEncoding(nn.Module):
    """
    Encodes frequency and time relationships in audio spectrograms.
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        # Learnable frequency and time embeddings
        num_freq_patches = config.input_fdim // config.patch_size
        num_time_patches = config.input_tdim // config.patch_size
        
        self.freq_embedding = nn.Parameter(torch.randn(num_freq_patches, config.embed_dim))
        self.time_embedding = nn.Parameter(torch.randn(num_time_patches, config.embed_dim))
        
        # Combining mechanism
        self.freq_time_combine = nn.Linear(config.embed_dim * 2, config.embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        freq_pos: torch.Tensor,
        time_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Add frequency and time positional encodings.
        
        Args:
            x: Input tensor [batch_size, num_patches, embed_dim]
            freq_pos: Frequency position indices [batch_size, num_patches]
            time_pos: Time position indices [batch_size, num_patches]
        """
        
        B, N, D = x.shape
        
        # Get frequency and time embeddings
        freq_emb = self.freq_embedding[freq_pos]  # [B, N, D]
        time_emb = self.time_embedding[time_pos]  # [B, N, D]
        
        # Combine frequency and time information
        freq_time_combined = torch.cat([freq_emb, time_emb], dim=-1)  # [B, N, 2*D]
        freq_time_emb = self.freq_time_combine(freq_time_combined)     # [B, N, D]
        
        # Add to input
        return x + freq_time_emb


class PatchRelationshipModeling(nn.Module):
    """
    Models relationships between audio patches considering spatial locality
    and frequency coherence in spectrograms.
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        # Learnable relationship encoding
        self.spatial_relationship = nn.Conv2d(
            config.embed_dim, config.embed_dim,
            kernel_size=3, padding=1, groups=config.embed_dim
        )
        
        # Frequency coherence modeling
        self.freq_coherence = nn.Linear(config.embed_dim, config.embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Model patch relationships.
        
        Args:
            x: Input tensor [batch_size, num_patches, embed_dim]
        """
        
        B, N, D = x.shape
        
        # Reshape for spatial processing (assuming square patches)
        sqrt_N = int(math.sqrt(N))
        if sqrt_N * sqrt_N == N:
            # Square arrangement
            x_2d = x.permute(0, 2, 1).reshape(B, D, sqrt_N, sqrt_N)
            
            # Apply spatial relationship modeling
            spatial_features = self.spatial_relationship(x_2d)
            spatial_features = spatial_features.reshape(B, D, N).permute(0, 2, 1)
            
            # Apply frequency coherence
            freq_features = self.freq_coherence(x)
            
            # Combine spatial and frequency features
            output = spatial_features + freq_features
        else:
            # Non-square arrangement, use only frequency coherence
            output = self.freq_coherence(x)
        
        return output


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for hierarchical audio processing.
    Useful for attending between different levels of audio representations.
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.scale_factor
        
        # Query projection (from target sequence)
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.use_bias)
        
        # Key and Value projections (from source sequence)
        self.kv_proj = nn.Linear(config.embed_dim, config.embed_dim * 2, bias=config.use_bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.proj_dropout = nn.Dropout(config.projection_dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query tensor [batch_size, target_len, embed_dim]
            key_value: Key-Value tensor [batch_size, source_len, embed_dim]
            attention_mask: Cross-attention mask [batch_size, target_len, source_len]
        """
        
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape
        
        # Generate Q from target, K,V from source
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, N_q, head_dim]
        
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_heads, N_kv, head_dim]
        k, v = kv[0], kv[1]
        
        # Scaled dot-product cross-attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, C)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)
        
        return output


class AttentionBlock(nn.Module):
    """
    Complete attention block combining multi-head attention with feed-forward network.
    Optimized for Audio Spectrogram Transformer architecture.
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        # Attention mechanism
        self.attention = AudioSpectrogramAttention(config)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(config)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        # Stochastic depth (optional)
        if hasattr(config, 'stochastic_depth_prob') and config.stochastic_depth_prob > 0:
            self.stochastic_depth = StochasticDepth(config.stochastic_depth_prob)
        else:
            self.stochastic_depth = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        freq_pos: Optional[torch.Tensor] = None,
        time_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of attention block.
        
        Args:
            x: Input tensor [batch_size, num_patches, embed_dim]
            freq_pos: Frequency position encodings
            time_pos: Time position encodings
            attention_mask: Attention mask
        """
        
        # Self-attention with residual connection
        residual1 = x
        x = self.norm1(x)
        attn_output = self.attention(x, freq_pos, time_pos, attention_mask)
        x = residual1 + self.stochastic_depth(attn_output)
        
        # Feed-forward with residual connection
        residual2 = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual2 + self.stochastic_depth(ff_output)
        
        return x


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network for transformer blocks.
    """
    
    def __init__(self, config: ASTAttentionConfig):
        super().__init__()
        self.config = config
        
        hidden_dim = int(config.embed_dim * 4)  # Standard 4x expansion
        
        self.linear1 = nn.Linear(config.embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(config.projection_dropout)
        self.linear2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout2 = nn.Dropout(config.projection_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward network forward pass."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class StochasticDepth(nn.Module):
    """
    Stochastic depth regularization for transformer blocks.
    """
    
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth."""
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# Utility functions for attention mask generation
def create_audio_attention_mask(
    patch_lengths: torch.Tensor,
    max_length: int
) -> torch.Tensor:
    """
    Create attention mask for variable-length audio sequences.
    
    Args:
        patch_lengths: Actual patch lengths for each sample [batch_size]
        max_length: Maximum sequence length
        
    Returns:
        Attention mask [batch_size, max_length, max_length]
    """
    batch_size = patch_lengths.size(0)
    device = patch_lengths.device
    
    # Create sequence positions
    positions = torch.arange(max_length, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask based on actual lengths
    mask = positions < patch_lengths.unsqueeze(1)
    
    # Expand to attention matrix
    attention_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    
    return attention_mask


def create_causal_attention_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask.
    
    Args:
        seq_length: Sequence length
        device: Target device
        
    Returns:
        Causal mask [seq_length, seq_length]
    """
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    return mask


def create_frequency_aware_mask(
    freq_positions: torch.Tensor,
    time_positions: torch.Tensor,
    freq_window: int = 8,
    time_window: int = 16
) -> torch.Tensor:
    """
    Create frequency-aware attention mask that allows attention within
    frequency neighborhoods and across time steps.
    
    Args:
        freq_positions: Frequency position indices [batch_size, seq_len]
        time_positions: Time position indices [batch_size, seq_len]
        freq_window: Frequency neighborhood size
        time_window: Time neighborhood size
        
    Returns:
        Frequency-aware mask [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = freq_positions.shape
    device = freq_positions.device
    
    # Create position difference matrices
    freq_diff = (freq_positions.unsqueeze(2) - freq_positions.unsqueeze(1)).abs()
    time_diff = (time_positions.unsqueeze(2) - time_positions.unsqueeze(1)).abs()
    
    # Create masks based on windows
    freq_mask = freq_diff <= freq_window
    time_mask = time_diff <= time_window
    
    # Combine masks (allow attention within freq OR time neighborhoods)
    attention_mask = freq_mask | time_mask
    
    return attention_mask


if __name__ == "__main__":
    # Example usage and testing
    config = ASTAttentionConfig(
        embed_dim=768,
        num_heads=12,
        input_tdim=1024,
        input_fdim=128,
        patch_size=16
    )
    
    # Test multi-head attention
    attention = MultiHeadSelfAttention(config)
    
    # Create test input
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, config.embed_dim)
    
    # Forward pass
    output = attention(x)
    print(f"Attention output shape: {output.shape}")
    
    # Test audio-specific attention
    audio_attention = AudioSpectrogramAttention(config)
    
    # Create position encodings
    freq_pos = torch.randint(0, 8, (batch_size, seq_len))
    time_pos = torch.randint(0, 64, (batch_size, seq_len))
    
    audio_output = audio_attention(x, freq_pos, time_pos)
    print(f"Audio attention output shape: {audio_output.shape}")
    
    print("AST Attention module test completed successfully!")
