#
# Plan:
# 1. Import necessary libraries: torch, torch.nn, math, typing
# 2. Create configuration dataclass for positional encoding parameters
# 3. Implement core positional encoding components:
#    a) SinusoidalPositionalEncoding:
#       - Standard transformer-style sinusoidal encoding
#       - Adaptable to different sequence lengths
#       - Support for 1D and 2D positional encoding
#    b) LearnedPositionalEncoding:
#       - Trainable positional embeddings
#       - Optimized for audio spectrogram patches
#       - Support for frequency and time dimensions
#    c) Audio2DPositionalEncoding:
#       - Specialized for frequency-time spectrograms
#       - Separate frequency and time position encoding
#       - Patch-aware positioning for AST architecture
#    d) RelativePositionalEncoding:
#       - Relative position bias for attention
#       - Audio-aware relative positioning
#       - Efficient computation for transformer blocks
# 4. Audio-specific enhancements:
#    - Frequency bin position encoding
#    - Time frame position encoding  
#    - Patch-based coordinate mapping
#    - Variable-length audio handling
# 5. Integration utilities for AST model
# 6. Performance optimizations for edge deployment
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionalEncodingConfig:
    """Configuration for AST positional encoding"""
    
    # Model dimensions
    embed_dim: int = 768
    max_seq_length: int = 1024
    
    # Audio spectrogram dimensions
    input_fdim: int = 128      # Frequency dimension (mel bins)
    input_tdim: int = 1024     # Time dimension (frames)
    patch_size: int = 16       # Patch size for both dimensions
    
    # Encoding type
    encoding_type: str = "learned"  # "learned", "sinusoidal", "2d", "relative"
    
    # Sinusoidal parameters
    temperature: float = 10000.0
    normalize: bool = True
    
    # 2D encoding parameters
    use_separate_freq_time: bool = True
    freq_encoding_dim: Optional[int] = None  # Will be embed_dim // 2 if None
    time_encoding_dim: Optional[int] = None  # Will be embed_dim // 2 if None
    
    # Relative encoding parameters
    max_relative_position: int = 128
    bidirectional: bool = True
    
    # Dropout
    dropout: float = 0.1
    
    # Initialization
    init_std: float = 0.02
    
    def __post_init__(self):
        # Calculate patch grid dimensions
        self.num_freq_patches = self.input_fdim // self.patch_size
        self.num_time_patches = self.input_tdim // self.patch_size
        self.total_patches = self.num_freq_patches * self.num_time_patches
        
        # Set default encoding dimensions for 2D
        if self.freq_encoding_dim is None:
            self.freq_encoding_dim = self.embed_dim // 2
        if self.time_encoding_dim is None:
            self.time_encoding_dim = self.embed_dim - self.freq_encoding_dim


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    Supports both 1D and 2D positional encoding for audio spectrograms.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.dropout = nn.Dropout(config.dropout)
        
        # Pre-compute positional encodings
        self.register_buffer(
            'pe_1d', 
            self._create_1d_encoding(config.max_seq_length, config.embed_dim)
        )
        
        if config.encoding_type == "2d":
            self.register_buffer(
                'pe_freq',
                self._create_1d_encoding(config.num_freq_patches, config.freq_encoding_dim)
            )
            self.register_buffer(
                'pe_time',
                self._create_1d_encoding(config.num_time_patches, config.time_encoding_dim)
            )
    
    def _create_1d_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create 1D sinusoidal positional encoding"""
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(self.config.temperature) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if self.config.normalize:
            pe = pe / math.sqrt(d_model)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(
        self, 
        x: torch.Tensor, 
        freq_positions: Optional[torch.Tensor] = None,
        time_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            freq_positions: Frequency position indices [batch_size, seq_len]
            time_positions: Time position indices [batch_size, seq_len]
            
        Returns:
            Tensor with positional encoding added
        """
        
        if self.config.encoding_type == "2d" and freq_positions is not None and time_positions is not None:
            return self._forward_2d(x, freq_positions, time_positions)
        else:
            return self._forward_1d(x)
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Standard 1D positional encoding"""
        seq_len = x.size(1)
        
        # Add positional encoding
        x = x + self.pe_1d[:, :seq_len, :]
        return self.dropout(x)
    
    def _forward_2d(
        self, 
        x: torch.Tensor, 
        freq_positions: torch.Tensor, 
        time_positions: torch.Tensor
    ) -> torch.Tensor:
        """2D positional encoding for frequency-time patches"""
        
        batch_size, seq_len, _ = x.shape
        
        # Get frequency and time encodings
        freq_pe = self.pe_freq[:, freq_positions.long()]  # [1, batch_size, seq_len, freq_dim]
        time_pe = self.pe_time[:, time_positions.long()]  # [1, batch_size, seq_len, time_dim]
        
        # Concatenate frequency and time encodings
        freq_pe = freq_pe.squeeze(0)  # [batch_size, seq_len, freq_dim]
        time_pe = time_pe.squeeze(0)  # [batch_size, seq_len, time_dim]
        
        pos_encoding = torch.cat([freq_pe, time_pe], dim=-1)  # [batch_size, seq_len, embed_dim]
        
        # Add to input
        x = x + pos_encoding
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings optimized for audio spectrogram patches.
    Supports both absolute and relative positional encoding.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        
        if config.encoding_type == "2d":
            # Separate frequency and time embeddings
            self.freq_embedding = nn.Parameter(
                torch.randn(config.num_freq_patches, config.freq_encoding_dim) * config.init_std
            )
            self.time_embedding = nn.Parameter(
                torch.randn(config.num_time_patches, config.time_encoding_dim) * config.init_std
            )
        else:
            # Standard 1D positional embedding
            self.position_embedding = nn.Parameter(
                torch.randn(config.max_seq_length, config.embed_dim) * config.init_std
            )
    
    def forward(
        self,
        x: torch.Tensor,
        freq_positions: Optional[torch.Tensor] = None,
        time_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add learned positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            freq_positions: Frequency position indices [batch_size, seq_len]
            time_positions: Time position indices [batch_size, seq_len]
        """
        
        if self.config.encoding_type == "2d" and freq_positions is not None and time_positions is not None:
            return self._forward_2d(x, freq_positions, time_positions)
        else:
            return self._forward_1d(x)
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Standard 1D learned positional encoding"""
        seq_len = x.size(1)
        
        # Add positional embeddings
        pos_emb = self.position_embedding[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
        x = x + pos_emb
        return self.dropout(x)
    
    def _forward_2d(
        self,
        x: torch.Tensor,
        freq_positions: torch.Tensor,
        time_positions: torch.Tensor
    ) -> torch.Tensor:
        """2D learned positional encoding"""
        
        batch_size, seq_len, _ = x.shape
        
        # Get frequency and time embeddings
        freq_emb = self.freq_embedding[freq_positions.long()]  # [batch_size, seq_len, freq_dim]
        time_emb = self.time_embedding[time_positions.long()]  # [batch_size, seq_len, time_dim]
        
        # Concatenate frequency and time embeddings
        pos_emb = torch.cat([freq_emb, time_emb], dim=-1)  # [batch_size, seq_len, embed_dim]
        
        # Add to input
        x = x + pos_emb
        return self.dropout(x)


class Audio2DPositionalEncoding(nn.Module):
    """
    Specialized 2D positional encoding for audio spectrograms.
    Handles frequency-time patch relationships and audio-specific positioning.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Frequency dimension encoding
        self.freq_encoding = self._create_freq_encoding(config)
        
        # Time dimension encoding
        self.time_encoding = self._create_time_encoding(config)
        
        # Patch interaction encoding
        self.patch_interaction = PatchInteractionEncoding(config)
        
        # Combination layer
        self.combine_proj = nn.Linear(config.embed_dim * 2, config.embed_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def _create_freq_encoding(self, config: PositionalEncodingConfig) -> nn.Module:
        """Create frequency-aware positional encoding"""
        
        class FrequencyEncoding(nn.Module):
            def __init__(self, num_freq_bins: int, embed_dim: int):
                super().__init__()
                self.num_freq_bins = num_freq_bins
                self.embed_dim = embed_dim
                
                # Learnable frequency scale
                self.freq_scale = nn.Parameter(torch.ones(1))
                
                # Frequency embedding
                self.freq_emb = nn.Parameter(torch.randn(num_freq_bins, embed_dim))
                
            def forward(self, freq_positions: torch.Tensor) -> torch.Tensor:
                """Encode frequency positions"""
                return self.freq_emb[freq_positions.long()] * self.freq_scale
        
        return FrequencyEncoding(config.num_freq_patches, config.freq_encoding_dim)
    
    def _create_time_encoding(self, config: PositionalEncodingConfig) -> nn.Module:
        """Create time-aware positional encoding"""
        
        class TimeEncoding(nn.Module):
            def __init__(self, max_time_steps: int, embed_dim: int):
                super().__init__()
                self.max_time_steps = max_time_steps
                self.embed_dim = embed_dim
                
                # Learnable time scale
                self.time_scale = nn.Parameter(torch.ones(1))
                
                # Time embedding with extrapolation capability
                self.time_emb = nn.Parameter(torch.randn(max_time_steps, embed_dim))
                
            def forward(self, time_positions: torch.Tensor) -> torch.Tensor:
                """Encode time positions with extrapolation for longer sequences"""
                max_pos = time_positions.max().item()
                
                if max_pos < self.max_time_steps:
                    # Direct lookup
                    return self.time_emb[time_positions.long()] * self.time_scale
                else:
                    # Interpolation for longer sequences
                    time_normalized = time_positions.float() / max_pos * (self.max_time_steps - 1)
                    time_floor = time_normalized.floor().long()
                    time_ceil = time_normalized.ceil().long()
                    time_frac = time_normalized - time_floor.float()
                    
                    emb_floor = self.time_emb[time_floor]
                    emb_ceil = self.time_emb[time_ceil]
                    
                    interpolated = emb_floor + time_frac.unsqueeze(-1) * (emb_ceil - emb_floor)
                    return interpolated * self.time_scale
        
        return TimeEncoding(config.num_time_patches, config.time_encoding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        freq_positions: torch.Tensor,
        time_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply 2D positional encoding for audio spectrograms.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            freq_positions: Frequency bin indices [batch_size, seq_len]
            time_positions: Time frame indices [batch_size, seq_len]
        """
        
        # Get frequency and time encodings
        freq_enc = self.freq_encoding(freq_positions)  # [batch_size, seq_len, freq_dim]
        time_enc = self.time_encoding(time_positions)  # [batch_size, seq_len, time_dim]
        
        # Combine frequency and time
        freq_time_enc = torch.cat([freq_enc, time_enc], dim=-1)
        
        # Add patch interaction information
        interaction_enc = self.patch_interaction(freq_positions, time_positions)
        
        # Combine all encodings
        combined_enc = torch.cat([freq_time_enc, interaction_enc], dim=-1)
        pos_encoding = self.combine_proj(combined_enc)
        
        # Add to input
        x = x + pos_encoding
        return self.dropout(x)


class PatchInteractionEncoding(nn.Module):
    """
    Encodes spatial relationships between audio patches.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Relative position embeddings
        self.freq_relative_emb = nn.Parameter(
            torch.randn(2 * config.num_freq_patches - 1, config.embed_dim // 4)
        )
        self.time_relative_emb = nn.Parameter(
            torch.randn(2 * config.num_time_patches - 1, config.embed_dim // 4)
        )
        
        # Distance encoding
        self.distance_proj = nn.Linear(2, config.embed_dim // 2)
        
    def forward(
        self,
        freq_positions: torch.Tensor,
        time_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute patch interaction encoding.
        
        Args:
            freq_positions: Frequency positions [batch_size, seq_len]
            time_positions: Time positions [batch_size, seq_len]
        """
        
        batch_size, seq_len = freq_positions.shape
        
        # Compute relative positions (for attention, we use simple absolute encoding here)
        freq_center = freq_positions.float().mean(dim=-1, keepdim=True)
        time_center = time_positions.float().mean(dim=-1, keepdim=True)
        
        freq_relative = freq_positions.float() - freq_center
        time_relative = time_positions.float() - time_center
        
        # Encode relative distances
        distances = torch.stack([freq_relative, time_relative], dim=-1)  # [batch_size, seq_len, 2]
        distance_enc = self.distance_proj(distances)  # [batch_size, seq_len, embed_dim//2]
        
        # Simple relative frequency encoding
        freq_rel_indices = (freq_relative + self.config.num_freq_patches - 1).clamp(
            0, 2 * self.config.num_freq_patches - 2
        ).long()
        freq_rel_enc = self.freq_relative_emb[freq_rel_indices]
        
        # Simple relative time encoding
        time_rel_indices = (time_relative + self.config.num_time_patches - 1).clamp(
            0, 2 * self.config.num_time_patches - 2
        ).long()
        time_rel_enc = self.time_relative_emb[time_rel_indices]
        
        # Combine all interaction encodings
        interaction_enc = torch.cat([distance_enc, freq_rel_enc, time_rel_enc], dim=-1)
        
        return interaction_enc


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for transformer attention.
    Provides relative position bias for attention computation.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        self.max_relative_position = config.max_relative_position
        self.num_heads = 12  # Will be set by parent model
        
        # Relative position embeddings
        self.relative_position_bias_table = nn.Parameter(
            torch.randn(
                (2 * config.max_relative_position - 1) ** 2,
                self.num_heads
            )
        )
        
        # Pre-compute relative position indices
        self._init_relative_position_index(config)
    
    def _init_relative_position_index(self, config: PositionalEncodingConfig):
        """Initialize relative position index for 2D grid"""
        
        # Create coordinate grids
        coords_h = torch.arange(config.num_freq_patches)
        coords_w = torch.arange(config.num_time_patches)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # Compute relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        
        # Shift to start from 0
        relative_coords[:, :, 0] += config.num_freq_patches - 1
        relative_coords[:, :, 1] += config.num_time_patches - 1
        
        # Convert to linear index
        relative_coords[:, :, 0] *= 2 * config.num_time_patches - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position bias for attention.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position bias [num_heads, seq_len, seq_len]
        """
        
        # Get bias from table
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(seq_len, seq_len, -1)  # seq_len, seq_len, num_heads
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, seq_len, seq_len
        
        return relative_position_bias


class PositionalEncodingFactory:
    """
    Factory class for creating different types of positional encodings.
    """
    
    @staticmethod
    def create_encoding(config: PositionalEncodingConfig) -> nn.Module:
        """
        Create positional encoding based on configuration.
        
        Args:
            config: Positional encoding configuration
            
        Returns:
            Positional encoding module
        """
        
        if config.encoding_type == "sinusoidal":
            return SinusoidalPositionalEncoding(config)
        elif config.encoding_type == "learned":
            return LearnedPositionalEncoding(config)
        elif config.encoding_type == "2d":
            return Audio2DPositionalEncoding(config)
        elif config.encoding_type == "relative":
            return RelativePositionalEncoding(config)
        else:
            raise ValueError(f"Unknown encoding type: {config.encoding_type}")


def create_patch_coordinates(
    input_fdim: int,
    input_tdim: int,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create frequency and time coordinates for audio patches.
    
    Args:
        input_fdim: Input frequency dimension
        input_tdim: Input time dimension
        patch_size: Patch size
        
    Returns:
        Tuple of (freq_coordinates, time_coordinates)
    """
    
    num_freq_patches = input_fdim // patch_size
    num_time_patches = input_tdim // patch_size
    
    # Create patch coordinates
    freq_coords = torch.arange(num_freq_patches).repeat_interleave(num_time_patches)
    time_coords = torch.arange(num_time_patches).repeat(num_freq_patches)
    
    return freq_coords, time_coords


def interpolate_positional_encoding(
    pos_encoding: torch.Tensor,
    old_size: int,
    new_size: int
) -> torch.Tensor:
    """
    Interpolate positional encoding for different sequence lengths.
    
    Args:
        pos_encoding: Original positional encoding [1, old_size, embed_dim]
        old_size: Original sequence length
        new_size: New sequence length
        
    Returns:
        Interpolated positional encoding [1, new_size, embed_dim]
    """
    
    if old_size == new_size:
        return pos_encoding
    
    # Interpolate using linear interpolation
    pos_encoding = pos_encoding.squeeze(0).transpose(0, 1)  # [embed_dim, old_size]
    pos_encoding = F.interpolate(
        pos_encoding.unsqueeze(0),
        size=new_size,
        mode='linear',
        align_corners=False
    )
    pos_encoding = pos_encoding.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, new_size, embed_dim]
    
    return pos_encoding


if __name__ == "__main__":
    # Example usage and testing
    config = PositionalEncodingConfig(
        embed_dim=768,
        input_fdim=128,
        input_tdim=1024,
        patch_size=16,
        encoding_type="2d"
    )
    
    # Test different encoding types
    for encoding_type in ["sinusoidal", "learned", "2d"]:
        config.encoding_type = encoding_type
        pos_encoder = PositionalEncodingFactory.create_encoding(config)
        
        # Create test input
        batch_size = 2
        seq_len = config.total_patches
        x = torch.randn(batch_size, seq_len, config.embed_dim)
        
        # Create patch coordinates
        freq_coords, time_coords = create_patch_coordinates(
            config.input_fdim, config.input_tdim, config.patch_size
        )
        
        # Expand for batch
        freq_positions = freq_coords.unsqueeze(0).expand(batch_size, -1)
        time_positions = time_coords.unsqueeze(0).expand(batch_size, -1)
        
        # Apply positional encoding
        if encoding_type == "2d":
            output = pos_encoder(x, freq_positions, time_positions)
        else:
            output = pos_encoder(x)
        
        print(f"{encoding_type} encoding output shape: {output.shape}")
    
    print("AST Positional Encoding module test completed successfully!")
