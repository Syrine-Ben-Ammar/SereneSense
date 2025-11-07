#
# Plan:
# 1. Import necessary libraries: torch, torch.nn, typing, numpy, math
# 2. Create configuration dataclass for BEATs tokenizer parameters
# 3. Implement core tokenization components:
#    a) AcousticTokenizer class:
#       - Vector quantization for continuous audio features
#       - Codebook learning for discrete representations
#       - Semantic clustering of audio patterns
#       - Support for multiple codebook hierarchies
#    b) SemanticEncoder class:
#       - Converts audio spectrograms to semantic tokens
#       - Handles bidirectional processing
#       - Military vehicle-specific acoustic patterns
#    c) TokenDecoder class:
#       - Reconstructs audio features from tokens
#       - Supports iterative refinement during training
#    d) TokenEmbedding class:
#       - Learnable embeddings for discrete tokens
#       - Position-aware token representations
# 4. BEATs-specific optimizations:
#    - Iterative pre-training support
#    - Discrete label prediction instead of reconstruction
#    - Semantic-rich representations for military audio
#    - Efficient tokenization for real-time processing
# 5. Integration with BEATs transformer architecture
# 6. Support for different tokenization strategies
# 7. Military vehicle acoustic pattern specialization
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TokenizationStrategy(Enum):
    """Different tokenization strategies for BEATs"""
    VECTOR_QUANTIZATION = "vq"
    K_MEANS_CLUSTERING = "kmeans" 
    HIERARCHICAL_CLUSTERING = "hierarchical"
    LEARNED_CLUSTERING = "learned"


@dataclass
class BEATsTokenizerConfig:
    """Configuration for BEATs acoustic tokenizer"""
    
    # Input audio features
    input_dim: int = 768                    # Input feature dimension
    num_mel_bins: int = 128                 # Number of mel bins in spectrogram
    
    # Tokenization parameters
    codebook_size: int = 8192               # Size of acoustic codebook
    num_codebooks: int = 1                  # Number of codebook hierarchies
    commitment_cost: float = 0.25           # Vector quantization commitment cost
    
    # Tokenizer architecture
    encoder_layers: int = 6                 # Number of encoder layers
    encoder_embed_dim: int = 512            # Encoder embedding dimension
    encoder_ffn_embed_dim: int = 2048       # Encoder FFN dimension
    encoder_attention_heads: int = 8        # Number of attention heads
    
    # Decoder parameters
    decoder_layers: int = 3                 # Number of decoder layers
    decoder_embed_dim: int = 512            # Decoder embedding dimension
    decoder_ffn_embed_dim: int = 2048       # Decoder FFN dimension
    decoder_attention_heads: int = 8        # Number of decoder attention heads
    
    # Training parameters
    tokenization_strategy: TokenizationStrategy = TokenizationStrategy.VECTOR_QUANTIZATION
    use_gumbel_softmax: bool = True         # Use Gumbel-Softmax for differentiable sampling
    gumbel_temperature: float = 1.0         # Temperature for Gumbel-Softmax
    
    # Military audio specific
    semantic_categories: int = 7            # Number of semantic categories (military vehicles)
    use_military_priors: bool = True        # Use military-specific acoustic priors
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    
    # Normalization
    normalize_before: bool = True           # Pre-norm vs post-norm
    
    # Initialization
    init_fn: str = "xavier_uniform"
    init_scale: float = 1.0


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for BEATs acoustic tokenization.
    Converts continuous audio features to discrete tokens via codebook lookup.
    """
    
    def __init__(self, config: BEATsTokenizerConfig):
        super().__init__()
        self.config = config
        
        self.embedding_dim = config.encoder_embed_dim
        self.codebook_size = config.codebook_size
        self.commitment_cost = config.commitment_cost
        
        # Codebook embeddings
        self.embeddings = nn.Embedding(config.codebook_size, config.encoder_embed_dim)
        self.embeddings.weight.data.uniform_(-1/config.codebook_size, 1/config.codebook_size)
        
        # Optional projection layer
        if config.input_dim != config.encoder_embed_dim:
            self.input_proj = nn.Linear(config.input_dim, config.encoder_embed_dim)
        else:
            self.input_proj = nn.Identity()
        
        # EMA for codebook updates (optional)
        self.use_ema = True
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(config.codebook_size))
            self.register_buffer("ema_w", torch.randn(config.codebook_size, config.encoder_embed_dim))
            self.ema_decay = 0.99
            self.epsilon = 1e-5
    
    def forward(
        self, 
        inputs: torch.Tensor,
        return_distances: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of vector quantization.
        
        Args:
            inputs: Input tensor [B, T, D] where T is time, D is feature dim
            return_distances: Whether to return quantization distances
            
        Returns:
            Dictionary containing:
            - quantized: Quantized features [B, T, D]
            - tokens: Discrete token indices [B, T]
            - vq_loss: Vector quantization loss
            - perplexity: Codebook usage metric
        """
        
        # Project input if necessary
        inputs = self.input_proj(inputs)
        
        B, T, D = inputs.shape
        
        # Flatten spatial dimensions
        flat_inputs = inputs.view(-1, D)  # [B*T, D]
        
        # Calculate distances to codebook entries
        distances = torch.sum(flat_inputs**2, dim=1, keepdim=True) + \
                   torch.sum(self.embeddings.weight**2, dim=1) - \
                   2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
        
        # Get closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*T, 1]
        encoding_one_hot = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
        encoding_one_hot.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized_flat = torch.matmul(encoding_one_hot, self.embeddings.weight)  # [B*T, D]
        quantized = quantized_flat.view(B, T, D)  # [B, T, D]
        
        # Update EMA if training
        if self.training and self.use_ema:
            self._update_ema(encoding_one_hot, flat_inputs)
        
        # Vector quantization loss
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Get token indices
        tokens = encoding_indices.view(B, T)
        
        result = {
            'quantized': quantized,
            'tokens': tokens,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encoding_indices': encoding_indices,
            'encodings': encoding_one_hot
        }
        
        if return_distances:
            result['distances'] = distances.view(B, T, self.codebook_size)
        
        return result
    
    def _update_ema(self, encoding_one_hot: torch.Tensor, flat_inputs: torch.Tensor):
        """Update EMA statistics for codebook"""
        with torch.no_grad():
            # Update cluster sizes
            cluster_size = torch.sum(encoding_one_hot, dim=0)
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            
            # Update embedding weights
            dw = torch.matmul(encoding_one_hot.t(), flat_inputs)
            self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
            
            # Normalize
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            
            self.embeddings.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
    
    def get_codebook_usage(self) -> float:
        """Get percentage of codebook entries being used"""
        if hasattr(self, 'ema_cluster_size'):
            used_entries = torch.sum(self.ema_cluster_size > 0.01).item()
            return used_entries / self.codebook_size
        return 1.0


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel-Softmax Vector Quantizer for differentiable tokenization.
    Allows gradient flow during training while maintaining discrete behavior.
    """
    
    def __init__(self, config: BEATsTokenizerConfig):
        super().__init__()
        self.config = config
        
        self.temperature = config.gumbel_temperature
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.encoder_embed_dim
        
        # Learnable codebook
        self.codebook = nn.Parameter(torch.randn(config.codebook_size, config.encoder_embed_dim))
        
        # Projection to logits
        self.to_logits = nn.Linear(config.input_dim, config.codebook_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.codebook)
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)
    
    def forward(self, inputs: torch.Tensor, hard: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Gumbel-Softmax sampling.
        
        Args:
            inputs: Input tensor [B, T, D]
            hard: Whether to use hard (one-hot) or soft sampling
            
        Returns:
            Dictionary with quantized features and tokens
        """
        
        B, T, D = inputs.shape
        
        # Get logits for each codebook entry
        logits = self.to_logits(inputs)  # [B, T, codebook_size]
        
        # Gumbel-Softmax sampling
        if self.training:
            # Add Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = (logits + gumbel_noise) / self.temperature
        
        # Softmax
        soft_tokens = F.softmax(logits, dim=-1)
        
        if hard:
            # Hard sampling with straight-through estimator
            hard_tokens = F.one_hot(torch.argmax(soft_tokens, dim=-1), self.codebook_size).float()
            tokens_for_lookup = hard_tokens
            
            # Straight-through estimator
            tokens_for_lookup = soft_tokens + (hard_tokens - soft_tokens).detach()
        else:
            tokens_for_lookup = soft_tokens
        
        # Lookup quantized features
        quantized = torch.matmul(tokens_for_lookup, self.codebook)  # [B, T, embed_dim]
        
        # Get discrete token indices
        token_indices = torch.argmax(soft_tokens, dim=-1)  # [B, T]
        
        # Calculate commitment loss
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        
        return {
            'quantized': quantized,
            'tokens': token_indices,
            'vq_loss': commitment_loss,
            'soft_tokens': soft_tokens,
            'logits': logits
        }


class SemanticEncoder(nn.Module):
    """
    Semantic encoder that converts audio spectrograms to token sequences.
    Optimized for military vehicle sound patterns.
    """
    
    def __init__(self, config: BEATsTokenizerConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.num_mel_bins, config.encoder_embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.encoder_embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_embed_dim,
            nhead=config.encoder_attention_heads,
            dim_feedforward=config.encoder_ffn_embed_dim,
            dropout=config.dropout,
            activation='gelu',
            norm_first=config.normalize_before,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        
        # Military vehicle acoustic priors
        if config.use_military_priors:
            self.military_priors = MilitaryAcousticPriors(config)
        
        # Semantic projection
        self.semantic_projection = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode mel spectrogram to semantic features.
        
        Args:
            mel_spectrogram: Input mel spectrogram [B, T, mel_bins]
            attention_mask: Attention mask for padding [B, T]
            
        Returns:
            Semantic features [B, T, embed_dim]
        """
        
        # Project input dimensions
        x = self.input_projection(mel_spectrogram)  # [B, T, embed_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply military acoustic priors if enabled
        if hasattr(self, 'military_priors'):
            x = self.military_priors(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (inverted: True = masked)
            transformer_mask = ~attention_mask.bool()
        else:
            transformer_mask = None
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        
        # Semantic projection and normalization
        x = self.semantic_projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class MilitaryAcousticPriors(nn.Module):
    """
    Military-specific acoustic priors for enhanced vehicle sound recognition.
    """
    
    def __init__(self, config: BEATsTokenizerConfig):
        super().__init__()
        self.config = config
        
        # Frequency band experts for different military vehicle types
        self.low_freq_expert = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim)  # Engines, tracks
        self.mid_freq_expert = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim)  # Rotors, mechanical
        self.high_freq_expert = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim) # Jet engines, electronics
        
        # Gating mechanism
        self.gate = nn.Linear(config.encoder_embed_dim, 3)
        
        # Military vehicle type embeddings
        self.vehicle_type_embeddings = nn.Parameter(
            torch.randn(config.semantic_categories, config.encoder_embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply military acoustic priors.
        
        Args:
            x: Input features [B, T, embed_dim]
            
        Returns:
            Enhanced features with military priors [B, T, embed_dim]
        """
        
        # Compute gating weights for frequency experts
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [B, T, 3]
        
        # Apply frequency experts
        low_freq_features = self.low_freq_expert(x)
        mid_freq_features = self.mid_freq_expert(x) 
        high_freq_features = self.high_freq_expert(x)
        
        # Weighted combination
        expert_features = (
            gate_weights[:, :, 0:1] * low_freq_features +
            gate_weights[:, :, 1:2] * mid_freq_features +
            gate_weights[:, :, 2:3] * high_freq_features
        )
        
        # Add vehicle type bias (learnable priors)
        # This encourages the model to learn military-specific patterns
        vehicle_bias = self.vehicle_type_embeddings.mean(dim=0, keepdim=True).unsqueeze(0)
        enhanced_features = expert_features + vehicle_bias
        
        return enhanced_features


class TokenDecoder(nn.Module):
    """
    Token decoder for reconstructing audio features from discrete tokens.
    Used during pre-training for iterative refinement.
    """
    
    def __init__(self, config: BEATsTokenizerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.codebook_size, config.decoder_embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config.decoder_embed_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_embed_dim,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_embed_dim,
            dropout=config.dropout,
            activation='gelu',
            norm_first=config.normalize_before,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.decoder_embed_dim, config.num_mel_bins)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.decoder_embed_dim)
        
    def forward(
        self,
        tokens: torch.Tensor,
        encoder_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode tokens back to audio features.
        
        Args:
            tokens: Discrete token indices [B, T]
            encoder_features: Encoder features for cross-attention [B, T, embed_dim]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Reconstructed audio features [B, T, mel_bins]
        """
        
        # Embed tokens
        x = self.token_embedding(tokens)  # [B, T, decoder_embed_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Prepare attention masks
        if attention_mask is not None:
            tgt_key_padding_mask = ~attention_mask.bool()
            memory_key_padding_mask = ~attention_mask.bool()
        else:
            tgt_key_padding_mask = None
            memory_key_padding_mask = None
        
        # Apply transformer decoder
        x = self.transformer_decoder(
            tgt=x,
            memory=encoder_features,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Layer norm and output projection
        x = self.layer_norm(x)
        reconstructed = self.output_projection(x)  # [B, T, mel_bins]
        
        return reconstructed


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return x


class BEATsTokenizer(nn.Module):
    """
    Complete BEATs tokenizer combining semantic encoding and vector quantization.
    Converts audio spectrograms to discrete semantic tokens for military vehicle detection.
    """
    
    def __init__(self, config: BEATsTokenizerConfig):
        super().__init__()
        self.config = config
        
        # Semantic encoder
        self.semantic_encoder = SemanticEncoder(config)
        
        # Vector quantizer
        if config.use_gumbel_softmax:
            self.quantizer = GumbelVectorQuantizer(config)
        else:
            self.quantizer = VectorQuantizer(config)
        
        # Token decoder (for pre-training)
        self.decoder = TokenDecoder(config)
        
        # Semantic classifier head (for military vehicle classification)
        self.semantic_classifier = nn.Linear(config.encoder_embed_dim, config.semantic_categories)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.init_fn == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight, gain=self.config.init_scale)
                elif self.config.init_fn == "xavier_normal":
                    nn.init.xavier_normal_(module.weight, gain=self.config.init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_tokens_only: bool = False,
        return_reconstructed: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BEATs tokenizer.
        
        Args:
            mel_spectrogram: Input mel spectrogram [B, T, mel_bins]
            attention_mask: Attention mask for variable length sequences [B, T]
            return_tokens_only: If True, only return tokens (for inference)
            return_reconstructed: If True, return reconstructed audio
            
        Returns:
            Dictionary containing tokens, features, and optional reconstructions
        """
        
        # Encode to semantic features
        semantic_features = self.semantic_encoder(mel_spectrogram, attention_mask)
        
        # Quantize to discrete tokens
        quantization_result = self.quantizer(semantic_features)
        
        tokens = quantization_result['tokens']
        quantized_features = quantization_result['quantized']
        vq_loss = quantization_result['vq_loss']
        
        result = {
            'tokens': tokens,
            'semantic_features': semantic_features,
            'quantized_features': quantized_features,
            'vq_loss': vq_loss
        }
        
        if return_tokens_only:
            return {'tokens': tokens}
        
        # Semantic classification (for military vehicle detection)
        if attention_mask is not None:
            # Average pool over valid positions
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(quantized_features)
            pooled_features = (quantized_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Global average pooling
            pooled_features = quantized_features.mean(dim=1)
        
        semantic_logits = self.semantic_classifier(pooled_features)
        result['semantic_logits'] = semantic_logits
        
        # Reconstruction (for pre-training)
        if return_reconstructed:
            reconstructed = self.decoder(tokens, semantic_features, attention_mask)
            result['reconstructed'] = reconstructed
            
            # Reconstruction loss
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(mel_spectrogram)
                reconstruction_loss = F.mse_loss(
                    reconstructed * mask_expanded,
                    mel_spectrogram * mask_expanded,
                    reduction='sum'
                ) / mask_expanded.sum()
            else:
                reconstruction_loss = F.mse_loss(reconstructed, mel_spectrogram)
            
            result['reconstruction_loss'] = reconstruction_loss
        
        return result
    
    def encode(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode audio to tokens (inference only).
        
        Args:
            mel_spectrogram: Input mel spectrogram [B, T, mel_bins]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Discrete tokens [B, T]
        """
        with torch.no_grad():
            result = self.forward(mel_spectrogram, attention_mask, return_tokens_only=True)
            return result['tokens']
    
    def get_codebook_usage(self) -> float:
        """Get codebook usage statistics"""
        if hasattr(self.quantizer, 'get_codebook_usage'):
            return self.quantizer.get_codebook_usage()
        return 1.0
    
    def get_semantic_embeddings(self) -> torch.Tensor:
        """Get learned semantic embeddings for military vehicle types"""
        if hasattr(self.semantic_encoder, 'military_priors'):
            return self.semantic_encoder.military_priors.vehicle_type_embeddings
        return None


def create_beats_tokenizer(
    input_dim: int = 128,
    codebook_size: int = 8192,
    semantic_categories: int = 7,
    use_military_priors: bool = True
) -> BEATsTokenizer:
    """
    Create a BEATs tokenizer with default configuration for military vehicle detection.
    
    Args:
        input_dim: Input mel bin dimension
        codebook_size: Size of acoustic codebook
        semantic_categories: Number of military vehicle categories
        use_military_priors: Whether to use military-specific priors
        
    Returns:
        Configured BEATs tokenizer
    """
    
    config = BEATsTokenizerConfig(
        num_mel_bins=input_dim,
        codebook_size=codebook_size,
        semantic_categories=semantic_categories,
        use_military_priors=use_military_priors
    )
    
    return BEATsTokenizer(config)


if __name__ == "__main__":
    # Example usage and testing
    config = BEATsTokenizerConfig(
        num_mel_bins=128,
        codebook_size=1024,  # Smaller for testing
        semantic_categories=7,
        encoder_layers=2,    # Smaller for testing
        decoder_layers=2
    )
    
    # Create tokenizer
    tokenizer = BEATsTokenizer(config)
    
    # Create test input
    batch_size = 2
    time_steps = 100
    mel_bins = 128
    
    mel_spectrogram = torch.randn(batch_size, time_steps, mel_bins)
    attention_mask = torch.ones(batch_size, time_steps)
    
    # Test forward pass
    with torch.no_grad():
        result = tokenizer(mel_spectrogram, attention_mask, return_reconstructed=True)
        
        print(f"Input shape: {mel_spectrogram.shape}")
        print(f"Tokens shape: {result['tokens'].shape}")
        print(f"Semantic logits shape: {result['semantic_logits'].shape}")
        print(f"Reconstructed shape: {result['reconstructed'].shape}")
        print(f"VQ Loss: {result['vq_loss'].item():.4f}")
        print(f"Reconstruction Loss: {result['reconstruction_loss'].item():.4f}")
        print(f"Codebook usage: {tokenizer.get_codebook_usage():.2%}")
    
    # Test encoding only
    tokens = tokenizer.encode(mel_spectrogram, attention_mask)
    print(f"Encoded tokens shape: {tokens.shape}")
    
    print("BEATs Tokenizer test completed successfully!")
