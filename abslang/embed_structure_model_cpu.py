"""
Structural Embedding Model for Protein Similarity Prediction

This module implements a transformer-based model that takes pre-computed protein 
sequence embeddings and generates structural embeddings that predict RMSD similarity
between protein pairs. The model is designed to run efficiently on CPU and be 
compatible with FAISS for similarity search.

Key features:
- CPU-optimized transformer architecture
- Multi-head attention pooling for variable sequence lengths
- Float32 outputs compatible with FAISS
"""

import json
import inspect
import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration class for the Structural Embedding Model"""
    
    # Model architecture parameters
    d_model: int = 1024                # Input embedding dimension
    nhead: int = 16                    # Number of attention heads
    num_layers: int = 8                # Number of transformer encoder layers
    dim_feedforward: int = 2048        # Feedforward network dimension
    out_dim: int = 128                 # Output embedding dimension
    dropout: float = 0.05              # Dropout rate
    activation: str = 'relu'           # Activation function
    hidden_dim: int = 1024             # Attention pooling hidden dimension
    batch_first: bool = True           # Batch first ordering
    
    # Training parameters
    lr0: float = 0.0001                # Initial learning rate
    warmup_steps: int = 300            # LR warmup steps
    
    # Legacy parameters (kept for compatibility)
    conv_kernel_size: int = 5
    conv_padding: int = 1
    conv_out_channels: int = 2560
    
    def isolate(self, config_function):
        """Extract only the parameters needed for a specific function"""
        specifics = inspect.signature(config_function).parameters
        my_specifics = {k: v for k, v in asdict(self).items() if k in specifics}
        return my_specifics

    def to_json(self, filename: str):
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_json(cls, filename: str):
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def build(self):
        """Build the model using this configuration"""
        return StructuralEmbeddingModel(self)


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling layer that converts variable-length sequences
    to fixed-size embeddings using standard PyTorch operations (CPU-compatible).
    """
    
    def __init__(self, d_model: int, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.query_projection = nn.Linear(d_model, hidden_dim)
        self.key_projection = nn.Linear(d_model, hidden_dim)
        self.value_projection = nn.Linear(d_model, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1024)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention pooling.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            padding_mask: Boolean mask of shape [batch_size, seq_len] where True indicates padding
            
        Returns:
            pooled: Output tensor of shape [batch_size, 1024]
        """
        batch_size, seq_len, _ = x.shape
        
        # Ensure float32 for CPU compatibility
        x = x.float()
        
        # Layer normalization and projections
        x_normalized = self.layer_norm(x)
        queries = self.query_projection(x_normalized)
        keys = self.key_projection(x_normalized)
        values = self.value_projection(x_normalized)
        
        # Reshape for multi-head attention: [batch_size, seq_len, num_heads, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale queries
        queries = queries * self.scaling
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Expand mask for broadcasting: [batch_size, 1, 1, seq_len]
            mask_expanded = padding_mask.bool().unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask_expanded, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Reshape back: [batch_size, seq_len, hidden_dim]
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, self.hidden_dim)
        
        # Pool over sequence length with proper masking
        if padding_mask is not None:
            # Create mask for valid tokens: [batch_size, seq_len, 1]
            valid_mask = (~padding_mask).float().unsqueeze(-1)
            valid_length = valid_mask.sum(dim=1).clamp(min=1e-9)  # [batch_size, 1]
            pooled = (attended_values * valid_mask).sum(dim=1) / valid_length
        else:
            pooled = attended_values.mean(dim=1)
        
        # Final projection
        pooled = self.output_projection(pooled)
        
        return pooled.float()


class StructuralEmbeddingModel(nn.Module):
    """
    Transformer-based model for learning structural embeddings from protein sequences.
    
    The model takes pre-computed protein sequence embeddings and generates fixed-size
    structural embeddings that can predict RMSD similarity between protein pairs.
    
    Architecture:
    1. Transformer encoder with standard PyTorch implementation
    2. Multi-head attention pooling to handle variable sequence lengths
    3. MLP head for final embedding generation
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Transformer encoder using standard PyTorch implementation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=config.batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Multi-head attention pooling for variable sequence lengths
        self.attention_pooling = MultiHeadAttentionPooling(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            num_heads=config.nhead,
            dropout=config.dropout
        )
        
        # Output MLP
        dropout_rate = 0.1
        self.dropout = nn.Dropout(dropout_rate)
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(config.hidden_dim, config.out_dim)
        )
        
        # Learnable scaling parameter (initialized as float32)
        self.embedding_scale = nn.Parameter(torch.tensor(math.sqrt(28.0), dtype=torch.float32))

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: ModelConfig):
        """Load model from checkpoint file with module name remapping"""
        import sys
        
        # Add module alias for backward compatibility with old checkpoints
        # This allows old checkpoints that reference 'embed_structure_model' to work
        if 'embed_structure_model' not in sys.modules:
            sys.modules['embed_structure_model'] = sys.modules[__name__]
        
        model = cls(config)
        
        # Load checkpoint with weights_only=False to support full unpickling
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle both Lightning and regular PyTorch checkpoints
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        return model

    @classmethod
    def load_with_config_flexibility(cls, checkpoint_path: str, config: ModelConfig):
        """Load model with flexible parameter matching"""
        import sys
        
        # Add module alias for backward compatibility
        if 'embed_structure_model' not in sys.modules:
            sys.modules['embed_structure_model'] = sys.modules[__name__]
        
        model = cls(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if "state_dict" in checkpoint:
            checkpoint_state = checkpoint["state_dict"]
        else:
            checkpoint_state = checkpoint
        
        model_state = model.state_dict()
        
        # Filter compatible parameters
        filtered_state = {}
        import warnings
        for key, value in checkpoint_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    warnings.warn(
                        f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}"
                    )
            else:
                warnings.warn(f"Parameter {key} not found in model")
        
        model.load_state_dict(filtered_state, strict=False)
        warnings.warn(f"Loaded {len(filtered_state)}/{len(model_state)} parameters from checkpoint")
        
        return model

    def forward(self, sequences: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            sequences: Input sequences of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (unused in current implementation)
            padding_mask: Boolean mask where True indicates padding positions
            
        Returns:
            embeddings: Structural embeddings of shape [batch_size, out_dim] in float32
        """
        # Get device and ensure float32
        device = next(self.parameters()).device
        sequences = sequences.to(device, torch.float32)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)
        
        # Transformer encoding
        encoded_sequences = self.transformer_encoder(
            sequences, 
            mask=attention_mask, 
            src_key_padding_mask=padding_mask
        )
        
        # Attention pooling to fixed size
        pooled_representation = self.attention_pooling(encoded_sequences, padding_mask)
        
        # Apply dropout
        pooled_representation = self.dropout(pooled_representation)
        
        # Generate final embeddings
        embeddings = self.output_mlp(pooled_representation)
        embeddings = embeddings * self.embedding_scale
        
        return embeddings.float()


# Legacy aliases for backward compatibility
trans_basic_block = StructuralEmbeddingModel
trans_basic_block_Config = ModelConfig
Config = ModelConfig
