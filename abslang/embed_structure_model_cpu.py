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
- Optional training dependencies for inference-only use
"""

import json
import inspect
import math
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# Optional imports for training/logging (not needed for inference)
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    import pytorch_lightning as pl
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False
    # Create dummy classes for compatibility
    class pl:
        class LightningModule:
            def __init__(self):
                pass
            def log(self, *args, **kwargs):
                pass
            def save_hyperparameters(self):
                pass
        class loggers:
            class WandbLogger:
                pass


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


class StructuralEmbeddingModel(nn.Module if not HAS_TRAINING_DEPS else pl.LightningModule):
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
        
        # Loss functions
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        
        # Training state
        self.validation_predictions = []
        self.validation_targets = []
        
        # Loss configuration
        self.loss_margin = 1.0
        self.loss_threshold = 26.5
        self.loss_mode = "distance"  # Options: "distance", "pearson", "contrastive"
        
        # Save hyperparameters if in training mode
        if HAS_TRAINING_DEPS:
            self.save_hyperparameters()
    
    def log(self, *args, **kwargs):
        """Dummy log method for non-Lightning use"""
        if HAS_TRAINING_DEPS and hasattr(super(), 'log'):
            super().log(*args, **kwargs)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: ModelConfig):
        """Load model from checkpoint file"""
        model = cls(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle both Lightning and regular PyTorch checkpoints
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        return model

    @classmethod
    def load_with_config_flexibility(cls, checkpoint_path: str, config: ModelConfig):
        """Load model with flexible parameter matching"""
        model = cls(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if "state_dict" in checkpoint:
            checkpoint_state = checkpoint["state_dict"]
        else:
            checkpoint_state = checkpoint
        
        model_state = model.state_dict()
        
        # Filter compatible parameters
        filtered_state = {}
        for key, value in checkpoint_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    print(f"Shape mismatch for {key}: "
                          f"checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                print(f"Parameter {key} not found in model")
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)}/{len(model_state)} parameters from checkpoint")
        
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

    # === Loss Functions ===
    
    def euclidean_distance_loss(self, embeddings_1: torch.Tensor, 
                               embeddings_2: torch.Tensor, 
                               target_distances: torch.Tensor) -> torch.Tensor:
        """Compute loss based on Euclidean distance between embeddings"""
        pairwise_distance = nn.PairwiseDistance(p=2)
        predicted_distances = pairwise_distance(embeddings_1, embeddings_2)
        return self.smooth_l1_loss(predicted_distances, target_distances.float())

    def dot_product_loss(self, embeddings_1: torch.Tensor, 
                        embeddings_2: torch.Tensor, 
                        target_scores: torch.Tensor) -> torch.Tensor:
        """Compute loss based on dot product similarity"""
        dot_products = torch.sum(embeddings_1 * embeddings_2, dim=1)
        return self.smooth_l1_loss(dot_products, target_scores.float())

    def pearson_correlation_loss(self, predictions: torch.Tensor, 
                               targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on Pearson correlation (1 - correlation)"""
        if torch.isnan(predictions).any() or torch.isnan(targets).any():
            return torch.tensor(1.0, device=predictions.device, requires_grad=True)
            
        pred_var = torch.var(predictions, unbiased=False)
        target_var = torch.var(targets, unbiased=False)
        
        if pred_var == 0 or target_var == 0:
            return torch.tensor(1.0, device=predictions.device, requires_grad=True)
        
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        
        covariance = (pred_centered * target_centered).mean()
        correlation = covariance / (torch.sqrt(pred_var * target_var) + 1e-8)
        correlation = torch.clamp(correlation, min=-1.0, max=1.0)
        
        return 1 - correlation

    def contrastive_loss(self, embeddings_1: torch.Tensor, 
                        embeddings_2: torch.Tensor, 
                        target_scores: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for similar/dissimilar pairs"""
        dot_products = torch.sum(embeddings_1 * embeddings_2, dim=1)
        is_similar = (target_scores >= self.loss_threshold).float()
        
        positive_loss = is_similar * 0.5 * (1.0 - dot_products).pow(2)
        negative_loss = (1 - is_similar) * 0.5 * F.relu(dot_products - self.loss_margin).pow(2)
        
        return (positive_loss + negative_loss).mean()

    # === Training Methods (only available with PyTorch Lightning) ===
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning"""
        if not HAS_TRAINING_DEPS:
            raise RuntimeError("Training dependencies not available")
            
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, targets = batch
        
        embeddings_1 = self.forward(sequence_1, padding_mask=pad_mask_1)
        embeddings_2 = self.forward(sequence_2, padding_mask=pad_mask_2)
        
        # Compute loss based on mode
        if self.loss_mode == 'distance':
            loss = self.euclidean_distance_loss(embeddings_1, embeddings_2, targets)
            # Also compute Pearson loss for monitoring
            pairwise_distance = nn.PairwiseDistance(p=2)
            predicted_distances = pairwise_distance(embeddings_1, embeddings_2)
            pearson_loss = self.pearson_correlation_loss(predicted_distances, targets.float())
        elif self.loss_mode == 'pearson':
            pairwise_distance = nn.PairwiseDistance(p=2)
            predicted_distances = pairwise_distance(embeddings_1, embeddings_2)
            loss = self.pearson_correlation_loss(predicted_distances, targets.float())
            pearson_loss = loss
        elif self.loss_mode == 'contrastive':
            loss = self.contrastive_loss(embeddings_1, embeddings_2, targets)
            dot_products = torch.sum(embeddings_1 * embeddings_2, dim=1)
            pearson_loss = self.pearson_correlation_loss(dot_products, targets.float())
        else:
            raise ValueError(f"Unknown loss mode: {self.loss_mode}")

        # Log metrics
        with torch.no_grad():
            correlation = 1 - pearson_loss.item()
            self.log('train_loss', loss, sync_dist=True)
            self.log('train_correlation', correlation, sync_dist=True)
            self.log('train_pearson_loss', pearson_loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        if not HAS_TRAINING_DEPS:
            raise RuntimeError("Training dependencies not available")
            
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, targets = batch
        
        embeddings_1 = self.forward(sequence_1, padding_mask=pad_mask_1)
        embeddings_2 = self.forward(sequence_2, padding_mask=pad_mask_2)
        
        # Compute validation loss (always using Euclidean distance)
        loss = self.euclidean_distance_loss(embeddings_1, embeddings_2, targets)
        
        # Compute predictions for correlation analysis
        pairwise_distance = nn.PairwiseDistance(p=2)
        predictions = pairwise_distance(embeddings_1, embeddings_2)
        
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        
        # Store for epoch-end analysis
        self.validation_predictions.append(predictions.detach())
        self.validation_targets.append(targets.detach())

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        """End of validation epoch processing"""
        if not HAS_TRAINING_DEPS or len(self.validation_predictions) == 0:
            return
            
        # Gather all predictions and targets
        all_predictions = torch.cat(self.validation_predictions, dim=0)
        all_targets = torch.cat(self.validation_targets, dim=0)
        
        # Handle distributed training
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            pred_list = [torch.zeros_like(all_predictions) for _ in range(world_size)]
            target_list = [torch.zeros_like(all_targets) for _ in range(world_size)]
            
            torch.distributed.all_gather(pred_list, all_predictions)
            torch.distributed.all_gather(target_list, all_targets)
            
            all_predictions = torch.cat(pred_list, dim=0)
            all_targets = torch.cat(target_list, dim=0)

        # Convert to numpy for analysis
        predictions_np = all_predictions.float().cpu().numpy()
        targets_np = all_targets.float().cpu().numpy()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(predictions_np) | np.isnan(targets_np))
        predictions_np = predictions_np[valid_mask]
        targets_np = targets_np[valid_mask]
        
        if len(predictions_np) > 1 and not np.all(predictions_np == predictions_np[0]):
            correlation, _ = pearsonr(predictions_np, targets_np)
            
            # Create validation plot
            plt.clf()
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(targets_np, predictions_np, alpha=0.5)
            plt.ylim(0, 12)
            plt.xlim(0, 12)
            plt.xlabel('Ground Truth')
            plt.ylabel('Predicted')
            plt.title(f'Validation: Predicted vs Ground Truth (r={correlation:.3f})')
            
            # Log to wandb if available
            if (self.logger is not None and 
                isinstance(self.logger, pl.loggers.WandbLogger)):
                self.logger.experiment.log({
                    "validation_scatter_plot": wandb.Image(fig)
                })
            plt.close(fig)
            
            self.log('val_pearson_correlation', correlation, sync_dist=True, prog_bar=True)
        else:
            correlation = 0.0
            self.log('val_pearson_correlation', correlation, sync_dist=True, prog_bar=True)
        
        # Clear validation data
        self.validation_predictions = []
        self.validation_targets = []

    def configure_optimizers(self):
        """Configure optimizers for PyTorch Lightning"""
        if not HAS_TRAINING_DEPS:
            raise RuntimeError("Training dependencies not available")
            
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr0,
            weight_decay=0.02
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,
            eta_min=self.config.lr0 * 0.1
        )
        
        return [optimizer], [scheduler]


# Legacy aliases for backward compatibility
trans_basic_block = StructuralEmbeddingModel
trans_basic_block_Config = ModelConfig
Config = ModelConfig