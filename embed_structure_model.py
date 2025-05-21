import json
import inspect
from functools import partial
from dataclasses import dataclass, asdict
import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from scipy.stats import pearsonr
import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_qkvpacked_func
from typing import Optional
import math
import matplotlib.pyplot as plt
torch.backends.cuda.enable_flash_sdp(True)

#use tensor cores in Ada???
torch.set_float32_matmul_precision('high')

@dataclass
class Config:
    def isolate(self, config):
        specifics = inspect.signature(config).parameters
        my_specifics = {k: v for k, v in asdict(self).items() if k in specifics}
        return config(**my_specifics)

    def to_json(self, filename):
        config = json.dumps(asdict(self), indent=2)
        with open(filename, 'w') as f:
            f.write(config)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            js = json.loads(f.read())
        config = cls(**js)
        return config



@dataclass
class trans_basic_block_Config(Config):
    d_model: int = 1152  # Input dimension for Transformer Encoder
    nhead: int = 16  # Number of attention heads
    num_layers: int = 10  # Number of Transformer Encoder layers
    dim_feedforward: int = 2304  # Hidden size in feedforward layers of Transformer
    out_dim: int = 512  # Output dimension of the final MLP (1024 for ESMC600)
    dropout: float = 0.05  # Dropout rate (changed to 0.05 for trainning from checkpoint)
    activation: str = 'relu'  # Activation function in Transformer
    hidden_dim: int = 1024 #(Must be divisible by num_heads) Hidden dimension for attention pooling (1024 for ESMC600)
    batch_first=True

    # Optimization parameters
    lr0: float = 0.0001  # Initial learning rate
    warmup_steps: int = 300  # Learning rate scheduler warmup steps

    # Convolution parameters
    conv_kernel_size: int = 5
    conv_padding: int = 1
    conv_out_channels: int = 2560

    def build(self):
        return trans_basic_block(self)



class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert (hidden_dim // num_heads) % 8 == 0, "head_dim must be multiple of 8"

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, hidden_dim)
        self.k_proj = nn.Linear(d_model, hidden_dim)
        self.v_proj = nn.Linear(d_model, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1024)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        B, L, _ = x.shape
        orig_dtype = x.dtype
        device = x.device

        # Use bfloat16 if supported, else fp16
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        x = x.to(dtype)

        # Layer norm and projections
        x_norm = self.layer_norm(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # [B, L, H, D]
        q = q.view(B, L, self.num_heads, self.head_dim) * self.scaling
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        if padding_mask is None:
            # No padding
            cu_seqlens = torch.arange(0, (B + 1)*L, step=L, dtype=torch.int32, device=q.device)
            max_seqlen = L

            # [B, L, 3, H, D]
            qkv = torch.stack([q, k, v], dim=2)  # [B, L, 3, H, D]

            # Flatten to [B*L, 3, H, D]
            qkv = qkv.reshape(B * L, 3, self.num_heads, self.head_dim)

            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                self.dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=False
            )
            # [B*L, H, D] -> [B, L, H, D]
            output = output_unpad.view(B, L, self.num_heads, self.head_dim)
        else:
            padding_mask = padding_mask.bool()
            keep_mask = ~padding_mask  # True means keep

            # Unpad q, k, v
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, keep_mask)
            k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, keep_mask)
            v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v, keep_mask)

            # Ensure consistency
            assert max_seqlen_q == max_seqlen_k == max_seqlen_v

            max_seqlen = max_seqlen_q
            cu_seqlens = cu_seqlens_q

            # Stack unpadded qkv: [total_tokens, 3, H, D]
            qkv = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)

            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                self.dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=False
            )
            # [total_tokens, H, D]

            # Pad back: [B, L, H, D]
            output = pad_input(output_unpad, indices_q, B, L)

        # [B, L, H, D] -> [B, L, D_model]
        output = output.reshape(B, L, self.hidden_dim)

        # Pooling over valid tokens
        if padding_mask is not None:
            # Create a mask for valid tokens
            valid_tokens = (~padding_mask).to(device=device, dtype=dtype).unsqueeze(-1)  # [B, L, 1]
            valid_length = valid_tokens.sum(dim=1).clamp(min=1e-9)                      # [B, 1]
            pooled = (output * valid_tokens).sum(dim=1) / valid_length                 # [B, hidden_dim]
        else:
            pooled = output.mean(dim=1)  # [B, hidden_dim]

        # Project to final dimension
        pooled = self.out_proj(pooled)  # [B, 1024]

        return pooled.to(orig_dtype)

#encoder to use flash attention in encoder
class FlashAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first: bool = True):
        super().__init__()
        self.self_attn = MultiHeadFlashAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        #feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: Optional[bool] = False) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

#Multihead attention for the encoder
class MultiHeadFlashAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, batch_first: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.batch_first = batch_first
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # query, key, value: [B, L, D]
        B, L, _ = query.shape
        orig_dtype = query.dtype

        # Use bfloat16 if supported, else fp16
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        query = query.to(dtype)
        key = key.to(dtype)
        value = value.to(dtype)

        # Linear projections
        q = self.q_proj(query)  # [B, L, D]
        k = self.k_proj(key)    # [B, L, D]
        v = self.v_proj(value)  # [B, L, D]

        # Reshape to [B, L, H, D_head]
        q = q.view(B, L, self.num_heads, self.head_dim) * self.scaling
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        if key_padding_mask is None:
            # No padding case
            # Just treat all sequences as full length
            cu_seqlens = torch.arange(0, (B + 1)*L, step=L, dtype=torch.int32, device=q.device)
            max_seqlen = L

            # [B, L, H, D] -> [B, L, 3, H, D]
            qkv = torch.stack([q, k, v], dim=2)  # [B, L, 3, H, D]

            # Flatten to [B*L, 3, H, D]
            total_tokens = B * L
            qkv = qkv.reshape(total_tokens, 3, self.num_heads, self.head_dim)

            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                self.dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=False
            )
            # [B*L, H, D] -> [B, L, H, D]
            output = output_unpad.view(B, L, self.num_heads, self.head_dim)

        else:
            # With padding
            # key_padding_mask: [B, L], True means padded
            key_padding_mask = key_padding_mask.bool()
            keep_mask = ~key_padding_mask  # True means keep

            # unpad_input expects: (batch, seqlen, ...)
            # q, k, v are already [B, L, H, D]
            # This is compatible with unpad_input, which treats "..." as (H, D)
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, keep_mask)
            k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, keep_mask)
            v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v, keep_mask)

            assert max_seqlen_q == max_seqlen_k == max_seqlen_v, "Mismatched max seq lengths"
            max_seqlen = max_seqlen_q
            cu_seqlens = cu_seqlens_q

            # q_unpad, k_unpad, v_unpad: [total_tokens, H, D]
            qkv = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # [total_tokens, 3, H, D]

            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                self.dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=False
            )
            # output_unpad: [total_tokens, H, D]

            # pad_input returns [B, L, H, D]
            output = pad_input(output_unpad, indices_q, B, L)

        # [B, L, H, D] -> [B, L, D_model]
        output = output.reshape(B, L, self.d_model)
        output = self.out_proj(output)
        output = output.to(orig_dtype)

        return output, None

    def handle_padding_mask(self, q, k, v, key_padding_mask):
        """Helper function to handle padding masks for variable length attention"""
        # Convert mask to bool and invert (True = keep, False = mask)
        attention_mask = (~key_padding_mask).bool()
        
        # Unpad inputs
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)
        k_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(k, attention_mask)
        v_unpad, _, _, _ = unpad_input(v, attention_mask)

        return (q_unpad, k_unpad, v_unpad, indices_q, 
                cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k)


class trans_basic_block(pl.LightningModule):
    def __init__(self, config: trans_basic_block_Config):
        super().__init__()
        self.config = config


        # Transformer Encoder
        encoder_args = {k: v for k, v in asdict(config).items() if k in inspect.signature(FlashAttentionEncoderLayer).parameters}
        num_layers = config.num_layers
        encoder_layer = FlashAttentionEncoderLayer(**encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Modified attention pooling with increased capacity
        self.attention_pooling = MultiHeadAttentionPooling(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            num_heads=config.nhead,
            dropout=config.dropout  # Use config value
        )

        # Reduced dropout and increased MLP capacity
        dropout_rate = 0.1  # Reduced from 0.1 (Before Dec20 = 0.05)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Increased MLP capacity with additional layers
        self.mlp = nn.Sequential(
            nn.Linear(1024, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(config.hidden_dim, config.out_dim)
        )
        
        self.scale = nn.Parameter(torch.tensor(math.sqrt(28.0), dtype=torch.bfloat16))

        # Loss Function
        #self.l1_loss = nn.L1Loss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.val_dot_products = []
        self.val_tm_scores = []

        self.alpha = 0.0  # starting value
        self.min_alpha = 0.0  # minimum weight for either loss
        self.max_alpha = 0.5  # maximum weight for either loss
        
        # Track validation metrics
        self.best_val_loss = float('inf')
        self.val_loss_history = []
        self.alpha_history = []
        
        self.log_every_n_steps = 500

        #part of the contrastive loss function
        self.margin = 1.0
        self.threshold = 26.5
        self.loss_mode = "distance"


        self.save_hyperparameters()

    # @classmethod
    # def load_with_config(cls, checkpoint_path, config):

    #     model = cls(config=config)  # Initialize with the config
    #     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #     model.load_state_dict(checkpoint["state_dict"], strict=False)  # Load weights
    #     return model



    def pool_sequence(self, x, padding_mask):
        """
        Apply multi-head attention pooling with proper masking.
        """
        return self.attention_pooling(x, padding_mask)

    def setup(self, stage=None):
        """Called by Lightning before training/validation/testing."""
        #flash attention on newer GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.to(torch.bfloat16)
        else:
            self.to(torch.float16)

    def forward(self, x, src_mask, src_key_padding_mask):
        # Transformer Encoder

        x = x.to(self.device, self.dtype)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)

        
        encoded = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling
        pooled = self.pool_sequence(encoded, src_key_padding_mask)



        # Layer norm and dropout
        pooled = self.dropout(pooled)

        # Final MLP
        output = self.mlp(pooled)
        #output = F.normalize(output, p=2, dim=-1) * math.sqrt(14.00)
        output = output * self.scale
        #output = F.normalize(output, p=2, dim=-1)


        return output
    
    def distance_loss_euclidean(self, output_seq1, output_seq2, tm_score):
        pdist_seq = nn.PairwiseDistance(p=2)
        dist_seq = pdist_seq(output_seq1, output_seq2)
        dist_tm = torch.cdist(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        return dist_tm

    def distance_loss(self, output_seq1, output_seq2, tm_score):
        dot_product = torch.sum(output_seq1 * output_seq2, dim=1)
        dist_tm = self.smooth_l1_loss(dot_product, tm_score.float())
        return dist_tm

#Feb 26, updating the loss to up-weight the high and low dot product value...
    def weighted_distance_loss(self, output_seq1, output_seq2, tm_score):
        dot_product = torch.sum(output_seq1 * output_seq2, dim=1)
        base_loss = F.smooth_l1_loss(dot_product, tm_score.float(), reduction='none')
        #weights = 1.0 / (1.0 + tm_score.float())
        normalized_tm = tm_score.float() / 27.9 #remember to change this
        #weights = 0.5 + 0.5 * (normalized_tm) # removed the squaring of normalized tm. 
        #weighted_loss = base_loss * weights
        weights = 0.5 + abs(normalized_tm - 0.5)
        weighted_loss = base_loss * weights
        return weighted_loss.mean()



    def pearson_correlation_loss(self, x, y):
        """
        Using pearson correlation as loss
        Returns 1 - correlation so that minimizing loss maximizes correlation.
        """
        if torch.isnan(x).any() or torch.isnan(y).any():
            return torch.tensor(1.0, device=x.device, requires_grad=True)
            
        x_var = torch.var(x, unbiased=False)
        y_var = torch.var(y, unbiased=False)
        
        # Check for zero variance
        if x_var == 0 or y_var == 0:
            return torch.tensor(1.0, device=x.device, requires_grad=True)
        
        # Center the variables
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Compute correlation
        numerator = (x_centered * y_centered).mean()
        denominator = torch.sqrt(x_var * y_var)
        
        # Add small epsilon to avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        # Clamp correlation to [-1, 1] range to handle numerical errors
        correlation = torch.clamp(correlation, min=-1.0, max=1.0)
        
        # Return 1 - correlation as loss (to minimize)
        return 1 - correlation
    
    def distance_loss_train_pearson_euclidean(self, output_seq1, output_seq2, tm_score):
        pdist_seq = nn.PairwiseDistance(p=2)
        dist_seq = pdist_seq(output_seq1, output_seq2)
        #dist_tm = torch.cdist(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        loss = self.pearson_correlation_loss(dist_seq, tm_score.float())
        return loss

    def distance_loss_train_pearson(self, output_seq1, output_seq2, tm_score):
        dot_product = torch.sum(output_seq1 * output_seq2, dim=1)
        loss = self.pearson_correlation_loss(dot_product, tm_score.float())
        return loss

    def combined_loss(self, output_seq1, output_seq2, tm_score):
        dot_product = torch.sum(output_seq1 * output_seq2, dim=1)
        
        pearson_loss = self.pearson_correlation_loss(dot_product, tm_score.float())
        dist_tm = self.smooth_l1_loss(dot_product, tm_score.float())
        
        # Combine losses with current alpha
        combined = self.alpha * pearson_loss + (1 - self.alpha) * dist_tm  # Corrected variable name
        
        return combined, pearson_loss, dist_tm  # Corrected variable name


    def distance_loss_val_euclidean(self, output_seq1, output_seq2, tm_score):
        pdist_seq = nn.PairwiseDistance(p=2)
        dot_produce = pdist_seq(output_seq1, output_seq2)
        dist_tm = torch.cdist(dot_produce.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        return dist_tm, dot_produce

    def distance_loss_val(self, output_seq1, output_seq2, tm_score):
        dot_product = torch.sum(output_seq1 * output_seq2, dim=1)
        dist_tm = self.smooth_l1_loss(dot_product, tm_score.float())
        # return dist_tm
        #Added the 'dot_product' output to allow for downstream pearson's R
        return dist_tm, dot_product

    def training_step(self, train_batch, batch_idx):
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = train_batch
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        #Note to change this if monitoring something different
        #toggle different loss terms
        if self.loss_mode == 'distance': #loss based on differnece between dot product and rmsd
            loss = self.distance_loss(out_seq1, out_seq2, tm_score)
            #loss = self.distance_loss_euclidean(out_seq1, out_seq2, tm_score)
            dot_product = torch.sum(out_seq1 * out_seq2, dim=1)
            #pdist_seq = nn.PairwiseDistance(p=2)
            #dot_product = pdist_seq(out_seq1, out_seq2)
            pearson_loss = self.pearson_correlation_loss(dot_product, tm_score.float())

        elif self.loss_mode == 'pearson': #pure pearson correlation loss function
            loss = self.distance_loss_train_pearson(out_seq1, out_seq2, tm_score)
            #loss = self.distance_loss_train_pearson_euclidean(out_seq1, out_seq2, tm_score)
            pearson_loss = loss
        elif self.loss_mode == 'contrastive': #bins based on large distance and small distance. Parameters defined in __init__
            dot_product = torch.sum(out_seq1 * out_seq2, dim=1)
            y = (tm_score >= self.threshold).float()
            pos_loss = y * 0.5 * (1.0 - dot_product).pow(2)
            neg_loss = (1 - y) * 0.5 * F.relu(dot_product - self.margin).pow(2)
            contrastive_loss = pos_loss + neg_loss
            loss = contrastive_loss.mean()

            pearson_loss = self.pearson_correlation_loss(dot_product, tm_score.float())



        with torch.no_grad():
            correlation = 1 - pearson_loss.item()
            self.log('train_correlation', correlation, sync_dist=True)

        self._log_attention_patterns(batch_idx)
        if batch_idx % self.log_every_n_steps == 0:
            return {
                "loss": loss,
                "out_seq1": out_seq1.detach(),
                "out_seq2": out_seq2.detach(),
            }

        self.log('train_loss', loss, sync_dist=True)
        self.log('train_pearson_loss', pearson_loss, sync_dist=True)
        return loss
    
    def on_after_backward(self):
        """Called after backward pass - gradients now exist"""
        if self.trainer.global_step % self.log_every_n_steps == 0:
            self._log_gradients()

    def validation_step(self, val_batch, batch_idx):
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = val_batch
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        loss, dot_product = self.distance_loss_val(out_seq1, out_seq2, tm_score)
        #loss, dot_product = self.distance_loss_val_euclidean(out_seq1, out_seq2, tm_score)
        
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        # validation ow returns dot product
        self.val_dot_products.append(dot_product.detach())
        self.val_tm_scores.append(tm_score.detach())

        return {'val_loss': loss}
    

    def on_epoch_start(self):
        if self.current_epoch < 1:
            self.loss_mode = 'distance'
        elif 1 < self.current_epoch < 2:
            self.loss_mode = 'pearson'
        else:
            self.loss_mode = 'distance'


    def _log_gradients(self):
        """Log gradient statistics for different components with safety checks"""
        # Query projection gradients
        if hasattr(self.attention_pooling.q_proj, 'weight') and self.attention_pooling.q_proj.weight.grad is not None:
            q_proj_grad_norm = self.attention_pooling.q_proj.weight.grad.norm(2).item()
            self.log('gradients/q_proj_grad_norm', q_proj_grad_norm, sync_dist=True)
        
        # Key projection gradients
        if hasattr(self.attention_pooling.k_proj, 'weight') and self.attention_pooling.k_proj.weight.grad is not None:
            k_proj_grad_norm = self.attention_pooling.k_proj.weight.grad.norm(2).item()
            self.log('gradients/k_proj_grad_norm', k_proj_grad_norm, sync_dist=True)
        
        # Value projection gradients
        if hasattr(self.attention_pooling.v_proj, 'weight') and self.attention_pooling.v_proj.weight.grad is not None:
            v_proj_grad_norm = self.attention_pooling.v_proj.weight.grad.norm(2).item()
            self.log('gradients/v_proj_grad_norm', v_proj_grad_norm, sync_dist=True)
        
        # Output projection gradients
        if hasattr(self.attention_pooling.out_proj, 'weight') and self.attention_pooling.out_proj.weight.grad is not None:
            out_proj_grad_norm = self.attention_pooling.out_proj.weight.grad.norm(2).item()
            self.log('gradients/out_proj_grad_norm', out_proj_grad_norm, sync_dist=True)
        
        # MLP gradients
        for idx, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear) and hasattr(layer.weight, 'grad') and layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm(2).item()
                self.log(f'gradients/mlp_layer_{idx}_norm', grad_norm, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr0,
            weight_decay=0.01 * 2 # 2x the default weight decay for training from checkpoint
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,
            eta_min=self.config.lr0 * 0.1
        )
        
        return [optimizer], [scheduler]