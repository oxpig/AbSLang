import torch
from .embed_structure_model_cpu import trans_basic_block, trans_basic_block_Config, ModelConfig

from esm.models.esmc import ESMC
from pathlib import Path 
from importlib import import_module

# Default paths for backward compatibility (will be overridden by user-provided paths)
DEFAULT_TM_CKPT = None
DEFAULT_TM_CFG = None

def load_mode(mode, device, tm_checkpoint_path=None, tm_config_path=None):
    """
    Load models for the specified mode.
    
    Args:
        mode: 'paired', 'hc', or 'nb'
        device: torch device to load models on
        tm_checkpoint_path: Path to transformer model checkpoint
        tm_config_path: Path to transformer model config JSON
    
    Returns:
        Tuple of (empty_dict, language_model, transformer_model, constant)
    """
    # Determine fallback model class based on mode
    fallback_cls = "PairedIgT5" if mode == "paired" else "ESMC"
    
    # No longer loading pre-computed embeddings - all sequences will be embedded from scratch
    seq_embeds = {}  # Empty dict instead of loading from file
    
    # Load the primary language model
    if fallback_cls == "ESMC":
        fallback = ESMC.from_pretrained("esmc_600m").to(device)
    else:  # PairedIgT5 lives in model.py
        from .model import PairedIgT5
        fallback = PairedIgT5()
    
    # Load transformer model if paths are provided
    if tm_checkpoint_path and tm_config_path:
        tm_model = trans_basic_block.load_from_checkpoint(
            tm_checkpoint_path,
            config=trans_basic_block_Config.from_json(tm_config_path)
        ).to(device).eval()
    else:
        # If no paths provided, return None for transformer model
        # This allows using the language models without the transformer
        tm_model = None
    
    # Return constant value (used for scaling, kept for compatibility)
    const = 28.0
    
    return seq_embeds, fallback, tm_model, const