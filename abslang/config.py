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
        tm_checkpoint_path: Path to transformer model checkpoint (required)
        tm_config_path: Path to transformer model config JSON (required)
    
    Returns:
        Tuple of (empty_dict, language_model, transformer_model, constant)
    
    Raises:
        ValueError: If transformer model paths are not provided
    """
    # Validate transformer model paths are provided
    if not tm_checkpoint_path or not tm_config_path:
        raise ValueError(
            "Transformer model is required for structure-aware embeddings. "
            "Please provide both tm_checkpoint_path and tm_config_path. "
            "The language model embeddings alone have no structural meaning."
        )
    
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
    
    # Load transformer model (now mandatory)
    try:
        tm_model = trans_basic_block.load_from_checkpoint(
            tm_checkpoint_path,
            config=trans_basic_block_Config.from_json(tm_config_path)
        ).to(device).eval()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load transformer model from {tm_checkpoint_path} "
            f"with config {tm_config_path}: {e}"
        )
    
    # Return constant value (used for scaling, kept for compatibility)
    const = 28.0
    
    return seq_embeds, fallback, tm_model, const