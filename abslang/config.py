import torch
from .embed_structure_model_cpu import trans_basic_block, trans_basic_block_Config, ModelConfig

from esm.models.esmc import ESMC
from pathlib import Path 
from importlib import import_module
base = Path("/vols/opig/users/ewang/Training")

# class ModelConfig(trans_basic_block_Config):
#     pass

# import sys
# sys.modules[__name__].ModelConfig = ModelConfig

MODE_CFG = {
    "paired": dict(
        precomputed = base/"IgT5_Paired_embeddings.pt",
        fallback_cls = "PairedIgT5",
        tm_ckpt  = base/"Models/cpu_compatible_June23_paired/checkpoints/0.2244.ckpt",
        tm_cfg   = base/"Models/cpu_compatible_June23_paired/params.json",
        const    = 28.0,
    ),
    "hc": dict(
        precomputed = base/"ESMC600M_ABB3_HC_Embedding.pt",
        fallback_cls = "ESMC",
        tm_ckpt  = base/"Models/cpu_compatible_June23_paired/checkpoints/0.2244.ckpt",
        tm_cfg   = base/"Models/cpu_compatible_June23_paired/params.json",
        const    = 28.0,
    ),
    "nb": dict(
        precomputed = base/"ESMC600M_ABB2_NB_Embedding.pt",
        fallback_cls = "ESMC",
        tm_ckpt  = base/"Models/cpu_compatible_June23_paired/checkpoints/0.2244.ckpt",
        tm_cfg   = base/"Models/cpu_compatible_June23_paired/params.json",
        const    = 28.0,
    ),
}

def load_mode(mode, device):
    cfg = MODE_CFG[mode]
 
    # pre‑computed sequence‑level embeddings
    seq_embeds = torch.load(cfg["precomputed"], weights_only=True)
 
    # fallback LM
    if cfg["fallback_cls"] == "ESMC":
        fallback = ESMC.from_pretrained("esmc_600m").to(device)
    else:  # PairedIgT5 lives in model.py
        from .model import PairedIgT5
        fallback = PairedIgT5()
 
    # TM distance predictor
    tm_model = trans_basic_block.load_from_checkpoint(
        cfg["tm_ckpt"],
        config=trans_basic_block_Config.from_json(cfg["tm_cfg"])
        #strict=False,
    ).to(device).eval()
 
    return seq_embeds, fallback, tm_model, cfg["const"]