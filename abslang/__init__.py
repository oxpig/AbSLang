"""
AbSLang: Antibody Structure Language Model

A deep learning framework for antibody structure prediction and analysis.
This package provides tools for embedding antibody sequences, building search indices,
and predicting structural properties like RMSD between antibody pairs.
"""

__version__ = "0.1.0"
__author__ = "Eric Wang"

# Core functionality imports
from .main_module import (
    IndexArtifacts,
    embed_sequences,
    build_ivfpq_index,
    build_ivfpq4_index,
    build_pq_flat_index,
    pairwise_l2,
)

from .Model_Embed import (
    embed_with_fallback,
    save_embeddings as save_model_embeddings,
    parse_paired_sequence,
    embed_IgT5,
    embed_ESM_single,
    embed_ESM_HC,
)

from .config import (
    load_mode,
)

# Search functionality
from .search_antibody_index import FaissSearcher

# Inference utilities
from .Large_RMSD_Inference import infer_rmsd

# Model components
from .model import (
    ProtTrans,
    ProtTransEmbedder,
    ProtT5,
    PairedIgT5,
    ProtBert,
)

from .embed_structure_model_cpu import (
    trans_basic_block,
    trans_basic_block_Config,
    ModelConfig,
)

# Adapter functionality
from .adapter import (
    read_sequences_csv,
    load_stacked_embeddings,
    save_faiss_index,
    save_embeddings,
    save_distance_matrix,
    save_sequence_list,
    dump_artifacts,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core classes and functions
    "IndexArtifacts",
    "embed_sequences",
    "build_ivfpq_index",
    "build_ivfpq4_index",
    "build_pq_flat_index",
    "pairwise_l2",
    "embed_with_fallback",
    "save_model_embeddings",
    "parse_paired_sequence",
    "embed_IgT5",
    "embed_ESM_single",
    "embed_ESM_HC",
    "load_mode",
    
    # Search
    "FaissSearcher",
    
    # Inference
    "infer_rmsd",
    
    # Models
    "ProtTrans",
    "ProtTransEmbedder",
    "ProtT5",
    "PairedIgT5",
    "ProtBert",
    "trans_basic_block",
    "trans_basic_block_Config",
    "ModelConfig",
    
    # Adapter utilities
    "read_sequences_csv",
    "load_stacked_embeddings",
    "save_faiss_index",
    "save_embeddings",
    "save_distance_matrix",
    "save_sequence_list",
    "dump_artifacts",
]