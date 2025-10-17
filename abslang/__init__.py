"""
AbSLang: Antibody Structure Language Model

A deep learning framework for antibody structure prediction and analysis.
This package provides tools for embedding antibody sequences, building search indices,
and predicting structural properties like RMSD between antibody pairs.
"""

__version__ = "0.1.0"
__author__ = "Eric Wang"

# Core public API - stable entry points only
from .search_antibody_index import FaissSearcher
from .large_rmsd_inference import predict_cdr_rmsd, predict_cdr_rmsd_batch
from .build_search_index import build_index_from_csv
from .config import load_mode

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Main API - Search
    "FaissSearcher",
    
    # Main API - RMSD Prediction
    "predict_cdr_rmsd",
    "predict_cdr_rmsd_batch",
    
    # Main API - Index Building
    "build_index_from_csv",
    
    # Configuration helper
    "load_mode",
]
