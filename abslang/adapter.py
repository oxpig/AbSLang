from pathlib import Path
import json
from typing import List, Dict

import torch
import faiss
import numpy as np
import pandas as pd
from .main_module import IndexArtifacts


#load
def read_sequences_csv(path: str | Path, id_col: str, seq_col: str) -> List[Dict]:
    """
    Read sequences from CSV file with validation.
    
    Args:
        path: Path to CSV file
        id_col: Column name for sequence IDs
        seq_col: Column name for sequences
    
    Returns:
        List of dicts with 'id' and 'sequence' keys
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Validate required columns exist
    missing_cols = []
    if id_col not in df.columns:
        missing_cols.append(id_col)
    if seq_col not in df.columns:
        missing_cols.append(seq_col)
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns {missing_cols} in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError(f"CSV file is empty: {path}")
    
    return [{"id": r[id_col], "sequence": r[seq_col]} for _, r in df.iterrows()]


# Removed: load_stacked_embeddings (unused in current pipeline)


#write
def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def save_embeddings(tensor: torch.Tensor, path: str | Path) -> None:
    torch.save(tensor, path)


def save_distance_matrix(mat: np.ndarray, path: str | Path) -> None:
    pd.DataFrame(mat).to_csv(path, index=False)


def save_sequence_list(seq_list: List[Dict], path: str | Path) -> None:
    # Pretty-print for readability and consistency with build_search_index
    Path(path).write_text(json.dumps(seq_list, indent=2))


#together!!!
def dump_artifacts(
    art: IndexArtifacts,
    out_dir: str | Path,
    *,
    write_index: bool = True,
    write_embeddings: bool = True,
    write_distance: bool = True,
    write_sequences: bool = True,
    index_type: str = "ivfpq",
) -> None:
    """
    Persist selected artefacts to *out_dir*.

    All write_* flags default to **True** so existing callers keep the
    old behaviour.  Example for "embeddings-only":

        dump_artifacts(art, "out",
                       write_index=False,
                       write_distance=False,
                       write_sequences=False)
    
    Args:
        art: IndexArtifacts object containing the data
        out_dir: Output directory
        write_index: Whether to write the FAISS index
        write_embeddings: Whether to write embeddings
        write_distance: Whether to write distance matrix
        write_sequences: Whether to write sequence list
        index_type: Type of index ('ivfpq', 'pq', 'flat') for filename
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if write_index:
        # Use appropriate extension based on index type
        index_ext = index_type if index_type in ('ivfpq', 'pq', 'flat') else 'index'
        save_faiss_index(art.faiss_index, out / f"index.{index_ext}")

    if write_embeddings:
        save_embeddings(art.embeddings, out / "embeddings.pt")

    if write_distance and art.distance_matrix is not None:
        save_distance_matrix(art.distance_matrix, out / "dist_mat.csv")

    if write_sequences and art.sequence_list is not None:
        save_sequence_list(art.sequence_list, out / "seq_list.json")
