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
    df = pd.read_csv(path)
    return [{"id": r[id_col], "sequence": r[seq_col]} for _, r in df.iterrows()]


def load_stacked_embeddings(pt_path: str | Path, device="cpu") -> torch.Tensor:
    return torch.load(pt_path, map_location=device)


#write
def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def save_embeddings(tensor: torch.Tensor, path: str | Path) -> None:
    torch.save(tensor, path)


def save_distance_matrix(mat: np.ndarray, path: str | Path) -> None:
    pd.DataFrame(mat).to_csv(path, index=False)


def save_sequence_list(seq_list: List[Dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(seq_list))


#together!!!
def dump_artifacts(
    art: IndexArtifacts,
    out_dir: str | Path,
    *,
    write_index: bool = True,
    write_embeddings: bool = True,
    write_distance: bool = True,
    write_sequences: bool = True,
) -> None:
    """
    Persist selected artefacts to *out_dir*.

    All write_* flags default to **True** so existing callers keep the
    old behaviour.  Example for “embeddings-only”:

        dump_artifacts(art, "out",
                       write_index=False,
                       write_distance=False,
                       write_sequences=False)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if write_index:
        save_faiss_index(art.faiss_index, out / "index.ivfpq")

    if write_embeddings:
        save_embeddings(art.embeddings, out / "embeddings.pt")

    if write_distance and art.distance_matrix is not None:
        save_distance_matrix(art.distance_matrix, out / "dist_mat.csv")

    if write_sequences and art.sequence_list is not None:
        save_sequence_list(art.sequence_list, out / "seq_list.json")