from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
import faiss
import numpy as np
from .model_embed import embed_with_fallback


#container
@dataclass
class IndexArtifacts:
    faiss_index: faiss.Index
    embeddings: torch.Tensor                  # (N, d) stacked
    distance_matrix: Optional[np.ndarray] = None
    sequence_list: Optional[List[dict]] = None


#embed sequence
def embed_sequences(
    seqs: Sequence[str],
    *,
    model,
    device: torch.device,
    lm_embeddings: dict,
    fallback_model,
    mode: str = "paired",
    batch_size: int = 256,
) -> torch.Tensor:
    seqs = list(seqs)
    chunks = [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]

    out_vecs = []
    for chunk in chunks:
        vecs = embed_with_fallback(
            sequences=chunk,
            model=model,
            fallback_lm=fallback_model,
            device=device,
            mode=mode,
        )
        out_vecs.extend(vecs)

    # Stack into a strict 2‑D (N, d) tensor and cast → bfloat16
    if not out_vecs:
        raise ValueError("No sequences to embed - input sequence list is empty")
    x = torch.stack(out_vecs)
    return x   # ensure 2-D


#building index code
def build_ivfpq_index(
    embeddings: torch.Tensor,
    *,
    nlist: int = 90,
    m: int = 32,
    nbits: int = 8,
) -> faiss.Index:
    xb = embeddings.to(torch.float32).cpu().numpy()
    _, d = xb.shape
    
    # Validate PQ parameters
    if d % m != 0:
        import math
        divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
        old_m = m
        m = min(divisors, key=lambda x: abs(x - old_m))
        print(f"Warning: Adjusted m from {old_m} to {m} to divide d={d}")
    
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, nbits)
    index.train(xb)
    index.add(xb)
    index.nprobe = 10
    return index


def build_ivfpq4_index(
    embeddings: torch.Tensor,
    *,
    nlist: int = 90,
    m: int = 32,
    nbits: int = 4,
) -> faiss.Index:
    """
    Build an IVFPQ index with 4-bit quantization.
    """
    xb = embeddings.to(torch.float32).cpu().numpy()
    _, d = xb.shape
    
    # Validate PQ parameters
    if d % m != 0:
        import math
        divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
        old_m = m
        m = min(divisors, key=lambda x: abs(x - old_m))
        print(f"Warning: Adjusted m from {old_m} to {m} to divide d={d}")
    
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, nbits)
    index.train(xb)
    index.add(xb)
    index.nprobe = 10
    return index


def build_pq_flat_index(
    embeddings: torch.Tensor,
    *,
    m: int = 32,
    nbits: int = 8,
) -> faiss.Index:
    """
    Build a flat Product Quantizer (PQ) index for exhaustive search.
    """
    xb = embeddings.to(torch.float32).cpu().numpy()
    _, d = xb.shape
    
    # Validate PQ parameters
    if d % m != 0:
        import math
        divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
        old_m = m
        m = min(divisors, key=lambda x: abs(x - old_m))
        print(f"Warning: Adjusted m from {old_m} to {m} to divide d={d}")
    
    index = faiss.IndexPQ(d, m, nbits)
    index.train(xb)
    index.add(xb)
    return index


#distance matrix
def pairwise_l2(embeddings: torch.Tensor) -> np.ndarray:
    return torch.cdist(embeddings.float(), embeddings.float(), p=2
                       ).cpu().numpy().astype(np.float32) #requires float32