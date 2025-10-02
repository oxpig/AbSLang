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

    # Accumulate embeddings on CPU to avoid GPU OOM for large datasets.
    out_vecs_cpu: list[torch.Tensor] = []
    for chunk in chunks:
        vecs_gpu = embed_with_fallback(
            sequences=chunk,
            model=model,
            fallback_lm=fallback_model,
            device=device,
            mode=mode,
        )
        # Move each item to CPU immediately to free GPU memory early
        out_vecs_cpu.extend([v.detach().to("cpu") for v in vecs_gpu])
        # Best-effort GPU memory trim
        if device.type == "cuda":
            del vecs_gpu
            torch.cuda.empty_cache()

    if not out_vecs_cpu:
        raise ValueError("No sequences to embed - input sequence list is empty")
    x = torch.stack(out_vecs_cpu)  # CPU tensor (N, d)
    return x


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
        import warnings
        divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
        old_m = m
        m = min(divisors, key=lambda x: abs(x - old_m))
        warnings.warn(f"Adjusted m from {old_m} to {m} to divide d={d}")
    
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, nbits)
    index.train(xb)
    index.add(xb)
    index.nprobe = 10
    return index


# Removed: build_ivfpq4_index (unused wrapper; use build_ivfpq_index with nbits=4)


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
        import warnings
        divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
        old_m = m
        m = min(divisors, key=lambda x: abs(x - old_m))
        warnings.warn(f"Adjusted m from {old_m} to {m} to divide d={d}")
    
    index = faiss.IndexPQ(d, m, nbits)
    index.train(xb)
    index.add(xb)
    return index


#distance matrix
def pairwise_l2(embeddings: torch.Tensor) -> np.ndarray:
    return torch.cdist(embeddings.float(), embeddings.float(), p=2
                       ).cpu().numpy().astype(np.float32) #requires float32
