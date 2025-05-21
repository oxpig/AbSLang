from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
import faiss
import numpy as np
from Model_Embed import embed_with_fallback


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
            lmembeddings=lm_embeddings,
            model=model,
            fallback_igt5=fallback_model,
            device=device,
            mode=mode,
        )
        out_vecs.extend(vecs)

    # Stack into a strict 2‑D (N, d) tensor and cast → bfloat16
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
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, nbits)
    index.train(xb)
    # index = faiss.IndexFlatL2(d) #try with flat l2
    index.add(xb)
    index.nprobe = 10
    return index


#distance matrix
def pairwise_l2(embeddings: torch.Tensor) -> np.ndarray:
    return torch.cdist(embeddings.float(), embeddings.float(), p=2
                       ).cpu().numpy().astype(np.float32) #requires float32