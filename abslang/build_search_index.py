import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # One gpu or else igt5 breaks?
import torch
import faiss
from pathlib import Path
import json
from .main_module import embed_sequences, build_ivfpq_index, pairwise_l2, IndexArtifacts, build_pq_flat_index
from . import adapter
from .config import load_mode


def build_index_from_csv(
    csv_path,
    mode="paired",
    *,
    id_column="id",
    seq_column="sequence_alignment_aa",
    out_dir="index_out",
    batch_size=256,
    index_type="pq",
    nlist=None,
    m=32,
    nbits=8,
    save_distance=False,
    write_index=True,
    write_embeddings=False,
    write_sequences=True,
    tm_checkpoint_path,
    tm_config_path,
):
    """
    Build index from CSV file.

    Parameters:
    - csv_path: Path to CSV with sequences
    - mode: 'paired', 'hc', or 'nb'
    - id_column: Column name for IDs
    - seq_column: Column name for sequences
    - out_dir: Output directory
    - batch_size: Batch size for embedding
    - index_type: Type of index ('ivfpq', 'pq', 'flat')
    - nlist: Number of clusters for IVF indices (auto-calculated if None)
    - m: Number of subquantizers for PQ
    - nbits: Number of bits per subquantizer
    - save_distance: whether to compute and save pairwise distances
    - write_index: whether to write the index file
    - write_embeddings: whether to write the embeddings file
    - write_sequences: whether to write the sequences file
    - tm_checkpoint_path: Path to transformer model checkpoint (required)
    - tm_config_path: Path to transformer model config JSON (required)
    
    Returns:
        IndexArtifacts object
    
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate index type
    if index_type not in ('ivfpq', 'pq', 'flat'):
        raise ValueError(f"Invalid index_type '{index_type}'. Must be 'ivfpq', 'pq', or 'flat'")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models (transformer is required)
    lm_emb, fallback_lm, trans_model, _ = load_mode(
        mode, device, tm_checkpoint_path=tm_checkpoint_path, tm_config_path=tm_config_path
    )

    # Read sequences (validation happens in adapter.read_sequences_csv)
    seq_items = adapter.read_sequences_csv(csv_path, id_column, seq_column)
    seqs = [s["sequence"] for s in seq_items]
    
    if not seqs:
        raise ValueError(f"No sequences found in CSV file: {csv_path}")
    
    # Validate sequence format matches mode for ALL sequences, not just first
    invalid_seqs = []
    if mode == 'paired':
        for i, seq in enumerate(seqs[:10]):  # Check first 10 for performance
            if '|' not in seq:
                invalid_seqs.append((i, seq[:50]))
        if invalid_seqs:
            raise ValueError(
                f"Mode 'paired' requires sequences with '|' separator (heavy|light). "
                f"Found {len(invalid_seqs)} invalid sequences. First invalid: row {invalid_seqs[0][0]}, "
                f"sequence: '{invalid_seqs[0][1]}...'"
            )
    elif mode in ('hc', 'nb'):
        for i, seq in enumerate(seqs[:10]):  # Check first 10 for performance
            if '|' in seq:
                invalid_seqs.append((i, seq[:50]))
        if invalid_seqs:
            raise ValueError(
                f"Mode '{mode}' requires single chain sequences without '|'. "
                f"Found {len(invalid_seqs)} invalid sequences. First invalid: row {invalid_seqs[0][0]}, "
                f"sequence: '{invalid_seqs[0][1]}...'"
            )

    # Embed sequences
    embs = embed_sequences(
        seqs,
        model=trans_model,
        device=device,
        lm_embeddings=lm_emb,
        fallback_model=fallback_lm,
        mode=mode,
        batch_size=batch_size,
    )
    
    # Build appropriate index type
    _, d = embs.shape
    
    if index_type == 'flat':
        faiss_idx = faiss.IndexFlatL2(d)
        faiss_idx.add(embs.to(torch.float32).cpu().numpy())
    elif index_type == 'pq':
        # Validate PQ parameters
        if d % m != 0:
            # Auto-adjust m to be a divisor of d
            import math
            divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
            old_m = m
            m = min(divisors, key=lambda x: abs(x - old_m))
            print(f"Warning: Adjusted m from {old_m} to {m} to divide d={d}")
        faiss_idx = build_pq_flat_index(embs, m=m, nbits=nbits)
    else:  # ivfpq
        # Calculate nlist if not provided
        n_samples = len(seqs)
        if nlist is None:
            import math
            nlist = min(4096, max(32, int(10 * math.sqrt(n_samples))))
            print(f"Auto-calculated nlist={nlist} for {n_samples} samples")
        
        # Validate PQ parameters
        if d % m != 0:
            import math
            divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
            old_m = m
            m = min(divisors, key=lambda x: abs(x - old_m))
            print(f"Warning: Adjusted m from {old_m} to {m} to divide d={d}")
        
        faiss_idx = build_ivfpq_index(embs, nlist=nlist, m=m, nbits=nbits)
    
    # Warn about expensive distance matrix
    if save_distance and len(seqs) > 10000:
        print(f"Warning: Computing pairwise distances for {len(seqs)} sequences (O(n²) memory)")
    dist = pairwise_l2(embs) if save_distance else None

    art = IndexArtifacts(faiss_idx, embs, dist, seq_items)
    
    # Save artifacts with appropriate naming
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if write_index:
        index_path = out / f"index.{index_type}"
        adapter.save_faiss_index(art.faiss_index, index_path)
    
    if write_embeddings:
        adapter.save_embeddings(art.embeddings, out / "embeddings.pt")
    
    if save_distance and art.distance_matrix is not None:
        adapter.save_distance_matrix(art.distance_matrix, out / "dist_mat.csv")
    
    if write_sequences and art.sequence_list is not None:
        # Save as pretty-printed JSON
        seq_path = out / "seq_list.json"
        import json
        with open(seq_path, 'w') as f:
            json.dump(art.sequence_list, f, indent=2)
    
    # Save metadata about the index
    metadata = {
        "index_type": index_type,
        "mode": mode,
        "dimension": d,
        "num_sequences": len(seqs),
        "metric": "L2",
        "parameters": {
            "m": m if index_type in ('pq', 'ivfpq') else None,
            "nbits": nbits if index_type in ('pq', 'ivfpq') else None,
            "nlist": nlist if index_type == 'ivfpq' else None
        }
    }
    
    with open(out / "artifacts.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return art

'''
artifacts = build_index_from_csv("/vols/opig/projects/ewang-HC_Search_Index/HC_Sequences.csv", id_column='PDB_file', seq_column='Sequence', save_distance=False)

from build_search_index import build_index_from_csv

build_index_from_csv(
    csv_path="/vols/opig/projects/ewang-HC_Search_Index/HC_Sequences.csv",
    mode="hc",
    id_column="PDB_file",
    seq_column="Sequence",
    save_distance=False,
    write_index=True,
    write_sequences=False
)
'''