import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # One gpu or else igt5 breaks?
import torch
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
    nlist=4,
    m=1,
    nbits=6,
    save_distance=False,
    write_index=True,
    write_embeddings=True,
    write_sequences=True,
    tm_checkpoint_path=None,
    tm_config_path=None,
):
    """
    Build index from CSV file.

    Parameters:
    - save_distance: whether to compute and save pairwise distances
    - write_index: whether to write the index file
    - write_embeddings: whether to write the embeddings file
    - write_sequences: whether to write the sequences file
    - tm_checkpoint_path: Path to transformer model checkpoint
    - tm_config_path: Path to transformer model config JSON
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_emb, fallback_lm, trans_model, _ = load_mode(
        mode, device, tm_checkpoint_path=tm_checkpoint_path, tm_config_path=tm_config_path
    )

    seq_items = adapter.read_sequences_csv(csv_path, id_column, seq_column)
    seqs = [s["sequence"] for s in seq_items]

    embs = embed_sequences(
        seqs,
        model=trans_model,
        device=device,
        lm_embeddings=lm_emb,
        fallback_model=fallback_lm,
        mode=mode,
        batch_size=batch_size,
    )
    
    faiss_idx = build_pq_flat_index(embs, m=m, nbits=nbits)
    dist = pairwise_l2(embs) if save_distance else None

    art = IndexArtifacts(faiss_idx, embs, dist, seq_items)
    adapter.dump_artifacts(
        art,
        out_dir,
        write_index=write_index,
        write_embeddings=write_embeddings,
        write_distance=save_distance,
        write_sequences=write_sequences,
    )
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