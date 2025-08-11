import faiss
import numpy as np
import json
import torch
import pandas as pd

from .main_module import embed_sequences
from .config import load_mode


class FaissSearcher:
    def __init__(
        self,
        faiss_index_file: str,
        *,
        csv_file: str | None = None,
        id_column: str = "id",
        seq_column: str = "sequence_alignment_aa",
        device: str = "cpu",
        mode: str = "paired",
        nprobe:int | None = None,
        tm_checkpoint_path: str,
        tm_config_path: str,
    ):
        """
        Initialize FAISS searcher for antibody similarity search.
        
        Args:
            faiss_index_file: Path to FAISS index file
            csv_file: Path to CSV with sequence metadata (required)
            id_column: Column name for sequence IDs (default: 'id')
            seq_column: Column name for sequences (default: 'sequence_alignment_aa')
            device: Device to run models on ('cpu' or 'cuda')
            mode: Embedding mode ('paired', 'hc', or 'nb')
            nprobe: Number of clusters to search (for IVF indices)
            tm_checkpoint_path: Path to transformer model checkpoint (required)
            tm_config_path: Path to transformer model config JSON (required)
        
        Raises:
            ValueError: If csv_file is None, required columns missing, or invalid mode
            FileNotFoundError: If required files don't exist
        """
        # Validate inputs
        if csv_file is None:
            raise ValueError("csv_file is required")
        if mode not in ('paired', 'hc', 'nb'):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'paired', 'hc', or 'nb'")
        
        self.device = torch.device(device)
        self.mode = mode
        
        # Load FAISS index
        try:
            self.faiss_index = faiss.read_index(faiss_index_file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load FAISS index from {faiss_index_file}: {e}")
        
        if nprobe is not None and hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = nprobe
        
        # Load and validate CSV
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load CSV from {csv_file}: {e}")
        
        # Validate required columns exist
        if seq_column not in self.df.columns:
            raise ValueError(
                f"Sequence column '{seq_column}' not found in CSV. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        self.id_column = id_column
        self.seq_column = seq_column
        # Validate sequence format matches mode
        sample_seq = self.df[self.seq_column].iloc[0] if len(self.df) > 0 else ""
        if self.mode == 'paired':
            if '|' not in sample_seq:
                raise ValueError(
                    f"Mode 'paired' requires sequences with '|' separator (heavy|light), "
                    f"but found sequence without '|': {sample_seq[:50]}..."
                )
        elif self.mode in ('hc', 'nb'):
            if '|' in sample_seq:
                raise ValueError(
                    f"Mode '{self.mode}' expects single chain sequences without '|', "
                    f"but found sequence with '|': {sample_seq[:50]}..."
                )
        
        # Build metadata
        if self.id_column in self.df.columns:
            id_series = self.df[self.id_column]
        else:
            # Use index if id_column doesn't exist
            id_series = self.df.reset_index().index
        
        self._metadata = [{'id': id_val, 'sequence': seq_val}
                         for id_val, seq_val in zip(id_series, self.df[self.seq_column])]
        
        meta_seq_key = 'sequence'
        self.seq_to_row = {d[meta_seq_key]: idx for idx, d in enumerate(self._metadata)}
        
        # Load models (transformer is now required)
        try:
            self.lm_emb, self.fallback_lm, self.trans_model, _ = load_mode(
                self.mode, 
                self.device,
                tm_checkpoint_path=tm_checkpoint_path,
                tm_config_path=tm_config_path
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def search(self, query_sequence: str, k: int = 5) -> list[dict]:

        q_vec = embed_sequences([query_sequence], model=self.trans_model, device=self.device, fallback_model=self.fallback_lm, mode=self.mode, lm_embeddings=self.lm_emb)
        q_np = q_vec.to(torch.float32).cpu().numpy()


        distance, indices = self.faiss_index.search(q_np, k)
        results = []
        for rank, (dist, idx) in enumerate(zip(distance[0], indices[0]), start=1): 
            if idx<0:
                continue
            meta = self._metadata[idx]
            results.append({
                'rank': rank,
                'distance': np.sqrt(dist),  # Convert L2 squared distance to Euclidean
                'id': meta['id'],
                'sequence': meta['sequence']
            })
        return results

    def search_by_threshold(
        self,
        query_sequence: str,
        *,
        threshold: float = 2.0,
        max_k: int = 100,
        reembed: bool = True,
    ) -> list[dict]:
        """
        Search for sequences within a distance threshold.
        
        This method is optimized to perform a single FAISS search with max_k candidates,
        then filters by threshold. This avoids repeated re-embedding of the query.
        
        Args:
            query_sequence: Query sequence
            threshold: Distance threshold (Euclidean distance)
            max_k: Maximum number of candidates to consider
            reembed: If True, re-embed sequences for exact distances.
                     If False, use FAISS distances (faster but approximate).
        
        Returns:
            List of sequences within threshold, sorted by distance
        """
        # Embed query sequence
        q_vec = embed_sequences(
            [query_sequence],
            model=self.trans_model,
            device=self.device,
            fallback_model=self.fallback_lm,
            mode=self.mode,
            lm_embeddings=self.lm_emb,
        )
        q_np = q_vec.to(torch.float32).cpu().numpy()
        
        # Search once with max_k
        distances, indices = self.faiss_index.search(q_np, max_k)
        
        results = []
        
        if reembed:
            # Re-embed for exact distances (slower but accurate)
            valid_pairs = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]
            if not valid_pairs:
                return []
            
            valid_idx = [p[0] for p in valid_pairs]
            seqs = [self._metadata[idx]["sequence"] for idx in valid_idx]
            
            # Re-embed candidates
            cand_emb = embed_sequences(
                seqs,
                model=self.trans_model,
                device=self.device,
                fallback_model=self.fallback_lm,
                mode=self.mode,
                lm_embeddings=self.lm_emb,
            )
            
            # Calculate exact distances
            q_vec_32 = q_vec.to(torch.float32)
            cand_emb_32 = cand_emb.to(torch.float32)
            exact_distances = torch.cdist(q_vec_32, cand_emb_32, p=2).squeeze(0).cpu().tolist()
            
            # Filter by threshold using exact distances
            rank = 1
            for idx, exact_dist in zip(valid_idx, exact_distances):
                if exact_dist <= threshold:
                    meta = self._metadata[idx]
                    results.append({
                        "rank": rank,
                        "distance": float(exact_dist),
                        "id": meta["id"],
                        "sequence": meta["sequence"],
                    })
                    rank += 1
        else:
            # Use FAISS distances directly (faster but approximate)
            rank = 1
            for idx, dist in zip(indices[0], distances[0]):
                if idx < 0:
                    continue
                # FAISS returns L2 squared distances, take sqrt for Euclidean
                euclidean_dist = np.sqrt(dist)
                if euclidean_dist <= threshold:
                    meta = self._metadata[idx]
                    results.append({
                        "rank": rank,
                        "distance": float(euclidean_dist),
                        "id": meta["id"],
                        "sequence": meta["sequence"],
                    })
                    rank += 1
        
        return results 








"""exampel: 
from search_antibody_index import FaissSearcher

searcher = FaissSearcher(
    model_checkpoint = "/vols/opig/users/ewang/Training/Models/Weighted_Euclidean_2U_Hidden_Paired_8Layer_FlashAttn_Feb28_Batch/checkpoints/epoch=2-step=369956-val_pearson_corr=0.7649.ckpt",
    model_config = "/vols/opig/users/ewang/Training/Models/Weighted_Euclidean_2U_Hidden_Paired_8Layer_FlashAttn_Feb28_Batch/params.json",
    embeddings_file = '/vols/opig/users/ewang/Training/testpaired_embeddings.pt',
    faiss_index_file = "/vols/opig/users/ewang/Training/oas_paired_IndexFlatL2_uncompressed.index",
    sequence_list_file = "/vols/opig/users/ewang/Training/oas_paired_combined_seq.json",
    device="cuda",
    mode='paired')

    
neighbors = searcher.search('EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVSS|DIQMTQSPSSLSASVGDRVTITCRASRDIKSYLNWYQQKPGKAPKVLIYYATSLAEGVPSRFSGSGSGTDYTLTISSLQPEDFATYYCLQHGESPWTFGQGTKVEIK', k=5)
   
"""

"""example for calling clustering: 
import faiss
from search_antibody_index import FaissSearcher
searcher = FaissSearcher(
    model_checkpoint = "/vols/opig/users/ewang/Training/Models/Weighted_Euclidean_2U_Hidden_Paired_8Layer_FlashAttn_Feb28_Batch/checkpoints/epoch=2-step=369956-val_pearson_corr=0.7649.ckpt",
    model_config = "/vols/opig/users/ewang/Training/Models/Weighted_Euclidean_2U_Hidden_Paired_8Layer_FlashAttn_Feb28_Batch/params.json",
    embeddings_file = '/vols/opig/users/ewang/Training/testpaired_embeddings.pt',
    faiss_index_file = "/vols/opig/users/ewang/Training/oas_paired_combined.index",
    sequence_list_file = "/vols/opig/users/ewang/Training/oas_paired_combined_seq.json",
    device="cuda",
    mode='paired')

k_candidates = range(2, 50)
threshold = 26.75

best_k, cluster_df = searcher.cluster(
    k_values=k_candidates,
    threshold=threshold,
    n_iter=30,
    metric=faiss.METRIC_INNER_PRODUCT
)

if best_k is None:
    print("No suitable k found.")
else:
    print(f"Chosen k={best_k}")
    print(cluster_df.head(20))
"""

"""
example to search by threshold
from search_antibody_index import FaissSearcher
searcher = FaissSearcher(
    faiss_index_file="/vols/opig/projects/ewang-HC_Search_Index/paired_numbered_nonredundant_embeddings_QT4bit.index",
    csv_file="/vols/opig/projects/ewang-HC_Search_Index/human_oas_paired_non_redundant_by_study.csv",
    device="cpu",
    mode="paired"
)

query_sequence = "QVQLVESGGGVVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDSKLCGGDCYPSGRGYFDYWGQGTLVTVSS|QSALTQPRSVSGSPGQSVTISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVPDRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSYTYVFGTGTKVTVL
"
results=searcher.search(query_sequence, k=10)
results = searcher.search_by_threshold(
    query_sequence,
    threshold=2.0,   # adjust as needed
    step=10,
    max_k=100
)
for hit in results:
    print(f"Rank: {hit['rank']}, Distance: {hit['distance']:.3f}, ID: {hit['id']}, Sequence: {hit['sequence']}")
"""