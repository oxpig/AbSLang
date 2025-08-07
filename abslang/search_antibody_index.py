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
        id_column: str = "Species",
        seq_column: str = "sequence_alignment_aa",
        device: str = "cpu",
        mode: str = "paired",
        nprobe:int | None = None,
        tm_checkpoint_path: str | None = None,
        tm_config_path: str | None = None,
    ):
        """
        Initialize FAISS searcher for antibody similarity search.
        
        Args:
            faiss_index_file: Path to FAISS index file
            csv_file: Path to CSV with sequence metadata
            id_column: Column name for sequence IDs
            seq_column: Column name for sequences
            device: Device to run models on ('cpu' or 'cuda')
            mode: Embedding mode ('paired', 'hc', or 'nb')
            nprobe: Number of clusters to search (for IVF indices)
            tm_checkpoint_path: Path to transformer model checkpoint
            tm_config_path: Path to transformer model config JSON
        """
        self.device = torch.device(device)
        self.faiss_index = faiss.read_index(faiss_index_file)
        if nprobe is not None and hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = nprobe
        if csv_file is None:
            raise ValueError("csv_file cannot be None")
        self.df = pd.read_csv(csv_file)
        self.id_column = id_column
        self.seq_column = seq_column
        self.mode = mode
        #self._metadata = self.df[[id_column, seq_column]].to_dict('records')

        # for d in self._metadata:
        #     if 'id' not in d:
        #         d['id'] = None
        # meta_seq_key = self.seq_column ifcsv_file is not None else 'sequence'
        if self.id_column in self.df.columns:
            id_series = self.df[self.id_column]
        else:
            id_series = self.df.reset_index().index
        self._metadata = [{'id': id_val, 'sequence': seq_val}
                            for id_val, seq_val in zip(id_series,
                            self.df[self.seq_column])]

        meta_seq_key = 'sequence'

        self.seq_to_row = {d[meta_seq_key]: idx for idx, d in enumerate(self._metadata)}
        self.lm_emb, self.fallback_lm, self.trans_model, _ = load_mode(
            self.mode, 
            self.device,
            tm_checkpoint_path=tm_checkpoint_path,
            tm_config_path=tm_config_path
        )

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
                'distance': np.sqrt(dist),
                'species': meta['id'],
                'sequence': meta['sequence']

            })
        return results

    def search_by_threshold(
        self,
        query_sequence: str,
        *,
        threshold: float = 2.0,
        step: int = 10,
        max_k: int = 100,
    ) -> list[dict]:
        q_vec = embed_sequences(
            [query_sequence],
            model=self.trans_model,
            device=self.device,
            fallback_model=self.fallback_lm,
            mode=self.mode,
            lm_embeddings=self.lm_emb,
        )

        k = step
        while True:
            q_np = q_vec.to(torch.float32).cpu().numpy()
            distance, indices = self.faiss_index.search(q_np, k)

            valid_idx = [idx for idx in indices[0] if idx >= 0]
            seqs = [self._metadata[idx]["sequence"] for idx in valid_idx]

            if not seqs:
                return []

            cand_emb = embed_sequences(
                seqs,
                model=self.trans_model,
                device=self.device,
                fallback_model=self.fallback_lm,
                mode=self.mode,
                lm_embeddings=self.lm_emb,
            )

            # Ensure both tensors are float32 for torch.cdist compatibility
            q_vec_32 = q_vec.to(torch.float32)
            cand_emb_32 = cand_emb.to(torch.float32)

            dist_vals = (
                torch.cdist(
                    q_vec_32, cand_emb_32, p=2
                )
                .squeeze(0)
                .cpu()
                .tolist()
            )

            if all(d <= threshold for d in dist_vals) and k < max_k:
                k += step
                continue
            break

        results = []
        rank = 1
        for idx, dist in zip(valid_idx, dist_vals):
            if dist <= threshold:
                meta = self._metadata[idx]
                results.append(
                    {
                        "rank": rank,
                        "distance": float(dist),
                        "id": meta["id"],
                        "sequence": meta["sequence"],
                    }
                )
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