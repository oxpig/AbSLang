import faiss
import numpy as np
import json
from pathlib import Path
import torch
import pandas as pd

from .main_module import embed_sequences
from .config import load_mode
from .index_csv_with_ids import decode_id_to_path_row


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
        idmap_meta_file: str | None = None,
        tm_checkpoint_path: str,
        tm_config_path: str,
    ):
        """FAISS-backed antibody similarity search.

        Supports two metadata modes:
        - Plain (csv_file): results index directly into the CSV rows
        - ID-mapped (idmap_meta_file): decodes FAISS labels to (csv_path, row_idx)
        """
        # Validate inputs
        if csv_file is None and idmap_meta_file is None:
            raise ValueError("Either csv_file or idmap_meta_file is required")
        if mode not in ('paired', 'hc', 'nb'):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'paired', 'hc', or 'nb'")
        
        self.device = torch.device(device)
        self.mode = mode
        self._use_idmap = idmap_meta_file is not None
        
        # Load FAISS index
        try:
            self.faiss_index = faiss.read_index(faiss_index_file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load FAISS index from {faiss_index_file}: {e}")
        
        if nprobe is not None and hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = nprobe
        
        # Metadata sources
        self.id_column = id_column
        self.seq_column = seq_column
        self._metadata = None
        self.seq_to_row = None
        self._idmap_meta = None
        self._csv_cache: dict[str, pd.DataFrame] = {}
        self._skiprows = 0

        if self._use_idmap:
            # Load meta JSON for ID decoding
            meta_path = Path(idmap_meta_file)
            if not meta_path.exists():
                raise FileNotFoundError(f"idmap_meta_file not found: {meta_path}")
            try:
                self._idmap_meta = json.loads(meta_path.read_text())
            except Exception as e:
                raise RuntimeError(f"Failed to read idmap meta JSON {meta_path}: {e}")
            # Prefer seq/id columns from meta if present
            self.seq_column = self._idmap_meta.get("seq_col", self.seq_column)
            id_col_meta = self._idmap_meta.get("id_col")
            if id_col_meta is not None:
                self.id_column = id_col_meta
            # Pull skiprows filter to align row indices
            self._skiprows = int(self._idmap_meta.get("filters", {}).get("skiprows", 0) or 0)
        else:
            # Load and validate single CSV
            try:
                self.df = pd.read_csv(csv_file)
            except Exception as e:
                raise FileNotFoundError(f"Failed to load CSV from {csv_file}: {e}")
            if self.seq_column not in self.df.columns:
                raise ValueError(
                    f"Sequence column '{self.seq_column}' not found in CSV. "
                    f"Available columns: {list(self.df.columns)}"
                )
            # Validate sequence format matches mode on sample
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
            # Build in-memory metadata list
            if self.id_column in self.df.columns:
                id_series = self.df[self.id_column]
            else:
                id_series = self.df.reset_index().index
            self._metadata = [
                {'id': id_val, 'sequence': seq_val}
                for id_val, seq_val in zip(id_series, self.df[self.seq_column])
            ]
            self.seq_to_row = {d['sequence']: idx for idx, d in enumerate(self._metadata)}
        
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

    def _decode_idmap_label(self, packed_id: int) -> dict:
        """Decode an ID-mapped FAISS label into {'id', 'sequence'} using meta JSON."""
        assert self._idmap_meta is not None, "ID map metadata not loaded"
        csv_path, row_idx = decode_id_to_path_row(self._idmap_meta, packed_id)
        if csv_path not in self._csv_cache:
            # Align with builder's skiprows if present
            df = pd.read_csv(csv_path, skiprows=self._skiprows)
            self._csv_cache[csv_path] = df
        df = self._csv_cache[csv_path]
        # Boundary checks
        if not (0 <= int(row_idx) < len(df)):
            raise IndexError(f"Decoded row_idx {row_idx} out of bounds for {csv_path} (len={len(df)})")
        row = df.iloc[int(row_idx)]
        seq_val = row[self.seq_column] if self.seq_column in df.columns else None
        id_val = None
        if self.id_column and self.id_column in df.columns:
            id_val = row[self.id_column]
        # Fallback ID if no column present: use packed label
        if id_val is None:
            id_val = int(packed_id)
        return {"id": id_val, "sequence": seq_val}

    def search(self, query_sequence: str, k: int = 5) -> list[dict]:
        q_vec = embed_sequences(
            [query_sequence],
            model=self.trans_model,
            device=self.device,
            fallback_model=self.fallback_lm,
            mode=self.mode,
            lm_embeddings=self.lm_emb,
        )
        q_np = q_vec.to(torch.float32).cpu().numpy()

        distance, indices = self.faiss_index.search(q_np, k)
        results = []
        for rank, (dist, idx) in enumerate(zip(distance[0], indices[0]), start=1):
            if idx < 0:
                continue
            if self._use_idmap:
                meta = self._decode_idmap_label(int(idx))
            else:
                meta = self._metadata[idx]
            results.append({
                'rank': rank,
                'distance': np.sqrt(dist),  # Convert L2 squared distance to Euclidean distance
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
                     If False, use FAISS distances (approximate).
        
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
            
            valid_idx = [int(p[0]) for p in valid_pairs]
            if self._use_idmap:
                seqs = [self._decode_idmap_label(idx)["sequence"] for idx in valid_idx]
            else:
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
            
            # Filter and re-rank by exact distances (ascending)
            filtered = [(idx, float(dist)) for idx, dist in zip(valid_idx, exact_distances) if dist <= threshold]
            filtered.sort(key=lambda x: x[1])
            for rank, (idx, exact_dist) in enumerate(filtered, start=1):
                meta = self._decode_idmap_label(idx) if self._use_idmap else self._metadata[idx]
                results.append({
                    "rank": rank,
                    "distance": exact_dist,
                    "id": meta["id"],
                    "sequence": meta["sequence"],
                })
        else:
            # Use FAISS distances directly (faster but approximate)
            rank = 1
            for idx, dist in zip(indices[0], distances[0]):
                if idx < 0:
                    continue
                # FAISS returns L2 squared distances, take sqrt for Euclidean
                euclidean_dist = np.sqrt(dist)
                if euclidean_dist <= threshold:
                    meta = self._decode_idmap_label(int(idx)) if self._use_idmap else self._metadata[idx]
                    results.append({
                        "rank": rank,
                        "distance": float(euclidean_dist),
                        "id": meta["id"],
                        "sequence": meta["sequence"],
                    })
                    rank += 1
        
        return results 







