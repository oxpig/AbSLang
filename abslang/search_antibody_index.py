from __future__ import annotations

import heapq
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch

from .main_module import embed_sequences
from .config import load_mode
from .index_csv_with_ids import decode_id_to_path_row
from .sequence_validation import validate_and_trim


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
        path_prefix_map: Optional[Dict[str, str] | str | Path] = None,
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
        self._path_prefix_map = _normalize_prefix_map(path_prefix_map)
        
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
        csv_path = _remap_path(csv_path, self._path_prefix_map)
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
        # Validate and trim to Fv prior to embedding
        query_sequence = validate_and_trim(query_sequence, self.mode)

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
        # Validate and trim query to Fv before embedding
        query_sequence = validate_and_trim(query_sequence, self.mode)

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


def pair_indices_and_meta(root: str | Path) -> List[Tuple[Path, Path, str]]:
    """
    Discover FAISS index files and their matching metadata JSON files.

    Supports both "<name>_index.<ext>" ↔ "<name>_meta.json" and
    "<name>.<ext>" ↔ "<name>.json" conventions. Returns tuples of
    (index_path, meta_path, study_name).
    """
    root = Path(root)
    index_exts = {".flat", ".pq", ".ivfpq"}
    index_files = [p for p in root.iterdir() if p.is_file() and p.suffix in index_exts]
    meta_lookup = {p.stem: p for p in root.glob("*.json")}

    pairs: List[Tuple[Path, Path, str]] = []
    for idx_path in index_files:
        stem = idx_path.stem
        base = stem[:-6] if stem.endswith("_index") else stem
        candidate_keys = (f"{base}_meta", base)

        meta_path: Optional[Path] = None
        for key in candidate_keys:
            meta_path = meta_lookup.get(key)
            if meta_path is not None:
                break
        if meta_path is None:
            fallback = root / f"{base}_meta.json"
            if fallback.exists():
                meta_path = fallback
        if meta_path is not None:
            pairs.append((idx_path, meta_path, base))
    return pairs


def embed_query_single_chain(
    query: str,
    *,
    mode: str,
    device: str,
    tm_checkpoint_path: str,
    tm_config_path: str,
) -> np.ndarray:
    """Validate, trim, and embed the query sequence once."""
    dev = torch.device(device)
    lm_emb, fallback_lm, trans_model, _ = load_mode(
        mode,
        dev,
        tm_checkpoint_path=tm_checkpoint_path,
        tm_config_path=tm_config_path,
    )
    trimmed = validate_and_trim(query, mode)
    embedding = embed_sequences(
        [trimmed],
        model=trans_model,
        device=dev,
        lm_embeddings=lm_emb,
        fallback_model=fallback_lm,
        mode=mode,
    )
    return embedding.to(torch.float32).cpu().numpy()


def _search_one_index(
    idx_path: str,
    per_index_k: int,
    q_vec: np.ndarray,
    faiss_threads: int,
    nprobe: Optional[int],
) -> Tuple[str, List[float], List[int]]:
    """Search a single FAISS index, returning squared distances and labels."""
    try:
        faiss.omp_set_num_threads(int(faiss_threads))
    except Exception:
        pass

    index = faiss.read_index(idx_path)
    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)
    if index.d != q_vec.shape[1]:
        raise RuntimeError(f"Dimension mismatch for {idx_path}: index.d={index.d}, query_d={q_vec.shape[1]}")

    distances, labels = index.search(q_vec, per_index_k)
    return idx_path, distances[0].tolist(), [int(x) for x in labels[0]]


@dataclass(order=True)
class _HeapItem:
    neg_dist2: float
    label: int
    index_path: str


def search_heavy_oas(
    directory: str | Path,
    query: str,
    *,
    mode: str = "hc",
    top_k: int = 5,
    per_index_k: Optional[int] = None,
    device: str = "cpu",
    tm_checkpoint_path: str,
    tm_config_path: str,
    nprobe: Optional[int] = None,
    workers: int = 1,
    faiss_threads_per_worker: Optional[int] = None,
    path_prefix_map: Optional[Dict[str, str] | str | Path] = None,
) -> List[Dict]:
    """
    Global top-K search across many ID-mapped indices with minimal memory use.

    Returns records sorted by ascending Euclidean distance with keys:
    {'study', 'distance', 'id', 'sequence', 'source_csv', 'row_idx', 'packed_id'}.
    """
    directory = Path(directory)
    pairs = pair_indices_and_meta(directory)
    if not pairs:
        raise RuntimeError(f"No (index, meta) pairs found in {directory}")

    prefix_map = _normalize_prefix_map(path_prefix_map)

    q_vec = embed_query_single_chain(
        query,
        mode=mode,
        device=device,
        tm_checkpoint_path=tm_checkpoint_path,
        tm_config_path=tm_config_path,
    )

    per_index_k = int(per_index_k or max(top_k, 10))
    cpu_cores = os.cpu_count() or 8
    faiss_threads = faiss_threads_per_worker or max(1, cpu_cores // max(1, workers))

    heap: List[_HeapItem] = []
    meta_paths: Dict[str, Path] = {}
    study_names: Dict[str, str] = {}

    if workers <= 1:
        for idx_path, meta_path, study in pairs:
            idx_key = str(idx_path)
            meta_paths[idx_key] = meta_path
            study_names[idx_key] = study
            idx_key, dist2_list, labels = _search_one_index(
                idx_key,
                per_index_k,
                q_vec,
                faiss_threads,
                nprobe,
            )
            for dist2, label in zip(dist2_list, labels):
                item = _HeapItem(neg_dist2=-float(dist2), label=int(label), index_path=idx_key)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item.neg_dist2 > heap[0].neg_dist2:
                    heapq.heapreplace(heap, item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx_path, meta_path, study in pairs:
                idx_key = str(idx_path)
                meta_paths[idx_key] = meta_path
                study_names[idx_key] = study
                futures.append(
                    executor.submit(
                        _search_one_index,
                        idx_key,
                        per_index_k,
                        q_vec,
                        faiss_threads,
                        nprobe,
                    )
                )

            for fut in as_completed(futures):
                idx_path, dist2_list, labels = fut.result()
                for dist2, label in zip(dist2_list, labels):
                    item = _HeapItem(neg_dist2=-float(dist2), label=int(label), index_path=idx_path)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                        continue
                    if item.neg_dist2 > heap[0].neg_dist2:
                        heapq.heapreplace(heap, item)

    best: List[Tuple[float, int, str]] = []
    while heap:
        item = heapq.heappop(heap)
        best.append((-item.neg_dist2, item.label, item.index_path))
    best.sort(key=lambda x: x[0])

    grouped: Dict[Path, List[Tuple[int, int, str, float, Path]]] = {}
    meta_cache: Dict[Path, dict] = {}
    for dist2, label, idx_path in best:
        meta_path = meta_paths[idx_path]
        study = study_names[idx_path]
        meta = meta_cache.get(meta_path)
        if meta is None:
            meta = json.loads(meta_path.read_text())
            meta_cache[meta_path] = meta
        csv_path_str, row_idx = decode_id_to_path_row(meta, label)
        csv_path = Path(_remap_path(csv_path_str, prefix_map))
        grouped.setdefault(csv_path, []).append(
            (int(row_idx), int(label), study, float(dist2), meta_path)
        )

    results: List[Dict] = []
    for csv_path, rows in grouped.items():
        meta_path = rows[0][4]
        meta = meta_cache.get(meta_path)
        if meta is None:
            meta = json.loads(meta_path.read_text())
            meta_cache[meta_path] = meta
        skiprows = int(meta.get("filters", {}).get("skiprows", 0) or 0)
        seq_col = meta.get("seq_col", "sequence_alignment_aa")
        id_col = meta.get("id_col")

        usecols = [seq_col]
        if id_col:
            usecols.append(id_col)

        df = pd.read_csv(str(csv_path), skiprows=skiprows, usecols=usecols)

        for row_idx, label, study, dist2, _ in rows:
            record = {
                "study": study,
                "distance": float(np.sqrt(dist2)),
                "source_csv": str(csv_path),
                "row_idx": row_idx,
                "sequence": df.iloc[row_idx][seq_col] if seq_col in df.columns else None,
                "id": df.iloc[row_idx][id_col] if id_col and id_col in df.columns else label,
                "packed_id": label,
            }
            results.append(record)

    results.sort(key=lambda r: r["distance"])
    return results[:top_k]


DEFAULT_SOURCE_PREFIX = Path("/vols/opig/datasets/oas")


def _normalize_prefix_map(prefix_spec: Optional[Dict[str, str] | Dict[Path, Path] | str | Path]) -> Dict[Path, Path]:
    if prefix_spec is None:
        return {}
    if isinstance(prefix_spec, (str, Path)):
        return {DEFAULT_SOURCE_PREFIX: Path(prefix_spec)}
    return {Path(k): Path(v) for k, v in prefix_spec.items()}


def _remap_path(original: str | Path, prefix_map: Dict[Path, Path]) -> str:
    """Rewrite an original source path using the first matching prefix map entry."""
    if not prefix_map:
        return str(original)
    original_path = Path(original)
    for src_prefix, dst_prefix in prefix_map.items():
        try:
            rel = original_path.relative_to(src_prefix)
        except ValueError:
            continue
        return str(dst_prefix / rel)
    return str(original_path)
