"""
Each vector is added with an explicit int64 label:
    id64 = (file_id << 32) | row_idx

Where:
  - file_id is a stable 32-bit ID derived from the CSV absolute path
  - row_idx is the original 0-based row index in the pandas DataFrame
    (i.e., excluding header). It is taken BEFORE filtering and passed through
    for kept rows only.

Outputs:
  - FAISS index file with ID-mapped labels (out_dir/index.<type>)
  - Metadata JSON (out_dir/meta.json) describing file_id mapping and params

Usage example:
  python -m abslang.index_csv_with_ids \
    --csv /Users/ericwang/Downloads/ERR220465_Heavy_Bulk.csv \
    --seq-col sequence_alignment \
    --mode hc \
    --tm-checkpoint /Users/ericwang/Documents/Part_II/Webapp/cpu_compatible_August9_paired/checkpoints/epoch=0-step=109581-val_pearson_correlation=0.7884.ckpt \
    --tm-config /Users/ericwang/Documents/Part_II/Webapp/cpu_compatible_August9_paired/params.json \
    --index-type flat \
    --out-dir /Users/ericwang/Downloads

Notes:
  - Filtering uses pandas.DataFrame.query syntax via --query
  - Mode 'paired' requires sequences with '|' delimiter; 'hc'/'nb' require none
  - pandas will auto-handle .gz by extension
"""
#--index-type ivfpq --nlist 2048 --m 32 --nbits 8 \
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path
from typing import Iterable, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
import gzip
import shutil

from .config import load_mode
from .main_module import embed_sequences


def stable_file_id(path: str | Path) -> int:
    """Compute a stable 32-bit file ID from absolute path using CRC32.

    CRC32 yields unsigned 32-bit in Python; cast to int for packing.
    """
    abspath = os.path.abspath(str(path))
    return zlib.crc32(abspath.encode("utf-8")) & 0xFFFFFFFF


def pack_id(file_id: int, row_idx: int) -> np.int64:
    """Pack (file_id, row_idx) into a signed int64 label.

    Uses two's-complement mapping to avoid OverflowError when the
    composed 64-bit value exceeds the signed range.
    """
    val = ((file_id & 0xFFFFFFFF) << 32) | (row_idx & 0xFFFFFFFF)
    if val >= (1 << 63):
        val -= (1 << 64)
    return np.int64(val)


def validate_sequences_mode(seqs: Iterable[str], mode: str) -> Tuple[list[str], list[int]]:
    """Filter sequences to match AbSLang mode.
    returns kept sequences and masks relate to input order

    placeholder, filtered by completed sequence
    """
    kept_idx = []
    kept_seqs = []
    if mode == "paired":
        for i, s in enumerate(seqs):
            if isinstance(s, str) and "|" in s:
                kept_idx.append(i)
                kept_seqs.append(s)
    elif mode in ("hc", "nb"):
        for i, s in enumerate(seqs):
            if isinstance(s, str) and "|" not in s:
                kept_idx.append(i)
                kept_seqs.append(s)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return kept_seqs, kept_idx


def build_index_with_ids(
    xb: np.ndarray,
    ids: np.ndarray,
    *,
    index_type: str = "ivfpq",
    m: int = 32,
    nbits: int = 8,
    nlist: int | None = None,
    nprobe: int | None = None,
) -> faiss.Index:
    """Build a FAISS index and add with explicit IDs.

    xb: (N, d) float32 array
    ids: (N,) int64 array
    """
    assert xb.dtype == np.float32 and xb.ndim == 2
    assert ids.dtype == np.int64 and ids.ndim == 1 and len(ids) == len(xb)
    N, d = xb.shape

    def adjust_m(m: int) -> int:
        if d % m != 0:
            divisors = [i for i in range(1, min(d, 64) + 1) if d % i == 0]
            if divisors:
                # pick divisor closest to requested m
                return min(divisors, key=lambda x: abs(x - m))
        return m

    index_type = index_type.lower()
    if index_type not in ("flat", "pq", "ivfpq"):
        raise ValueError("index_type must be one of: flat, pq, ivfpq")

    if index_type == "flat":
        base = faiss.IndexFlatL2(d)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(xb, ids)
        return index

    if index_type == "pq":
        m = adjust_m(m)
        base = faiss.IndexPQ(d, m, nbits)
        base.train(xb)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(xb, ids)
        return index

    # ivfpq
    m = adjust_m(m)
    if nlist is None:
        import math
        nlist = min(4096, max(32, int(10 * math.sqrt(max(N, 1)))))
    quant = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFPQ(quant, d, int(nlist), int(m), int(nbits))
    ivf.train(xb)
    if nprobe is not None:
        ivf.nprobe = int(nprobe)
    index = faiss.IndexIDMap2(ivf)
    index.add_with_ids(xb, ids)
    return index

def unpack_id(packed_id: np.int64) -> Tuple[int, int]:
    """Decode composite ID back to (file_id, row_idx).

    Works for any signed/unsigned representation by reinterpreting as uint64.
    """
    u = np.uint64(packed_id)
    file_id = int(u >> np.uint64(32))
    row_idx = int(u & np.uint64(0xFFFFFFFF))
    return file_id, row_idx


def decode_id_to_path_row(meta: dict | str | Path, packed_id: int | np.int64) -> Tuple[str, int]:
    """Decode a FAISS ID to (csv_path, row_idx) using meta.json inputs.

    Args:
        meta: The loaded meta dict or a path to meta.json
        packed_id: The FAISS label (int64) returned from search

    Returns:
        (csv_path, row_idx) where row_idx is 0-based index after the CSV's
        first metadata row is skipped (skiprows=1 at ingestion).
    """
    # Load meta if a path is provided
    if isinstance(meta, (str, Path)):
        meta = json.loads(Path(meta).read_text())
    inputs = meta.get("inputs")
    if not inputs:
        # Single-file legacy format
        csv_path = meta.get("csv_path")
        file_id_meta = int(meta.get("file_id")) if meta.get("file_id") is not None else None
        fid, ridx = unpack_id(np.int64(packed_id))
        if file_id_meta is not None and fid != file_id_meta:
            raise ValueError(f"FAISS ID file_id {fid} does not match meta file_id {file_id_meta}")
        return csv_path, ridx

    # Multi-file: build lookup map once per call
    # inputs: [{"path": str, "file_id": int, ...}, ...]
    fid, ridx = unpack_id(np.int64(packed_id))
    for rec in inputs:
        if int(rec.get("file_id")) == fid:
            return rec.get("path"), ridx
    raise KeyError(f"file_id {fid} not found in meta inputs")

def main():
    p = argparse.ArgumentParser(description="Index CSV(s) with AbSLang embeddings and composite FAISS IDs")
    p.add_argument("--csv", required=True, help="Path to CSV file or a directory of CSVs (.csv or .csv.gz)")
    p.add_argument("--seq-col", default="sequence_alignment_aa", help="Sequence column name")
    p.add_argument("--mode", choices=["paired", "hc", "nb"], default="paired", help="Embedding mode")
    p.add_argument("--id-col", default=None, help="Optional ID column (metadata only)")
    p.add_argument("--out-dir", default="index_out", help="Output directory for index and meta")
    p.add_argument("--work-dir", default=None, help="Optional working directory to place uncompressed CSVs (for .csv.gz)")
    p.add_argument("--index-type", choices=["flat", "pq", "ivfpq"], default="flat")
    p.add_argument("--nlist", type=int, default=None, help="IVFPQ: number of clusters (auto if None)")
    p.add_argument("--nprobe", type=int, default=None, help="IVFPQ: search probes to set on index")
    p.add_argument("--m", type=int, default=32, help="PQ/IVFPQ: subquantizers")
    p.add_argument("--nbits", type=int, default=8, help="PQ/IVFPQ: bits per subquantizer")
    p.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--tm-checkpoint", required=True, help="Path to transformer checkpoint")
    p.add_argument("--tm-config", required=True, help="Path to transformer config JSON")
    args = p.parse_args()

    root = Path(args.csv)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    # Collect input CSVs
    if root.is_dir():
        csv_paths = [
            p
            for p in sorted(root.iterdir())
            if p.is_file()
            and (
                p.suffix.lower() == ".csv"
                or p.suffix.lower() == ".gz"
                or p.name.lower().endswith(".csv.gz")
            )
        ]
        # For heavy-chain mode, ignore files whose names suggest light chains
        if args.mode == "hc":
            before = len(csv_paths)
            csv_paths = [p for p in csv_paths if "light" not in p.name.lower()]
            if not csv_paths:
                raise FileNotFoundError(f"No CSV files found in directory after excluding Light files: {root}")
            if before != len(csv_paths):
                print(f"Excluded {before - len(csv_paths)} Light file(s); indexing {len(csv_paths)} heavy file(s).")
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in directory: {root}")
    else:
        csv_paths = [root]

    # Load models
    device = torch.device(args.device)
    # Informative device print removed to reduce noise
    lm_emb, fallback_lm, trans_model, _ = load_mode(
        args.mode, device, tm_checkpoint_path=args.tm_checkpoint, tm_config_path=args.tm_config
    )

    # Prepare working directory for optional decompression
    work_dir: Path | None = None
    if args.work_dir is not None:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    def _materialize_csv_if_needed(src: Path) -> Path:
        """If src is .gz and work_dir is provided, decompress into work_dir and return the new path.
        Otherwise, return src.
        """
        name = src.name.lower()
        is_gz = name.endswith(".gz")
        if work_dir is None or not is_gz:
            return src
        # derive output filename (strip only the trailing .gz)
        out_name = src.name[:-3]  # remove '.gz'
        dst = work_dir / out_name
        if not dst.exists() or dst.stat().st_size == 0:
            # stream copy
            with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        return dst

    # Accumulate sequences and IDs across files
    all_sequences: list[str] = []
    all_ids_u: list[np.ndarray] = []
    files_meta = []

    for csv_path in csv_paths:
        # If requested, materialize gz into work_dir; otherwise pandas can read compressed directly
        read_path = _materialize_csv_if_needed(csv_path)
        # Load CSV; skip first metadata row
        df = pd.read_csv(read_path, skiprows=1)
        if args.seq_col not in df.columns:
            raise ValueError(f"Sequence column '{args.seq_col}' not in CSV {read_path}. Available: {list(df.columns)}")

        # Preserve original row indices for ID mapping before filtering
        df = df.reset_index().rename(columns={"index": "__orig_row_idx__"})

        filt = 'ANARCI_status'
        if filt in df.columns:
            has_bad_terms = df[filt].astype(str).str.contains(r"\b(?:cdr|conserved)\b", flags=re.I, na=False)
            print(f"{csv_path.name}: total={len(df)} kept_after_bad_terms={len(df[~has_bad_terms])}")
            df = df.loc[~has_bad_terms].reset_index(drop=True)
        else:
            print(f"{csv_path.name}: total={len(df)} (no {filt} column; skipping bad-terms filter)")

        # Validate sequences with respect to mode and drop invalids
        seqs_raw = df[args.seq_col].astype(str).tolist()
        kept_seqs, kept_positions = validate_sequences_mode(seqs_raw, args.mode)
        if not kept_seqs:
            print(f"{csv_path.name}: no sequences left after filtering + mode validation; skipping file")
            continue
        kept_df = df.iloc[kept_positions].copy()

        # Original row indices to encode into IDs
        orig_row_idx = kept_df["__orig_row_idx__"].astype(int).to_numpy()

        # Build IDs for this file
        file_id = stable_file_id(csv_path)
        row_idx_u = orig_row_idx.astype(np.uint64) & np.uint64(0xFFFFFFFF)
        ids_u = (np.uint64(file_id) << np.uint64(32)) | row_idx_u

        # Accumulate
        all_sequences.extend(kept_seqs)
        all_ids_u.append(ids_u)
        files_meta.append({
            "path": str(csv_path),  # original (possibly compressed) path for lookup
            "file_id": int(file_id),
            "kept": int(len(kept_seqs)),
            "read_from": str(read_path) if read_path != csv_path else None,
        })

    if not all_sequences:
        raise ValueError("No sequences collected from provided files after filtering.")

    # Embed all sequences
    embs = embed_sequences(
        all_sequences,
        model=trans_model,
        device=device,
        lm_embeddings=lm_emb,
        fallback_model=fallback_lm,
        mode=args.mode,
        batch_size=args.batch_size,
    )
    xb = embs.to(torch.float32).cpu().numpy()

    # Concatenate IDs from all files and view as signed int64
    ids_u_all = np.concatenate(all_ids_u, axis=0).astype(np.uint64, copy=False)
    ids = ids_u_all.view(np.int64)

    # Build index and add with IDs
    index = build_index_with_ids(
        xb,
        ids,
        index_type=args.index_type,
        m=args.m,
        nbits=args.nbits,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )

    # Persist outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_ext = {
        "flat": "flat",
        "pq": "pq",
        "ivfpq": "ivfpq",
    }[args.index_type]
    # Use the name of the directory being indexed (or the parent dir of a single CSV)
    prefix_name = root.name if root.is_dir() else root.parent.name
    index_path = out_dir / f"{prefix_name}_index.{index_ext}"
    faiss.write_index(index, str(index_path))

    meta = {
        "inputs": files_meta,
        "id_scheme": "(file_id<<32)|row_idx",
        "row_index_base": 0,
        "mode": args.mode,
        "seq_col": args.seq_col,
        "id_col": args.id_col,
        "index": {
            "type": args.index_type,
            "m": int(args.m) if args.index_type in ("pq", "ivfpq") else None,
            "nbits": int(args.nbits) if args.index_type in ("pq", "ivfpq") else None,
            "nlist": int(args.nlist) if args.index_type == "ivfpq" and args.nlist is not None else None,
            "nprobe": int(args.nprobe) if args.index_type == "ivfpq" and args.nprobe is not None else None,
            "dimension": int(xb.shape[1]),
            "count": int(xb.shape[0]),
        },
        "filters": {
            "skiprows": 1,
            "bad_terms": {"column": "ANARCI_status", "regex": r"\\b(?:cdrs|conserved)\\b", "case_insensitive": True},
        },
        "faiss_version": faiss.__version__ if hasattr(faiss, "__version__") else None,
    }
    meta_path = out_dir / f"{prefix_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Indexed {len(files_meta)} file(s); total vectors: {len(ids)}")
    print(f"Wrote metadata to: {meta_path}")


if __name__ == "__main__":
    main()
