import argparse
from pathlib import Path
import torch
import pandas as pd
from torch import nn
from .model_embed import embed_with_fallback
from .config import load_mode


#data = pd.read_csv('/vols/opig/users/ewang/Training/ABB3_pdb_split/paired_test.csv')
def infer_rmsd(
    csv_path: str,
    *,
    mode: str = "paired",          
    seq1_col: str = "Seq1",
    seq2_col: str = "Seq2",
    batch_size: int = 128,
    out_path: str | None = None,
    tm_checkpoint_path: str,
    tm_config_path: str,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_embeddings, fallback_model, model, _ = load_mode(
        mode, device, tm_checkpoint_path=tm_checkpoint_path, tm_config_path=tm_config_path
    )

    df = pd.read_csv(csv_path)
    if seq1_col not in df.columns or seq2_col not in df.columns:
        raise ValueError(f"Need '{seq1_col}' and '{seq2_col}' columns in {csv_path}")

    unique = list(set(df[seq1_col]).union(df[seq2_col]))
    cache: dict[str, torch.Tensor] = {}

    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        vecs = embed_with_fallback(
            sequences=batch,
            model=model,
            fallback_lm=fallback_model,
            device=device,
            mode=mode,
        )
        for s, v in zip(batch, vecs):
            cache[s] = v.cpu()

    pdist = nn.PairwiseDistance(p=2)
    df["Predicted_RMSD"] = [
        pdist(cache[s1].to(device).squeeze(), cache[s2].to(device).squeeze()).item()
        for s1, s2 in zip(df[seq1_col], df[seq2_col])
    ]

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Results written to {out_path}")

    return df


def predict_cdr_rmsd(
    seq1: str,
    seq2: str,
    *,
    mode: str = "paired",
    device: str | None = None,
    tm_checkpoint_path: str,
    tm_config_path: str,
) -> float:

    # Input validation
    if not seq1 or not seq2:
        raise ValueError("Both sequences must be non-empty")
        
    if mode == "paired":
        if "|" not in seq1 or "|" not in seq2:
            raise ValueError("Paired mode requires sequences with '|' separator (heavy|light)")
    elif mode in ("hc", "nb"):
        if "|" in seq1 or "|" in seq2:
            raise ValueError(f"Mode '{mode}' expects single chain sequences without '|' separator")
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'paired', 'hc', or 'nb'")
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    # Load models
    try:
        lm_embeddings, fallback_model, model, _ = load_mode(
            mode, device_obj, tm_checkpoint_path=tm_checkpoint_path, tm_config_path=tm_config_path
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")
    
    # Embed both sequences
    try:
        vecs = embed_with_fallback(
            sequences=[seq1, seq2],
            model=model,
            fallback_lm=fallback_model,
            device=device_obj,
            mode=mode,
        )
        
        if len(vecs) != 2:
            raise RuntimeError(f"Expected 2 embeddings, got {len(vecs)}")
            
        emb1, emb2 = vecs[0], vecs[1]
        
    except Exception as e:
        raise RuntimeError(f"Failed to embed sequences: {e}")
    
    # Calculate pairwise distance on 1D transformer embeddings
    emb1_dev = emb1.to(device_obj)
    emb2_dev = emb2.to(device_obj)
    # Allow a harmless leading batch dim of 1
    if emb1_dev.dim() == 2 and emb1_dev.size(0) == 1:
        emb1_dev = emb1_dev.squeeze(0)
    if emb2_dev.dim() == 2 and emb2_dev.size(0) == 1:
        emb2_dev = emb2_dev.squeeze(0)
    if emb1_dev.dim() != 1 or emb2_dev.dim() != 1:
        raise RuntimeError(
            f"Unexpected embedding shapes: emb1={tuple(emb1_dev.shape)}, emb2={tuple(emb2_dev.shape)}; "
            "transformer should output 1D vectors per sequence"
        )
    
    # Calculate L2 distance
    rmsd = torch.norm(emb1_dev - emb2_dev, p=2).item()
    
    return rmsd


def predict_cdr_rmsd_batch(
    sequence_pairs: list[tuple[str, str]],
    *,
    mode: str = "paired",
    device: str | None = None,
    tm_checkpoint_path: str,
    tm_config_path: str,
    batch_size: int = 128,
) -> list[float]:
    """
    Predict CDR RMSD for multiple sequence pairs efficiently.
    
    Args:
        sequence_pairs: List of (seq1, seq2) tuples
        mode: paired, hc, nb
        batch_size: Batch size for processing
    """
    if not sequence_pairs:
        return []
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    

    lm_embeddings, fallback_model, model, _ = load_mode(
        mode, device_obj, tm_checkpoint_path=tm_checkpoint_path, tm_config_path=tm_config_path
    )
    
    # Get unique sequences and build cache
    unique_seqs = list(set(seq for pair in sequence_pairs for seq in pair))
    cache: dict[str, torch.Tensor] = {}
    
    for i in range(0, len(unique_seqs), batch_size):
        batch = unique_seqs[i : i + batch_size]
        vecs = embed_with_fallback(
            sequences=batch,
            model=model,
            fallback_lm=fallback_model,
            device=device_obj,
            mode=mode,
        )
        for s, v in zip(batch, vecs):
            cache[s] = v.cpu()
    
    results = []
    for s1, s2 in sequence_pairs:
        emb1 = cache[s1].to(device_obj)
        emb2 = cache[s2].to(device_obj)
        
        # Ensure 1D transformer embeddings (tolerate leading batch dim of 1)
        if emb1.dim() == 2 and emb1.size(0) == 1:
            emb1 = emb1.squeeze(0)
        if emb2.dim() == 2 and emb2.size(0) == 1:
            emb2 = emb2.squeeze(0)
        if emb1.dim() != 1 or emb2.dim() != 1:
            raise RuntimeError(
                f"Unexpected embedding shapes: emb1={tuple(emb1.shape)}, emb2={tuple(emb2.shape)}; "
                "transformer should output 1D vectors per sequence"
            )
        # Calculate L2 distance
        rmsd = torch.norm(emb1 - emb2, p=2).item()
        results.append(rmsd)
    
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict CDR RMSD for sequence pairs in a CSV.")
    ap.add_argument("csv", help="Input CSV")
    ap.add_argument("--mode", choices=["paired", "hc", "nb"], default="paired")
    ap.add_argument("--seq1-col", default="Seq1")
    ap.add_argument("--seq2-col", default="Seq2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", help="Destination CSV (default: <csv>_<mode>_pred.csv)")
    ap.add_argument("--tm-checkpoint", required=True, dest="tm_checkpoint_path", help="Path to transformer checkpoint (.ckpt)")
    ap.add_argument("--tm-config", required=True, dest="tm_config_path", help="Path to transformer config JSON")
    args = ap.parse_args()

    outfile = args.out
    if outfile is None:
        stem = Path(args.csv).with_suffix("")
        outfile = f"{stem}_{args.mode}_prediction.csv"

    infer_rmsd(
        csv_path=args.csv,
        mode=args.mode,
        seq1_col=args.seq1_col,
        seq2_col=args.seq2_col,
        batch_size=args.batch_size,
        out_path=outfile,
        tm_checkpoint_path=args.tm_checkpoint_path,
        tm_config_path=args.tm_config_path,
    )

    """
    CLI usage example:

    python -m abslang.large_rmsd_inference \
        /vols/opig/users/ewang/Training/ABB3_pdb_split/paired_test.csv \
        --mode paired \
        --tm-checkpoint /path/to/checkpoint.ckpt \
        --tm-config /path/to/params.json \
        --out /vols/opig/users/ewang/Training/Predictions/paired_test_api.csv

    """
