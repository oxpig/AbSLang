import argparse
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from .Model_Embed import embed_with_fallback
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
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_embeddings, fallback_model, model, _ = load_mode(mode, device)

    df = pd.read_csv(csv_path)
    if seq1_col not in df.columns or seq2_col not in df.columns:
        raise ValueError(f"Need '{seq1_col}' and '{seq2_col}' columns in {csv_path}")

    unique = list(set(df[seq1_col]).union(df[seq2_col]))
    cache: dict[str, torch.Tensor] = {}

    for i in tqdm(range(0, len(unique), batch_size), desc="Embedding unique sequences"):
        batch = unique[i : i + batch_size]
        vecs = embed_with_fallback(
            sequences=batch,
            lmembeddings=lm_embeddings,
            model=model,
            fallback_igt5=fallback_model,
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict CDR RMSD for sequence pairs in a CSV.")
    ap.add_argument("csv", help="Input CSV")
    ap.add_argument("--mode", choices=["paired", "hc", "nb"], default="paired")
    ap.add_argument("--seq1-col", default="Seq1")
    ap.add_argument("--seq2-col", default="Seq2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", help="Destination CSV (default: <csv>_<mode>_pred.csv)")
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
    )

    """
    
    python Large_RMSD_Inference.py /vols/opig/users/ewang/Training/ABB3_pdb_split/paired_test.csv --mode paired --out /vols/opig/users/ewang/Training/Predictions/paired_test_api.csv
    
    """