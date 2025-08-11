import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import itertools
import sys
from .embed_structure_model_cpu import trans_basic_block, trans_basic_block_Config
from .model import PairedIgT5
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.lines as mlines
from torch.amp import autocast
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import os


def model_embed_with_precomputed(sequences, lmembeddings, model, device):
    """
    Embeds a batch of sequences using precomputed embeddings and the provided model.

    Args:
        sequences (list): List of sequences to embed.
        lmembeddings (dict): Dictionary of precomputed embeddings.
        model (torch.nn.Module): Model to use for embedding.
        device (torch.device): Device to run the model on.

    Returns:
        torch.Tensor: Batched embeddings for the input sequences.
    """
    embeddings_list = []
    missing_sequences = []
    for seq in sequences:
        embedding = lmembeddings.get(seq)
        if embedding is not None:
            embeddings_list.append(embedding.to(device))
        else:
            missing_sequences.append(seq)
            print(f"Sequence {seq} not found in precomputed embeddings.")

    if not embeddings_list:
        return torch.empty(0)

    embedding_tensor = torch.stack(embeddings_list)

    if len(embedding_tensor.shape) == 2:
        embedding_tensor = embedding_tensor.unsqueeze(0)

    batch_size, seq_len, _ = embedding_tensor.shape
    padding = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    with torch.no_grad():
        embedded_sequence = model(embedding_tensor, attention_mask=None, padding_mask=padding)

    return embedded_sequence


def process_paired_precomputed(data, sequence_embeddings, model, device, batch_size=32):
    """
    Processes paired sequences in batches using precomputed embeddings.

    Args:
        data (pd.DataFrame): DataFrame containing sequences in 'Sequence' column.
        sequence_embeddings (dict): Dictionary of precomputed embeddings.
        model (torch.nn.Module): Model to use for embedding.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for processing.

    Returns:
        tuple: raw_sequences (list), embeddings (list of torch.Tensor)
    """
    raw_sequences = []
    embeddings = []
    all_sequences = data['Sequence'].tolist()

    for i in tqdm(range(0, len(all_sequences), batch_size), desc="Processing batches"):
        batch_sequences = all_sequences[i:i + batch_size]
        batch_raw_sequences = []
        batch_embeddings = []

        valid_batch_sequences = []
        for seq in batch_sequences:
            if seq in sequence_embeddings:
                valid_batch_sequences.append(seq)
                batch_raw_sequences.append(seq)
            else:
                print(f"Sequence {seq} not found in precomputed embeddings. Skipping from precomputed batch.")

        if valid_batch_sequences:
            try:
                batch_embedding_tensor = model_embed_with_precomputed(valid_batch_sequences, sequence_embeddings, model, device)
                if not torch.is_tensor(batch_embedding_tensor) or batch_embedding_tensor.numel() == 0:
                    print("Warning: model_embed_with_precomputed returned empty embeddings for batch.")
                else:
                    batch_embeddings.extend(list(batch_embedding_tensor))
            except KeyError as e:
                print(f"KeyError in batch processing: {e}")
                print("Skipping batch.")
                continue

        raw_sequences.extend(batch_raw_sequences)
        embeddings.extend(batch_embeddings)

    return raw_sequences, embeddings

def save_embeddings(embeddings, output_file):
    """Saves embeddings to a numpy file."""
    if embeddings is not None and len(embeddings) > 0:
        if isinstance(embeddings[0], torch.Tensor):
            embedding_matrix = torch.stack(embeddings).cpu().numpy().astype(np.float32)
        else:
            embedding_matrix = np.array(embeddings, dtype=np.float32)
    else:
        embedding_matrix = np.array([])

    np.save(output_file, embedding_matrix)
    print(f'saved as {output_file}')
    return embedding_matrix

def parse_paired_sequence(seq):
    """Parse paired sequence separated by '|' into heavy and light chains."""
    parts = seq.split("|", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Sequence '{seq[:50]}...' does not contain '|' separating heavy and light chains. "
            f"Use '|' to separate heavy and light chains for paired mode."
        )
    hc, lc = parts[0].strip(), parts[1].strip()
    return hc, lc

def embed_IgT5(sequences, fallback):
    """Embeds a list of paired sequences using IgT5."""
    heavy_seqs = []
    light_seqs = []
    for sequence in sequences:
        heavy_seq, light_seq = parse_paired_sequence(sequence)
        heavy_seqs.append(heavy_seq)
        light_seqs.append(light_seq)
    embeddings_list = fallback.get_embeddings(heavy_seqs, light_seqs)
    return embeddings_list

def embed_ESM_single(sequences: list, model) -> list:
    """Embeds a list of sequences using ESM-single."""
    embeddings = []
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    for seq in sequences:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        logits_output = model.logits(protein_tensor, logits_config)
        residue_embeddings = logits_output.embeddings.detach()
        embeddings.append(residue_embeddings)
    return embeddings

def embed_ESM_HC(sequences: list, fallback_model) -> list:
    """Embeds a list of sequences using ESM-HC."""
    embeddings = embed_ESM_single(sequences, fallback_model)
    return embeddings

def embed_with_fallback(sequences, model, fallback_lm, device, mode='paired', **kwargs):
    """
    Embed sequences using language model followed by mandatory transformer.
    
    Args:
        sequences: List of sequences to embed
        model: Transformer model for final embedding (required, cannot be None)
        fallback_lm: Primary language model (IgT5 for paired, ESMC for hc/nb)
        device: Device to run models on
        mode: 'paired', 'hc', or 'nb'
        **kwargs: Additional arguments (for backward compatibility)
    
    Returns:
        List of structure-aware embedded sequences
    
    Raises:
        ValueError: If model is None or sequences are invalid for the mode
    """
    if not sequences:
        return []
    
    if model is None:
        raise ValueError(
            "Transformer model is required for structure-aware embeddings. "
            "Please provide tm_checkpoint_path and tm_config_path. "
            "Language model embeddings alone have no structural meaning."
        )
    
    # Validate sequence format matches mode
    for seq in sequences:
        if mode == 'paired':
            if '|' not in seq:
                raise ValueError(
                    f"Mode 'paired' requires sequences with '|' separator (heavy|light). "
                    f"Found sequence without '|': {seq[:50]}..."
                )
        elif mode in ('hc', 'nb'):
            if '|' in seq:
                raise ValueError(
                    f"Mode '{mode}' requires single chain sequences without '|'. "
                    f"Found sequence with '|': {seq[:50]}..."
                )
    
    # Get raw embeddings from language model
    if mode == 'paired':
        raw = embed_IgT5(sequences, fallback_lm)
    elif mode in ('hc', 'nb'):
        raw = embed_ESM_HC(sequences, fallback_lm)
    else:
        raise ValueError(f"Invalid embedding mode: '{mode}'. Must be 'paired', 'hc', or 'nb'")
    
    # Process embeddings through transformer model (mandatory)
    fb_embs = [(x.squeeze(0) if x.dim() == 3 else x).to(device) for x in raw]
    
    def pad_stack(txs):
        m = max(t.shape[0] for t in txs)
        d = txs[0].shape[1]
        b = len(txs)
        out = txs[0].new_zeros((b, m, d))
        mask = torch.zeros((b, m), dtype=torch.bool, device=device)
        for j, t in enumerate(txs):
            l = t.shape[0]
            out[j, :l] = t
            mask[j, l:] = True
        return out, mask
    
    batch, pad = pad_stack(fb_embs)
    with torch.no_grad():
        proc = model(batch, attention_mask=None, padding_mask=pad)
    
    results = [proc[i] for i in range(len(sequences))]
    return results

def process_paired_precomputed_fallback(data, model, fallback_lm, device, mode="paired", batch_size=128, seq_column=None):
    """
    Processes sequences in batches using language model with transformer.

    Args:
        data (pd.DataFrame): DataFrame containing sequences.
        model (torch.nn.Module): Transformer model for final embedding.
        fallback_lm: Language model (IgT5 for paired, ESMC for hc/nb).
        device (torch.device): Device to run the model on.
        mode (str): Embedding mode ('paired', 'hc', 'nb').
        batch_size (int): Batch size for processing.
        seq_column (str): Column name for sequences (auto-detected if None).

    Returns:
        tuple: raw_sequences (list), embeddings (list of torch.Tensor)
    """
    raw_sequences = []
    embeddings = []
    
    # Auto-detect sequence column if not specified
    if seq_column is None:
        for col in ["sequence_alignment_aa", "Sequence", "sequence"]:
            if col in data.columns:
                seq_column = col
                break
        if seq_column is None:
            raise ValueError(f"No sequence column found. Available columns: {list(data.columns)}")
    elif seq_column not in data.columns:
        raise ValueError(f"Column '{seq_column}' not found. Available columns: {list(data.columns)}")
    
    all_sequences = data[seq_column].tolist()

    for i in tqdm(range(0, len(all_sequences), batch_size), desc="Processing batches"):
        batch_sequences = all_sequences[i:i + batch_size]
        batch_embeddings = embed_with_fallback(
            batch_sequences, 
            model=model,
            fallback_lm=fallback_lm,
            device=device,
            mode=mode
        )
        
        raw_sequences.extend(batch_sequences)
        embeddings.extend(batch_embeddings)

    return raw_sequences, embeddings