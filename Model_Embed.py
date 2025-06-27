import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import itertools
import sys
#from embed_structure_model import trans_basic_block, trans_basic_block_Config
from embed_structure_model_cpu import trans_basic_block, trans_basic_block_Config
from model import PairedIgT5
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.lines as mlines
from torch.amp import autocast
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
# from huggingface_hub import login
import os


# def model_embed_with_precomputed(sequence, lmembeddings, model, device): 
#     #precomputed dictionary
#     embedding = lmembeddings[sequence].to(device)
#     if len(embedding.shape) == 2:
#         embedding = embedding.unsqueeze(0)
#     padding = torch.zeros(embedding.shape[0:2]).type(torch.BoolTensor).to(device)
#     with torch.no_grad():
#         with autocast(device.type, dtype=torch.bfloat16):
#             embedded_sequence = model(embedding, src_mask=None, src_key_padding_mask=padding)
#             embedded_sequence = embedded_sequence.squeeze(0)

#     return embedded_sequence

# def process_paired_precomputed(data, sequence_embeddings, model, device):
#     raw_sequences = []
#     embeddings = []
#     for sequences in tqdm(data['Sequence']):
#         try:
#             embedding = model_embed_with_precomputed(sequences, sequence_embeddings, model, device)
#             raw_sequences.append(sequences)
#             embeddings.append(embedding)
#         except KeyError as e:
#             print(f"Sequence {sequences} not found in precomputed embeddings. Skipping.")
#             continue
#     return raw_sequences, embeddings

# def save_embeddings(embeddings, output_file):
#     embedding_matrix = np.array(embeddings, dtype=np.float32)
#     np.save(output_file, embedding_matrix)
#     print(f' saved as {output_file}')
#     return embedding_matrix

# #When provided sequences do not have pre-computed embeddings
# def parse_paired_sequence(seq):
#     parts = seq.split("|", maxsplit=1)
#     if len(parts) != 2:
#         raise ValueError(
#             f"Row '{seq}' does not contain '|' separating heavy and light. Use | to separate heavy and light chains."
#         )
#     hc, lc = parts[0].strip(), parts[1].strip()
#     return hc, lc

# def embed_IgT5(sequence, fallback): 
#     heavy_seq, light_seq = parse_paired_sequence(sequence)
#     embeddings = fallback.get_embeddings([heavy_seq], [light_seq])
#     return embeddings

# def embed_ESM_single(sequence: str, model) -> torch.Tensor:
#     #model = model.to(device)
#     protein = ESMProtein(sequence=sequence)      
#     protein_tensor = model.encode(protein)       

#     logits_config = LogitsConfig(sequence=True, return_embeddings=True)
#     logits_output = model.logits(protein_tensor, logits_config)

#     residue_embeddings = logits_output.embeddings.detach()

#     if residue_embeddings.dim() == 3:
#         residue_embeddings = residue_embeddings.squeeze(0)

#     return residue_embeddings

# def embed_ESM_HC(sequence: str, fallback_model) -> torch.Tensor:
#     embedding = embed_ESM_single(sequence, fallback_model)
#     return embedding

# def embed_with_fallback(sequence, lmembeddings, model, fallback_igt5, device, mode='paired'):

#         new_embedding = embed_IgT5(sequence, fallback_igt5)
#         new_embedding = new_embedding[0]
#         new_embedding = new_embedding.unsqueeze(0)

#         padding = torch.zeros(new_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
#         with torch.no_grad():
#             with autocast(device.type, dtype=torch.bfloat16):
#                 embedded_sequence = model(
#                     new_embedding,
#                     src_mask=None,
#                     src_key_padding_mask=padding
#                 )
#                 embedded_sequence = embedded_sequence.squeeze(0)

#         return embedded_sequence



# def embed_with_fallback(sequence, lmembeddings, model, fallback_igt5, device, mode='paired'):
#     try: 
#         return model_embed_with_precomputed(sequence, lmembeddings, model, device)
    
#     except KeyError:
#         #print(f'sequence not in precomputed embeddings, embedding paired using IgT5/ESM')
#         if mode == 'paired':
#             new_embedding = embed_IgT5(sequence, fallback_igt5)
#             new_embedding = new_embedding[0]
#             new_embedding = new_embedding.unsqueeze(0)
#             #print(new_embedding.shape)
#         elif mode in ('hc', 'nb'):
#             new_embedding = embed_ESM_HC(sequence, fallback_igt5)  #ignore name, legacy from when paired only ffs
#             new_embedding = new_embedding.unsqueeze(0)
#             if len(new_embedding.shape) == 2:
#                 new_embedding = new_embedding.unsqueeze(0)
#             #print(new_embedding.shape)
#         else:
#             raise ValueError(f'what is you tryna embed?')

#         padding = torch.zeros(new_embedding.shape[0:2]).type(torch.BoolTensor).to(device)

#         with torch.no_grad():
#             with autocast(device.type, dtype=torch.bfloat16):
#                 embedded_sequence = model(
#                     new_embedding,
#                     src_mask=None,
#                     src_key_padding_mask=padding
#                 )
#                 embedded_sequence = embedded_sequence.squeeze(0)

#         return embedded_sequence

# def embed_with_fallback(sequence, lmembeddings, model, fallback_igt5, device, mode='paired'):
#     raw_sequences = list(sequence["Sequence"])
#     results = [None] * len(raw_sequences)
    
#     fallback_indices = []
#     fallback_seqs = []
    
#     # Separate sequences based on availability of precomputed embeddings.
#     for idx, seq in enumerate(raw_sequences):
#         if seq in sequence_embeddings:
#             # Process precomputed embedding for this sequence.
#             results[idx] = model_embed_with_precomputed(seq, sequence_embeddings, model, device)
#         else:
#             fallback_indices.append(idx)
#             fallback_seqs.append(seq)
#     if fallback_seqs:
#         if mode == 'paired': 
#             new_embedding = embed_IgT5(sequence, fallback_igt5)
#             new_embedding = new_embedding[0]
#             new_embedding = new_embedding.unsqueeze(0)
#         elif mode in ('hc', 'nb'):
#             new_embedding = embed_ESM_HC(sequence, fallback_igt5)  #ignore name, legacy from when paired only ffs
#             new_embedding = new_embedding.unsqueeze(0)
#             if len(new_embedding.shape) == 2:
#                 new_embedding = new_embedding.unsqueeze(0)
#             #print(new_embedding.shape)
#         else:
#             raise ValueError(f'what is you tryna embed?')
#         padding = torch.zeros(new_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
#         with torch.no_grad():
#             with autocast(device.type, dtype=torch.bfloat16):
#                 embedded_sequence = model(
#                     new_embedding,
#                     src_mask=None,
#                     src_key_padding_mask=padding
#                 )
#                 embedded_sequence = embedded_sequence.squeeze(0)

#         return embedded_sequence
        
    

# def process_paired_precomputed_fallback(data, sequence_embeddings, model, fallback_igt5, device, mode="paired"):
#     raw_sequences = []
#     embeddings = []
   
#     for seq in tqdm(data["Sequence"]):
#         embedding = embed_with_fallback(seq, sequence_embeddings, model, fallback_igt5, device, mode)
#         raw_sequences.append(seq)
#         embeddings.append(embedding)
    
#     return raw_sequences, embeddings







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
            missing_sequences.append(seq) # Handle missing sequences later if needed, for now just skip and print.
            print(f"Sequence {seq} not found in precomputed embeddings.")
            #raise KeyError(f"Sequence {seq} not found in precomputed embeddings.") # decide how to handle missing. For batching, better to handle outside.

    if not embeddings_list: # if all sequences are missing, return empty tensor or handle as needed.
        return torch.empty(0) # Or raise an exception, or return None, based on desired behaviour.

    # Stack embeddings to create a batch
    embedding_tensor = torch.stack(embeddings_list)

    if len(embedding_tensor.shape) == 2: # Handle case where single sequence embedding was loaded (though batching should avoid this)
        embedding_tensor = embedding_tensor.unsqueeze(0) # Correct dimension for batching to work

    batch_size, seq_len, _ = embedding_tensor.shape
    padding = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    with torch.no_grad():
        #with autocast(device.type, dtype=torch.bfloat16): #Not for cpu version
            embedded_sequence = model(embedding_tensor, attention_mask=None, padding_mask=padding)
            # embedded_sequence shape will be (batch_size, seq_len, embedding_dim)

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
    all_sequences = data['Sequence'].tolist() # Get all sequences once

    for i in tqdm(range(0, len(all_sequences), batch_size), desc="Processing batches"):
        batch_sequences = all_sequences[i:i + batch_size]
        batch_raw_sequences = [] # To keep track of successfully embedded sequences in this batch
        batch_embeddings = []

        valid_batch_sequences = [] # Sequences that have precomputed embeddings
        for seq in batch_sequences:
            if seq in sequence_embeddings:
                valid_batch_sequences.append(seq)
                batch_raw_sequences.append(seq) # Only add to raw_sequences if embedding found
            else:
                print(f"Sequence {seq} not found in precomputed embeddings. Skipping from precomputed batch.")

        if valid_batch_sequences: # Only process if there are valid sequences in the batch
            try:
                batch_embedding_tensor = model_embed_with_precomputed(valid_batch_sequences, sequence_embeddings, model, device)
                if not torch.is_tensor(batch_embedding_tensor) or batch_embedding_tensor.numel() == 0: # Handle cases where model_embed_with_precomputed returns empty tensor
                    print("Warning: model_embed_with_precomputed returned empty embeddings for batch.")
                else:
                    batch_embeddings.extend(list(batch_embedding_tensor)) # Convert batch tensor to list of tensors
            except KeyError as e: # Catch KeyError from model_embed_with_precomputed in batch mode if needed.
                print(f"KeyError in batch processing: {e}")
                print("Skipping batch.")
                continue # or break, depending on error handling policy

        raw_sequences.extend(batch_raw_sequences)
        embeddings.extend(batch_embeddings)


    return raw_sequences, embeddings

def save_embeddings(embeddings, output_file):
    """Saves embeddings to a numpy file."""
    # Check if embeddings is non-empty.
    if embeddings is not None and len(embeddings) > 0:
        if isinstance(embeddings[0], torch.Tensor):
            embedding_matrix = torch.stack(embeddings).cpu().numpy().astype(np.float32)
        else:
            embedding_matrix = np.array(embeddings, dtype=np.float32)
    else:
        embedding_matrix = np.array([])  # Handle empty embeddings list

    np.save(output_file, embedding_matrix)
    print(f'saved as {output_file}')
    return embedding_matrix

#When provided sequences do not have pre-computed embeddings
def parse_paired_sequence(seq):
    parts = seq.split("|", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Row '{seq}' does not contain '|' separating heavy and light. Use | to separate heavy and light chains."
        )
    hc, lc = parts[0].strip(), parts[1].strip()
    return hc, lc

def embed_IgT5(sequences, fallback): # Modified to take list of sequences
    """Embeds a list of paired sequences using IgT5."""
    heavy_seqs = []
    light_seqs = []
    for sequence in sequences:
        heavy_seq, light_seq = parse_paired_sequence(sequence)
        heavy_seqs.append(heavy_seq)
        light_seqs.append(light_seq)
    embeddings_list = fallback.get_embeddings(heavy_seqs, light_seqs)
    return embeddings_list # Returns list of embedding tensors

def embed_ESM_single(sequences: list, model) -> list:
    """Embeds a list of sequences using ESM-single."""
    embeddings = []
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    for seq in sequences:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        logits_output = model.logits(protein_tensor, logits_config)
        residue_embeddings = logits_output.embeddings.detach()  # Shape: (seq_len, embed_dim) or (1, seq_len, embed_dim)
        embeddings.append(residue_embeddings)
    return embeddings

def embed_ESM_HC(sequences: list, fallback_model) -> list: # Modified to take list of sequences
    """Embeds a list of sequences using ESM-HC."""
    embeddings = embed_ESM_single(sequences, fallback_model)
    return embeddings

# def embed_with_fallback(sequences, lmembeddings, model, fallback_igt5, device, mode='paired'):

#     processed_fallback_embeddings = []

#     precomputed_seqs = []
#     fallback_seqs = []
#     precomputed_indices = []
#     fallback_indices = []
#     results = [None] * len(sequences) # Initialize results list to maintain order

#     for idx, seq in enumerate(sequences):
#         if seq in lmembeddings:
#             precomputed_seqs.append(seq)
#             precomputed_indices.append(idx)
#         else:
#             fallback_seqs.append(seq)
#             fallback_indices.append(idx)

#     precomputed_embeddings = []
#     for seq in precomputed_seqs:
#         emb = lmembeddings[seq].to(device)
#         if emb.dim() == 2:
#             emb = emb.unsqueeze(0)
#         # For a single sequence, we assume no padding is needed.
#         batch_size, seq_len, _ = emb.shape
#         padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
#         with torch.no_grad():
#             with autocast(device.type, dtype=torch.bfloat16):
#                 processed = model(emb, src_mask=None, src_key_padding_mask=padding_mask)
        
#         processed = processed.squeeze(0)
#         precomputed_embeddings.append(processed)


#     fallback_embeddings = []
#     if fallback_seqs:
#         if mode == 'paired':
#             fallback_embeddings = embed_IgT5(fallback_seqs, fallback_igt5) # Already returns a list
#         elif mode in ('hc', 'nb'):
#             fallback_embeddings = embed_ESM_HC(fallback_seqs, fallback_igt5) # Already returns a list
#         else:
#             raise ValueError(f'Invalid embedding mode: {mode}')

#         # Process fallback embeddings through the main model
#         # processed_fallback_embeddings = []
#         if fallback_embeddings:
#             for emb in fallback_embeddings: # Process each fallback embedding individually
#                 if emb.dim() == 2:
#                     fallback_embedding_tensor = emb.unsqueeze(0).to(device)
#                 elif emb.dim() == 3:
#                     fallback_embedding_tensor = emb.to(device)
#                 else:
#                     raise ValueError(f"Unexpected embedding dimension: {emb.dim()}")
#                 padding = torch.zeros(fallback_embedding_tensor.shape[0:2]).type(torch.BoolTensor).to(device)
#                 with torch.no_grad():
#                     with autocast(device.type, dtype=torch.bfloat16):
#                         embedded_sequence = model( # Process each embedding individually
#                             fallback_embedding_tensor,
#                             src_mask=None,
#                             src_key_padding_mask=padding
#                         )
#                         processed_fallback_embeddings.append(embedded_sequence.squeeze(0)) # Append individual embedding


#     # Place embeddings back into the results list in original order
#     for i, emb in zip(precomputed_indices, precomputed_embeddings):
#         results[i] = emb
#     for i, emb in zip(fallback_indices, processed_fallback_embeddings):
#         results[i] = emb

#     return results # Return list of embeddings in original input order

def embed_with_fallback(sequences, lmembeddings, model, fallback_igt5, device, mode='paired'):
    n = len(sequences)
    results = [None] * n

    pre_idx, pre_embs = [], []
    fb_idx, fb_seqs = [], []
    for i, seq in enumerate(sequences):
        emb = lmembeddings.get(seq)
        if emb is not None:
            pre_idx.append(i)
            pre_embs.append(emb.to(device))
        else:
            fb_idx.append(i)
            fb_seqs.append(seq)

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

    if pre_embs:
        pre_embs_clean = [(e.squeeze(0) if e.dim() == 3 else e) for e in pre_embs]
        batch, pad = pad_stack(pre_embs_clean)
        with torch.no_grad():
            #with autocast(device.type, dtype=torch.bfloat16):
                proc = model(batch, attention_mask=None, padding_mask=pad)
        for k, idx in enumerate(pre_idx):
            results[idx] = proc[k]

    if fb_seqs:
        if mode == 'paired':
            raw = embed_IgT5(fb_seqs, fallback_igt5)
        elif mode in ('hc', 'nb'):
            raw = embed_ESM_HC(fb_seqs, fallback_igt5)
        else:
            raise ValueError('Invalid embedding mode')
        fb_embs = [(x.squeeze(0) if x.dim() == 3 else x).to(device) for x in raw]
        batch, pad = pad_stack(fb_embs)
        with torch.no_grad():
            #with autocast(device.type, dtype=torch.bfloat16):
                proc = model(batch, attention_mask=None, padding_mask=pad)
        for k, idx in enumerate(fb_idx):
            results[idx] = proc[k]

    return results

def process_paired_precomputed_fallback(data, sequence_embeddings, model, fallback_igt5, device, mode="paired", batch_size=128): # set batch_size to 128 as requested
    """
    Processes paired sequences in batches, using precomputed embeddings with fallback.

    Args:
        data (pd.DataFrame): DataFrame containing sequences in 'Sequence' column.
        sequence_embeddings (dict): Dictionary of precomputed embeddings.
        model (torch.nn.Module): Model to use for embedding.
        fallback_igt5: Fallback IgT5 model.
        device (torch.device): Device to run the model on.
        mode (str): Embedding mode ('paired', 'hc', 'nb').
        batch_size (int): Batch size for processing.

    Returns:
        tuple: raw_sequences (list), embeddings (list of torch.Tensor)
    """
    raw_sequences = []
    embeddings = []
    # Use flexible column name - this function seems to expect a specific column
    # This should be updated to accept a configurable column name parameter
    seq_column = "sequence_alignment_aa"  # Default, but should be configurable
    if seq_column not in data.columns:
        # Try common alternatives
        if "Sequence" in data.columns:
            seq_column = "Sequence"
        elif "sequence" in data.columns:
            seq_column = "sequence"
        else:
            raise ValueError(f"No sequence column found. Available columns: {list(data.columns)}")
    
    all_sequences = data[seq_column].tolist()

    for i in tqdm(range(0, len(all_sequences), batch_size), desc="Processing batches"):
        batch_sequences = all_sequences[i:i + batch_size]
        batch_raw_sequences = batch_sequences # Keep all sequences for raw sequence output
        batch_embeddings = embed_with_fallback(batch_sequences, sequence_embeddings, model, fallback_igt5, device, mode) # Process batch

        raw_sequences.extend(batch_raw_sequences)
        embeddings.extend(batch_embeddings) # Extend with list of embeddings from batch

    return raw_sequences, embeddings






