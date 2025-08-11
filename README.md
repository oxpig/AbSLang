# AbSLang
Search for structurally similar antibodies (CDRs) from sequence. 

## Features

- **Structure-aware embeddings**: Sequence level embeddings generated from sequences. 
- **Similarity search**: FAISS index used for efficient search.
- **RMSD prediction**: Estimate structural distance between antibody pairs directly from sequence.
- **Supported inputs**: `paired` (heavy|light), `hc` (heavy chain), `nb` (VHH).

## Installation

### Prerequisites

- Python 3.10

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/ericjidawang/AbSLang.git
cd AbSLang
```

2. Create the conda environment:
```bash
conda env create -f env.yml
#for cpu compatible version
conda env create -f cpu_env.yml
```

3. Activate the environment:
```bash
conda activate database
```



## Start

### Build an Index from CSV

```python
from abslang.build_search_index import build_index_from_csv

artifacts = build_index_from_csv(
    csv_path="/path/to/sequences.csv",
    mode="paired",  # 'paired', 'hc', or 'nb'
    id_column="id", 
    seq_column="sequence_alignment_aa", #header for sequences
    out_dir="index_out", #directory that the outputs will be saved to
    index_type="pq",  # 'flat' | 'pq' | 'ivfpq'
    tm_checkpoint_path="/path/to/checkpoint.ckpt",
    tm_config_path="/path/to/config.json",
)

# Writes: index.{type}, seq_list.json, artifacts.json
```

CSV requirements:
- Default columns: `id`, `sequence_alignment_aa` (customizable via `id_column`, `seq_column`).
- Mode vs sequence format:
  - `paired`: sequences must contain `|` (format: `HC|LC`).
  - `hc`/`nb`: sequences must not contain `|`.

### Search an Index

```python
from abslang.search_antibody_index import FaissSearcher

searcher = FaissSearcher(
    faiss_index_file="index_out/index.pq",
    csv_file="/path/to/sequences.csv",  # corresponding CSV used to build the index
    device="cpu",
    mode="paired",
    tm_checkpoint_path="/path/to/checkpoint.ckpt",
    tm_config_path="/path/to/config.json",
)

# Query (HC|LC for paired mode)
query = "HeavyChain|LightChain"

# Top‑k search (distances are Euclidean; FAISS returns L2 squared distance which is sqrt transformed)
results = searcher.search(query, k=10) #configure the number of nearest matches returned. 

# Threshold search
results_thresh = searcher.search_by_threshold(
    query,
    threshold=2.0,     # Euclidean distance
    max_k=100,         # candidates to retrieve once
    reembed=True,     # True:Returns exact distance
)
```

### Predict RMSD Between Sequences

```python
from abslang import predict_cdr_rmsd, predict_cdr_rmsd_batch

# Single pair
r = predict_cdr_rmsd(
    "HC1...|LC1...", "HC2...|LC2...", 
    mode="paired",
    tm_checkpoint_path="/path/to/checkpoint.ckpt",
    tm_config_path="/path/to/config.json",
)

# Batch
pairs = [("HC1|LC1", "HC2|LC2"), ("HC3|LC3", "HC4|LC4")] #list
rs = predict_cdr_rmsd_batch(
    pairs,
    mode="paired",
    tm_checkpoint_path="/path/to/checkpoint.ckpt",
    tm_config_path="/path/to/config.json",
)
```

## Index Types

- `flat`: Exact, large memory footprint, fast queries for small datasets.
- `pq`: Product quantization (approximate), small index, good baseline.
- `ivfpq`: Inverted file + PQ (approximate), scalable to large datasets. Tune `nprobe` in FAISS for recall.

PQ parameter validation:
- If the embedding dimension `d` is not divisible by `m`, `m` is auto-adjusted to a nearby divisor.
- For IVFPQ, `nlist` is auto‑set from dataset size if omitted.

## Configuration

- AbSLang requires a checkpoint and a matching JSON config.
- `paired` mode uses IgT5 as the language model; `hc`/`nb` use ESMC (esmc_600m).