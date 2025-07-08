# AbSLang

AbSLang is a deep learning framework for antibody structure prediction and analysis. It provides tools for embedding antibody sequences, building search indices, and predicting structural properties like RMSD between antibody pairs.

## Features

- **Sequence Embedding**: Convert antibody sequences into high-dimensional embeddings using pre-trained language models
- **Structure Prediction**: Predict structural distances (RMSD) between antibody pairs
- **Similarity Search**: Build and query FAISS indices for fast antibody similarity search
- **Multiple Input Modes**: Support for paired (heavy+light chain), heavy chain only, and nanobody sequences
- **Pre-computed Embeddings**: Leverage pre-computed embeddings for faster inference

## Installation

### Prerequisites

- Python 3.10
- Conda to install dependencies

### Setting up the Environment

1. Clone the repository:
```bash
git clone https://github.com/ericjidawang/AbSLang.git
cd AbSLang
```

2. Create the conda environment called database: 
```bash
conda env create -f env.yml
```

3. Activate the environment:
```bash
conda activate database
```

## Quick Start

See `search_example.ipynb` for an example.

### Searching an Antibody Index

```python
# Searching an index using a specific sequence
from abslang.search_antibody_index import FaissSearcher

# Initialize searcher with pre-built index
searcher = FaissSearcher(
    faiss_index_file="ewang-HC_Search_Index/paired_numbered_nonredundant_embeddings_QT4bit.index",  # index file
    csv_file="ewang-HC_Search_Index/human_oas_paired_non_redundant_by_study.csv",  # corresponding csv with sequence and global IDs
    device="cpu",
    mode="paired"
)

# Define query sequence (HC|LC format)
query_sequence = "QVQLVESGGGVVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDSKLCGGDCYPSGRGYFDYWGQGTLVTVSS|QSALTQPRSVSGSPGQSVTISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVPDRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSYTYVFGTGTKVTVL"

# Search for k most similar antibodies
results = searcher.search(query_sequence, k=10)  # number of results to return
print(results)

# Alternatively search by CDR RMSD threshold
results = searcher.search_by_threshold(
    query_sequence,
    threshold=2.0,   # Average CDR RMSD threshold
    step=10,         # Step size for the search
    max_k=100
)
# Note: igt5 is pretty slow on cpu, so maybe use a smaller max_k or smaller step
```

## Model Modes

AbSLang supports three input modes:

- **paired**: Heavy + Light chain pairs (format: "HC|LC", separated by "|")
- **hc**: Heavy chain only sequences
- **nb**: VHH sequences

## Configuration

The model paths and configurations are defined in `config.py`. You maybe need to update the base path to point to your model files:

```python
base = Path("path_to_checkpoints")
```
