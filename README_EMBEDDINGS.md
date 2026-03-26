# Embedding Generation for LLM QA System

Generate dense vector embeddings from processed PDF chunks using sentence-transformers on V100 GPU.

---

## Overview

This module converts text chunks into numerical vectors (embeddings) that capture semantic meaning. These embeddings enable:
- Semantic search
- Similar document retrieval
- Question-answering systems
- Document clustering

### Model: all-MiniLM-L6-v2

- **Dimensions**: 384
- **Performance**: Fast, efficient
- **Quality**: Excellent for semantic similarity
- **Size**: ~90MB
- **Speed**: ~1000-2000 chunks/second on V100 GPU

---

## Quick Start

### 1. Prerequisites

Ensure you have:
- ✅ Processed chunks: `data/processed_chunks.json`
- ✅ GPU access (V100 recommended)
- ✅ Dependencies installed

```bash
# Install dependencies
pip install torch sentence-transformers numpy
```

### 2. Test on Sample

```bash
# Test with 5 sample chunks
python3 src/test_embeddings.py
```

### 3. Generate Embeddings

**Option A: SLURM Job (Recommended)**
```bash
sbatch jobs/generate_embeddings.sb
```

**Option B: Interactive**
```bash
python3 src/generate_embeddings.py
```

---

## Input & Output

### Input

**File**: `data/processed_chunks.json`

**Format**:
```json
[
  {
    "text": "Bean breeding programs...",
    "source_file": "file.pdf",
    "year_range": "1961-2006",
    "chunk_id": 0,
    "page_number": 1,
    "token_count": 487
  },
  ...
]
```

### Output

**1. Embeddings**: `data/embeddings.npy`
- Numpy array of shape `(n_chunks, 384)`
- Normalized for cosine similarity
- Float32 dtype

**2. Metadata**: `data/embeddings_metadata.json`
```json
{
  "generation_date": "2026-03-21T14:30:00",
  "model_name": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "statistics": {
    "total_chunks": 15234,
    "successful_embeddings": 15234,
    "processing_time_seconds": 12.5,
    "chunks_per_second": 1218.72,
    "gpu_used": true,
    "gpu_name": "Tesla V100S-PCIE-32GB"
  },
  "chunk_metadata": [...]
}
```

---

## Usage

### Command-Line Arguments

```bash
python3 src/generate_embeddings.py \
    --input-file data/processed_chunks.json \
    --output-embeddings data/embeddings.npy \
    --output-metadata data/embeddings_metadata.json \
    --log-file logs/embeddings.log \
    --model all-MiniLM-L6-v2 \
    --batch-size 32 \
    --device cuda
```

### Python API

```python
from generate_embeddings import EmbeddingGenerator

# Create generator
generator = EmbeddingGenerator(
    input_file="data/processed_chunks.json",
    output_embeddings="data/embeddings.npy",
    output_metadata="data/embeddings_metadata.json",
    log_file="logs/embeddings.log",
    model_name="all-MiniLM-L6-v2",
    batch_size=32,
    device="cuda"
)

# Run generation
success = generator.run()
```

---

## Configuration

### Batch Size Optimization

| GPU | VRAM | Recommended Batch Size |
|-----|------|----------------------|
| V100 | 32GB | 32-64 |
| A100 | 40GB | 64-128 |
| CPU | N/A | 8-16 |

### Model Options

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | General purpose (recommended) |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality needed |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | QA-optimized |

---

## Performance

### Expected Processing Times

**For 15,000 chunks** (typical for 1,067 PDFs):

| Hardware | Batch Size | Time |
|----------|------------|------|
| V100 GPU | 32 | ~10-15 seconds |
| A100 GPU | 64 | ~5-10 seconds |
| CPU (8 cores) | 16 | ~5-10 minutes |

### Resource Requirements

**SLURM Job Configuration**:
```bash
#SBATCH --gpus=1             # 1 GPU required
#SBATCH --cpus-per-task=4    # 4 CPU cores
#SBATCH --mem=16GB           # 16GB RAM
#SBATCH --time=02:00:00      # 2 hours (plenty of buffer)
#SBATCH --constraint=v100    # Request V100 GPU
```

**Memory Usage**:
- Model: ~500MB
- Embeddings (15K chunks): ~23MB (in memory)
- Peak GPU memory: ~2-3GB
- Peak RAM: ~4-6GB

---

## Validation

The script performs automatic validation:

### 1. Dimension Check
✓ Embeddings have correct dimension (384 for all-MiniLM-L6-v2)

### 2. NaN/Inf Check
✓ No invalid values in embeddings

### 3. Normalization Check
✓ Embeddings are L2-normalized (for cosine similarity)

### 4. Count Check
✓ Number of embeddings matches number of chunks

---

## Using the Embeddings

### Load Embeddings

```python
import numpy as np
import json

# Load embeddings
embeddings = np.load('data/embeddings.npy')
print(f"Shape: {embeddings.shape}")  # (15234, 384)

# Load metadata
with open('data/embeddings_metadata.json') as f:
    metadata = json.load(f)

chunks_metadata = metadata['chunk_metadata']
```

### Semantic Search

```python
from numpy.linalg import norm

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (norm(a) * norm(b))

# Search query
query = "What are bean breeding techniques?"

# Generate query embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode(query, normalize_embeddings=True)

# Compute similarities
similarities = embeddings @ query_embedding  # Matrix multiplication

# Get top 5 results
top_5_indices = np.argsort(similarities)[-5:][::-1]

print("Top 5 most relevant chunks:")
for idx in top_5_indices:
    score = similarities[idx]
    chunk = chunks_metadata[idx]
    print(f"Score: {score:.4f} - {chunk['source_file']}")
```

### Find Similar Chunks

```python
# Find chunks similar to chunk 0
query_embedding = embeddings[0]
similarities = embeddings @ query_embedding

# Get top 10 similar chunks (excluding itself)
top_10 = np.argsort(similarities)[-11:-1][::-1]

for idx in top_10:
    print(f"Similarity: {similarities[idx]:.4f}")
    print(f"File: {chunks_metadata[idx]['source_file']}")
    print()
```

### Clustering

```python
from sklearn.cluster import KMeans

# Cluster chunks into 10 groups
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Analyze clusters
for i in range(10):
    cluster_chunks = [chunks_metadata[j] for j, c in enumerate(clusters) if c == i]
    print(f"Cluster {i}: {len(cluster_chunks)} chunks")
    print(f"  Years: {set(c['year_range'] for c in cluster_chunks)}")
```

---

## Monitoring

### During Processing

```bash
# Check job status
squeue -u $USER

# Monitor log in real-time
tail -f logs/embeddings_JOBID.log

# Watch SLURM output
tail -f logs/embeddings_JOBID.out
```

### After Processing

```bash
# Check output files
ls -lh data/embeddings.npy
ls -lh data/embeddings_metadata.json

# Verify embeddings
python3 -c "
import numpy as np
emb = np.load('data/embeddings.npy')
print(f'Shape: {emb.shape}')
print(f'Dtype: {emb.dtype}')
print(f'Size: {emb.nbytes / 1024 / 1024:.2f} MB')
"
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```bash
   python3 src/generate_embeddings.py --batch-size 16
   ```
2. Use smaller model:
   ```bash
   python3 src/generate_embeddings.py --model all-MiniLM-L6-v2
   ```
3. Use CPU (slower):
   ```bash
   python3 src/generate_embeddings.py --device cpu
   ```

### Issue: No GPU Available

**Symptoms**: `GPU used: false` in logs

**Solutions**:
1. Check GPU allocation:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   nvidia-smi
   ```
2. Request GPU in SLURM:
   ```bash
   #SBATCH --gpus=1
   ```
3. Check PyTorch CUDA:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### Issue: Model Download Fails

**Symptoms**: `Connection error` or `Download failed`

**Solutions**:
1. Set cache directory:
   ```bash
   export SENTENCE_TRANSFORMERS_HOME=/mnt/scratch/$USER/llm_cache/sentence_transformers
   ```
2. Pre-download model:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```
3. Check internet connectivity

### Issue: Input File Not Found

**Symptoms**: `FileNotFoundError: Input file not found`

**Solutions**:
1. Verify processed chunks exist:
   ```bash
   ls -lh data/processed_chunks.json
   ```
2. Run PDF processor first:
   ```bash
   python3 src/pdf_processor.py
   ```
3. Check file path in command

---

## Next Steps

After generating embeddings:

### 1. Set Up Vector Database

**Option A: ChromaDB**
```python
import chromadb
from chromadb.utils import embedding_functions

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="bean_research",
    metadata={"description": "Bean research papers 1961-2026"}
)

# Add embeddings
collection.add(
    embeddings=embeddings.tolist(),
    documents=[chunk['text'] for chunk in chunks_metadata],
    metadatas=chunks_metadata,
    ids=[f"chunk_{i}" for i in range(len(embeddings))]
)
```

**Option B: FAISS**
```python
import faiss

# Create FAISS index
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(index, "data/faiss_index.bin")
```

### 2. Build QA System

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

# Set up vector store
vectorstore = Chroma(
    collection_name="bean_research",
    embedding_function=embedding_function
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# Ask questions
answer = qa_chain.run("What are the main bean breeding techniques?")
print(answer)
```

---

## Files Created

| File | Description | Size |
|------|-------------|------|
| `src/generate_embeddings.py` | Main embedding generator | ~15KB |
| `src/test_embeddings.py` | Test script | ~5KB |
| `jobs/generate_embeddings.sb` | SLURM job script | ~3KB |
| `data/embeddings.npy` | Generated embeddings | ~23MB (for 15K chunks) |
| `data/embeddings_metadata.json` | Metadata & statistics | ~5-10MB |
| `logs/embeddings.log` | Processing log | Varies |

---

## Advanced Usage

### Custom Model

```python
generator = EmbeddingGenerator(
    input_file="data/processed_chunks.json",
    output_embeddings="data/embeddings_custom.npy",
    output_metadata="data/metadata_custom.json",
    log_file="logs/embeddings_custom.log",
    model_name="sentence-transformers/all-mpnet-base-v2",  # Different model
    batch_size=16,  # Adjust for model size
    device="cuda"
)
```

### Process Subset

```python
import json

# Load all chunks
with open('data/processed_chunks.json') as f:
    all_chunks = json.load(f)

# Filter subset (e.g., only 2007-2026)
subset = [c for c in all_chunks if c['year_range'] == '2007-2026']

# Save subset
with open('data/subset_chunks.json', 'w') as f:
    json.dump(subset, f)

# Generate embeddings for subset
# ... use subset_chunks.json as input
```

### Incremental Updates

```python
# Load existing embeddings
existing = np.load('data/embeddings.npy')

# Generate new embeddings
new_embeddings = generator.generate_embeddings(new_chunks)

# Concatenate
combined = np.vstack([existing, new_embeddings])

# Save
np.save('data/embeddings.npy', combined)
```

---

## Best Practices

1. **Always test first**: Use `test_embeddings.py` before full run
2. **Use GPU**: 100x faster than CPU
3. **Normalize embeddings**: Enable for cosine similarity
4. **Save metadata**: Essential for tracking chunks
5. **Validate output**: Check shape, NaN, normalization
6. **Set cache directory**: Avoid filling home directory
7. **Monitor GPU memory**: Adjust batch size if needed
8. **Version control**: Save model name and version

---

## References

- **Sentence-Transformers**: https://www.sbert.net/
- **Model Card**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **ChromaDB**: https://docs.trychroma.com/
- **FAISS**: https://github.com/facebookresearch/faiss

---

**Ready to generate embeddings?**

```bash
# Test first
python3 src/test_embeddings.py

# Then run full generation
sbatch jobs/generate_embeddings.sb
```
