# Embedding Generation - Quick Start

Convert your processed PDF chunks into vector embeddings for semantic search and QA.

---

## ⚡ Quick Commands

### Test (5 sample chunks)
```bash
python3 src/test_embeddings.py
```

### Generate All Embeddings
```bash
# Submit to SLURM (recommended)
sbatch jobs/generate_embeddings.sb

# Or run interactively
python3 src/generate_embeddings.py
```

### Monitor Progress
```bash
squeue -u $USER
tail -f logs/embeddings_*.out
```

---

## 📊 What You Get

**Input**: `data/processed_chunks.json` (15,000-20,000 chunks)

**Output**:
- `data/embeddings.npy` - Vector embeddings (~23MB)
- `data/embeddings_metadata.json` - Metadata & stats

**Processing Time**: 10-15 seconds on V100 GPU

---

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `all-MiniLM-L6-v2` | Embedding model |
| `--batch-size` | `32` | GPU batch size |
| `--device` | `auto` | cuda/cpu/auto |

---

## ✅ Validation

Automatic checks:
- ✓ Correct dimensions (384)
- ✓ No NaN/Inf values
- ✓ L2 normalized
- ✓ Count matches chunks

---

## 🚀 Next Steps

```python
import numpy as np

# Load embeddings
embeddings = np.load('data/embeddings.npy')

# Use for semantic search
# Build vector database (ChromaDB/FAISS)
# Create QA system
```

---

**Full Documentation**: [README_EMBEDDINGS.md](README_EMBEDDINGS.md)
