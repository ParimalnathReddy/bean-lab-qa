# Vector Store - Quick Start

Build a searchable database from your embeddings in 3 steps.

---

## ⚡ Quick Commands

### Test (5 sample chunks)
```bash
python3 src/test_vector_store.py
```

### Build Full Database
```bash
# Run directly
python3 src/build_vector_store.py

# Or submit SLURM job
sbatch jobs/build_vector_store.sb
```

### Verify
```bash
ls -lh vector_db/
cat vector_db/database_statistics.json
```

---

## 📊 What You Get

**Input**:
- `embeddings.npy` (vector embeddings)
- `embeddings_metadata.json` (metadata)
- `processed_chunks.json` (text)

**Output**:
- `vector_db/` - Persistent ChromaDB (~75-100MB)
- Searchable by semantic meaning
- Filterable by year, source, etc.

**Processing Time**: 10-30 seconds for 15K chunks

---

## 🔍 Using the Database

```python
import chromadb

# Connect
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection("bean_research_docs")

# Search
results = collection.query(
    query_texts=["bean breeding techniques"],
    n_results=5
)

# Filter by year
results = collection.query(
    query_texts=["drought resistance"],
    n_results=5,
    where={"year_range": "2007-2026"}
)
```

---

## ✅ Features

- ✓ Semantic search (meaning-based)
- ✓ Metadata filtering (year, source)
- ✓ Persistent storage
- ✓ Fast queries (<100ms)
- ✓ Validation included

---

## 🚀 Next Steps

1. Build QA system with LangChain
2. Create search interface
3. Deploy web app

---

**Full Documentation**: [README_VECTOR_STORE.md](README_VECTOR_STORE.md)
