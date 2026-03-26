# Vector Store with ChromaDB

Build a persistent vector database for semantic search and document QA.

---

## Overview

This module creates a ChromaDB vector store from your embeddings, enabling:
- **Semantic Search**: Find documents by meaning, not keywords
- **Metadata Filtering**: Filter by year, source file, etc.
- **Persistent Storage**: Database saved to disk
- **Fast Retrieval**: Optimized for quick searches

---

## Quick Start

### 1. Prerequisites

Ensure you have:
- ✅ Embeddings generated: `data/embeddings.npy`
- ✅ Metadata file: `data/embeddings_metadata.json`
- ✅ Chunks file: `data/processed_chunks.json`
- ✅ ChromaDB installed: `pip install chromadb`

### 2. Test

```bash
python3 src/test_vector_store.py
```

### 3. Build Vector Store

```bash
# Run directly
python3 src/build_vector_store.py

# Or submit SLURM job
sbatch jobs/build_vector_store.sb
```

---

## Input & Output

### Input Files

| File | Description |
|------|-------------|
| `data/embeddings.npy` | Vector embeddings (n_chunks, 384) |
| `data/embeddings_metadata.json` | Embedding metadata |
| `data/processed_chunks.json` | Original text chunks |

### Output

**Vector Database**: `vector_db/`
- Persistent ChromaDB database
- Collection: `bean_research_docs`
- Size: ~50-100 MB (for 15K chunks)

**Statistics**: `vector_db/database_statistics.json`
```json
{
  "creation_date": "2026-03-21T15:00:00",
  "collection_name": "bean_research_docs",
  "statistics": {
    "total_chunks": 15234,
    "chunks_added": 15234,
    "database_size_mb": 85.2,
    "unique_sources": 1067,
    "year_ranges": ["1961-2006", "2007-2026"]
  }
}
```

---

## Usage

### Command-Line

```bash
python3 src/build_vector_store.py \
    --embeddings-file data/embeddings.npy \
    --metadata-file data/embeddings_metadata.json \
    --chunks-file data/processed_chunks.json \
    --db-path vector_db \
    --collection-name bean_research_docs \
    --batch-size 1000
```

### Python API

```python
from build_vector_store import VectorStoreBuilder

# Create builder
builder = VectorStoreBuilder(
    embeddings_file="data/embeddings.npy",
    metadata_file="data/embeddings_metadata.json",
    chunks_file="data/processed_chunks.json",
    db_path="vector_db",
    log_file="logs/vector_store.log",
    collection_name="bean_research_docs",
    batch_size=1000
)

# Build store
success = builder.run()
```

---

## Features

### 1. Semantic Search

Find documents by meaning:

```python
import chromadb

# Connect to database
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection("bean_research_docs")

# Search
results = collection.query(
    query_texts=["bean breeding techniques"],
    n_results=5
)

# Print results
for doc_id, doc, metadata, distance in zip(
    results['ids'][0],
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
):
    print(f"Distance: {distance:.4f}")
    print(f"Source: {metadata['source_file']}")
    print(f"Text: {doc[:200]}...")
    print()
```

### 2. Metadata Filtering

Filter by year range:

```python
# Find documents from 2007-2026 about drought
results = collection.query(
    query_texts=["drought resistance"],
    n_results=5,
    where={"year_range": "2007-2026"}
)
```

Filter by source file:

```python
# Find chunks from specific paper
results = collection.query(
    query_texts=["genetic modification"],
    n_results=5,
    where={"source_file": "10.2135_cropsci2004.3510.pdf"}
)
```

### 3. Direct Retrieval

Get specific chunks:

```python
# Get chunk by ID
result = collection.get(
    ids=["chunk_0"],
    include=["documents", "metadatas"]
)

print(result['documents'][0])
print(result['metadatas'][0])
```

### 4. Batch Queries

Query multiple at once:

```python
queries = [
    "bean breeding",
    "drought resistance",
    "disease management"
]

results = collection.query(
    query_texts=queries,
    n_results=3
)

# results['ids'] is list of lists (one per query)
```

---

## Validation Queries

The builder runs automatic validation:

### 1. Semantic Search Test
```
Query: 'bean breeding techniques'
✓ Found 3 results
```

### 2. Filtered Search Test
```
Query: 'drought resistance' WHERE year_range = '2007-2026'
✓ Found 3 results from 2007-2026
```

### 3. Direct Retrieval Test
```
✓ Retrieved chunk_0
```

### 4. Count Verification
```
✓ Collection contains 15234 documents
✓ Count matches expected
```

---

## Statistics

After building, you'll see:

```
VECTOR STORE STATISTICS
================================================================================
Collection Name:       bean_research_docs
Database Path:         vector_db
Total Chunks:          15234
Chunks Added:          15234
Failed Chunks:         0
Embedding Dimension:   384
Database Size:         85.2 MB
Unique Sources:        1067

Date Range Coverage:
  - 1961-2006
  - 2007-2026
================================================================================
```

---

## Advanced Usage

### Update Existing Collection

```python
# Connect to existing database
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection("bean_research_docs")

# Add new chunks
collection.add(
    embeddings=new_embeddings.tolist(),
    documents=new_texts,
    metadatas=new_metadata,
    ids=[f"chunk_{i}" for i in range(15234, 15234 + len(new_texts))]
)
```

### Delete Collection

```python
client.delete_collection("bean_research_docs")
```

### List Collections

```python
collections = client.list_collections()
for col in collections:
    print(f"Collection: {col.name}")
    print(f"  Count: {col.count()}")
```

### Custom Embedding Function

```python
from chromadb.utils import embedding_functions

# Use sentence-transformers
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_func
)
```

---

## Integration with QA System

### LangChain Integration

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector store
vectorstore = Chroma(
    persist_directory="vector_db",
    collection_name="bean_research_docs",
    embedding_function=embeddings
)

# Use as retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# Get relevant documents
docs = retriever.get_relevant_documents("What are bean breeding techniques?")
```

### Build QA Chain

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# Ask questions
answer = qa_chain.run("What are the main bean breeding techniques?")
print(answer)
```

---

## Performance

### Processing Time

**For 15,000 chunks**:
- Building database: ~10-30 seconds
- Batch insertion (1000 chunks/batch): ~1-2 seconds per batch

### Database Size

| Chunks | Embedding Dim | Disk Size |
|--------|---------------|-----------|
| 1,000 | 384 | ~5-10 MB |
| 5,000 | 384 | ~25-50 MB |
| 15,000 | 384 | ~75-100 MB |
| 50,000 | 384 | ~250-300 MB |

### Query Speed

- Simple search: <100ms
- Filtered search: <200ms
- Batch queries (10): <500ms

---

## Troubleshooting

### Issue: Collection Already Exists

**Symptoms**: `Collection 'bean_research_docs' already exists`

**Solution**: The script automatically deletes and recreates. To keep existing:
```python
# Modify script to get instead of create
collection = client.get_collection(name="bean_research_docs")
```

### Issue: Import Error

**Symptoms**: `ImportError: No module named 'chromadb'`

**Solution**:
```bash
pip install chromadb
```

### Issue: Metadata Not Filterable

**Symptoms**: `where` filter doesn't work

**Solution**: Ensure metadata fields are strings or numbers:
```python
# Good
metadata = {"year_range": "2007-2026", "page_number": 5}

# Bad
metadata = {"year_range": ["2007", "2026"]}  # Lists not supported
```

### Issue: Database Size Growing

**Symptoms**: Database larger than expected

**Solution**:
- ChromaDB stores embeddings and documents
- To save space, only store references to documents
- Use smaller batch sizes during development

---

## Best Practices

1. **Test First**: Always run `test_vector_store.py` before full build
2. **Backup Database**: Copy `vector_db/` periodically
3. **Use Batch Insert**: Don't add one chunk at a time
4. **Normalize Embeddings**: Ensure embeddings are L2-normalized
5. **Validate Metadata**: Check metadata structure before adding
6. **Monitor Size**: Large databases may slow queries
7. **Use Filters**: Narrow searches with `where` clause
8. **Version Collections**: Use different names for different versions

---

## Metadata Schema

Each chunk has this metadata structure:

```python
{
    "source_file": str,        # PDF filename
    "year_range": str,         # "1961-2006" or "2007-2026"
    "chunk_id": int,           # Sequential ID within document
    "page_number": int,        # Page number in PDF
    "embedding_index": int     # Index in embeddings array
}
```

### Filterable Fields

All metadata fields are filterable:

```python
# Filter by year
where={"year_range": "2007-2026"}

# Filter by page
where={"page_number": 5}

# Multiple filters (AND)
where={"year_range": "2007-2026", "page_number": 5}

# Greater than/less than
where={"chunk_id": {"$gt": 10}}
```

---

## Next Steps

After building the vector store:

### 1. Build Simple Search Interface

```python
def search_documents(query, n_results=5, year_filter=None):
    """Search bean research documents."""

    where_clause = {"year_range": year_filter} if year_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )

    return results
```

### 2. Create QA System

- Use LangChain for orchestration
- Integrate with LLM (GPT, LLaMA, etc.)
- Build retrieval-augmented generation

### 3. Deploy Web Interface

- Streamlit for simple UI
- FastAPI for REST API
- Gradio for demo

---

## Files Created

| File | Description | Size |
|------|-------------|------|
| `src/build_vector_store.py` | Main builder script | ~20KB |
| `src/test_vector_store.py` | Test script | ~5KB |
| `jobs/build_vector_store.sb` | SLURM job | ~2KB |
| `vector_db/` | ChromaDB database | ~75-100MB |
| `vector_db/database_statistics.json` | Statistics | ~5KB |

---

## References

- **ChromaDB**: https://docs.trychroma.com/
- **LangChain**: https://python.langchain.com/
- **Sentence Transformers**: https://www.sbert.net/

---

**Ready to build?**

```bash
# Test first
python3 src/test_vector_store.py

# Build full database
python3 src/build_vector_store.py

# Or submit job
sbatch jobs/build_vector_store.sb
```
