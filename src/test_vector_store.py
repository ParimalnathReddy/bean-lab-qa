#!/usr/bin/env python3
"""
Test script for vector store builder
Tests ChromaDB functionality with sample data
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_vector_store():
    """Test vector store creation with sample data."""

    print("="*80)
    print("VECTOR STORE TEST")
    print("="*80)
    print()

    # Check dependencies
    print("Checking dependencies...")
    try:
        import chromadb
        print(f"✓ ChromaDB installed")
    except ImportError:
        print("✗ ChromaDB not installed")
        print("Install with: pip install chromadb")
        return False

    print()

    # Setup paths
    base_dir = Path("/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa")
    test_dir = base_dir / "data" / "test_vector_store"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test data
    print("Creating test data...")

    # Sample embeddings (5 chunks, 384 dimensions)
    np.random.seed(42)
    test_embeddings = np.random.randn(5, 384).astype(np.float32)
    # Normalize for cosine similarity
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    # Sample metadata
    test_metadata = {
        "chunk_metadata": [
            {"source_file": "test1.pdf", "year_range": "1961-2006", "chunk_id": 0, "page_number": 1},
            {"source_file": "test2.pdf", "year_range": "2007-2026", "chunk_id": 0, "page_number": 1},
            {"source_file": "test3.pdf", "year_range": "2007-2026", "chunk_id": 0, "page_number": 2},
            {"source_file": "test4.pdf", "year_range": "1961-2006", "chunk_id": 1, "page_number": 3},
            {"source_file": "test5.pdf", "year_range": "2007-2026", "chunk_id": 0, "page_number": 1}
        ]
    }

    # Sample chunks
    test_chunks = [
        {"text": "Bean breeding programs have shown significant improvements in yield.", **test_metadata["chunk_metadata"][0]},
        {"text": "Drought resistance is crucial for bean cultivation in arid regions.", **test_metadata["chunk_metadata"][1]},
        {"text": "Genetic modification techniques have enabled better disease resistance.", **test_metadata["chunk_metadata"][2]},
        {"text": "Climate adaptation strategies are essential for sustainable agriculture.", **test_metadata["chunk_metadata"][3]},
        {"text": "Photosynthesis efficiency in beans affects overall crop productivity.", **test_metadata["chunk_metadata"][4]}
    ]

    # Save test files
    test_emb_file = test_dir / "test_embeddings.npy"
    test_meta_file = test_dir / "test_metadata.json"
    test_chunks_file = test_dir / "test_chunks.json"

    np.save(test_emb_file, test_embeddings)
    with open(test_meta_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    with open(test_chunks_file, 'w') as f:
        json.dump(test_chunks, f, indent=2)

    print(f"✓ Test data created:")
    print(f"  Embeddings: {test_emb_file}")
    print(f"  Metadata: {test_meta_file}")
    print(f"  Chunks: {test_chunks_file}")
    print()

    # Build vector store
    print("-"*80)
    print("Building vector store...")
    print("-"*80)

    from build_vector_store import VectorStoreBuilder

    builder = VectorStoreBuilder(
        embeddings_file=str(test_emb_file),
        metadata_file=str(test_meta_file),
        chunks_file=str(test_chunks_file),
        db_path=str(test_dir / "test_db"),
        log_file=str(base_dir / "logs" / "test_vector_store.log"),
        collection_name="test_collection",
        batch_size=10
    )

    success = builder.run()

    print("-"*80)
    print()

    if success:
        print("✓ Vector store built successfully!")
        print()

        # Test queries
        print("Testing queries...")
        print()

        collection = builder.collection

        # Test 1: Semantic search
        print("1. Semantic search: 'bean breeding'")
        results = collection.query(
            query_texts=["bean breeding"],
            n_results=2
        )
        print(f"   Found {len(results['ids'][0])} results")
        for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            print(f"   Result {i+1}: {doc_id} - {metadata.get('source_file')}")
        print()

        # Test 2: Filtered search
        print("2. Filtered search: 'drought' WHERE year_range='2007-2026'")
        results = collection.query(
            query_texts=["drought resistance"],
            n_results=2,
            where={"year_range": "2007-2026"}
        )
        print(f"   Found {len(results['ids'][0])} results from 2007-2026")
        for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            print(f"   Result {i+1}: {doc_id} - {metadata.get('year_range')}")
        print()

        # Test 3: Count
        print("3. Total documents in collection:")
        count = collection.count()
        print(f"   {count} documents")
        print()

        print("="*80)
        print("TEST PASSED!")
        print("="*80)
        print()
        print("If test looks good, build full vector store with:")
        print(f"  python3 {base_dir}/src/build_vector_store.py")
        print()
        return True

    else:
        print("="*80)
        print("TEST FAILED!")
        print("="*80)
        print("Check log file for errors:")
        print(f"  {base_dir}/logs/test_vector_store.log")
        print()
        return False


if __name__ == "__main__":
    success = test_vector_store()
    exit(0 if success else 1)
