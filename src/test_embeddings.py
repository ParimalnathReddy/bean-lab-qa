#!/usr/bin/env python3
"""
Test script for embedding generation
Tests on a small sample before processing all chunks
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_embeddings import EmbeddingGenerator


def create_test_chunks():
    """Create a small test dataset."""
    test_chunks = [
        {
            "text": "Bean breeding programs have shown significant improvements in yield.",
            "source_file": "test1.pdf",
            "year_range": "1961-2006",
            "chunk_id": 0,
            "page_number": 1,
            "token_count": 12
        },
        {
            "text": "Drought resistance is crucial for bean cultivation in arid regions.",
            "source_file": "test2.pdf",
            "year_range": "2007-2026",
            "chunk_id": 0,
            "page_number": 1,
            "token_count": 11
        },
        {
            "text": "Genetic modification techniques have enabled better disease resistance.",
            "source_file": "test3.pdf",
            "year_range": "2007-2026",
            "chunk_id": 0,
            "page_number": 1,
            "token_count": 10
        },
        {
            "text": "Climate adaptation strategies are essential for sustainable agriculture.",
            "source_file": "test4.pdf",
            "year_range": "2007-2026",
            "chunk_id": 0,
            "page_number": 1,
            "token_count": 10
        },
        {
            "text": "Photosynthesis efficiency in beans affects overall crop productivity.",
            "source_file": "test5.pdf",
            "year_range": "1961-2006",
            "chunk_id": 0,
            "page_number": 1,
            "token_count": 10
        }
    ]
    return test_chunks


def test_embeddings():
    """Test embedding generation on small sample."""

    print("="*80)
    print("EMBEDDING GENERATION TEST")
    print("="*80)
    print()

    # Setup paths
    base_dir = Path("/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa")
    test_dir = base_dir / "data" / "test_embeddings"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_input = test_dir / "test_chunks.json"
    test_embeddings = test_dir / "test_embeddings.npy"
    test_metadata = test_dir / "test_metadata.json"
    test_log = base_dir / "logs" / "test_embeddings.log"

    print(f"Test directory: {test_dir}")
    print()

    # Create test chunks
    print("Creating test dataset...")
    test_chunks = create_test_chunks()
    print(f"Created {len(test_chunks)} test chunks")
    print()

    # Save test chunks
    with open(test_input, 'w') as f:
        json.dump(test_chunks, f, indent=2)
    print(f"✓ Test chunks saved to: {test_input}")
    print()

    # Print sample chunk
    print("Sample chunk:")
    print(json.dumps(test_chunks[0], indent=2))
    print()

    # Check dependencies
    print("Checking dependencies...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        print("Install with: pip install torch")
        return

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers available")
    except ImportError as e:
        print(f"✗ sentence-transformers not available: {e}")
        print("Install with: pip install sentence-transformers")
        return

    print()

    # Create generator
    print("Initializing embedding generator...")
    generator = EmbeddingGenerator(
        input_file=str(test_input),
        output_embeddings=str(test_embeddings),
        output_metadata=str(test_metadata),
        log_file=str(test_log),
        model_name="all-MiniLM-L6-v2",
        batch_size=2,  # Small batch for testing
        device=None  # Auto-detect
    )

    print()
    print("-"*80)

    # Run generation
    success = generator.run()

    print("-"*80)
    print()

    if success:
        print("✓ Test PASSED!")
        print()

        # Load and display results
        print("Loading results...")
        embeddings = np.load(test_embeddings)
        with open(test_metadata) as f:
            metadata = json.load(f)

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Expected: ({len(test_chunks)}, 384)")
        print()

        print("Sample embedding (first 10 values):")
        print(embeddings[0][:10])
        print()

        print("Statistics:")
        stats = metadata['statistics']
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Successful: {stats['successful_embeddings']}")
        print(f"  Failed: {stats['failed_embeddings']}")
        print(f"  Time: {stats['processing_time_seconds']:.2f}s")
        print(f"  Speed: {stats['chunks_per_second']:.2f} chunks/s")
        print(f"  GPU used: {stats['gpu_used']}")
        print()

        # Test similarity
        print("Testing semantic similarity...")
        from numpy.linalg import norm

        # Compute cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))

        # Compare similar texts
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        print(f"Similarity (bean breeding vs drought): {sim_01:.4f}")

        sim_04 = cosine_similarity(embeddings[0], embeddings[4])
        print(f"Similarity (bean breeding vs photosynthesis): {sim_04:.4f}")

        print()
        print("Output files:")
        print(f"  Embeddings: {test_embeddings}")
        print(f"  Metadata: {test_metadata}")
        print(f"  Log: {test_log}")

    else:
        print("✗ Test FAILED!")
        print("Check log file for errors:")
        print(f"  {test_log}")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print()

    if success:
        print("If the test looks good, process all chunks with:")
        print(f"  python3 {base_dir}/src/generate_embeddings.py")
        print()
        print("Or submit as SLURM job:")
        print(f"  sbatch {base_dir}/jobs/generate_embeddings.sb")
    else:
        print("Fix errors before processing all chunks.")

    print()


if __name__ == "__main__":
    test_embeddings()
