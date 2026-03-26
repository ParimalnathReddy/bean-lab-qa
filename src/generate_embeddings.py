#!/usr/bin/env python3
"""
================================================================================
EMBEDDING GENERATOR FOR LLM-BASED DOCUMENT QA SYSTEM
================================================================================

Purpose:
    Generate dense vector embeddings from processed text chunks using
    sentence-transformers on V100 GPU.

Input:
    - data/processed_chunks.json (from PDF processor)

Output:
    - data/embeddings.npy (numpy array of embeddings)
    - data/embeddings_metadata.json (metadata and validation info)

Model:
    - all-MiniLM-L6-v2 (384-dimensional embeddings)
    - Optimized for semantic similarity
    - Fast and efficient

GPU Optimization:
    - Batch processing (batch_size=32)
    - CUDA acceleration on V100
    - Mixed precision support
    - Memory-efficient batching

Author: MSU HPCC Bean Lab
Date: 2026-03-21
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import traceback

# Deep learning and embeddings
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error: Missing dependencies: {e}")
    print("Install with: pip install torch sentence-transformers")
    exit(1)

# Progress tracking
from tqdm import tqdm


# ============================================================================
# MAIN CLASS - EmbeddingGenerator
# ============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings from text chunks using sentence-transformers.

    Optimized for:
    - V100 GPU (32GB VRAM)
    - Batch processing
    - Memory efficiency
    - Progress tracking
    """

    def __init__(
        self,
        input_file: str,                          # Path to processed_chunks.json
        output_embeddings: str,                   # Path to save embeddings.npy
        output_metadata: str,                     # Path to save metadata.json
        log_file: str,                            # Path to log file
        model_name: str = "all-MiniLM-L6-v2",    # Sentence transformer model
        batch_size: int = 32,                     # Batch size for GPU
        device: str = None                        # Device (auto-detect if None)
    ):
        """
        Initialize the embedding generator.

        Args:
            input_file: Path to processed_chunks.json
            output_embeddings: Path to save embeddings array
            output_metadata: Path to save metadata
            log_file: Path to log file
            model_name: Sentence-transformers model name
            batch_size: Batch size for GPU processing
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """

        # ====================================================================
        # Store configuration parameters
        # ====================================================================

        self.input_file = Path(input_file)
        self.output_embeddings = Path(output_embeddings)
        self.output_metadata = Path(output_metadata)
        self.log_file = Path(log_file)
        self.model_name = model_name
        self.batch_size = batch_size

        # ====================================================================
        # Detect and set device (GPU or CPU)
        # ====================================================================

        if device is None:
            # Auto-detect: use CUDA if available, else CPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # ====================================================================
        # Set up logging
        # ====================================================================

        self._setup_logging()

        # ====================================================================
        # Initialize model (will be loaded later)
        # ====================================================================

        self.model = None
        self.embedding_dim = None  # Will be set when model loads

        # ====================================================================
        # Initialize statistics tracking
        # ====================================================================

        self.stats = {
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "embedding_dimension": 0,
            "processing_time_seconds": 0,
            "chunks_per_second": 0,
            "gpu_used": False,
            "gpu_name": None,
            "errors": []
        }

    def _setup_logging(self):
        """
        Configure logging to write to both file and console.
        """

        # Create log directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),  # Write to file
                logging.StreamHandler()               # Print to console
            ]
        )

        self.logger = logging.getLogger(__name__)

        # Log initialization
        self.logger.info("="*80)
        self.logger.info("Embedding Generator Initialized")
        self.logger.info(f"Input file: {self.input_file}")
        self.logger.info(f"Output embeddings: {self.output_embeddings}")
        self.logger.info(f"Output metadata: {self.output_metadata}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("="*80)

    def check_gpu(self):
        """
        Check GPU availability and log information.

        Returns:
            bool: True if GPU is available and will be used
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("GPU Check")
        self.logger.info("="*80)

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        self.logger.info(f"CUDA available: {cuda_available}")

        if cuda_available:
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"Number of GPUs: {gpu_count}")

            # Get current GPU name
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU 0: {gpu_name}")

            # Get GPU memory info
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU Memory: {gpu_mem_total:.2f} GB")

            # Store in stats
            self.stats["gpu_used"] = True
            self.stats["gpu_name"] = gpu_name

            self.logger.info("="*80 + "\n")
            return True
        else:
            self.logger.warning("No GPU available, will use CPU (slower)")
            self.logger.warning("To use GPU, ensure CUDA is installed and accessible")
            self.stats["gpu_used"] = False
            self.logger.info("="*80 + "\n")
            return False

    def load_model(self):
        """
        Load sentence-transformers model.

        Model Details:
            - all-MiniLM-L6-v2: 384-dimensional embeddings
            - Fast and efficient
            - Good for semantic similarity
        """

        self.logger.info("Loading sentence-transformers model...")
        self.logger.info(f"Model: {self.model_name}")

        try:
            # Load model
            # SentenceTransformer will automatically download if not cached
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.stats["embedding_dimension"] = self.embedding_dim

            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Embedding dimension: {self.embedding_dim}")
            self.logger.info(f"Device: {self.device}")

            # If using GPU, log memory usage
            if self.device == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                mem_reserved = torch.cuda.memory_reserved(0) / 1e9
                self.logger.info(f"GPU Memory allocated: {mem_allocated:.2f} GB")
                self.logger.info(f"GPU Memory reserved: {mem_reserved:.2f} GB")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def load_chunks(self) -> List[Dict]:
        """
        Load processed chunks from JSON file.

        Returns:
            List[Dict]: List of chunk dictionaries
        """

        self.logger.info(f"\nLoading chunks from: {self.input_file}")

        # Check if file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        try:
            # Load JSON file
            with open(self.input_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Validate chunks
            if not isinstance(chunks, list):
                raise ValueError("Input file must contain a list of chunks")

            if len(chunks) == 0:
                raise ValueError("Input file contains no chunks")

            # Log statistics
            self.stats["total_chunks"] = len(chunks)
            self.logger.info(f"Loaded {len(chunks)} chunks")

            # Log sample chunk structure
            if chunks:
                sample_keys = list(chunks[0].keys())
                self.logger.info(f"Chunk structure: {sample_keys}")

            return chunks

        except Exception as e:
            self.logger.error(f"Failed to load chunks: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def generate_embeddings(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries (must have 'text' field)

        Returns:
            Tuple[np.ndarray, List[Dict]]:
                - embeddings: numpy array of shape (n_chunks, embedding_dim)
                - metadata_list: list of metadata dicts for each chunk
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Generating Embeddings")
        self.logger.info("="*80)

        # ====================================================================
        # Extract texts and metadata
        # ====================================================================

        texts = []
        metadata_list = []

        for i, chunk in enumerate(chunks):
            # Extract text
            if 'text' not in chunk:
                self.logger.warning(f"Chunk {i} missing 'text' field, skipping")
                continue

            texts.append(chunk['text'])

            # Create metadata (exclude text to save space)
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            metadata['embedding_index'] = len(metadata_list)  # Add index
            metadata_list.append(metadata)

        self.logger.info(f"Texts extracted: {len(texts)}")

        # ====================================================================
        # Generate embeddings in batches
        # ====================================================================

        embeddings_list = []

        # Calculate number of batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        self.logger.info(f"Processing in {num_batches} batches of {self.batch_size}")
        self.logger.info(f"Total texts to embed: {len(texts)}")

        # Track timing
        import time
        start_time = time.time()

        # Process in batches with progress bar
        with tqdm(total=len(texts), desc="Generating embeddings", unit="chunk") as pbar:
            for i in range(0, len(texts), self.batch_size):
                # Get batch
                batch_texts = texts[i:i + self.batch_size]

                try:
                    # Generate embeddings for batch
                    # show_progress_bar=False because we have our own tqdm
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,  # Return as numpy array
                        normalize_embeddings=True  # L2 normalization for cosine similarity
                    )

                    # Add to list
                    embeddings_list.append(batch_embeddings)

                    # Update successful count
                    self.stats["successful_embeddings"] += len(batch_texts)

                    # Update progress bar
                    pbar.update(len(batch_texts))

                except Exception as e:
                    self.logger.error(f"Error processing batch {i//self.batch_size}: {e}")
                    self.stats["failed_embeddings"] += len(batch_texts)
                    self.stats["errors"].append({
                        "batch": i // self.batch_size,
                        "error": str(e)
                    })
                    # Continue with next batch
                    pbar.update(len(batch_texts))

        # ====================================================================
        # Combine all batches into single array
        # ====================================================================

        if not embeddings_list:
            raise RuntimeError("No embeddings were generated successfully")

        embeddings = np.vstack(embeddings_list)

        # ====================================================================
        # Calculate statistics
        # ====================================================================

        end_time = time.time()
        processing_time = end_time - start_time

        self.stats["processing_time_seconds"] = processing_time
        self.stats["chunks_per_second"] = len(texts) / processing_time if processing_time > 0 else 0

        self.logger.info(f"\nEmbeddings generated: {embeddings.shape[0]}")
        self.logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Speed: {self.stats['chunks_per_second']:.2f} chunks/second")

        if self.device == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(0) / 1e9
            self.logger.info(f"Peak GPU memory: {max_mem:.2f} GB")

        self.logger.info("="*80)

        return embeddings, metadata_list

    def validate_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> bool:
        """
        Validate generated embeddings.

        Checks:
        1. Shape is correct
        2. No NaN or Inf values
        3. Embeddings are normalized (if using cosine similarity)
        4. Metadata matches embeddings count

        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries

        Returns:
            bool: True if validation passes
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Validating Embeddings")
        self.logger.info("="*80)

        validation_passed = True

        # ====================================================================
        # Check 1: Shape
        # ====================================================================

        expected_dim = self.embedding_dim
        actual_dim = embeddings.shape[1]

        if actual_dim != expected_dim:
            self.logger.error(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")
            validation_passed = False
        else:
            self.logger.info(f"✓ Dimension check passed: {actual_dim}")

        # ====================================================================
        # Check 2: NaN and Inf values
        # ====================================================================

        nan_count = np.isnan(embeddings).sum()
        inf_count = np.isinf(embeddings).sum()

        if nan_count > 0:
            self.logger.error(f"Found {nan_count} NaN values in embeddings")
            validation_passed = False
        else:
            self.logger.info("✓ No NaN values found")

        if inf_count > 0:
            self.logger.error(f"Found {inf_count} Inf values in embeddings")
            validation_passed = False
        else:
            self.logger.info("✓ No Inf values found")

        # ====================================================================
        # Check 3: Normalization (if using cosine similarity)
        # ====================================================================

        # Calculate norms of each embedding vector
        norms = np.linalg.norm(embeddings, axis=1)

        # Check if normalized (norm should be close to 1.0)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        self.logger.info(f"Embedding norms - mean: {mean_norm:.6f}, std: {std_norm:.6f}")

        # If mean is close to 1.0, embeddings are normalized
        if abs(mean_norm - 1.0) < 0.01:
            self.logger.info("✓ Embeddings are normalized")
        else:
            self.logger.warning(f"⚠ Embeddings may not be normalized (mean norm: {mean_norm:.6f})")

        # ====================================================================
        # Check 4: Count matches
        # ====================================================================

        if len(embeddings) != len(metadata):
            self.logger.error(f"Count mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata")
            validation_passed = False
        else:
            self.logger.info(f"✓ Count match: {len(embeddings)} embeddings and metadata entries")

        # ====================================================================
        # Summary
        # ====================================================================

        if validation_passed:
            self.logger.info("\n✓ All validation checks passed!")
        else:
            self.logger.error("\n✗ Validation failed - check errors above")

        self.logger.info("="*80)

        return validation_passed

    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Save embeddings and metadata to files.

        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Saving Embeddings")
        self.logger.info("="*80)

        # ====================================================================
        # Create output directory if needed
        # ====================================================================

        self.output_embeddings.parent.mkdir(parents=True, exist_ok=True)

        # ====================================================================
        # Save embeddings as numpy array
        # ====================================================================

        try:
            np.save(self.output_embeddings, embeddings)

            # Check file size
            file_size_mb = self.output_embeddings.stat().st_size / (1024 * 1024)

            self.logger.info(f"✓ Embeddings saved to: {self.output_embeddings}")
            self.logger.info(f"  File size: {file_size_mb:.2f} MB")
            self.logger.info(f"  Shape: {embeddings.shape}")

        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
            raise

        # ====================================================================
        # Save metadata as JSON
        # ====================================================================

        # Create comprehensive metadata file
        metadata_output = {
            "generation_date": datetime.now().isoformat(),
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "statistics": self.stats,
            "configuration": {
                "batch_size": self.batch_size,
                "device": self.device,
                "input_file": str(self.input_file),
                "output_file": str(self.output_embeddings)
            },
            "chunk_metadata": metadata  # Individual chunk metadata
        }

        try:
            with open(self.output_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata_output, f, indent=2, ensure_ascii=False)

            # Check file size
            file_size_mb = self.output_metadata.stat().st_size / (1024 * 1024)

            self.logger.info(f"✓ Metadata saved to: {self.output_metadata}")
            self.logger.info(f"  File size: {file_size_mb:.2f} MB")

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise

        self.logger.info("="*80)

    def print_statistics(self):
        """
        Print processing statistics.
        """

        print("\n" + "="*80)
        print("EMBEDDING GENERATION STATISTICS")
        print("="*80)
        print(f"Model:                   {self.model_name}")
        print(f"Embedding dimension:     {self.stats['embedding_dimension']}")
        print(f"Total chunks:            {self.stats['total_chunks']}")
        print(f"Successful embeddings:   {self.stats['successful_embeddings']}")
        print(f"Failed embeddings:       {self.stats['failed_embeddings']}")
        print(f"Processing time:         {self.stats['processing_time_seconds']:.2f} seconds")
        print(f"Speed:                   {self.stats['chunks_per_second']:.2f} chunks/second")
        print(f"GPU used:                {self.stats['gpu_used']}")

        if self.stats['gpu_used']:
            print(f"GPU name:                {self.stats['gpu_name']}")

        if self.stats['errors']:
            print(f"\nErrors encountered:      {len(self.stats['errors'])}")
            print("Check log file for details.")

        print("="*80 + "\n")

    def run(self) -> bool:
        """
        Main pipeline to generate embeddings.

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            # Step 1: Check GPU
            self.check_gpu()

            # Step 2: Load model
            self.load_model()

            # Step 3: Load chunks
            chunks = self.load_chunks()

            # Step 4: Generate embeddings
            embeddings, metadata = self.generate_embeddings(chunks)

            # Step 5: Validate embeddings
            validation_passed = self.validate_embeddings(embeddings, metadata)

            if not validation_passed:
                self.logger.error("Validation failed! Review errors above.")
                return False

            # Step 6: Save results
            self.save_embeddings(embeddings, metadata)

            # Step 7: Print statistics
            self.print_statistics()

            self.logger.info("✓ Embedding generation complete!")

            return True

        except Exception as e:
            self.logger.error(f"Fatal error in embedding generation: {e}")
            self.logger.debug(traceback.format_exc())
            return False


# ============================================================================
# MAIN FUNCTION - Command-line interface
# ============================================================================

def main():
    """
    Main entry point for command-line usage.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings from processed text chunks"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/processed_chunks.json",
        help="Input JSON file with chunks"
    )

    parser.add_argument(
        "--output-embeddings",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/embeddings.npy",
        help="Output file for embeddings (.npy)"
    )

    parser.add_argument(
        "--output-metadata",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/embeddings_metadata.json",
        help="Output file for metadata (.json)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/logs/embeddings.log",
        help="Log file path"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU processing"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help="Device to use (auto for auto-detect)"
    )

    args = parser.parse_args()

    # Convert 'auto' to None for auto-detection
    device = None if args.device == 'auto' else args.device

    # Create generator
    generator = EmbeddingGenerator(
        input_file=args.input_file,
        output_embeddings=args.output_embeddings,
        output_metadata=args.output_metadata,
        log_file=args.log_file,
        model_name=args.model,
        batch_size=args.batch_size,
        device=device
    )

    # Run embedding generation
    success = generator.run()

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
