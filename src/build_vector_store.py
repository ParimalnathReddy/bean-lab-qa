#!/usr/bin/env python3
"""
================================================================================
VECTOR STORE BUILDER FOR LLM-BASED DOCUMENT QA SYSTEM
================================================================================

Purpose:
    Build a persistent ChromaDB vector store from generated embeddings.
    Enables semantic search and retrieval for QA system.

Input:
    - data/embeddings.npy (vector embeddings)
    - data/embeddings_metadata.json (chunk metadata)
    - data/processed_chunks.json (original text chunks)

Output:
    - vector_db/ (persistent ChromaDB database)

Features:
    - Persistent storage
    - Metadata filtering (by year_range, source, etc.)
    - Semantic search
    - Validation queries
    - Statistics reporting

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
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import traceback
from collections import Counter

# ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"Error: ChromaDB not installed: {e}")
    print("Install with: pip install chromadb")
    exit(1)

# Progress tracking
from tqdm import tqdm


# ============================================================================
# MAIN CLASS - VectorStoreBuilder
# ============================================================================

class VectorStoreBuilder:
    """
    Build and manage ChromaDB vector store for document QA.

    Features:
    - Persistent storage
    - Metadata filtering
    - Batch insertion
    - Validation queries
    - Statistics tracking
    """

    def __init__(
        self,
        embeddings_file: str,           # Path to embeddings.npy
        metadata_file: str,              # Path to embeddings_metadata.json
        chunks_file: str,                # Path to processed_chunks.json
        db_path: str,                    # Path to vector database directory
        log_file: str,                   # Path to log file
        collection_name: str = "bean_research_docs",  # Collection name
        batch_size: int = 1000           # Batch size for insertion
    ):
        """
        Initialize vector store builder.

        Args:
            embeddings_file: Path to embeddings numpy file
            metadata_file: Path to metadata JSON file
            chunks_file: Path to processed chunks JSON file
            db_path: Path to database directory
            log_file: Path to log file
            collection_name: Name for ChromaDB collection
            batch_size: Batch size for inserting vectors
        """

        # ====================================================================
        # Store configuration parameters
        # ====================================================================

        self.embeddings_file = Path(embeddings_file)
        self.metadata_file = Path(metadata_file)
        self.chunks_file = Path(chunks_file)
        self.db_path = Path(db_path)
        self.log_file = Path(log_file)
        self.collection_name = collection_name
        self.batch_size = batch_size

        # ====================================================================
        # Set up logging
        # ====================================================================

        self._setup_logging()

        # ====================================================================
        # Initialize ChromaDB client and collection (will be set later)
        # ====================================================================

        self.client = None
        self.collection = None

        # ====================================================================
        # Initialize statistics tracking
        # ====================================================================

        self.stats = {
            "total_chunks": 0,
            "chunks_added": 0,
            "failed_chunks": 0,
            "database_size_mb": 0,
            "year_ranges": [],
            "unique_sources": 0,
            "embedding_dimension": 0,
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
        self.logger.info("Vector Store Builder Initialized")
        self.logger.info(f"Embeddings file: {self.embeddings_file}")
        self.logger.info(f"Metadata file: {self.metadata_file}")
        self.logger.info(f"Chunks file: {self.chunks_file}")
        self.logger.info(f"Database path: {self.db_path}")
        self.logger.info(f"Collection name: {self.collection_name}")
        self.logger.info("="*80)

    def load_data(self) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """
        Load embeddings, metadata, and text chunks.

        Returns:
            Tuple containing:
            - embeddings: numpy array of embeddings
            - metadata_list: list of metadata dictionaries
            - texts: list of text strings
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Loading Data")
        self.logger.info("="*80)

        # ====================================================================
        # Load embeddings
        # ====================================================================

        self.logger.info(f"Loading embeddings from: {self.embeddings_file}")

        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

        try:
            embeddings = np.load(self.embeddings_file)
            self.logger.info(f"✓ Loaded embeddings: shape {embeddings.shape}")
            self.stats["embedding_dimension"] = embeddings.shape[1]
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            raise

        # ====================================================================
        # Load metadata
        # ====================================================================

        self.logger.info(f"Loading metadata from: {self.metadata_file}")

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata_obj = json.load(f)

            # Extract chunk metadata
            metadata_list = metadata_obj.get('chunk_metadata', [])
            self.logger.info(f"✓ Loaded metadata: {len(metadata_list)} entries")

        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise

        # ====================================================================
        # Load text chunks
        # ====================================================================

        self.logger.info(f"Loading chunks from: {self.chunks_file}")

        if not self.chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")

        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Extract text from each chunk
            texts = [chunk['text'] for chunk in chunks]
            self.logger.info(f"✓ Loaded texts: {len(texts)} chunks")

        except Exception as e:
            self.logger.error(f"Failed to load chunks: {e}")
            raise

        # ====================================================================
        # Validate counts match
        # ====================================================================

        if not (len(embeddings) == len(metadata_list) == len(texts)):
            self.logger.error(
                f"Count mismatch: {len(embeddings)} embeddings, "
                f"{len(metadata_list)} metadata, {len(texts)} texts"
            )
            raise ValueError("Data count mismatch")

        self.stats["total_chunks"] = len(embeddings)
        self.logger.info(f"\n✓ Data validation passed: {self.stats['total_chunks']} chunks")
        self.logger.info("="*80)

        return embeddings, metadata_list, texts

    def create_database(self):
        """
        Create or connect to ChromaDB database.
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Creating ChromaDB Database")
        self.logger.info("="*80)

        # ====================================================================
        # Create database directory
        # ====================================================================

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Database directory: {self.db_path}")

        # ====================================================================
        # Create persistent ChromaDB client
        # ====================================================================

        try:
            # Create client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry
                    allow_reset=True             # Allow reset for development
                )
            )

            self.logger.info("✓ ChromaDB client created")

        except Exception as e:
            self.logger.error(f"Failed to create ChromaDB client: {e}")
            raise

        # ====================================================================
        # Create or get collection
        # ====================================================================

        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                self.logger.warning(f"⚠ Collection '{self.collection_name}' already exists")
                self.logger.warning("Deleting existing collection...")

                # Delete and recreate for fresh start
                self.client.delete_collection(name=self.collection_name)

            except Exception:
                # Collection doesn't exist, which is fine
                pass

            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Bean research documents from 1961-2026",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "creation_date": datetime.now().isoformat()
                }
            )

            self.logger.info(f"✓ Collection '{self.collection_name}' created")

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise

        self.logger.info("="*80)

    def add_to_database(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict],
        texts: List[str]
    ):
        """
        Add embeddings, metadata, and texts to ChromaDB collection.

        Args:
            embeddings: Numpy array of embeddings
            metadata_list: List of metadata dictionaries
            texts: List of text strings
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Adding Data to Database")
        self.logger.info("="*80)

        total_chunks = len(embeddings)
        num_batches = (total_chunks + self.batch_size - 1) // self.batch_size

        self.logger.info(f"Total chunks: {total_chunks}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of batches: {num_batches}")

        # ====================================================================
        # Process in batches with progress bar
        # ====================================================================

        with tqdm(total=total_chunks, desc="Adding to database", unit="chunk") as pbar:
            for batch_idx in range(num_batches):
                # Calculate batch range
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_chunks)

                # Get batch data
                batch_embeddings = embeddings[start_idx:end_idx]
                batch_metadata = metadata_list[start_idx:end_idx]
                batch_texts = texts[start_idx:end_idx]

                # Create IDs for this batch
                batch_ids = [f"chunk_{i}" for i in range(start_idx, end_idx)]

                try:
                    # Add batch to collection
                    self.collection.add(
                        embeddings=batch_embeddings.tolist(),  # Convert to list
                        metadatas=batch_metadata,               # Metadata dicts
                        documents=batch_texts,                  # Text content
                        ids=batch_ids                           # Unique IDs
                    )

                    # Update statistics
                    self.stats["chunks_added"] += len(batch_texts)

                except Exception as e:
                    self.logger.error(f"Error adding batch {batch_idx}: {e}")
                    self.stats["failed_chunks"] += len(batch_texts)
                    self.stats["errors"].append({
                        "batch": batch_idx,
                        "error": str(e)
                    })

                # Update progress bar
                pbar.update(len(batch_texts))

        # ====================================================================
        # Log completion
        # ====================================================================

        self.logger.info(f"\n✓ Added {self.stats['chunks_added']} chunks to database")

        if self.stats["failed_chunks"] > 0:
            self.logger.warning(f"⚠ Failed to add {self.stats['failed_chunks']} chunks")

        self.logger.info("="*80)

    def analyze_metadata(self, metadata_list: List[Dict]):
        """
        Analyze metadata to extract statistics.

        Args:
            metadata_list: List of metadata dictionaries
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Analyzing Metadata")
        self.logger.info("="*80)

        # ====================================================================
        # Extract unique values
        # ====================================================================

        # Year ranges
        year_ranges = [m.get('year_range', 'unknown') for m in metadata_list]
        unique_years = set(year_ranges)
        year_counts = Counter(year_ranges)

        self.stats["year_ranges"] = sorted(unique_years)

        self.logger.info("\nYear Range Distribution:")
        for year, count in sorted(year_counts.items()):
            percentage = (count / len(metadata_list)) * 100
            self.logger.info(f"  {year}: {count} chunks ({percentage:.1f}%)")

        # Source files
        source_files = [m.get('source_file', 'unknown') for m in metadata_list]
        unique_sources = set(source_files)

        self.stats["unique_sources"] = len(unique_sources)

        self.logger.info(f"\nUnique source files: {len(unique_sources)}")

        # Sample sources
        sample_sources = sorted(list(unique_sources))[:5]
        self.logger.info("Sample sources:")
        for src in sample_sources:
            self.logger.info(f"  - {src}")

        self.logger.info("="*80)

    def run_validation_queries(self):
        """
        Run validation queries to test database functionality.
        """

        self.logger.info("\n" + "="*80)
        self.logger.info("Running Validation Queries")
        self.logger.info("="*80)

        # ====================================================================
        # Query 1: Simple semantic search
        # ====================================================================

        self.logger.info("\n1. Semantic Search Test")
        self.logger.info("-" * 40)

        test_query = "bean breeding techniques"
        self.logger.info(f"Query: '{test_query}'")

        try:
            results = self.collection.query(
                query_texts=[test_query],
                n_results=3
            )

            self.logger.info(f"✓ Found {len(results['ids'][0])} results")

            for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                self.logger.info(f"\nResult {i+1}:")
                self.logger.info(f"  ID: {doc_id}")
                self.logger.info(f"  Source: {metadata.get('source_file', 'unknown')}")
                self.logger.info(f"  Year: {metadata.get('year_range', 'unknown')}")
                self.logger.info(f"  Distance: {distance:.4f}")
                self.logger.info(f"  Text preview: {doc[:100]}...")

        except Exception as e:
            self.logger.error(f"✗ Semantic search failed: {e}")

        # ====================================================================
        # Query 2: Filtered search by year range
        # ====================================================================

        self.logger.info("\n2. Filtered Search Test (2007-2026)")
        self.logger.info("-" * 40)

        test_query = "drought resistance"
        self.logger.info(f"Query: '{test_query}' WHERE year_range = '2007-2026'")

        try:
            results = self.collection.query(
                query_texts=[test_query],
                n_results=3,
                where={"year_range": "2007-2026"}  # Filter by year range
            )

            self.logger.info(f"✓ Found {len(results['ids'][0])} results from 2007-2026")

            for i, (doc_id, metadata) in enumerate(zip(
                results['ids'][0],
                results['metadatas'][0]
            )):
                self.logger.info(f"\nResult {i+1}:")
                self.logger.info(f"  ID: {doc_id}")
                self.logger.info(f"  Source: {metadata.get('source_file', 'unknown')}")
                self.logger.info(f"  Year: {metadata.get('year_range', 'unknown')}")

        except Exception as e:
            self.logger.error(f"✗ Filtered search failed: {e}")

        # ====================================================================
        # Query 3: Retrieve by ID
        # ====================================================================

        self.logger.info("\n3. Direct Retrieval Test")
        self.logger.info("-" * 40)

        try:
            # Get first chunk
            result = self.collection.get(
                ids=["chunk_0"],
                include=["documents", "metadatas"]
            )

            self.logger.info("✓ Retrieved chunk_0:")
            if result['metadatas']:
                metadata = result['metadatas'][0]
                self.logger.info(f"  Source: {metadata.get('source_file', 'unknown')}")
                self.logger.info(f"  Year: {metadata.get('year_range', 'unknown')}")
                self.logger.info(f"  Page: {metadata.get('page_number', 'unknown')}")

        except Exception as e:
            self.logger.error(f"✗ Direct retrieval failed: {e}")

        # ====================================================================
        # Query 4: Count check
        # ====================================================================

        self.logger.info("\n4. Count Verification")
        self.logger.info("-" * 40)

        try:
            # Get collection count
            count = self.collection.count()
            self.logger.info(f"✓ Collection contains {count} documents")

            if count == self.stats["total_chunks"]:
                self.logger.info("✓ Count matches expected")
            else:
                self.logger.warning(f"⚠ Count mismatch: expected {self.stats['total_chunks']}")

        except Exception as e:
            self.logger.error(f"✗ Count check failed: {e}")

        self.logger.info("\n" + "="*80)

    def calculate_database_size(self):
        """
        Calculate total database size on disk.
        """

        total_size = 0

        # Walk through database directory
        for root, dirs, files in os.walk(self.db_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)

        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        self.stats["database_size_mb"] = round(size_mb, 2)

        return size_mb

    def print_statistics(self):
        """
        Print comprehensive statistics.
        """

        print("\n" + "="*80)
        print("VECTOR STORE STATISTICS")
        print("="*80)

        print(f"Collection Name:       {self.collection_name}")
        print(f"Database Path:         {self.db_path}")
        print(f"Total Chunks:          {self.stats['total_chunks']}")
        print(f"Chunks Added:          {self.stats['chunks_added']}")
        print(f"Failed Chunks:         {self.stats['failed_chunks']}")
        print(f"Embedding Dimension:   {self.stats['embedding_dimension']}")
        print(f"Database Size:         {self.stats['database_size_mb']} MB")
        print(f"Unique Sources:        {self.stats['unique_sources']}")

        print(f"\nDate Range Coverage:")
        for year_range in sorted(self.stats['year_ranges']):
            print(f"  - {year_range}")

        if self.stats['errors']:
            print(f"\nErrors Encountered:    {len(self.stats['errors'])}")
            print("Check log file for details.")

        print("="*80 + "\n")

    def save_statistics(self):
        """
        Save statistics to JSON file.
        """

        stats_file = self.db_path / "database_statistics.json"

        stats_output = {
            "creation_date": datetime.now().isoformat(),
            "collection_name": self.collection_name,
            "database_path": str(self.db_path),
            "statistics": self.stats
        }

        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_output, f, indent=2)

            self.logger.info(f"✓ Statistics saved to: {stats_file}")

        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")

    def run(self) -> bool:
        """
        Main pipeline to build vector store.

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            # Step 1: Load data
            embeddings, metadata_list, texts = self.load_data()

            # Step 2: Create database
            self.create_database()

            # Step 3: Add data to database
            self.add_to_database(embeddings, metadata_list, texts)

            # Step 4: Analyze metadata
            self.analyze_metadata(metadata_list)

            # Step 5: Calculate database size
            db_size = self.calculate_database_size()
            self.logger.info(f"\nDatabase size: {db_size:.2f} MB")

            # Step 6: Run validation queries
            self.run_validation_queries()

            # Step 7: Save statistics
            self.save_statistics()

            # Step 8: Print statistics
            self.print_statistics()

            self.logger.info("✓ Vector store built successfully!")

            return True

        except Exception as e:
            self.logger.error(f"Fatal error building vector store: {e}")
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
        description="Build ChromaDB vector store from embeddings"
    )

    parser.add_argument(
        "--embeddings-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/embeddings.npy",
        help="Path to embeddings numpy file"
    )

    parser.add_argument(
        "--metadata-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/embeddings_metadata.json",
        help="Path to embeddings metadata JSON file"
    )

    parser.add_argument(
        "--chunks-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/processed_chunks.json",
        help="Path to processed chunks JSON file"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/vector_db",
        help="Path to vector database directory"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/logs/vector_store.log",
        help="Path to log file"
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="bean_research_docs",
        help="Name for ChromaDB collection"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for inserting vectors"
    )

    args = parser.parse_args()

    # Create builder
    builder = VectorStoreBuilder(
        embeddings_file=args.embeddings_file,
        metadata_file=args.metadata_file,
        chunks_file=args.chunks_file,
        db_path=args.db_path,
        log_file=args.log_file,
        collection_name=args.collection_name,
        batch_size=args.batch_size
    )

    # Run builder
    success = builder.run()

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
