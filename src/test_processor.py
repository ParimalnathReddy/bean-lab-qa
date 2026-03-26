#!/usr/bin/env python3
"""
Test script for PDF processor
Tests on a small sample before processing all 1,067 PDFs
"""

import os
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf_processor import PDFProcessor


def test_small_sample():
    """Test processor on a small sample of PDFs."""

    print("="*80)
    print("PDF PROCESSOR TEST")
    print("="*80)
    print()

    # Setup paths
    base_dir = Path("/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa")
    input_dir = base_dir / "data" / "pdfs"
    output_dir = base_dir / "data" / "test_output"
    log_dir = base_dir / "logs"

    # Create test output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "test_chunks.json"
    log_file = log_dir / "test_processing.log"

    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Log file: {log_file}")
    print()

    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        print()
        print("Please create symbolic links or copy PDFs to data/pdfs/ first:")
        print("Run one of these commands:")
        print()
        print("# Option 1: Create symbolic links")
        print(f"mkdir -p {input_dir}")
        print(f"ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006 {input_dir}/1961-2006")
        print(f"ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026 {input_dir}/2007-2026")
        print()
        print("# Option 2: Copy PDFs")
        print(f"bash {base_dir}/scripts/analyze_and_copy_pdfs.sh")
        return

    # Find PDFs
    pdf_files = list(input_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} total PDFs")

    if len(pdf_files) == 0:
        print("❌ No PDFs found in input directory!")
        return

    # Limit to first 5 for testing
    test_limit = min(5, len(pdf_files))
    print(f"Testing with first {test_limit} PDFs")
    print()

    # Create test directory with only sample PDFs
    test_input_dir = output_dir / "test_pdfs"
    test_input_dir.mkdir(exist_ok=True)

    # Create symbolic links to test PDFs
    print("Sample PDFs:")
    for i, pdf in enumerate(pdf_files[:test_limit]):
        link_path = test_input_dir / pdf.name
        if not link_path.exists():
            link_path.symlink_to(pdf)
        print(f"  {i+1}. {pdf.name}")
    print()

    # Initialize processor
    print("Initializing processor...")
    processor = PDFProcessor(
        input_dir=str(test_input_dir),
        output_file=str(output_file),
        log_file=str(log_file),
        chunk_size=512,
        chunk_overlap=50,
        use_ocr=False  # Disable OCR for faster testing
    )

    # Run processing
    print("Processing PDFs...")
    print("-"*80)
    chunks = processor.run()
    print("-"*80)
    print()

    # Display sample results
    if chunks:
        print(f"✅ Successfully created {len(chunks)} chunks")
        print()
        print("Sample chunk (first one):")
        print("-"*80)
        sample = chunks[0]
        print(f"Source: {sample['source_file']}")
        print(f"Year Range: {sample['year_range']}")
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Page: {sample['page_number']}")
        print(f"Tokens: {sample['token_count']}")
        print(f"Text preview (first 200 chars):")
        print(sample['text'][:200] + "...")
        print("-"*80)
        print()

        # Show distribution
        print("Chunks per file:")
        file_chunks = {}
        for chunk in chunks:
            file_chunks[chunk['source_file']] = file_chunks.get(chunk['source_file'], 0) + 1

        for filename, count in file_chunks.items():
            print(f"  {filename}: {count} chunks")
        print()

        print(f"✅ Test output saved to: {output_file}")
        print(f"✅ Test log saved to: {log_file}")
    else:
        print("❌ No chunks were created!")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print()
    print("If the test looks good, run the full processor with:")
    print(f"python {base_dir}/src/pdf_processor.py")
    print()
    print("Or submit as SLURM job:")
    print(f"sbatch {base_dir}/jobs/process_pdfs.sb")
    print()


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    print()

    dependencies = {
        "PyPDF2": "PyPDF2",
        "tiktoken": "tiktoken",
        "tqdm": "tqdm",
    }

    optional_deps = {
        "pytesseract": "pytesseract (for OCR)",
        "pdf2image": "pdf2image (for OCR)",
        "PIL": "Pillow (for OCR)",
    }

    missing = []
    missing_optional = []

    # Check required
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
            missing.append(name)

    # Check optional
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - optional")
            missing_optional.append(name)

    print()

    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print()
        print("Install with:")
        print(f"pip install {' '.join(missing)}")
        print()
        print("Or install all dependencies:")
        print("pip install -r requirements.txt")
        return False

    if missing_optional:
        print("⚠️  Missing optional dependencies (for OCR):")
        for dep in missing_optional:
            print(f"   - {dep}")
        print()
        print("OCR will not be available without these packages.")
        print("To enable OCR, install with:")
        print("pip install pytesseract pdf2image Pillow")

    print()
    return True


if __name__ == "__main__":
    # Check dependencies first
    if check_dependencies():
        print()
        # Run test
        test_small_sample()
    else:
        print("Please install missing dependencies before running the test.")
        sys.exit(1)
