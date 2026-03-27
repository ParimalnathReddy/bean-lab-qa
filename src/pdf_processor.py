#!/usr/bin/env python3
"""
PDF Processor for LLM-based Document QA System
Processes agricultural research PDFs from the Bean Lab collection

Improvements over v1:
  - Section detection: tags chunks with abstract/intro/methods/results/discussion/conclusion
  - Table preservation: keeps table rows with surrounding explanatory text
  - Increased overlap: 128 tokens (was 50) for better cross-chunk context
  - Section metadata stored in each chunk for section-aware retrieval
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import traceback

# PDF and text processing
try:
    from PyPDF2 import PdfReader
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError as e:
    print(f"Warning: Some OCR dependencies not available: {e}")
    print("Install with: pip install PyPDF2 pytesseract pdf2image Pillow")

# Text chunking
try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken not available. Install with: pip install tiktoken")
    tiktoken = None

from tqdm import tqdm


# ── Section detection ─────────────────────────────────────────────────────────
# Ordered list of (section_name, regex_pattern) tuples.
# Patterns match common scientific paper section headers (case-insensitive).
# Sections are detected per-page and propagated forward until a new section starts.

SECTION_PATTERNS = [
    ("abstract",      r"^\s*(abstract)\s*$"),
    ("introduction",  r"^\s*(introduction|background)\s*$"),
    ("methods",       r"^\s*(materials?\s+and\s+methods?|methods?|methodology|experimental)\s*$"),
    ("results",       r"^\s*(results?(\s+and\s+discussion)?)\s*$"),
    ("discussion",    r"^\s*(discussion)\s*$"),
    ("conclusion",    r"^\s*(conclusions?|summary)\s*$"),
    ("references",    r"^\s*(references?|bibliography|literature\s+cited)\s*$"),
]

# Compiled patterns for speed
_COMPILED_SECTION_PATTERNS = [
    (name, re.compile(pat, re.IGNORECASE | re.MULTILINE))
    for name, pat in SECTION_PATTERNS
]


def detect_section(line: str) -> Optional[str]:
    """Return section name if line matches a section header, else None."""
    stripped = line.strip()
    for name, pattern in _COMPILED_SECTION_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def is_table_row(line: str) -> bool:
    """
    Heuristic: a line is part of a table if it contains at least 2 tab-separated
    or multiple-space-separated columns, or pipe characters, typical of PDF tables.
    """
    if "|" in line:
        return True
    # Two or more groups of 3+ spaces between tokens suggest tabular alignment
    if re.search(r'\S\s{3,}\S', line) and len(line.strip()) > 10:
        return True
    return False


def group_table_blocks(lines: List[str]) -> List[Dict]:
    """
    Scan lines and group consecutive table rows into blocks.
    Returns list of dicts: {"type": "text"|"table", "content": str}
    """
    blocks = []
    i = 0
    while i < len(lines):
        if is_table_row(lines[i]):
            # Collect consecutive table rows
            table_lines = []
            while i < len(lines) and is_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "content": "\n".join(table_lines)})
        else:
            blocks.append({"type": "text", "content": lines[i]})
            i += 1
    return blocks


class PDFProcessor:
    """
    Memory-efficient PDF processor for HPCC environment.
    Handles text extraction, chunking, and metadata creation.
    """

    def __init__(
        self,
        input_dir: str,
        output_file: str,
        log_file: str,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_ocr: bool = False,
        encoding_model: str = "cl100k_base"  # GPT-3.5/4 tokenizer
    ):
        """
        Initialize PDF processor.

        Args:
            input_dir: Directory containing PDF files
            output_file: Path to output JSON file
            log_file: Path to log file
            chunk_size: Number of tokens per chunk
            chunk_overlap: Overlapping tokens between chunks (128 for good cross-chunk context)
            use_ocr: Whether to use OCR for scanned documents
            encoding_model: Tokenizer model name for tiktoken
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.log_file = Path(log_file)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr

        # Initialize tokenizer
        if tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_model)
            except Exception as e:
                logging.warning(f"Failed to load tiktoken encoder: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Setup logging
        self._setup_logging()

        # Statistics
        self.stats = {
            "total_pdfs": 0,
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_pages": 0,
            "ocr_used": 0,
            "errors": []
        }

    def _setup_logging(self):
        """Configure logging to file and console."""
        # Create log directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info("PDF Processor Initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output file: {self.output_file}")
        self.logger.info(f"Chunk size: {self.chunk_size} tokens")
        self.logger.info(f"Chunk overlap: {self.chunk_overlap} tokens")
        self.logger.info("="*80)

    def find_all_pdfs(self) -> List[Path]:
        """
        Recursively find all PDF files in input directory.
        Uses os.walk with followlinks=True to follow symlinked directories.

        Returns:
            List of Path objects for all PDFs found
        """
        pdf_files = []
        for root, dirs, files in os.walk(str(self.input_dir), followlinks=True):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(Path(root) / file)
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def extract_year_range(self, file_path: Path) -> str:
        """
        Extract year range from directory structure.

        Args:
            file_path: Path to PDF file

        Returns:
            Year range string (e.g., "1961-2006" or "unknown")
        """
        # Look for year patterns in path (e.g., "1961-2006", "2007-2026")
        path_str = str(file_path)
        year_pattern = r'(\d{4})-(\d{4})'
        match = re.search(year_pattern, path_str)

        if match:
            return f"{match.group(1)}-{match.group(2)}"

        # Try to extract from filename (DOI format may contain year)
        filename = file_path.stem
        year_in_filename = re.search(r'(19|20)\d{2}', filename)
        if year_in_filename:
            return year_in_filename.group(0)

        return "unknown"

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)

            text_content = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()

                    # If text is minimal and OCR is enabled, try OCR
                    if self.use_ocr and (not text or len(text.strip()) < 50):
                        self.logger.debug(f"Using OCR for {pdf_path.name} page {page_num}")
                        ocr_text = self._ocr_page(pdf_path, page_num)
                        if ocr_text:
                            text = ocr_text
                            self.stats["ocr_used"] += 1

                    if text:
                        # Add page marker for metadata
                        text_content.append(f"[PAGE {page_num}]\n{text}")

                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num} from {pdf_path.name}: {e}")
                    continue

            full_text = "\n\n".join(text_content)
            self.stats["total_pages"] += num_pages

            return full_text, num_pages

        except Exception as e:
            self.logger.error(f"Failed to read PDF {pdf_path.name}: {e}")
            raise

    def _ocr_page(self, pdf_path: Path, page_num: int) -> Optional[str]:
        """
        Perform OCR on a specific page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to OCR (1-indexed)

        Returns:
            Extracted text or None if OCR fails
        """
        try:
            # Convert PDF page to image
            images = convert_from_path(
                str(pdf_path),
                first_page=page_num,
                last_page=page_num,
                dpi=300
            )

            if images:
                # Perform OCR
                text = pytesseract.image_to_string(images[0])
                return text

        except Exception as e:
            self.logger.warning(f"OCR failed for {pdf_path.name} page {page_num}: {e}")

        return None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate tokens (1 token ≈ 4 characters)
            return len(text) // 4

    def chunk_text(self, text: str, source_file: str, year_range: str) -> List[Dict]:
        """
        Split text into overlapping chunks with section detection and table preservation.

        Changes from v1:
          - Section headers detected per line; current section propagated to all chunks
          - Table rows grouped and kept intact (not split mid-table)
          - Table blocks always emitted as a single chunk even if > chunk_size
          - Text chunks use 128-token overlap for cross-chunk context

        Args:
            text: Full text extracted from PDF (with [PAGE N] markers)
            source_file: Source PDF filename for metadata
            year_range: Year range metadata (e.g. "1961-2006")

        Returns:
            List of chunk dicts with keys: text, source_file, year_range,
            chunk_id, page_number, token_count, section
        """
        chunks = []

        # Split by page markers to track page numbers
        page_sections = re.split(r'\[PAGE (\d+)\]', text)

        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        current_page = 1
        current_section = ""          # tracks detected section name

        def flush_chunk(section: str, page: int):
            """Save current_chunk as a completed chunk."""
            nonlocal current_chunk, current_tokens, chunk_id
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source_file": source_file,
                    "year_range": year_range,
                    "chunk_id": chunk_id,
                    "page_number": page,
                    "token_count": current_tokens,
                    "section": section,
                })
                chunk_id += 1

        for i in range(1, len(page_sections), 2):
            if i + 1 >= len(page_sections):
                break

            page_num = int(page_sections[i])
            page_text = page_sections[i + 1]

            # Split page into lines for section detection and table grouping
            lines = page_text.split("\n")
            blocks = group_table_blocks(lines)   # list of {type, content}

            for block in blocks:
                btype = block["type"]
                content = block["content"].strip()
                if not content:
                    continue

                if btype == "table":
                    # ── Table block: flush current text chunk, emit table as its own chunk ──
                    flush_chunk(current_section, page_num)
                    current_chunk = ""
                    current_tokens = 0

                    table_tokens = self.count_tokens(content)
                    chunks.append({
                        "text": content,
                        "source_file": source_file,
                        "year_range": year_range,
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "token_count": table_tokens,
                        "section": current_section,
                    })
                    chunk_id += 1

                else:
                    # ── Text block: check for section header on each line ──────────────────
                    sentences = re.split(r'(?<=[.!?])\s+', content)

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue

                        # Check if this line is a section header
                        detected = detect_section(sentence)
                        if detected:
                            # Flush before starting new section
                            flush_chunk(current_section, page_num)
                            current_chunk = ""
                            current_tokens = 0
                            current_section = detected
                            continue   # don't add header text to chunk

                        sentence_tokens = self.count_tokens(sentence)

                        if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                            # Flush and start new chunk with overlap
                            flush_chunk(current_section, page_num)
                            overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                            current_chunk = (overlap_text + " " + sentence).strip()
                            current_tokens = self.count_tokens(current_chunk)
                        else:
                            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                            current_tokens += sentence_tokens

                current_page = page_num

        # Flush any remaining text
        flush_chunk(current_section, current_page)

        self.stats["total_chunks"] += len(chunks)
        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens from text for overlap.

        Args:
            text: Full text
            overlap_tokens: Number of tokens to overlap

        Returns:
            Overlapping text portion
        """
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > overlap_tokens:
                overlap_token_ids = tokens[-overlap_tokens:]
                return self.tokenizer.decode(overlap_token_ids)

        # Fallback: use character-based approximation
        overlap_chars = overlap_tokens * 4
        return text[-overlap_chars:] if len(text) > overlap_chars else text

    def process_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of chunks with metadata
        """
        try:
            # Extract text
            text, num_pages = self.extract_text_from_pdf(pdf_path)

            if not text or len(text.strip()) < 10:
                self.logger.warning(f"No text extracted from {pdf_path.name}")
                return []

            # Get metadata
            year_range = self.extract_year_range(pdf_path)
            source_file = pdf_path.name

            # Create chunks
            chunks = self.chunk_text(text, source_file, year_range)

            self.logger.info(
                f"Processed {pdf_path.name}: {num_pages} pages → {len(chunks)} chunks"
            )

            return chunks

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {e}")
            self.logger.debug(traceback.format_exc())
            self.stats["errors"].append({
                "file": str(pdf_path),
                "error": str(e)
            })
            return []

    def process_all_pdfs(self, batch_size: int = 50) -> List[Dict]:
        """
        Process all PDFs in input directory.

        Args:
            batch_size: Number of PDFs to process before saving checkpoint

        Returns:
            List of all chunks from all PDFs
        """
        pdf_files = self.find_all_pdfs()
        self.stats["total_pdfs"] = len(pdf_files)

        all_chunks = []

        # Process with progress bar
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for idx, pdf_path in enumerate(pdf_files):
                try:
                    chunks = self.process_single_pdf(pdf_path)
                    if chunks:
                        all_chunks.extend(chunks)
                        self.stats["successful"] += 1
                    else:
                        self.stats["failed"] += 1

                    # Save checkpoint every batch_size files
                    if (idx + 1) % batch_size == 0:
                        self._save_checkpoint(all_chunks, idx + 1)

                except Exception as e:
                    self.logger.error(f"Unexpected error with {pdf_path.name}: {e}")
                    self.stats["failed"] += 1

                finally:
                    pbar.update(1)

        return all_chunks

    def _save_checkpoint(self, chunks: List[Dict], num_processed: int):
        """Save intermediate results."""
        checkpoint_file = self.output_file.parent / f"checkpoint_{num_processed}.json"
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Checkpoint saved: {checkpoint_file} ({len(chunks)} chunks)")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def save_chunks(self, chunks: List[Dict]):
        """
        Save chunks to JSON file.

        Args:
            chunks: List of chunk dictionaries
        """
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(chunks)} chunks to {self.output_file}")

            # Also save metadata
            metadata = {
                "processing_date": datetime.now().isoformat(),
                "statistics": self.stats,
                "configuration": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "use_ocr": self.use_ocr
                },
                "total_chunks": len(chunks)
            }

            metadata_file = self.output_file.parent / "processing_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved metadata to {metadata_file}")

        except Exception as e:
            self.logger.error(f"Failed to save chunks: {e}")
            raise

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*80)
        print("PROCESSING STATISTICS")
        print("="*80)
        print(f"Total PDFs found:      {self.stats['total_pdfs']}")
        print(f"Successfully processed: {self.stats['successful']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"Total pages:           {self.stats['total_pages']}")
        print(f"Total chunks created:  {self.stats['total_chunks']}")
        if self.use_ocr:
            print(f"OCR used (pages):      {self.stats['ocr_used']}")

        if self.stats['total_chunks'] > 0:
            avg_chunks = self.stats['total_chunks'] / max(self.stats['successful'], 1)
            print(f"Average chunks/PDF:    {avg_chunks:.1f}")

        if self.stats['errors']:
            print(f"\nErrors encountered:    {len(self.stats['errors'])}")
            print("Check log file for details.")

        print("="*80 + "\n")

    def run(self):
        """Main processing pipeline."""
        try:
            self.logger.info("Starting PDF processing pipeline...")

            # Process all PDFs
            chunks = self.process_all_pdfs()

            # Save results
            if chunks:
                self.save_chunks(chunks)
            else:
                self.logger.warning("No chunks were created!")

            # Print statistics
            self.print_statistics()

            self.logger.info("Processing complete!")

            return chunks

        except Exception as e:
            self.logger.error(f"Fatal error in processing pipeline: {e}")
            self.logger.debug(traceback.format_exc())
            raise


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDF documents for LLM-based QA system"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs",
        help="Input directory containing PDFs"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/processed_chunks.json",
        help="Output JSON file for chunks"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/logs/processing.log",
        help="Log file path"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Overlap between chunks in tokens"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR for scanned documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Save checkpoint every N files"
    )

    args = parser.parse_args()

    # Create processor
    processor = PDFProcessor(
        input_dir=args.input_dir,
        output_file=args.output_file,
        log_file=args.log_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr=args.use_ocr
    )

    # Run processing
    processor.run()


if __name__ == "__main__":
    main()
