#!/usr/bin/env python3
"""
================================================================================
PDF PROCESSOR FOR LLM-BASED DOCUMENT QA SYSTEM
================================================================================

Purpose:
    Extract text from agricultural research PDFs, split into chunks, and
    prepare for vector database ingestion in a QA system.

Input:
    - Directory of PDF files (1,067 PDFs from 1961-2026)

Output:
    - JSON file with text chunks and metadata
    - Processing statistics and logs

Features:
    - Recursive PDF discovery
    - Text extraction from PDFs
    - OCR support for scanned documents (optional)
    - Smart text chunking with token-based overlap
    - Comprehensive metadata tracking
    - Memory-efficient processing for HPCC
    - Error handling and logging
    - Progress tracking
    - Checkpoint saves

Author: MSU HPCC Bean Lab
Date: 2026-03-21
================================================================================
"""

# ============================================================================
# IMPORTS - Standard Library
# ============================================================================

import os              # Operating system interface (file operations)
import json            # JSON encoding/decoding for data serialization
import logging         # Logging facility for tracking execution
import re              # Regular expressions for pattern matching
from pathlib import Path           # Object-oriented filesystem paths
from typing import List, Dict, Optional, Tuple  # Type hints for better code documentation
from datetime import datetime      # Date and time manipulation
import traceback       # Stack trace extraction for debugging

# ============================================================================
# IMPORTS - PDF Processing Libraries
# ============================================================================

# Try to import PDF and OCR libraries
# Using try/except to handle optional dependencies gracefully
try:
    from PyPDF2 import PdfReader           # PDF reading and text extraction
    import pytesseract                     # OCR (Optical Character Recognition) engine
    from pdf2image import convert_from_path # Convert PDF pages to images for OCR
    from PIL import Image                  # Python Imaging Library for image processing
except ImportError as e:
    # If imports fail, warn user but don't crash
    # This allows the script to run without OCR capabilities
    print(f"Warning: Some OCR dependencies not available: {e}")
    print("Install with: pip install PyPDF2 pytesseract pdf2image Pillow")

# ============================================================================
# IMPORTS - Text Tokenization
# ============================================================================

# tiktoken: OpenAI's tokenizer for accurate token counting
# This is critical for proper text chunking for LLMs
try:
    import tiktoken
except ImportError:
    # If tiktoken not available, we'll fall back to character-based approximation
    print("Warning: tiktoken not available. Install with: pip install tiktoken")
    tiktoken = None  # Set to None so we can check later

# ============================================================================
# IMPORTS - Progress Tracking
# ============================================================================

from tqdm import tqdm  # Progress bar library for visual feedback


# ============================================================================
# MAIN CLASS - PDFProcessor
# ============================================================================

class PDFProcessor:
    """
    Memory-efficient PDF processor designed for HPCC environment.

    This class handles:
    1. Finding PDF files recursively in directory structure
    2. Extracting text from PDFs (with optional OCR for scanned docs)
    3. Splitting text into chunks with token-based overlap
    4. Creating structured metadata for each chunk
    5. Saving results to JSON format
    6. Comprehensive logging and error handling

    Memory Efficiency:
    - Processes one PDF at a time (not all in memory)
    - Uses generators where possible
    - Saves checkpoints periodically
    - Cleans up resources after each file
    """

    def __init__(
        self,
        input_dir: str,              # Path to directory containing PDFs
        output_file: str,            # Path where processed chunks will be saved
        log_file: str,               # Path for logging output
        chunk_size: int = 512,       # Number of tokens per text chunk (default: 512)
        chunk_overlap: int = 50,     # Number of tokens to overlap between chunks (default: 50)
        use_ocr: bool = False,       # Whether to use OCR for scanned documents
        encoding_model: str = "cl100k_base"  # Tokenizer model (GPT-3.5/4 compatible)
    ):
        """
        Initialize the PDF processor with configuration parameters.

        Args:
            input_dir: Directory containing PDF files (can have subdirectories)
            output_file: Full path to output JSON file for processed chunks
            log_file: Full path to log file
            chunk_size: Target number of tokens per chunk (affects retrieval granularity)
            chunk_overlap: Number of overlapping tokens between chunks (helps context continuity)
            use_ocr: Enable OCR for scanned/image-based PDFs (slower but handles more formats)
            encoding_model: Tokenizer model name ('cl100k_base' for GPT-3.5/4)

        Notes:
            - Larger chunk_size = fewer chunks, more context per chunk
            - Smaller chunk_size = more chunks, more precise retrieval
            - chunk_overlap helps maintain context across chunk boundaries
            - OCR requires additional dependencies and is much slower
        """

        # ====================================================================
        # STEP 1: Store configuration parameters as instance variables
        # ====================================================================

        # Convert string paths to Path objects for better path manipulation
        self.input_dir = Path(input_dir)      # Input directory as Path object
        self.output_file = Path(output_file)  # Output file as Path object
        self.log_file = Path(log_file)        # Log file as Path object

        # Store chunking parameters
        self.chunk_size = chunk_size          # Tokens per chunk (e.g., 512)
        self.chunk_overlap = chunk_overlap    # Overlapping tokens (e.g., 50)
        self.use_ocr = use_ocr                # Boolean flag for OCR usage

        # ====================================================================
        # STEP 2: Initialize tokenizer for accurate token counting
        # ====================================================================

        # Check if tiktoken library is available
        if tiktoken:
            try:
                # Load the specified encoding model
                # cl100k_base is used by GPT-3.5-turbo and GPT-4
                self.tokenizer = tiktoken.get_encoding(encoding_model)
            except Exception as e:
                # If loading fails, log warning and fall back to approximation
                logging.warning(f"Failed to load tiktoken encoder: {e}")
                self.tokenizer = None  # Will use character-based approximation instead
        else:
            # tiktoken not installed, will use fallback method
            self.tokenizer = None

        # ====================================================================
        # STEP 3: Set up logging system
        # ====================================================================

        # Call internal method to configure logging to both file and console
        self._setup_logging()

        # ====================================================================
        # STEP 4: Initialize statistics tracking dictionary
        # ====================================================================

        # This dictionary tracks all processing statistics
        self.stats = {
            "total_pdfs": 0,       # Total number of PDF files found
            "successful": 0,       # Number of successfully processed PDFs
            "failed": 0,           # Number of failed PDFs
            "total_chunks": 0,     # Total number of chunks created
            "total_pages": 0,      # Total number of pages processed
            "ocr_used": 0,         # Number of times OCR was used
            "errors": []           # List of error details for failed files
        }

    # ========================================================================
    # METHOD: _setup_logging
    # ========================================================================

    def _setup_logging(self):
        """
        Configure logging to write to both file and console.

        This method:
        1. Creates log directory if it doesn't exist
        2. Configures logging format and handlers
        3. Logs initial configuration information

        Logging Levels:
            DEBUG: Detailed information for diagnosing problems
            INFO: General informational messages
            WARNING: Something unexpected but not critical
            ERROR: Serious problem that prevented an operation
        """

        # ====================================================================
        # STEP 1: Ensure log directory exists
        # ====================================================================

        # Get parent directory of log file (e.g., 'logs/')
        # mkdir creates directory, parents=True creates intermediate dirs
        # exist_ok=True doesn't raise error if directory already exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # ====================================================================
        # STEP 2: Configure logging system
        # ====================================================================

        # basicConfig sets up the root logger
        logging.basicConfig(
            level=logging.INFO,    # Log INFO level and above (INFO, WARNING, ERROR)
            format='%(asctime)s - %(levelname)s - %(message)s',  # Format: timestamp - level - message
            handlers=[
                # Handler 1: Write to file
                logging.FileHandler(self.log_file),
                # Handler 2: Print to console (stdout)
                logging.StreamHandler()
            ]
        )

        # ====================================================================
        # STEP 3: Get logger instance for this module
        # ====================================================================

        # Create logger specific to this module (__name__ = current module name)
        self.logger = logging.getLogger(__name__)

        # ====================================================================
        # STEP 4: Log initialization information
        # ====================================================================

        # Log a visual separator line (80 equals signs)
        self.logger.info("="*80)

        # Log that processor has been initialized
        self.logger.info("PDF Processor Initialized")

        # Log configuration parameters so they're recorded in log file
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output file: {self.output_file}")
        self.logger.info(f"Chunk size: {self.chunk_size} tokens")
        self.logger.info(f"Chunk overlap: {self.chunk_overlap} tokens")

        # Log another separator line
        self.logger.info("="*80)

    # ========================================================================
    # METHOD: find_all_pdfs
    # ========================================================================

    def find_all_pdfs(self) -> List[Path]:
        """
        Recursively find all PDF files in the input directory.

        This method searches through the input directory and all subdirectories
        to find files with .pdf extension.

        Returns:
            List[Path]: List of Path objects pointing to all PDF files found

        Example:
            If input_dir = 'data/pdfs/' containing:
                data/pdfs/1961-2006/file1.pdf
                data/pdfs/1961-2006/file2.pdf
                data/pdfs/2007-2026/file3.pdf

            Returns: [Path('data/pdfs/1961-2006/file1.pdf'),
                     Path('data/pdfs/1961-2006/file2.pdf'),
                     Path('data/pdfs/2007-2026/file3.pdf')]
        """

        # ====================================================================
        # STEP 1: Use rglob to recursively find all PDF files
        # ====================================================================

        # rglob("*.pdf") recursively searches for files matching pattern "*.pdf"
        # The ** in the pattern means "any number of directories deep"
        # Convert iterator to list to get all matches at once
        pdf_files = list(self.input_dir.rglob("*.pdf"))

        # ====================================================================
        # STEP 2: Log how many PDFs were found
        # ====================================================================

        # Log the count so we know what to expect in processing
        self.logger.info(f"Found {len(pdf_files)} PDF files")

        # ====================================================================
        # STEP 3: Return the list of PDF file paths
        # ====================================================================

        return pdf_files

    # ========================================================================
    # METHOD: extract_year_range
    # ========================================================================

    def extract_year_range(self, file_path: Path) -> str:
        """
        Extract year range from file path or filename.

        This function looks for year patterns in the directory structure
        (e.g., "1961-2006", "2007-2026") or in the filename itself.
        This metadata helps categorize documents by time period.

        Args:
            file_path: Path object pointing to the PDF file

        Returns:
            str: Year range string (e.g., "1961-2006", "2007", or "unknown")

        Examples:
            Path: /data/pdfs/1961-2006/file.pdf -> "1961-2006"
            Path: /data/pdfs/file_2008.pdf -> "2008"
            Path: /data/pdfs/file.pdf -> "unknown"
        """

        # ====================================================================
        # STEP 1: Convert path to string for regex matching
        # ====================================================================

        path_str = str(file_path)  # Convert Path object to string

        # ====================================================================
        # STEP 2: Define regex pattern for year range (YYYY-YYYY)
        # ====================================================================

        # Pattern explanation:
        # (\d{4}) - Capture group 1: exactly 4 digits (first year)
        # -       - Literal hyphen character
        # (\d{4}) - Capture group 2: exactly 4 digits (second year)
        year_pattern = r'(\d{4})-(\d{4})'

        # ====================================================================
        # STEP 3: Search for year range pattern in full path
        # ====================================================================

        # search() looks for pattern anywhere in the string
        match = re.search(year_pattern, path_str)

        # ====================================================================
        # STEP 4: If year range found, return it
        # ====================================================================

        if match:
            # match.group(1) = first year, match.group(2) = second year
            # Format as "YYYY-YYYY"
            return f"{match.group(1)}-{match.group(2)}"

        # ====================================================================
        # STEP 5: If no range found, try to find single year in filename
        # ====================================================================

        # Get filename without extension (stem)
        # Example: "file_2008.pdf" -> "file_2008"
        filename = file_path.stem

        # Pattern for year: 19XX or 20XX (1900s or 2000s)
        # (19|20) - Either 19 or 20
        # \d{2}   - Followed by exactly 2 more digits
        year_in_filename = re.search(r'(19|20)\d{2}', filename)

        # If single year found, return it
        if year_in_filename:
            return year_in_filename.group(0)  # Return the matched year

        # ====================================================================
        # STEP 6: If no year information found, return "unknown"
        # ====================================================================

        return "unknown"

    # ========================================================================
    # METHOD: extract_text_from_pdf
    # ========================================================================

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        """
        Extract text content from a PDF file.

        This method:
        1. Opens PDF file using PyPDF2
        2. Extracts text from each page
        3. Optionally uses OCR if text extraction yields minimal results
        4. Adds page markers for metadata tracking
        5. Returns combined text and page count

        Args:
            pdf_path: Path object pointing to the PDF file

        Returns:
            Tuple[str, int]:
                - str: Full extracted text with page markers
                - int: Number of pages in the PDF

        Raises:
            Exception: If PDF cannot be read or is corrupted

        Example Output:
            ("[PAGE 1]\nThis is page one text...\n\n[PAGE 2]\nThis is page two...", 2)
        """

        try:
            # ================================================================
            # STEP 1: Open and read the PDF file
            # ================================================================

            # Create PdfReader object to read the PDF
            # Convert Path to string as PyPDF2 expects string path
            reader = PdfReader(str(pdf_path))

            # Get total number of pages in PDF
            num_pages = len(reader.pages)

            # ================================================================
            # STEP 2: Initialize list to collect text from all pages
            # ================================================================

            # Empty list to store text from each page
            text_content = []

            # ================================================================
            # STEP 3: Loop through each page and extract text
            # ================================================================

            # enumerate() gives us both index and page object
            # Start counting from 1 (not 0) for human-readable page numbers
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    # ========================================================
                    # STEP 3a: Extract text from current page
                    # ========================================================

                    # extract_text() method pulls text from page
                    text = page.extract_text()

                    # ========================================================
                    # STEP 3b: Check if OCR is needed
                    # ========================================================

                    # If OCR is enabled AND (no text found OR very little text)
                    # Then this might be a scanned/image-based PDF
                    if self.use_ocr and (not text or len(text.strip()) < 50):
                        # Log that we're using OCR for this page
                        self.logger.debug(f"Using OCR for {pdf_path.name} page {page_num}")

                        # Call OCR method (defined below)
                        ocr_text = self._ocr_page(pdf_path, page_num)

                        # If OCR succeeded, use that text instead
                        if ocr_text:
                            text = ocr_text  # Replace extracted text with OCR text
                            self.stats["ocr_used"] += 1  # Increment OCR usage counter

                    # ========================================================
                    # STEP 3c: Add page text to collection (if exists)
                    # ========================================================

                    if text:
                        # Add page marker and text to list
                        # Format: "[PAGE N]\ntext content"
                        # This marker helps track which page each chunk comes from
                        text_content.append(f"[PAGE {page_num}]\n{text}")

                except Exception as e:
                    # ========================================================
                    # STEP 3d: Handle errors for individual pages
                    # ========================================================

                    # If error on this page, log warning but continue
                    # Don't fail entire PDF because one page has issues
                    self.logger.warning(f"Error extracting page {page_num} from {pdf_path.name}: {e}")
                    continue  # Skip to next page

            # ================================================================
            # STEP 4: Combine all page texts into single string
            # ================================================================

            # Join all page texts with double newline separator
            # "\n\n" creates clear separation between pages
            full_text = "\n\n".join(text_content)

            # ================================================================
            # STEP 5: Update statistics
            # ================================================================

            # Add this PDF's pages to total page count
            self.stats["total_pages"] += num_pages

            # ================================================================
            # STEP 6: Return extracted text and page count
            # ================================================================

            return full_text, num_pages

        except Exception as e:
            # ================================================================
            # STEP 7: Handle PDF-level errors
            # ================================================================

            # Log error - this is serious (entire PDF failed)
            self.logger.error(f"Failed to read PDF {pdf_path.name}: {e}")

            # Re-raise exception so caller knows processing failed
            raise

    # ========================================================================
    # METHOD: _ocr_page (PRIVATE METHOD - indicated by leading underscore)
    # ========================================================================

    def _ocr_page(self, pdf_path: Path, page_num: int) -> Optional[str]:
        """
        Perform OCR (Optical Character Recognition) on a specific PDF page.

        This is used when standard text extraction fails or returns minimal text,
        indicating the PDF page might be a scanned image rather than digital text.

        Process:
        1. Convert PDF page to high-resolution image (300 DPI)
        2. Run Tesseract OCR on the image
        3. Return extracted text

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to OCR (1-indexed, not 0-indexed)

        Returns:
            Optional[str]: Extracted text if successful, None if OCR fails

        Note:
            OCR is much slower than direct text extraction and requires
            additional dependencies (tesseract, pdf2image, poppler)
        """

        try:
            # ================================================================
            # STEP 1: Convert PDF page to image
            # ================================================================

            # convert_from_path converts PDF pages to PIL Image objects
            images = convert_from_path(
                str(pdf_path),           # PDF file path as string
                first_page=page_num,     # Start page (inclusive)
                last_page=page_num,      # End page (inclusive) - so just one page
                dpi=300                  # Resolution: 300 DPI for good OCR quality
            )
            # Result: list of image objects, should contain exactly 1 image

            # ================================================================
            # STEP 2: Perform OCR if image was created
            # ================================================================

            if images:  # Check if list is not empty
                # images[0] = first (and only) image in list
                # pytesseract.image_to_string() runs OCR on the image
                text = pytesseract.image_to_string(images[0])
                return text  # Return extracted text

        except Exception as e:
            # ================================================================
            # STEP 3: Handle OCR errors
            # ================================================================

            # Log warning if OCR fails
            # This is expected sometimes (bad image quality, missing dependencies)
            self.logger.warning(f"OCR failed for {pdf_path.name} page {page_num}: {e}")

        # ================================================================
        # STEP 4: Return None if OCR failed or no images generated
        # ================================================================

        return None  # Indicates OCR was unsuccessful

    # ========================================================================
    # METHOD: count_tokens
    # ========================================================================

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Tokens are the basic units that LLMs process. A token can be:
        - A word (e.g., "hello")
        - Part of a word (e.g., "ing" in "running")
        - Punctuation (e.g., "!")

        Accurate token counting is crucial for:
        - Staying within LLM context limits
        - Creating properly-sized chunks
        - Estimating processing costs

        Args:
            text: String to count tokens in

        Returns:
            int: Number of tokens in the text

        Methods:
            1. If tiktoken available: Use exact tokenization (preferred)
            2. If tiktoken not available: Use approximation (1 token ≈ 4 chars)
        """

        # ================================================================
        # STEP 1: Check if we have access to tiktoken tokenizer
        # ================================================================

        if self.tokenizer:  # If tokenizer was successfully loaded in __init__
            # ============================================================
            # METHOD 1: Exact token counting with tiktoken
            # ============================================================

            # encode() converts text to list of token IDs
            # len() of that list = number of tokens
            # Example: "Hello world" -> [15496, 1917] -> 2 tokens
            return len(self.tokenizer.encode(text))

        else:
            # ============================================================
            # METHOD 2: Approximate token counting (fallback)
            # ============================================================

            # Approximation: 1 token ≈ 4 characters (rough estimate)
            # This is based on empirical observation that English text
            # averages about 4-5 characters per token
            return len(text) // 4  # Integer division

    # ========================================================================
    # METHOD: chunk_text
    # ========================================================================

    def chunk_text(self, text: str, source_file: str, year_range: str) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.

        This is a critical method that:
        1. Splits long documents into manageable pieces for LLM processing
        2. Maintains context continuity via overlapping chunks
        3. Preserves metadata (source, page number, etc.) for each chunk
        4. Respects token limits to avoid exceeding LLM context windows

        Chunking Strategy:
        - Split by sentences (not mid-word or mid-sentence)
        - Target chunk_size tokens per chunk
        - Overlap chunks by chunk_overlap tokens
        - Track page numbers for citation

        Args:
            text: Full text to be chunked (with [PAGE N] markers)
            source_file: Name of source PDF file
            year_range: Year range metadata (e.g., "1961-2006")

        Returns:
            List[Dict]: List of chunk dictionaries, each containing:
                - text: The chunk text content
                - source_file: Source PDF filename
                - year_range: Time period metadata
                - chunk_id: Sequential chunk number
                - page_number: Page where chunk originated
                - token_count: Number of tokens in chunk

        Example:
            Input: "Very long text..." (2000 tokens)
            chunk_size: 512, overlap: 50
            Output: ~4-5 chunks with 50 tokens overlap between adjacent chunks
        """

        # ================================================================
        # STEP 1: Initialize tracking variables
        # ================================================================

        chunks = []            # List to store completed chunks
        current_chunk = ""     # Text accumulator for current chunk being built
        current_tokens = 0     # Token count for current chunk
        chunk_id = 0           # Sequential ID for chunks (0, 1, 2, ...)
        current_page = 1       # Page number tracker (starts at 1)

        # ================================================================
        # STEP 2: Split text by page markers
        # ================================================================

        # Split text at [PAGE N] markers
        # Pattern: \[PAGE (\d+)\]
        #   \[     - Literal left bracket (escaped because [ is special in regex)
        #   PAGE   - Literal text "PAGE"
        #   \s     - Whitespace
        #   (\d+)  - Capture group: one or more digits (the page number)
        #   \]     - Literal right bracket (escaped)

        # re.split() splits string and includes captured groups
        # Result: ['', '1', 'page 1 text', '2', 'page 2 text', ...]
        # Odd indices (1, 3, 5...) are page numbers
        # Even indices (0, 2, 4...) are page text (index 0 is empty)
        page_sections = re.split(r'\[PAGE (\d+)\]', text)

        # ================================================================
        # STEP 3: Process each page's text
        # ================================================================

        # Start at index 1 (first page number), step by 2
        # This iterates through page numbers at indices 1, 3, 5, ...
        for i in range(1, len(page_sections), 2):
            # ============================================================
            # STEP 3a: Get page number and text
            # ============================================================

            # Check if we have both page number and corresponding text
            if i + 1 < len(page_sections):
                page_num = int(page_sections[i])      # Page number (convert to int)
                page_text = page_sections[i + 1]      # Text for this page

                # ========================================================
                # STEP 3b: Split page text into sentences
                # ========================================================

                # Split on sentence boundaries for cleaner chunks
                # Pattern: (?<=[.!?])\s+
                #   (?<=...)  - Positive lookbehind (match if preceded by...)
                #   [.!?]     - Period, exclamation, or question mark
                #   \s+       - One or more whitespace characters
                # This splits AFTER punctuation, not before
                sentences = re.split(r'(?<=[.!?])\s+', page_text)

                # ========================================================
                # STEP 3c: Process each sentence
                # ========================================================

                for sentence in sentences:
                    # Strip whitespace from sentence
                    sentence = sentence.strip()

                    # Skip empty sentences
                    if not sentence:
                        continue

                    # Count tokens in this sentence
                    sentence_tokens = self.count_tokens(sentence)

                    # ====================================================
                    # STEP 3d: Check if adding sentence exceeds chunk size
                    # ====================================================

                    # If adding this sentence would exceed chunk_size
                    # AND we already have some content in current chunk
                    if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                        # ================================================
                        # CHUNK IS FULL - Save it and start new chunk
                        # ================================================

                        # Create chunk dictionary with all metadata
                        chunks.append({
                            "text": current_chunk.strip(),        # Remove extra whitespace
                            "source_file": source_file,           # PDF filename
                            "year_range": year_range,             # Time period
                            "chunk_id": chunk_id,                 # Sequential ID
                            "page_number": current_page,          # Page number
                            "token_count": current_tokens         # Actual token count
                        })

                        # Increment chunk ID for next chunk
                        chunk_id += 1

                        # ============================================
                        # STEP 3e: Handle overlap between chunks
                        # ============================================

                        # If overlap is configured (> 0)
                        if self.chunk_overlap > 0:
                            # Get last N tokens from previous chunk
                            # This maintains context across chunk boundaries
                            overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)

                            # Start new chunk with overlap + current sentence
                            current_chunk = overlap_text + " " + sentence

                            # Recalculate token count for new chunk
                            current_tokens = self.count_tokens(current_chunk)
                        else:
                            # No overlap - start fresh with current sentence
                            current_chunk = sentence
                            current_tokens = sentence_tokens

                    else:
                        # ================================================
                        # CHUNK NOT FULL - Add sentence to current chunk
                        # ================================================

                        if current_chunk:
                            # If chunk already has content, add space + sentence
                            current_chunk += " " + sentence
                        else:
                            # If chunk is empty, start with this sentence
                            current_chunk = sentence

                        # Add sentence tokens to running total
                        current_tokens += sentence_tokens

                    # Update current page tracker
                    current_page = page_num

        # ================================================================
        # STEP 4: Save final chunk (if any content remains)
        # ================================================================

        # After processing all text, there may be remaining text in current_chunk
        if current_chunk.strip():  # If there's any non-whitespace content
            # Add final chunk to list
            chunks.append({
                "text": current_chunk.strip(),
                "source_file": source_file,
                "year_range": year_range,
                "chunk_id": chunk_id,
                "page_number": current_page,
                "token_count": current_tokens
            })

        # ================================================================
        # STEP 5: Update statistics and return chunks
        # ================================================================

        # Add number of chunks created to running total
        self.stats["total_chunks"] += len(chunks)

        # Return list of all chunks for this document
        return chunks

    # ========================================================================
    # METHOD: _get_overlap_text (PRIVATE HELPER METHOD)
    # ========================================================================

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Extract the last N tokens from text for chunk overlap.

        Overlap maintains context continuity between adjacent chunks.
        For example, if chunk 1 ends with "...photosynthesis in beans"
        and chunk 2 starts with "photosynthesis in beans is affected by...",
        the LLM has better context.

        Args:
            text: Full text to extract overlap from
            overlap_tokens: Number of tokens to extract from end

        Returns:
            str: Last N tokens of text (or entire text if shorter)

        Methods:
            1. If tiktoken available: Extract exact N tokens from end
            2. If not available: Approximate using characters (N tokens ≈ 4N chars)
        """

        # ================================================================
        # METHOD 1: Exact token extraction (if tiktoken available)
        # ================================================================

        if self.tokenizer:  # If we have access to tokenizer
            # Encode text to token IDs
            # Example: "Hello world" -> [15496, 1917]
            tokens = self.tokenizer.encode(text)

            # Check if text has more tokens than we want to extract
            if len(tokens) > overlap_tokens:
                # Get last N token IDs using negative indexing
                # Example: tokens[-50:] gets last 50 tokens
                overlap_token_ids = tokens[-overlap_tokens:]

                # Decode token IDs back to text
                return self.tokenizer.decode(overlap_token_ids)

        # ================================================================
        # METHOD 2: Approximate character extraction (fallback)
        # ================================================================

        # Estimate: 1 token ≈ 4 characters
        overlap_chars = overlap_tokens * 4

        # Extract last N characters (or entire text if shorter)
        # text[-N:] gets last N characters
        # Ternary: use last N chars if text long enough, else use entire text
        return text[-overlap_chars:] if len(text) > overlap_chars else text

    # ========================================================================
    # METHOD: process_single_pdf
    # ========================================================================

    def process_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Process a single PDF file from start to finish.

        This is the main orchestration method that:
        1. Extracts text from PDF
        2. Extracts metadata
        3. Chunks the text
        4. Returns list of chunks with metadata

        Args:
            pdf_path: Path to PDF file to process

        Returns:
            List[Dict]: List of chunk dictionaries with metadata
                       Empty list if processing fails

        Error Handling:
            - Catches all exceptions
            - Logs errors
            - Records error in statistics
            - Returns empty list (doesn't crash entire processing)
        """

        try:
            # ============================================================
            # STEP 1: Extract text from PDF
            # ============================================================

            # Call extract_text_from_pdf method (defined above)
            # Returns: (full_text_string, page_count)
            text, num_pages = self.extract_text_from_pdf(pdf_path)

            # ============================================================
            # STEP 2: Validate extracted text
            # ============================================================

            # Check if we got meaningful text
            # Conditions: no text OR text too short (< 10 chars after stripping)
            if not text or len(text.strip()) < 10:
                # Log warning - not an error, just no extractable text
                self.logger.warning(f"No text extracted from {pdf_path.name}")
                return []  # Return empty list

            # ============================================================
            # STEP 3: Extract metadata
            # ============================================================

            # Get year range from path/filename (method defined above)
            year_range = self.extract_year_range(pdf_path)

            # Get just the filename (not full path)
            # Example: Path("/data/pdfs/file.pdf").name -> "file.pdf"
            source_file = pdf_path.name

            # ============================================================
            # STEP 4: Chunk the text
            # ============================================================

            # Call chunk_text method (defined above)
            # This splits text into chunks with metadata
            chunks = self.chunk_text(text, source_file, year_range)

            # ============================================================
            # STEP 5: Log success
            # ============================================================

            # Log informational message about processing result
            self.logger.info(
                f"Processed {pdf_path.name}: {num_pages} pages → {len(chunks)} chunks"
            )

            # ============================================================
            # STEP 6: Return chunks
            # ============================================================

            return chunks

        except Exception as e:
            # ============================================================
            # STEP 7: Handle errors gracefully
            # ============================================================

            # Log error with filename
            self.logger.error(f"Error processing {pdf_path.name}: {e}")

            # Log full stack trace at debug level (for detailed diagnostics)
            self.logger.debug(traceback.format_exc())

            # Record error in statistics for reporting
            self.stats["errors"].append({
                "file": str(pdf_path),  # Full path as string
                "error": str(e)         # Error message
            })

            # Return empty list (processing continues with next file)
            return []

    # ========================================================================
    # METHOD: process_all_pdfs
    # ========================================================================

    def process_all_pdfs(self, batch_size: int = 50) -> List[Dict]:
        """
        Process all PDF files in the input directory.

        This is the main processing loop that:
        1. Finds all PDFs
        2. Processes each one
        3. Collects all chunks
        4. Saves checkpoints periodically
        5. Shows progress bar

        Memory Efficiency:
        - Processes one PDF at a time (not all at once)
        - Periodic checkpoints prevent losing all work if crash occurs
        - Progress bar provides user feedback

        Args:
            batch_size: Save checkpoint after every N files (default: 50)

        Returns:
            List[Dict]: Combined list of all chunks from all PDFs
        """

        # ================================================================
        # STEP 1: Find all PDF files
        # ================================================================

        # Call find_all_pdfs method (returns list of Path objects)
        pdf_files = self.find_all_pdfs()

        # Record total count in statistics
        self.stats["total_pdfs"] = len(pdf_files)

        # ================================================================
        # STEP 2: Initialize chunk collection list
        # ================================================================

        # This will hold all chunks from all PDFs
        all_chunks = []

        # ================================================================
        # STEP 3: Process all PDFs with progress bar
        # ================================================================

        # tqdm creates a progress bar
        # with statement ensures progress bar is closed properly
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:

            # Loop through each PDF file with enumeration for checkpoint tracking
            # enumerate() gives us (index, value) pairs
            # Example: (0, file1.pdf), (1, file2.pdf), ...
            for idx, pdf_path in enumerate(pdf_files):
                try:
                    # ====================================================
                    # STEP 3a: Process single PDF
                    # ====================================================

                    # Call process_single_pdf method (returns list of chunks)
                    chunks = self.process_single_pdf(pdf_path)

                    # ====================================================
                    # STEP 3b: Update statistics based on result
                    # ====================================================

                    if chunks:  # If we got chunks (processing succeeded)
                        # Add this PDF's chunks to combined list
                        # extend() adds all items from chunks list to all_chunks
                        all_chunks.extend(chunks)

                        # Increment successful count
                        self.stats["successful"] += 1
                    else:  # No chunks (processing failed or no text)
                        # Increment failed count
                        self.stats["failed"] += 1

                    # ====================================================
                    # STEP 3c: Save checkpoint if batch_size reached
                    # ====================================================

                    # Check if we've processed a multiple of batch_size files
                    # Example: idx=49 -> (49+1) % 50 = 0 -> save checkpoint
                    if (idx + 1) % batch_size == 0:
                        # Save checkpoint (method defined below)
                        self._save_checkpoint(all_chunks, idx + 1)

                except Exception as e:
                    # ====================================================
                    # STEP 3d: Handle unexpected errors
                    # ====================================================

                    # This catches errors not caught by process_single_pdf
                    self.logger.error(f"Unexpected error with {pdf_path.name}: {e}")

                    # Increment failed count
                    self.stats["failed"] += 1

                finally:
                    # ====================================================
                    # STEP 3e: Update progress bar (happens no matter what)
                    # ====================================================

                    # finally block always executes
                    # Update progress bar by 1 step
                    pbar.update(1)

        # ================================================================
        # STEP 4: Return all collected chunks
        # ================================================================

        return all_chunks

    # ========================================================================
    # METHOD: _save_checkpoint (PRIVATE METHOD)
    # ========================================================================

    def _save_checkpoint(self, chunks: List[Dict], num_processed: int):
        """
        Save intermediate results to prevent data loss.

        Checkpoints are important because:
        - Processing 1,067 PDFs takes hours
        - If crash occurs, checkpoints let you recover work
        - Can manually resume from checkpoint if needed

        Args:
            chunks: Current list of all chunks processed so far
            num_processed: Number of PDFs processed (for filename)

        Checkpoint Filename:
            checkpoint_50.json  (after 50 files)
            checkpoint_100.json (after 100 files)
            etc.
        """

        # ================================================================
        # STEP 1: Create checkpoint filename
        # ================================================================

        # Get output file's parent directory (e.g., 'data/')
        # Create filename: checkpoint_N.json
        checkpoint_file = self.output_file.parent / f"checkpoint_{num_processed}.json"

        # ================================================================
        # STEP 2: Try to save checkpoint
        # ================================================================

        try:
            # Open file in write mode with UTF-8 encoding
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                # Write chunks as JSON
                # indent=2: Pretty-print with 2-space indentation
                # ensure_ascii=False: Keep unicode characters as-is
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            # Log successful checkpoint save
            self.logger.info(f"Checkpoint saved: {checkpoint_file} ({len(chunks)} chunks)")

        except Exception as e:
            # ================================================================
            # STEP 3: Handle checkpoint save errors
            # ================================================================

            # Log warning (not error - checkpoint failure shouldn't stop processing)
            self.logger.warning(f"Failed to save checkpoint: {e}")

    # ========================================================================
    # METHOD: save_chunks
    # ========================================================================

    def save_chunks(self, chunks: List[Dict]):
        """
        Save final processed chunks to JSON file.

        This method:
        1. Creates output directory if needed
        2. Saves chunks to main output file
        3. Saves processing metadata to separate file

        Args:
            chunks: Complete list of all processed chunks

        Output Files:
            1. processed_chunks.json - The chunk data
            2. processing_metadata.json - Statistics and configuration
        """

        # ================================================================
        # STEP 1: Ensure output directory exists
        # ================================================================

        # Get parent directory of output file
        # Create it if it doesn't exist
        # parents=True: Create intermediate directories too
        # exist_ok=True: Don't error if already exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # ============================================================
            # STEP 2: Save chunks to JSON file
            # ============================================================

            # Open output file for writing with UTF-8 encoding
            with open(self.output_file, 'w', encoding='utf-8') as f:
                # Write chunks as JSON
                # indent=2: Pretty-print for readability
                # ensure_ascii=False: Preserve unicode characters
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            # Log success message
            self.logger.info(f"Saved {len(chunks)} chunks to {self.output_file}")

            # ============================================================
            # STEP 3: Create metadata dictionary
            # ============================================================

            metadata = {
                # ISO format timestamp (YYYY-MM-DDTHH:MM:SS)
                "processing_date": datetime.now().isoformat(),

                # All statistics collected during processing
                "statistics": self.stats,

                # Configuration parameters used
                "configuration": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "use_ocr": self.use_ocr
                },

                # Total chunk count (redundant but convenient)
                "total_chunks": len(chunks)
            }

            # ============================================================
            # STEP 4: Save metadata to separate file
            # ============================================================

            # Create metadata filename in same directory as chunks
            metadata_file = self.output_file.parent / "processing_metadata.json"

            # Write metadata as JSON
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            # Log metadata save
            self.logger.info(f"Saved metadata to {metadata_file}")

        except Exception as e:
            # ============================================================
            # STEP 5: Handle save errors
            # ============================================================

            # Log error
            self.logger.error(f"Failed to save chunks: {e}")

            # Re-raise exception (this is critical - can't continue without saving)
            raise

    # ========================================================================
    # METHOD: print_statistics
    # ========================================================================

    def print_statistics(self):
        """
        Print comprehensive processing statistics to console.

        This provides a summary of:
        - How many PDFs were processed
        - Success/failure counts
        - Total pages and chunks
        - OCR usage (if enabled)
        - Average chunks per PDF
        - Error count
        """

        # ================================================================
        # Print formatted statistics table
        # ================================================================

        # Print visual separator (80 equals signs)
        print("\n" + "="*80)

        # Print header
        print("PROCESSING STATISTICS")

        # Print separator
        print("="*80)

        # Print each statistic with aligned labels
        print(f"Total PDFs found:      {self.stats['total_pdfs']}")
        print(f"Successfully processed: {self.stats['successful']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"Total pages:           {self.stats['total_pages']}")
        print(f"Total chunks created:  {self.stats['total_chunks']}")

        # Only show OCR stats if OCR was enabled
        if self.use_ocr:
            print(f"OCR used (pages):      {self.stats['ocr_used']}")

        # Calculate and print average chunks per PDF
        if self.stats['total_chunks'] > 0:
            # Use max() to avoid division by zero
            avg_chunks = self.stats['total_chunks'] / max(self.stats['successful'], 1)
            # Format to 1 decimal place
            print(f"Average chunks/PDF:    {avg_chunks:.1f}")

        # Print error summary if any errors occurred
        if self.stats['errors']:
            print(f"\nErrors encountered:    {len(self.stats['errors'])}")
            print("Check log file for details.")

        # Print closing separator
        print("="*80 + "\n")

    # ========================================================================
    # METHOD: run
    # ========================================================================

    def run(self):
        """
        Main processing pipeline - orchestrates the entire workflow.

        This is the top-level method that:
        1. Starts processing
        2. Processes all PDFs
        3. Saves results
        4. Prints statistics
        5. Handles any fatal errors

        Returns:
            List[Dict]: All processed chunks (or raises exception on fatal error)
        """

        try:
            # ============================================================
            # STEP 1: Log pipeline start
            # ============================================================

            self.logger.info("Starting PDF processing pipeline...")

            # ============================================================
            # STEP 2: Process all PDFs
            # ============================================================

            # Call process_all_pdfs (returns list of all chunks)
            chunks = self.process_all_pdfs()

            # ============================================================
            # STEP 3: Save results
            # ============================================================

            if chunks:  # If we got any chunks
                # Save to file
                self.save_chunks(chunks)
            else:  # No chunks created at all
                # Log warning
                self.logger.warning("No chunks were created!")

            # ============================================================
            # STEP 4: Print statistics summary
            # ============================================================

            self.print_statistics()

            # ============================================================
            # STEP 5: Log completion
            # ============================================================

            self.logger.info("Processing complete!")

            # ============================================================
            # STEP 6: Return chunks
            # ============================================================

            return chunks

        except Exception as e:
            # ============================================================
            # STEP 7: Handle fatal errors
            # ============================================================

            # Log error
            self.logger.error(f"Fatal error in processing pipeline: {e}")

            # Log full stack trace
            self.logger.debug(traceback.format_exc())

            # Re-raise exception (let caller handle)
            raise


# ============================================================================
# MAIN FUNCTION - Command-line interface
# ============================================================================

def main():
    """
    Main entry point for command-line usage.

    This function:
    1. Parses command-line arguments
    2. Creates PDFProcessor instance
    3. Runs processing

    Command-line Arguments:
        --input-dir: Where to find PDFs
        --output-file: Where to save chunks
        --log-file: Where to write logs
        --chunk-size: Tokens per chunk
        --chunk-overlap: Overlapping tokens
        --use-ocr: Enable OCR flag
        --batch-size: Checkpoint frequency

    Usage:
        python pdf_processor.py --chunk-size 256 --use-ocr
    """

    # ========================================================================
    # STEP 1: Import argparse for command-line argument parsing
    # ========================================================================

    import argparse

    # ========================================================================
    # STEP 2: Create argument parser
    # ========================================================================

    # ArgumentParser handles command-line arguments
    parser = argparse.ArgumentParser(
        description="Process PDF documents for LLM-based QA system"
    )

    # ========================================================================
    # STEP 3: Define all command-line arguments
    # ========================================================================

    # Input directory argument
    parser.add_argument(
        "--input-dir",
        type=str,  # Argument type
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs",  # Default value
        help="Input directory containing PDFs"  # Help text
    )

    # Output file argument
    parser.add_argument(
        "--output-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/processed_chunks.json",
        help="Output JSON file for chunks"
    )

    # Log file argument
    parser.add_argument(
        "--log-file",
        type=str,
        default="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/logs/processing.log",
        help="Log file path"
    )

    # Chunk size argument
    parser.add_argument(
        "--chunk-size",
        type=int,  # Integer type
        default=512,
        help="Chunk size in tokens"
    )

    # Chunk overlap argument
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks in tokens"
    )

    # OCR flag (action="store_true" means presence of flag sets to True)
    parser.add_argument(
        "--use-ocr",
        action="store_true",  # Boolean flag (no value needed)
        help="Enable OCR for scanned documents"
    )

    # Batch size argument
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Save checkpoint every N files"
    )

    # ========================================================================
    # STEP 4: Parse arguments
    # ========================================================================

    # parse_args() reads sys.argv and returns namespace object
    args = parser.parse_args()

    # ========================================================================
    # STEP 5: Create PDFProcessor instance with parsed arguments
    # ========================================================================

    processor = PDFProcessor(
        input_dir=args.input_dir,          # From --input-dir argument
        output_file=args.output_file,      # From --output-file argument
        log_file=args.log_file,            # From --log-file argument
        chunk_size=args.chunk_size,        # From --chunk-size argument
        chunk_overlap=args.chunk_overlap,  # From --chunk-overlap argument
        use_ocr=args.use_ocr               # From --use-ocr flag
    )

    # ========================================================================
    # STEP 6: Run processing
    # ========================================================================

    # Call run() method to start processing
    processor.run()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# This block only executes if script is run directly (not imported as module)
if __name__ == "__main__":
    main()  # Call main function

# ============================================================================
# END OF FILE
# ============================================================================
