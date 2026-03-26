# Complete Code Documentation

## Overview

This document provides comprehensive documentation for all code in the PDF processing system.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Structure](#file-structure)
3. [Main Components](#main-components)
4. [Code Walkthrough](#code-walkthrough)
5. [Function Reference](#function-reference)
6. [Data Flow](#data-flow)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF PROCESSING PIPELINE                   │
└─────────────────────────────────────────────────────────────┘

Input: PDF Files (1,067 documents)
    │
    ├── 1961-2006/ (964 PDFs)
    └── 2007-2026/ (103 PDFs)
    │
    ▼
┌─────────────────────┐
│  PDFProcessor Class │
└─────────────────────┘
    │
    ├── 1. Find all PDFs recursively
    │
    ├── 2. For each PDF:
    │   ├── Extract text (PyPDF2)
    │   ├── Optional: OCR if needed
    │   ├── Extract metadata (year, page)
    │   └── Chunk text (token-based)
    │
    ├── 3. Save checkpoints (every 50 PDFs)
    │
    └── 4. Output results
        │
        ├── processed_chunks.json (15,000-20,000 chunks)
        └── processing_metadata.json (statistics)
```

### Key Design Principles

1. **Memory Efficiency**: Process one PDF at a time
2. **Fault Tolerance**: Checkpoint saves, error handling
3. **Metadata Preservation**: Track source, page, year
4. **Token Accuracy**: Use tiktoken for exact counting
5. **Context Continuity**: Overlapping chunks

---

## File Structure

```
hpcc-llm-qa/
├── src/
│   ├── pdf_processor.py              # Main processor (production)
│   ├── pdf_processor_commented.py    # Fully commented version
│   └── test_processor.py             # Test suite
│
├── data/
│   ├── pdfs/                         # Input PDFs (symlinks)
│   ├── processed_chunks.json         # Output: all chunks
│   ├── processing_metadata.json      # Output: statistics
│   └── checkpoint_*.json             # Intermediate saves
│
├── jobs/
│   └── process_pdfs.sb               # SLURM batch script
│
├── logs/
│   ├── processing.log                # Main log file
│   └── pdf_processing_*.{out,err}    # SLURM logs
│
├── scripts/
│   ├── setup_pdf_processing.sh       # Setup automation
│   └── analyze_and_copy_pdfs.sh      # PDF organization
│
├── config/
│   ├── module_setup.sh               # HPCC module loading
│   └── HPCC_SETUP_GUIDE.md          # HPCC documentation
│
├── requirements.txt                  # Python dependencies
├── README_PDF_PROCESSOR.md          # User guide
├── QUICK_START.md                   # Quick reference
└── CODE_DOCUMENTATION.md            # This file
```

---

## Main Components

### 1. PDFProcessor Class

**Location**: `src/pdf_processor.py`

**Purpose**: Main class that orchestrates all PDF processing

**Key Attributes**:
```python
self.input_dir        # Path to PDF directory
self.output_file      # Path to output JSON
self.log_file         # Path to log file
self.chunk_size       # Target tokens per chunk (default: 512)
self.chunk_overlap    # Overlapping tokens (default: 50)
self.use_ocr          # Boolean: use OCR for scanned PDFs
self.tokenizer        # tiktoken tokenizer instance
self.logger           # Logging instance
self.stats            # Statistics dictionary
```

**Main Methods**:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `__init__()` | Initialize processor | Config params | None |
| `find_all_pdfs()` | Find PDFs recursively | None | List[Path] |
| `extract_text_from_pdf()` | Extract text from PDF | Path | (text, pages) |
| `chunk_text()` | Split text into chunks | text, metadata | List[Dict] |
| `process_single_pdf()` | Process one PDF | Path | List[Dict] |
| `process_all_pdfs()` | Process all PDFs | batch_size | List[Dict] |
| `save_chunks()` | Save to JSON | chunks | None |
| `run()` | Main pipeline | None | List[Dict] |

---

## Code Walkthrough

### Step 1: Initialization

```python
processor = PDFProcessor(
    input_dir="data/pdfs",
    output_file="data/processed_chunks.json",
    log_file="logs/processing.log",
    chunk_size=512,
    chunk_overlap=50,
    use_ocr=False
)
```

**What happens**:
1. Convert paths to Path objects
2. Load tiktoken tokenizer
3. Set up logging (file + console)
4. Initialize statistics dictionary

### Step 2: Find PDFs

```python
pdf_files = processor.find_all_pdfs()
# Returns: [Path('data/pdfs/1961-2006/file1.pdf'), ...]
```

**What happens**:
1. Use `rglob("*.pdf")` to find all PDFs recursively
2. Return list of Path objects
3. Log count to console and file

### Step 3: Process Each PDF

```python
for pdf_path in pdf_files:
    chunks = processor.process_single_pdf(pdf_path)
```

**What happens for each PDF**:

#### 3a. Extract Text
```python
text, num_pages = extract_text_from_pdf(pdf_path)
```
- Open PDF with PyPDF2
- Loop through pages
- Extract text from each page
- Add [PAGE N] markers
- Optionally use OCR if text minimal

#### 3b. Extract Metadata
```python
year_range = extract_year_range(pdf_path)  # "1961-2006"
source_file = pdf_path.name                # "file.pdf"
```

#### 3c. Chunk Text
```python
chunks = chunk_text(text, source_file, year_range)
```

**Chunking algorithm**:
1. Split by page markers
2. Split pages by sentences
3. Build chunks sentence by sentence
4. When chunk reaches target size:
   - Save current chunk
   - Start new chunk with overlap from previous
5. Track page numbers throughout

**Example chunk creation**:
```
Input text: 1000 tokens
Chunk size: 512 tokens
Overlap: 50 tokens

Chunk 0: tokens 0-512     (512 tokens)
Chunk 1: tokens 462-974   (512 tokens, 50 overlap with chunk 0)
Chunk 2: tokens 924-1000  (76 tokens, final chunk)
```

### Step 4: Save Results

```python
processor.save_chunks(all_chunks)
```

**Creates two files**:

1. **processed_chunks.json**:
```json
[
  {
    "text": "Bean breeding programs...",
    "source_file": "10.2135_cropsci1969.pdf",
    "year_range": "1961-2006",
    "chunk_id": 0,
    "page_number": 1,
    "token_count": 487
  },
  ...
]
```

2. **processing_metadata.json**:
```json
{
  "processing_date": "2026-03-21T14:30:00",
  "statistics": {
    "total_pdfs": 1067,
    "successful": 1067,
    "failed": 0,
    "total_chunks": 15234,
    "total_pages": 8456
  },
  "configuration": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "use_ocr": false
  }
}
```

---

## Function Reference

### Core Processing Functions

#### `find_all_pdfs()`

**Purpose**: Recursively find all PDF files

**Algorithm**:
```python
1. Use Path.rglob("*.pdf") to find all PDFs
2. Convert iterator to list
3. Log count
4. Return list of Paths
```

**Time Complexity**: O(n) where n = total files in directory tree

**Example**:
```python
# Input directory:
# data/pdfs/
#   ├── 1961-2006/
#   │   ├── file1.pdf
#   │   └── file2.pdf
#   └── 2007-2026/
#       └── file3.pdf

pdf_files = find_all_pdfs()
# Returns: [Path(.../file1.pdf), Path(.../file2.pdf), Path(.../file3.pdf)]
```

---

#### `extract_year_range(file_path)`

**Purpose**: Extract year metadata from path/filename

**Algorithm**:
```python
1. Convert path to string
2. Search for pattern YYYY-YYYY (e.g., "1961-2006")
   - If found, return range
3. If not found, search for single year (19XX or 20XX)
   - If found, return year
4. If neither found, return "unknown"
```

**Regex Patterns**:
- Year range: `(\d{4})-(\d{4})`
- Single year: `(19|20)\d{2}`

**Examples**:
```python
extract_year_range("/data/1961-2006/file.pdf")    # "1961-2006"
extract_year_range("/data/file_2008.pdf")         # "2008"
extract_year_range("/data/file.pdf")              # "unknown"
```

---

#### `extract_text_from_pdf(pdf_path)`

**Purpose**: Extract all text from PDF file

**Algorithm**:
```python
1. Open PDF with PyPDF2.PdfReader
2. Get page count
3. For each page:
   a. Extract text with page.extract_text()
   b. If OCR enabled AND text minimal:
      - Convert page to image (300 DPI)
      - Run Tesseract OCR
      - Use OCR text instead
   c. Add "[PAGE N]\n" marker
   d. Append to text_content list
4. Join all pages with "\n\n"
5. Return (full_text, num_pages)
```

**Error Handling**:
- Page-level errors: Log warning, continue to next page
- PDF-level errors: Log error, raise exception

**Output Format**:
```
[PAGE 1]
First page text content here...

[PAGE 2]
Second page text content here...
```

---

#### `count_tokens(text)`

**Purpose**: Count tokens in text string

**Algorithm**:
```python
IF tiktoken available:
    1. Encode text to token IDs
    2. Return length of token ID list
ELSE (fallback):
    1. Calculate: len(text) // 4
    2. Return approximate count
```

**Why Accurate Counting Matters**:
- LLMs have token limits (e.g., 8192, 16384, 32768)
- Pricing often based on tokens
- Need precise chunks for optimal retrieval

**Example**:
```python
text = "Hello, how are you doing today?"

# With tiktoken:
count_tokens(text)  # Returns: 8 (exact)

# Without tiktoken (approximation):
count_tokens(text)  # Returns: 7 (32 chars / 4)
```

---

#### `chunk_text(text, source_file, year_range)`

**Purpose**: Split text into overlapping chunks with metadata

**Algorithm**:
```python
1. Split text by [PAGE N] markers
   - Extract page numbers
   - Extract page text

2. For each page:
   a. Split into sentences (at .!? boundaries)
   b. For each sentence:
      i.   Count tokens in sentence
      ii.  If adding sentence exceeds chunk_size:
           - Save current chunk
           - Start new chunk with overlap
      iii. Else:
           - Add sentence to current chunk
      iv.  Track page number

3. Save final chunk if any content remains

4. Return list of chunk dictionaries
```

**Chunk Dictionary Structure**:
```python
{
    "text": str,           # Chunk text content
    "source_file": str,    # Source PDF filename
    "year_range": str,     # Time period metadata
    "chunk_id": int,       # Sequential ID (0, 1, 2, ...)
    "page_number": int,    # Page where chunk originated
    "token_count": int     # Actual token count
}
```

**Overlap Mechanism**:
```python
# When chunk is full:
if chunk_overlap > 0:
    # Get last N tokens from previous chunk
    overlap_text = get_overlap_text(current_chunk, chunk_overlap)
    # Start new chunk with: overlap + current sentence
    new_chunk = overlap_text + " " + sentence
```

**Why Overlap**:
- Maintains context across chunk boundaries
- Helps LLM understand connections
- Improves retrieval quality

**Example**:
```
Full text (1000 tokens):
"Bean varieties have shown resistance to drought.
Drought resistance is crucial for climate adaptation.
Climate adaptation strategies include genetic modification..."

With chunk_size=512, overlap=50:

Chunk 0: "Bean varieties... drought resistance is..." (512 tokens)
Chunk 1: "...drought resistance is... genetic modification..." (512 tokens)
         ↑ This 50 tokens appears in both chunks (overlap)
```

---

### Helper Functions

#### `_get_overlap_text(text, overlap_tokens)`

**Purpose**: Extract last N tokens from text for overlap

**Algorithm**:
```python
IF tokenizer available:
    1. Encode text to tokens
    2. Get last N token IDs: tokens[-overlap_tokens:]
    3. Decode back to text
    4. Return overlap text
ELSE (fallback):
    1. Estimate: N tokens ≈ 4N characters
    2. Get last 4N characters
    3. Return overlap text
```

---

#### `_ocr_page(pdf_path, page_num)`

**Purpose**: Perform OCR on a PDF page

**Algorithm**:
```python
1. Convert PDF page to image (300 DPI)
   - Uses pdf2image library
   - Requires poppler-utils

2. Run Tesseract OCR on image
   - pytesseract.image_to_string()

3. Return extracted text (or None if failed)
```

**When Used**:
- Only if `use_ocr=True`
- Only if normal text extraction yields < 50 characters
- Indicates scanned/image-based PDF

**Performance**:
- Much slower than text extraction (~10-100x)
- Requires additional dependencies

---

#### `_save_checkpoint(chunks, num_processed)`

**Purpose**: Save intermediate results

**Why Important**:
- Processing 1000+ PDFs takes hours
- Prevents data loss if crash/timeout
- Allows manual recovery

**Checkpoint Files**:
```
checkpoint_50.json   # After 50 PDFs
checkpoint_100.json  # After 100 PDFs
checkpoint_150.json  # After 150 PDFs
...
```

**Algorithm**:
```python
1. Create filename: checkpoint_{num}.json
2. Write chunks to JSON
3. Log success (or warning if failed)
```

---

## Data Flow

### Complete Pipeline Diagram

```
┌─────────────┐
│   Input     │
│ 1067 PDFs   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│ 1. Find All PDFs            │
│    - Recursive glob search  │
│    - Filter .pdf extension  │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ 2. For Each PDF:            │
│                             │
│  ┌─────────────────────┐   │
│  │ Extract Text        │   │
│  │ - PyPDF2            │   │
│  │ - Optional: OCR     │   │
│  └──────┬──────────────┘   │
│         │                   │
│         ▼                   │
│  ┌─────────────────────┐   │
│  │ Extract Metadata    │   │
│  │ - Year range        │   │
│  │ - Source file       │   │
│  └──────┬──────────────┘   │
│         │                   │
│         ▼                   │
│  ┌─────────────────────┐   │
│  │ Chunk Text          │   │
│  │ - Split by sentence │   │
│  │ - 512 token chunks  │   │
│  │ - 50 token overlap  │   │
│  └──────┬──────────────┘   │
│         │                   │
└─────────┼───────────────────┘
          │
          ▼
   ┌──────────────┐
   │  All Chunks  │
   │ (in memory)  │
   └──────┬───────┘
          │
          ├─────────────────┐
          │                 │
          ▼                 ▼
   ┌─────────────┐   ┌─────────────┐
   │  Save Main  │   │    Save     │
   │   Output    │   │  Metadata   │
   └──────┬──────┘   └──────┬──────┘
          │                 │
          ▼                 ▼
┌──────────────────┐ ┌─────────────────┐
│processed_        │ │ processing_     │
│chunks.json       │ │ metadata.json   │
│                  │ │                 │
│ 15,000-20,000    │ │ Statistics      │
│ chunks           │ │ Configuration   │
└──────────────────┘ └─────────────────┘
```

### Data Transformations

**Stage 1: PDF → Raw Text**
```
Input: file.pdf (binary)
Process: PyPDF2 extraction
Output: "Bean varieties have..."
```

**Stage 2: Raw Text → Marked Text**
```
Input: "Bean varieties have..."
Process: Add page markers
Output: "[PAGE 1]\nBean varieties have..."
```

**Stage 3: Marked Text → Chunks**
```
Input: "[PAGE 1]\nBean varieties...\n\n[PAGE 2]\nDrought resistance..."
Process: Token-based chunking
Output: [
  {text: "Bean varieties...", chunk_id: 0, page: 1},
  {text: "...drought resistance...", chunk_id: 1, page: 2}
]
```

**Stage 4: Chunks → JSON**
```
Input: Python list of dictionaries
Process: JSON serialization
Output: File on disk
```

---

## Error Handling

### Error Hierarchy

```
1. Non-Fatal Errors (logged, processing continues)
   ├── Single page extraction failure
   ├── OCR failure
   ├── Checkpoint save failure
   └── Minimal text extraction

2. File-Level Errors (logged, file skipped)
   ├── Corrupted PDF
   ├── Permission denied
   ├── Encoding issues
   └── Empty file

3. Fatal Errors (logged, processing stops)
   ├── Output directory not writable
   ├── Cannot save final output
   └── Out of memory
```

### Error Handling Strategy

**Page-Level Errors**:
```python
try:
    text = page.extract_text()
except Exception as e:
    logger.warning(f"Page {page_num} failed: {e}")
    continue  # Skip page, process next
```

**File-Level Errors**:
```python
try:
    chunks = process_single_pdf(pdf_path)
except Exception as e:
    logger.error(f"File {pdf_path} failed: {e}")
    stats["failed"] += 1
    return []  # Skip file, process next
```

**Fatal Errors**:
```python
try:
    save_chunks(chunks)
except Exception as e:
    logger.error(f"Cannot save output: {e}")
    raise  # Stop processing, propagate error
```

### Logged Information

**INFO Level** (normal operation):
- Processing start/complete
- File counts
- Successful processing
- Checkpoint saves

**WARNING Level** (unexpected but recoverable):
- Page extraction failures
- OCR failures
- No text extracted
- Checkpoint save failures

**ERROR Level** (serious problems):
- File processing failures
- Cannot read PDF
- Fatal errors

**DEBUG Level** (detailed diagnostics):
- OCR usage
- Token counts
- Full stack traces

---

## Performance Considerations

### Memory Usage

**Per PDF**:
- PDF object: 5-50 MB
- Extracted text: 0.1-5 MB
- Chunks: 0.5-2 MB

**Total in Memory**:
- One PDF at a time: ~10-100 MB
- All chunks list: 100-500 MB
- Total: ~200-600 MB typical

**Optimization Strategies**:
1. Process one PDF at a time (not all at once)
2. Clear variables after each file
3. Use generators where possible
4. Periodic checkpoint saves

### Processing Speed

**Without OCR**:
- ~5-10 PDFs/minute
- 1,067 PDFs in ~1-2 hours

**With OCR**:
- ~1-2 PDFs/minute
- 1,067 PDFs in ~6-10 hours

**Bottlenecks**:
1. PDF parsing (PyPDF2)
2. OCR (if enabled) - 10-100x slower
3. Token encoding (tiktoken)
4. File I/O (writing checkpoints)

### Optimization Tips

**For Faster Processing**:
1. Disable OCR if not needed
2. Increase batch_size for fewer checkpoints
3. Use SSD for faster I/O
4. Process on compute node (not login node)

**For Lower Memory**:
1. Decrease batch_size for more frequent checkpoints
2. Process fewer files at once
3. Delete checkpoints after processing

---

## Testing Strategy

### Unit Tests (test_processor.py)

**What It Tests**:
1. Dependency availability
2. PDF discovery
3. Text extraction (5 samples)
4. Chunking logic
5. Metadata extraction
6. Output format

**How to Run**:
```bash
python3 src/test_processor.py
```

**Expected Output**:
```
✅ Dependencies OK
✅ Found 1067 PDFs
✅ Testing with 5 samples
✅ Created 73 chunks
✅ Sample chunk looks correct
```

### Integration Test (Full Pipeline)

```bash
# Test with small subset
python3 src/pdf_processor.py \
    --input-dir data/test_subset \
    --output-file data/test_output.json
```

---

## Troubleshooting Guide

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module named 'PyPDF2'` | Missing dependency | `pip install PyPDF2` |
| `No PDFs found` | Wrong directory | Check `--input-dir` path |
| `Permission denied` | File permissions | Check file/directory permissions |
| `Out of memory` | Too many chunks | Increase `--batch-size` |
| `Encoding error` | Special characters | Already handled (utf-8) |
| `OCR failed` | Missing tesseract | Install tesseract or disable OCR |

### Debug Commands

```bash
# Check dependencies
python3 -c "import PyPDF2, tiktoken, tqdm; print('OK')"

# Find PDFs manually
find data/pdfs -name "*.pdf" | wc -l

# Check output directory
ls -la data/

# View logs
tail -f logs/processing.log

# Test single PDF
python3 -c "from src.pdf_processor import PDFProcessor; p = PDFProcessor(...); p.process_single_pdf(Path('file.pdf'))"
```

---

## Next Steps

After processing completes:

1. **Verify Output**:
   ```bash
   ls -lh data/processed_chunks.json
   python3 -c "import json; print(len(json.load(open('data/processed_chunks.json'))))"
   ```

2. **Create Embeddings**:
   - Use sentence-transformers
   - Convert chunks to vectors
   - Store in vector database

3. **Set Up Vector Database**:
   - ChromaDB or FAISS
   - Index chunks by embeddings
   - Enable semantic search

4. **Build QA Interface**:
   - LangChain for orchestration
   - LLM for answer generation
   - Retrieval-augmented generation

---

## References

- **PyPDF2**: https://pypdf2.readthedocs.io/
- **tiktoken**: https://github.com/openai/tiktoken
- **tqdm**: https://tqdm.github.io/
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **MSU HPCC**: https://docs.icer.msu.edu/

---

**Document Version**: 1.0
**Last Updated**: 2026-03-21
**Author**: MSU HPCC Bean Lab
