# PDF Processor for LLM QA System

Comprehensive PDF processing pipeline for the Bean Lab agricultural research document collection (1,067 PDFs from 1961-2026).

## Overview

This processor extracts text from PDFs, chunks them into manageable pieces, and creates structured metadata for vector database ingestion.

### Features

- ✅ **Recursive PDF discovery** in subdirectories
- ✅ **Text extraction** from PDF files using PyPDF2
- ✅ **OCR support** for scanned documents (optional)
- ✅ **Smart chunking** with token-based overlap
- ✅ **Metadata tracking** (source, year, page, chunk ID)
- ✅ **Progress tracking** with tqdm
- ✅ **Error handling** and logging
- ✅ **Memory efficient** for HPCC
- ✅ **Checkpoint saves** during processing
- ✅ **Statistics reporting**

---

## Quick Start

### 1. Install Dependencies

```bash
cd /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa

# Activate your conda environment
conda activate bean_llm

# Install required packages
pip install -r requirements.txt
```

### 2. Set Up PDF Directory

You need PDFs in `data/pdfs/`. Choose one option:

**Option A: Symbolic Links (Recommended - saves 347MB)**
```bash
mkdir -p data/pdfs
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006 data/pdfs/1961-2006
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026 data/pdfs/2007-2026
```

**Option B: Copy PDFs**
```bash
bash scripts/analyze_and_copy_pdfs.sh
```

### 3. Test on Sample

```bash
# Test with 5 PDFs first
python3 src/test_processor.py
```

### 4. Process All PDFs

**Interactive (on login node - not recommended for 1,067 PDFs):**
```bash
python3 src/pdf_processor.py
```

**SLURM Job (Recommended):**
```bash
sbatch jobs/process_pdfs.sb
```

---

## Usage Examples

### Basic Usage

```python
from pdf_processor import PDFProcessor

# Create processor
processor = PDFProcessor(
    input_dir="data/pdfs",
    output_file="data/processed_chunks.json",
    log_file="logs/processing.log",
    chunk_size=512,
    chunk_overlap=50
)

# Run processing
chunks = processor.run()
```

### Command Line

```bash
python3 src/pdf_processor.py \
    --input-dir data/pdfs \
    --output-file data/processed_chunks.json \
    --log-file logs/processing.log \
    --chunk-size 512 \
    --chunk-overlap 50 \
    --batch-size 50
```

### With OCR (for scanned PDFs)

```bash
python3 src/pdf_processor.py --use-ocr
```

**Note:** OCR requires additional dependencies:
```bash
pip install pytesseract pdf2image
# On HPCC, you may also need: module load tesseract
```

---

## Output Format

### Processed Chunks JSON

Each chunk has this structure:

```json
{
  "text": "The actual text content of this chunk...",
  "source_file": "10.2135_cropsci1969.0011183X000900030028x.pdf",
  "year_range": "1961-2006",
  "chunk_id": 0,
  "page_number": 1,
  "token_count": 487
}
```

### Processing Metadata JSON

Statistics and configuration:

```json
{
  "processing_date": "2026-03-21T14:30:00",
  "statistics": {
    "total_pdfs": 1067,
    "successful": 1067,
    "failed": 0,
    "total_chunks": 15234,
    "total_pages": 8456,
    "ocr_used": 0
  },
  "configuration": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "use_ocr": false
  },
  "total_chunks": 15234
}
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dir` | `data/pdfs` | Directory containing PDFs |
| `output_file` | `data/processed_chunks.json` | Output JSON file |
| `log_file` | `logs/processing.log` | Log file path |
| `chunk_size` | `512` | Tokens per chunk |
| `chunk_overlap` | `50` | Overlapping tokens |
| `use_ocr` | `False` | Enable OCR for scans |
| `batch_size` | `50` | Checkpoint frequency |

### Chunk Size Recommendations

- **512 tokens** (default): Good balance for most LLMs
- **256 tokens**: Smaller, more precise chunks
- **1024 tokens**: Larger context, fewer chunks

**For your collection (1,067 PDFs):**
- 512 tokens → ~15,000-20,000 chunks (estimated)
- 256 tokens → ~30,000-40,000 chunks
- 1024 tokens → ~7,500-10,000 chunks

---

## SLURM Job Submission

### Submit Job

```bash
sbatch jobs/process_pdfs.sb
```

### Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch job output
tail -f logs/pdf_processing_JOBID.out

# View processing log
tail -f logs/processing_JOBID.log
```

### Job Configuration

The job script requests:
- **Time**: 4 hours (adjust if needed)
- **CPUs**: 8 cores
- **Memory**: 32GB
- **Node**: Standard compute node

**Estimated runtime:** 1-2 hours for 1,067 PDFs

---

## Directory Structure

```
hpcc-llm-qa/
├── src/
│   ├── pdf_processor.py          # Main processor
│   └── test_processor.py         # Test script
├── data/
│   ├── pdfs/                     # Input PDFs (symlinks or copies)
│   │   ├── 1961-2006/           # 964 PDFs
│   │   └── 2007-2026/           # 103 PDFs
│   ├── processed_chunks.json     # Output chunks
│   ├── processing_metadata.json  # Statistics
│   └── test_output/             # Test results
├── jobs/
│   └── process_pdfs.sb          # SLURM job script
├── logs/
│   ├── processing.log           # Main log
│   └── pdf_processing_*.{out,err} # SLURM logs
├── requirements.txt             # Python dependencies
└── README_PDF_PROCESSOR.md      # This file
```

---

## Error Handling

### Common Issues

**1. No PDFs found**
```
Error: Input directory does not exist
```
**Solution:** Create symlinks or copy PDFs to `data/pdfs/`

**2. Missing dependencies**
```
ImportError: No module named 'PyPDF2'
```
**Solution:** `pip install -r requirements.txt`

**3. OCR failures**
```
Warning: OCR failed for file.pdf page 1
```
**Solution:** Install tesseract or disable OCR with `--use-ocr False`

**4. Memory errors**
```
MemoryError: Unable to allocate array
```
**Solution:** Request more memory in SLURM script (`#SBATCH --mem=64GB`)

### Logs

All errors are logged to:
- Console output
- `logs/processing.log`
- SLURM `.err` file (for job submission)

Check logs for:
- Failed PDFs
- Extraction errors
- OCR issues
- Encoding problems

---

## Performance Optimization

### For HPCC

1. **Use symbolic links** instead of copying PDFs (saves disk space)
2. **Submit as SLURM job** (don't run on login node)
3. **Adjust batch size** for checkpoint frequency
4. **Disable OCR** unless needed (much faster)
5. **Use scratch space** for cache if needed

### Memory Usage

- **Per PDF**: ~5-50MB (depends on size)
- **Total chunks in memory**: ~100-500MB
- **Recommended memory**: 32GB (allows headroom)

### Processing Speed

- **Without OCR**: ~5-10 PDFs/minute
- **With OCR**: ~1-2 PDFs/minute

**Estimated times for 1,067 PDFs:**
- Without OCR: 1-2 hours
- With OCR: 6-10 hours

---

## Next Steps

After processing PDFs:

1. **Verify output:**
   ```bash
   ls -lh data/processed_chunks.json
   python3 -c "import json; chunks = json.load(open('data/processed_chunks.json')); print(f'Total chunks: {len(chunks)}')"
   ```

2. **Create embeddings:**
   - Use `sentence-transformers` to create vector embeddings
   - Store in ChromaDB or FAISS

3. **Build QA system:**
   - Set up retrieval pipeline
   - Integrate with LLM (GPT, LLaMA, etc.)
   - Create query interface

---

## Example: Using Processed Chunks

```python
import json

# Load chunks
with open('data/processed_chunks.json', 'r') as f:
    chunks = json.load(f)

# Example: Find chunks from specific year range
chunks_1960s = [c for c in chunks if c['year_range'] == '1961-2006']
print(f"Chunks from 1961-2006: {len(chunks_1960s)}")

# Example: Get all chunks from a specific file
file_chunks = [c for c in chunks if 'cropsci1969' in c['source_file']]
for chunk in file_chunks:
    print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")

# Example: Analyze chunk distribution
from collections import Counter
year_distribution = Counter(c['year_range'] for c in chunks)
print("Chunks by year range:")
for year, count in year_distribution.items():
    print(f"  {year}: {count} chunks")
```

---

## Testing

### Run Test Suite

```bash
# Test on 5 sample PDFs
python3 src/test_processor.py
```

### Manual Testing

```python
from pdf_processor import PDFProcessor

# Test single PDF
processor = PDFProcessor(
    input_dir="data/pdfs/1961-2006",
    output_file="test_output.json",
    log_file="test.log"
)

# Process just one file
pdf_path = Path("data/pdfs/1961-2006/10.2135_cropsci1969.0011183X000900030028x.pdf")
chunks = processor.process_single_pdf(pdf_path)

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"\nChunk {chunk['chunk_id']}:")
    print(chunk['text'][:200])
```

---

## Troubleshooting

### Check Dependencies

```bash
python3 src/test_processor.py  # Includes dependency check
```

### Validate PDF Files

```bash
# Check if PDFs are readable
find data/pdfs -name "*.pdf" | head -5 | xargs -I {} file "{}"
```

### Check Disk Space

```bash
# Check quota
quota -s

# Check output size
du -sh data/
```

### Resume from Checkpoint

If processing fails, checkpoint files are saved:

```bash
# Find checkpoint
ls -lh data/checkpoint_*.json

# Resume by loading checkpoint and continuing
# (Manual merge required - or just reprocess)
```

---

## Support

For issues or questions:

1. Check logs in `logs/processing.log`
2. Review SLURM output: `logs/pdf_processing_JOBID.out`
3. Test with small sample first
4. Consult HPCC documentation: https://docs.icer.msu.edu/

---

## Credits

- Built for MSU HPCC Bean Lab
- Uses PyPDF2 for text extraction
- Tokenization via tiktoken (OpenAI)
- Designed for 1,067 agricultural research PDFs (1961-2026)
