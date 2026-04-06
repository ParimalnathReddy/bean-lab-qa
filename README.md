# HPCC LLM-Based Document QA System

**🟢 LIVE DEMO (Production QA System):** [https://huggingface.co/spaces/Parimalanath/bean-lab-qa](https://huggingface.co/spaces/Parimalanath/bean-lab-qa)

## 📚 Complete Documentation Index

Welcome to the Bean Lab PDF Processing System for MSU HPCC. This README serves as your central navigation hub for all documentation.

---

## 🚀 Quick Navigation

### **New Users Start Here** ⭐
1. Read [QUICK_START.md](QUICK_START.md) - Get running in 5 minutes
2. Run `bash scripts/setup_pdf_processing.sh` - Automated setup
3. Test with `python3 src/test_processor.py` - Verify installation

### **Developers**
1. Read [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) - Complete code walkthrough
2. Review [src/pdf_processor_commented.py](src/pdf_processor_commented.py) - Line-by-line comments
3. Check [Function Reference](#function-reference) below

### **HPCC Users**
1. Read [config/HPCC_SETUP_GUIDE.md](config/HPCC_SETUP_GUIDE.md) - Module loading, SLURM jobs
2. Review [jobs/process_pdfs.sb](jobs/process_pdfs.sb) - Batch processing
3. See [HPCC Best Practices](#hpcc-best-practices) below

### **Researchers**
1. Read [README_PDF_PROCESSOR.md](README_PDF_PROCESSOR.md) - Full user guide
2. Check [PDF_ANALYSIS.md](PDF_ANALYSIS.md) - Dataset statistics
3. See [Output Format](#output-format) below

---

## 📖 Documentation Map

### Core Documentation

| Document | Purpose | Audience | Read Time |
|----------|---------|----------|-----------|
| **[QUICK_START.md](QUICK_START.md)** | Get started in 5 minutes | All users | 5 min |
| **[README_PDF_PROCESSOR.md](README_PDF_PROCESSOR.md)** | Complete user guide | Users | 15 min |
| **[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)** | Code architecture & walkthrough | Developers | 30 min |
| **[PDF_ANALYSIS.md](PDF_ANALYSIS.md)** | Dataset analysis | Researchers | 10 min |

### HPCC-Specific

| Document | Purpose |
|----------|---------|
| **[config/HPCC_SETUP_GUIDE.md](config/HPCC_SETUP_GUIDE.md)** | HPCC modules, conda, SLURM |
| **[config/module_setup.sh](config/module_setup.sh)** | Module loading script |

### Code Files

| File | Description | Lines | Comments |
|------|-------------|-------|----------|
| **[src/pdf_processor.py](src/pdf_processor.py)** | Main processor (production) | 595 | Standard |
| **[src/pdf_processor_commented.py](src/pdf_processor_commented.py)** | Fully documented version | 800+ | Every line |
| **[src/test_processor.py](src/test_processor.py)** | Test suite | 200+ | Detailed |

### Setup Scripts

| Script | Purpose |
|--------|---------|
| **[scripts/setup_pdf_processing.sh](scripts/setup_pdf_processing.sh)** | Automated setup |
| **[scripts/analyze_and_copy_pdfs.sh](scripts/analyze_and_copy_pdfs.sh)** | PDF organization |

### Job Scripts

| Script | Purpose |
|--------|---------|
| **[jobs/process_pdfs.sb](jobs/process_pdfs.sb)** | SLURM batch job |

---

## 📊 Project Overview

### Dataset

- **Total PDFs**: 1,067 documents
- **Size**: ~347MB
- **Time Range**: 1961-2026
- **Subject**: Agricultural/bean research
- **Format**: Scientific papers (DOI-based filenames)

### Processing Pipeline

```
1,067 PDFs → Text Extraction → Chunking → 15,000-20,000 Chunks → Vector Database
```

### Output

- **Chunks**: 15,000-20,000 (estimated)
- **Chunk Size**: 512 tokens
- **Overlap**: 50 tokens
- **Metadata**: Source, year, page, token count

---

## 🎯 Common Tasks

### Installation

```bash
# Activate conda environment
conda activate bean_llm

# Install dependencies
pip install -r requirements.txt
```

### Setup PDFs

```bash
# Create symbolic links (recommended)
mkdir -p data/pdfs
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006 data/pdfs/1961-2006
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026 data/pdfs/2007-2026
```

### Testing

```bash
# Run test on 5 sample PDFs
python3 src/test_processor.py
```

### Processing

```bash
# Option 1: Submit SLURM job (recommended)
sbatch jobs/process_pdfs.sb

# Option 2: Run interactively
python3 src/pdf_processor.py
```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/processing.log
tail -f logs/pdf_processing_*.out
```

### Verification

```bash
# Check output exists
ls -lh data/processed_chunks.json

# Count chunks
python3 -c "import json; print(f'Chunks: {len(json.load(open(\"data/processed_chunks.json\")))}')"

# View sample chunk
python3 -c "import json; import pprint; pprint.pprint(json.load(open('data/processed_chunks.json'))[0])"
```

---

## 🔧 Configuration Options

### Command-Line Arguments

```bash
python3 src/pdf_processor.py \
    --input-dir data/pdfs \              # PDF directory
    --output-file data/output.json \     # Output file
    --log-file logs/processing.log \     # Log file
    --chunk-size 512 \                   # Tokens per chunk
    --chunk-overlap 50 \                 # Overlap tokens
    --use-ocr \                          # Enable OCR
    --batch-size 50                      # Checkpoint frequency
```

### Environment Variables

```bash
# Set in ~/.bashrc or job script
export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/scratch/$USER/llm_cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/mnt/scratch/$USER/llm_cache/sentence_transformers
```

---

## 📁 Directory Structure

```
hpcc-llm-qa/
├── README.md                         # This file - START HERE
├── QUICK_START.md                    # 5-minute quick start
├── README_PDF_PROCESSOR.md           # Complete user guide
├── CODE_DOCUMENTATION.md             # Code architecture
├── PDF_ANALYSIS.md                   # Dataset analysis
│
├── src/
│   ├── pdf_processor.py              # Main processor
│   ├── pdf_processor_commented.py    # Fully commented version
│   └── test_processor.py             # Test suite
│
├── data/
│   ├── pdfs/                         # Input PDFs (symlinks)
│   │   ├── 1961-2006/  (964 PDFs)
│   │   └── 2007-2026/  (103 PDFs)
│   ├── processed_chunks.json         # Output chunks
│   ├── processing_metadata.json      # Statistics
│   └── checkpoint_*.json             # Intermediate saves
│
├── jobs/
│   └── process_pdfs.sb               # SLURM job script
│
├── logs/
│   ├── processing.log                # Main log
│   └── pdf_processing_*.{out,err}    # SLURM logs
│
├── scripts/
│   ├── setup_pdf_processing.sh       # Setup automation
│   └── analyze_and_copy_pdfs.sh      # PDF organization
│
├── config/
│   ├── HPCC_SETUP_GUIDE.md          # HPCC documentation
│   └── module_setup.sh               # Module loading
│
└── requirements.txt                  # Python dependencies
```

---

## 📝 Output Format

### Chunk Structure

Each chunk is a JSON object with:

```json
{
  "text": "Bean breeding programs have demonstrated...",
  "source_file": "10.2135_cropsci1969.0011183X000900030028x.pdf",
  "year_range": "1961-2006",
  "chunk_id": 0,
  "page_number": 1,
  "token_count": 487
}
```

### Metadata Structure

```json
{
  "processing_date": "2026-03-21T14:30:00.123456",
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

## 🖥️ HPCC Best Practices

### 1. **Always Use SLURM for Processing**
```bash
# DON'T run on login node:
# python3 src/pdf_processor.py  ❌

# DO submit as job:
sbatch jobs/process_pdfs.sb  ✅
```

### 2. **Set Cache Directories to Scratch**
```bash
export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface
```

### 3. **Use Symbolic Links for PDFs**
```bash
# Saves 347MB of disk space
ln -s /source/path data/pdfs/directory
```

### 4. **Monitor Disk Quota**
```bash
quota -s  # Check home directory
du -sh /mnt/research/BeanLab/Parimal  # Check research storage
```

### 5. **Load Modules or Use Conda**
```bash
# Option 1: Conda (recommended)
conda activate bean_llm

# Option 2: Modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

---

## 🔍 Function Reference

### Main Functions

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `find_all_pdfs()` | None | List[Path] | Find PDFs recursively |
| `extract_text_from_pdf(path)` | Path | (str, int) | Extract text & page count |
| `extract_year_range(path)` | Path | str | Get year metadata |
| `count_tokens(text)` | str | int | Count tokens |
| `chunk_text(text, meta...)` | str, ... | List[Dict] | Split into chunks |
| `process_single_pdf(path)` | Path | List[Dict] | Process one PDF |
| `process_all_pdfs()` | None | List[Dict] | Process all PDFs |
| `save_chunks(chunks)` | List[Dict] | None | Save to JSON |
| `run()` | None | List[Dict] | Main pipeline |

See [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) for detailed API reference.

---

## ⚡ Performance

### Expected Processing Times

| Configuration | Time for 1,067 PDFs |
|---------------|---------------------|
| Without OCR (default) | 1-2 hours |
| With OCR | 6-10 hours |

### Resource Requirements

| Resource | Recommended |
|----------|-------------|
| CPUs | 8 cores |
| Memory | 32GB |
| Time Limit | 4 hours |
| Storage | 500MB (output) |

---

## 🐛 Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| `No module named 'PyPDF2'` | `pip install -r requirements.txt` |
| `No PDFs found` | Check `data/pdfs/` directory exists |
| `Permission denied` | Check file permissions |
| `Out of memory` | Request more memory in SLURM |
| `Job timeout` | Increase time limit in SLURM script |

### Debug Commands

```bash
# Check dependencies
python3 src/test_processor.py

# Find PDFs manually
find data/pdfs -name "*.pdf" | wc -l

# Test single PDF
python3 -c "from pathlib import Path; from src.pdf_processor import PDFProcessor; p = PDFProcessor('data/pdfs', 'test.json', 'test.log'); p.process_single_pdf(Path('data/pdfs/1961-2006/yourfile.pdf'))"

# Check logs
tail -f logs/processing.log
```

---

## 📚 Additional Resources

### External Documentation

- **PyPDF2**: https://pypdf2.readthedocs.io/
- **tiktoken**: https://github.com/openai/tiktoken
- **MSU HPCC**: https://docs.icer.msu.edu/
- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://docs.trychroma.com/

### Related Projects

- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract

---

## 🎓 Learning Path

### For Complete Beginners

1. Read [QUICK_START.md](QUICK_START.md)
2. Run `bash scripts/setup_pdf_processing.sh`
3. Test with `python3 src/test_processor.py`
4. Read [README_PDF_PROCESSOR.md](README_PDF_PROCESSOR.md)
5. Submit job: `sbatch jobs/process_pdfs.sb`

### For Developers

1. Read [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)
2. Review [src/pdf_processor_commented.py](src/pdf_processor_commented.py)
3. Understand chunking algorithm
4. Modify configuration as needed
5. Run tests before production

### For Researchers

1. Read [PDF_ANALYSIS.md](PDF_ANALYSIS.md)
2. Understand output format
3. Process PDFs
4. Analyze chunk distribution
5. Build QA system on top

---

## 🚦 Next Steps

After PDF processing completes:

### 1. **Create Embeddings**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([chunk['text'] for chunk in chunks])
```

### 2. **Set Up Vector Database**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("bean_research")
collection.add(
    documents=[chunk['text'] for chunk in chunks],
    metadatas=[{k:v for k,v in chunk.items() if k != 'text'} for chunk in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
```

### 3. **Build QA Interface**
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

answer = qa_chain.run("What are the main bean breeding techniques?")
```

---

## 📞 Support

### Getting Help

1. **Check documentation** (you're reading it!)
2. **Review logs**: `logs/processing.log`
3. **Run test**: `python3 src/test_processor.py`
4. **Check HPCC docs**: https://docs.icer.msu.edu/

### Reporting Issues

Include:
- Error message
- Log file excerpt
- Command used
- Environment (conda list, module list)

---

## 📜 Project Information

- **Project**: HPCC LLM Document QA System
- **Lab**: MSU Bean Lab
- **Dataset**: 1,067 agricultural research PDFs (1961-2026)
- **Purpose**: Enable LLM-based question answering over research corpus
- **Version**: 1.0
- **Last Updated**: 2026-03-21

---

## ✅ Checklist

### Setup Checklist
- [ ] Read QUICK_START.md
- [ ] Conda environment created
- [ ] Dependencies installed
- [ ] PDFs linked/copied to data/pdfs/
- [ ] Test passed
- [ ] SLURM job submitted

### Processing Checklist
- [ ] Job running
- [ ] Logs monitored
- [ ] Checkpoints saving
- [ ] Job completed
- [ ] Output verified

### Next Steps Checklist
- [ ] Embeddings created
- [ ] Vector database set up
- [ ] QA system built
- [ ] System tested

---

**🚀 Ready to start? → [QUICK_START.md](QUICK_START.md)**
