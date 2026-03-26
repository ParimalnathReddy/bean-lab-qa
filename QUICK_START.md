# Quick Start Guide - PDF Processor

Process 1,067 agricultural research PDFs for LLM QA system.

---

## 🚀 Quick Setup (5 minutes)

```bash
cd /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa

# Run automated setup
bash scripts/setup_pdf_processing.sh
```

This will:
1. ✅ Check conda environment
2. ✅ Install dependencies
3. ✅ Set up PDF directories
4. ✅ Run a test on 5 sample PDFs

---

## 📋 Manual Setup

### 1. Install Dependencies

```bash
conda activate bean_llm
pip install -r requirements.txt
```

### 2. Link PDFs

```bash
mkdir -p data/pdfs
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006 data/pdfs/1961-2006
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026 data/pdfs/2007-2026
```

### 3. Test

```bash
python3 src/test_processor.py
```

---

## ▶️ Run Processing

### Option 1: SLURM Job (Recommended)

```bash
sbatch jobs/process_pdfs.sb

# Monitor
squeue -u $USER
tail -f logs/pdf_processing_*.out
```

### Option 2: Interactive

```bash
python3 src/pdf_processor.py
```

---

## 📊 Expected Output

After processing completes:

```
data/
├── processed_chunks.json        # ~15,000-20,000 chunks
└── processing_metadata.json     # Statistics
```

**Sample chunk:**
```json
{
  "text": "Bean research shows...",
  "source_file": "10.2135_cropsci1969...pdf",
  "year_range": "1961-2006",
  "chunk_id": 0,
  "page_number": 1,
  "token_count": 487
}
```

---

## ⚙️ Customization

### Change Chunk Size

```bash
python3 src/pdf_processor.py --chunk-size 256  # Smaller chunks
python3 src/pdf_processor.py --chunk-size 1024 # Larger chunks
```

### Enable OCR (for scanned PDFs)

```bash
pip install pytesseract pdf2image
python3 src/pdf_processor.py --use-ocr
```

---

## 🔍 Verify Results

```bash
# Check output exists
ls -lh data/processed_chunks.json

# Count chunks
python3 -c "import json; print(f'Chunks: {len(json.load(open(\"data/processed_chunks.json\")))}')"

# View sample
python3 -c "import json; print(json.load(open('data/processed_chunks.json'))[0])"
```

---

## ⏱️ Estimated Times

- **Setup**: 5 minutes
- **Test (5 PDFs)**: 1 minute
- **Full processing (1,067 PDFs)**: 1-2 hours
- **With OCR**: 6-10 hours

---

## 📁 Files Created

| File | Description |
|------|-------------|
| [src/pdf_processor.py](src/pdf_processor.py) | Main processor |
| [src/test_processor.py](src/test_processor.py) | Test script |
| [jobs/process_pdfs.sb](jobs/process_pdfs.sb) | SLURM job |
| [requirements.txt](requirements.txt) | Dependencies |
| [README_PDF_PROCESSOR.md](README_PDF_PROCESSOR.md) | Full documentation |

---

## ❓ Troubleshooting

**No PDFs found?**
```bash
# Check if PDFs exist
ls data/pdfs/*/
```

**Module not found?**
```bash
pip install -r requirements.txt
```

**Out of memory?**
```bash
# Edit jobs/process_pdfs.sb
#SBATCH --mem=64GB  # Increase memory
```

**Need help?**
- Read [README_PDF_PROCESSOR.md](README_PDF_PROCESSOR.md)
- Check logs: `logs/processing.log`
- HPCC docs: https://docs.icer.msu.edu/

---

## 🎯 Next Steps

After processing:

1. **Create embeddings** (sentence-transformers)
2. **Set up vector database** (ChromaDB)
3. **Build QA interface** (LangChain + LLM)

---

**Ready to start?**

```bash
bash scripts/setup_pdf_processing.sh
```
