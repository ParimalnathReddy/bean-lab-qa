# PDF Analysis and Organization Commands

## Current Status

Your PDFs are located in:
- **1961-2006**: `/mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/`
- **2007-2026**: `/mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/`

## Quick Stats

| Directory | PDF Count | Total Size |
|-----------|-----------|------------|
| 1961-2006 | **964** | **236M** |
| 2007-2026 | **103** | **111M** |
| **TOTAL** | **1,067** | **~347M** |

All PDFs are valid and readable (verified by `file` command).

---

## Individual Commands (Lightweight for Login Node)

### 1. Count Total PDFs

```bash
# Count in 1961-2006
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/ -type f -name "*.pdf" | wc -l

# Count in 2007-2026
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/ -type f -name "*.pdf" | wc -l

# Count both (total)
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/ -type f -name "*.pdf" | wc -l
```

### 2. Check Total File Size

```bash
# Size of 1961-2006
du -sh /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/

# Size of 2007-2026
du -sh /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/

# Total size
du -sh /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/

# Detailed breakdown
du -sh /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/*/
```

### 3. Show Sample Filenames (First 10)

```bash
# First 10 from 1961-2006
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/ -type f -name "*.pdf" | head -10

# First 10 from 2007-2026
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/ -type f -name "*.pdf" | head -10

# Show basenames only
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/ -type f -name "*.pdf" -exec basename {} \; | head -10
```

### 4. Check If PDFs Are Readable/Valid

```bash
# Validate 5 random PDFs from 1961-2006
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/ -type f -name "*.pdf" | shuf | head -5 | xargs -I {} file "{}"

# Validate 5 random PDFs from 2007-2026
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/ -type f -name "*.pdf" | shuf | head -5 | xargs -I {} file "{}"

# Check for corrupted PDFs (will show non-PDF files)
find /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/ -type f -name "*.pdf" | xargs -I {} file "{}" | grep -v "PDF document"
```

### 5. Copy All PDFs Preserving Organization

#### Option A: Manual Commands (Step by Step)

```bash
# Create destination directories
mkdir -p /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/1961-2006
mkdir -p /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/2007-2026

# Copy PDFs from 1961-2006
cp /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/*.pdf \
   /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/1961-2006/

# Copy PDFs from 2007-2026
cp /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/*.pdf \
   /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/2007-2026/

# Verify copy
find /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/ -type f -name "*.pdf" | wc -l
```

#### Option B: Using rsync (Safer, with Progress)

```bash
# Create destination directories
mkdir -p /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs

# Sync with rsync (preserves structure)
rsync -av --progress \
  /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/ \
  /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/1961-2006/

rsync -av --progress \
  /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/ \
  /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/2007-2026/
```

#### Option C: Using the Automated Script

```bash
# Run the interactive script
bash /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/scripts/analyze_and_copy_pdfs.sh
```

---

## Alternative: Create Symbolic Links (No Copy)

If you want to **avoid duplicating 347MB** of data, create symbolic links instead:

```bash
# Create destination directory
mkdir -p /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs

# Create symbolic links to original directories
ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006 \
      /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/1961-2006

ln -s /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026 \
      /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/2007-2026

# Verify links
ls -la /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/
```

**Advantages of symlinks:**
- No disk space used
- Always up-to-date with source
- Faster "copy"

**Disadvantages:**
- If source is deleted, links break
- Some programs may not follow symlinks

---

## Verify After Copy

```bash
# Count PDFs in destination
echo "1961-2006: $(find /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/1961-2006/ -type f -name "*.pdf" | wc -l)"
echo "2007-2026: $(find /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/2007-2026/ -type f -name "*.pdf" | wc -l)"

# Check total size
du -sh /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/

# List directory structure
tree -L 2 /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/
# or if tree not available:
find /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/ -type d
```

---

## Sample Filenames Found

### From 1961-2006 (DOI format):
- `10.2135_cropsci1969.0011183X000900030028x.pdf`
- `10.2135_cropsci1967.0011183X000700020009x.pdf`
- `10.1002_plr2.20289.pdf`

### From 2007-2026 (Newer publications):
- `10.3198_jpr2008.09.0562crg.pdf`
- `10.1002_plr2.20051.pdf`
- `10.3198_jpr2014.05.0028crc.pdf`

All files follow DOI naming convention (Digital Object Identifier).

---

## Recommended Next Steps

1. **Use symlinks** to avoid duplicating 347MB of data (unless you need a separate copy)
2. **Test PDF reading** with a small sample before processing all 1,067 PDFs
3. **Create metadata** about the PDFs (year ranges help organize chronologically)
4. **Set up vector database** to handle this corpus efficiently

---

## Quick Copy Command (One-liner)

```bash
mkdir -p /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/{1961-2006,2007-2026} && \
cp /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006/*.pdf \
   /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/1961-2006/ && \
cp /mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026/*.pdf \
   /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/2007-2026/ && \
echo "✅ Copied $(find /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/pdfs/ -type f -name "*.pdf" | wc -l) PDFs"
```
