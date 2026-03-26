# Ollama Quick Start - MSU HPCC

Get Ollama running on HPCC in 3 steps.

---

## ⚡ Quick Setup

### Step 1: Initial Setup
```bash
bash scripts/setup_ollama.sh
```

### Step 2: Download Model (on compute node!)
```bash
# Get compute node
salloc --mem=16GB --time=1:00:00

# Download llama3.1:8b (~4.7GB, takes 5-15 min)
bash scripts/download_llama_model.sh

# Exit
exit
```

### Step 3: Test
```bash
# Get compute node with GPU
salloc --mem=16GB --gpus=1 --time=1:00:00

# Run tests
bash scripts/test_ollama.sh

# Exit
exit
```

---

## 📋 Key Information

**Module**: `Ollama/0.15.5`

**Model Storage**:
- Location: `hpcc-llm-qa/models/ollama/`
- Model: llama3.1:8b (~4.7GB)

**Environment Variables**:
```bash
export OLLAMA_MODELS="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/models/ollama"
export OLLAMA_HOST="0.0.0.0:11434"
```

---

## 🚀 Usage

### Interactive
```bash
# 1. Get compute node
salloc --mem=16GB --gpus=1 --time=2:00:00

# 2. Start server
bash scripts/start_ollama.sh

# 3. Use Ollama
ollama run llama3.1:8b "Your question here"

# 4. Cleanup
pkill ollama
exit
```

### SLURM Job
```bash
sbatch jobs/run_ollama_qa.sb
```

---

## ⚠️ Important Rules

1. **Never run on login node** - Use `salloc` or `sbatch`
2. **Models in research space** - Not home directory (quota!)
3. **Set OLLAMA_HOST** - For SLURM job access
4. **Kill server when done** - `pkill ollama`

---

## 🔍 Test Commands

```bash
# Check module
module load Ollama/0.15.5
ollama --version

# List models
ollama list

# Test inference
ollama run llama3.1:8b "What is 2+2?"
```

---

## 📖 Full Documentation

See [README_OLLAMA.md](README_OLLAMA.md) for complete guide.

---

## 🆘 Quick Troubleshooting

**Model won't download?**
→ Check internet: `ping ollama.com`

**Server won't start?**
→ Kill existing: `pkill ollama`

**Out of memory?**
→ Request more: `salloc --mem=32GB`

**On login node?**
→ Get compute node: `salloc --mem=16GB --gpus=1`
